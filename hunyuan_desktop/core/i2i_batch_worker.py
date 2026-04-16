"""QThread worker for image-to-image batch generation.

Handles batch I2I processing with:
- Global reference images or per-prompt image overrides
- Bot task / think mode selection
- Round-robin iteration through prompts/styles/variations
- Wildcard processing and Ollama enhancement
- JSON sidecar saving
"""

import time
import random
import re
import gc
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from models.i2i_batch_config import I2IBatchConfig


class I2IBatchWorker(QThread):
    """Executes image-to-image batch generation."""

    progress = Signal(int, int, str)       # current, total, status_text
    image_ready = Signal(str)               # image_path
    cot_received = Signal(str)              # chain-of-thought text
    completed = Signal(str, int)            # batch_dir, total_count
    stopped = Signal(str, int)              # batch_dir, count_so_far
    error = Signal(str)                     # error_msg

    def __init__(self, config: I2IBatchConfig):
        super().__init__()
        self.config = config
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run_batch()
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

    def _run_batch(self):
        from core.model_manager import get_model, is_model_loaded
        from ui.state import get_state
        from ui.constants import OUTPUT_DIR, DEFAULT_STYLE_PRESETS

        import torch

        state = get_state()
        config = self.config

        if not is_model_loaded():
            # Model not ready yet — wait for it if a load is in progress
            if state.model_load_lock.locked():
                self.progress.emit(0, 0, "Waiting for model to finish loading...")
                print("[I2I BATCH] Model loading in progress, waiting...")
                poll_count = 0
                while not is_model_loaded() and not self._stop_requested:
                    time.sleep(1)
                    poll_count += 1
                    if poll_count % 5 == 0:
                        self.progress.emit(0, 0, f"Waiting for model... ({poll_count}s)")
                    if not state.model_load_lock.locked() and not is_model_loaded():
                        self.error.emit("Model load failed. Check the log and try again.")
                        return
                if self._stop_requested:
                    self.stopped.emit("", 0)
                    return
                print(f"[I2I BATCH] Model ready after {poll_count}s wait")
            else:
                self.error.emit("Model not loaded. Load an Instruct or Distil model first.")
                return

        if state.model_type not in ("instruct", "distil", "nf4", "distil_nf4", "deepgen", "firered", "instruct_int8", "distil_int8"):
            self.error.emit(
                f"I2I batch requires Instruct, Distil, DeepGen, or FireRed model (not Base). "
                f"Current model: {state.model_type}"
            )
            return

        # Pre-scan folder slots so we can cycle through them per generation.
        # folder_images[slot_idx] is a sorted list of paths (or [] if no folder).
        IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        folder_images = [[], [], []]
        slot_folders = list(config.global_image_folders or []) + ["", "", ""]
        slot_singles = list(config.global_image_slots or []) + ["", "", ""]
        for i in range(3):
            folder = slot_folders[i]
            if folder:
                p = Path(folder)
                if p.is_dir():
                    folder_images[i] = sorted(
                        str(f) for f in p.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                    )
                if folder_images[i]:
                    print(f"[I2I BATCH] Slot {i+1} folder: {folder} "
                          f"({len(folder_images[i])} images, will cycle)")

        has_any_slot_image = any(folder_images) or any(slot_singles[:3])
        if (not config.global_images
                and not config.prompt_image_overrides
                and not has_any_slot_image):
            self.error.emit("No reference images provided. Add at least one image.")
            return

        def _images_for_generation(prompt_idx: int, gen_index: int) -> list:
            """Build the per-generation reference image list.

            Per-prompt [img:] overrides take precedence. Otherwise build from
            slot folders (cycling) and slot singles. Falls back to legacy
            global_images if no slot data is present.
            """
            if prompt_idx in config.prompt_image_overrides:
                return list(config.prompt_image_overrides[prompt_idx])
            built = []
            for slot_i in range(3):
                if folder_images[slot_i]:
                    imgs = folder_images[slot_i]
                    built.append(imgs[gen_index % len(imgs)])
                elif slot_singles[slot_i]:
                    built.append(slot_singles[slot_i])
            if built:
                return built
            return list(config.global_images)

        # Create batch output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c for c in config.batch_name[:30] if c.isalnum() or c in " -_"
        ).strip().replace(" ", "_") or "i2i_batch"
        batch_dir = OUTPUT_DIR / "batches" / f"{safe_name}_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        model = get_model()
        if model is None:
            self.error.emit("Model not available")
            return

        # Initialize Ollama / LM Studio enhancer (with lazy init)
        ollama_model_name = None
        print(f"\n{'='*60}")
        print(f"[I2I BATCH OLLAMA] Enhancement Status Check:")
        print(f"[I2I BATCH OLLAMA]   config.enhance: {config.enhance}")
        print(f"[I2I BATCH OLLAMA]   ollama_available: {state.ollama_available}")
        print(f"[I2I BATCH OLLAMA]   ollama_enhancer exists: {state.ollama_enhancer is not None}")

        # Lazy init: if enhance requested but enhancer not ready, set it up now
        if config.enhance and (not state.ollama_available or state.ollama_enhancer is None):
            print(f"[I2I BATCH OLLAMA]   Lazy-initializing enhancer...")
            self.progress.emit(0, config.total_images(), "Initializing prompt enhancer...")
            try:
                from ollama_prompts import PromptEnhancer
                state.ollama_enhancer = PromptEnhancer()
                state.ollama_available = True
                print(f"[I2I BATCH OLLAMA] ✓ Enhancer initialized")
            except Exception as e:
                print(f"[I2I BATCH OLLAMA] ✗ Failed to init enhancer: {e}")

        if config.enhance and state.ollama_available and state.ollama_enhancer:
            ollama_model_name = config.ollama_model
            if any(size in ollama_model_name.lower() for size in ("30b", "24b", "20b")):
                ollama_model_name = "qwen2.5:7b-instruct"
            print(f"[I2I BATCH OLLAMA] ✓ ENHANCEMENT ENABLED — model: {ollama_model_name}")
        elif config.enhance:
            print(f"[I2I BATCH OLLAMA] ✗ ENHANCEMENT REQUESTED BUT UNAVAILABLE")
        print(f"{'='*60}\n")

        total_images = config.total_images()
        total_count = 0
        gen_index = 0  # increments per generation, used to cycle through folders
        generated_images = []

        prompts = config.prompts
        styles = config.styles if config.styles else ["None"]

        # Round-robin: variation -> prompt -> style -> images_per_combo
        for iteration_idx in range(config.variations_per_prompt):
            if self._stop_requested:
                break

            for prompt_idx, prompt_text in enumerate(prompts):
                if self._stop_requested:
                    break

                for style in styles:
                    if self._stop_requested:
                        break

                    for img_idx in range(config.images_per_combo):
                        if self._stop_requested:
                            break

                        # Resolve per-generation reference images (cycles folders)
                        img_paths = _images_for_generation(prompt_idx, gen_index)
                        if not img_paths:
                            self.progress.emit(
                                total_count, total_images,
                                f"Skipping prompt {prompt_idx+1} (no images)"
                            )
                            gen_index += 1
                            continue

                        # Build prompt with style suffix
                        full_prompt = prompt_text
                        style_suffix = DEFAULT_STYLE_PRESETS.get(style, "")
                        if style_suffix:
                            full_prompt += style_suffix

                        # Generate seed
                        if config.random_seeds:
                            current_seed = random.randint(0, 2**31 - 1)
                        else:
                            current_seed = (
                                iteration_idx * 10000
                                + prompt_idx * 1000
                                + img_idx
                            )

                        # Wildcard processing
                        if state.wildcard_available and state.wildcard_manager:
                            try:
                                full_prompt = state.wildcard_manager.process_prompt(
                                    full_prompt
                                )
                            except Exception:
                                pass

                        # Ollama enhancement
                        print(f"[I2I BATCH] Ollama enhancement check: enhance={config.enhance}, available={state.ollama_available}, enhancer_exists={state.ollama_enhancer is not None}")
                        if (config.enhance and state.ollama_available
                                and state.ollama_enhancer):
                            print(f"[I2I BATCH] ⚠ ENHANCING PROMPT with Ollama model: {ollama_model_name}")
                            try:
                                actual_length = config.ollama_length
                                actual_complexity = config.ollama_complexity
                                if str(actual_length).lower() == "random":
                                    actual_length = random.choice([
                                        "minimal", "short", "medium", "long",
                                        "detailed", "cinematic", "experimental"
                                    ])
                                if str(actual_complexity).lower() == "random":
                                    actual_complexity = random.choice([
                                        "simple", "moderate", "detailed",
                                        "complex", "cinematic", "experimental"
                                    ])

                                enhanced = state.ollama_enhancer.enhance(
                                    full_prompt,
                                    length=str(actual_length),
                                    complexity=str(actual_complexity),
                                    model=ollama_model_name,
                                    max_length=config.max_prompt_length or 0,
                                )

                                # Clean LLM artifacts
                                clean = enhanced
                                for prefix in [
                                    "Here is the enhanced prompt:",
                                    "Enhanced version:",
                                    "Here's the rewritten description:",
                                ]:
                                    if clean.startswith(prefix):
                                        clean = clean[len(prefix):].strip()
                                if "ORIGINAL PROMPT:" in clean:
                                    clean = clean.split("ORIGINAL PROMPT:")[-1].strip()

                                full_prompt = clean
                            except Exception:
                                pass

                        status = (
                            f"Iter {iteration_idx+1}/{config.variations_per_prompt} | "
                            f"#{total_count+1}/{total_images}: "
                            f"{prompt_text[:25]}... ({style})"
                        )
                        self.progress.emit(total_count, total_images, status)

                        try:
                            start_time = time.time()

                            # I2I generation - pass paths directly
                            img_arg = (
                                img_paths if len(img_paths) > 1
                                else img_paths[0]
                            )
                            print(f"[I2I BATCH] Generating image {total_count+1}/{total_images}")
                            print(f"[I2I BATCH] Input images: {img_arg}")
                            print(f"[I2I BATCH] Prompt: {full_prompt[:80]}...")
                            print(f"[I2I BATCH] bot_task={config.bot_task}, guidance={config.guidance_scale}")

                            try:
                                cot_text = None
                                image = None
                                if state.model_type in ("instruct_int8", "distil_int8"):
                                    from core.model_manager import generate_int8, is_model_loaded
                                    from ui.constants import is_int8_model
                                    def _progress_cb(msg):
                                        self.progress.emit(total_count, total_images, msg)
                                    int8_result = generate_int8(
                                        model=model,
                                        prompt=full_prompt,
                                        seed=current_seed,
                                        image_size="auto",
                                        bot_task=config.bot_task,
                                        steps=config.get_steps(state.model_type),
                                        guidance_scale=config.guidance_scale,
                                        i2i_images=img_paths,
                                        progress_callback=_progress_cb,
                                    )
                                    # generate_int8 returns (image, seed) or (image, seed, cot_text)
                                    if len(int8_result) == 3:
                                        image, current_seed, cot_text = int8_result
                                    else:
                                        image, current_seed = int8_result
                                    # Re-fetch model in case OOM recovery deleted it
                                    if not is_model_loaded():
                                        self.error.emit("Model unloaded during VAE OOM recovery. Reload to continue.")
                                        self._stop_requested = True
                                        break
                                    model = get_model()
                                elif state.model_type == "firered":
                                    from core.model_manager import generate_firered
                                    from ui.constants import get_default_guidance
                                    firered_cfg = config.guidance_scale
                                    if firered_cfg == 5.0:
                                        firered_cfg = get_default_guidance("firered")
                                    # FireRed accepts up to 3 images as file paths
                                    # Use None for width/height to auto-detect from input
                                    result, current_seed = generate_firered(
                                        model=model,
                                        prompt=full_prompt,
                                        seed=current_seed,
                                        width=None,
                                        height=None,
                                        steps=config.get_steps(state.model_type),
                                        true_cfg_scale=firered_cfg,
                                        src_image_path=img_paths,
                                    )
                                elif state.model_type == "deepgen":
                                    from core.model_manager import generate_deepgen
                                    from ui.constants import get_default_guidance
                                    src_path = img_paths[0] if isinstance(img_arg, list) else img_arg
                                    deepgen_guidance = config.guidance_scale
                                    if deepgen_guidance == 5.0:
                                        deepgen_guidance = get_default_guidance("deepgen")
                                    result, current_seed = generate_deepgen(
                                        model=model,
                                        prompt=full_prompt,
                                        seed=current_seed,
                                        width=512,
                                        height=512,
                                        steps=config.get_steps(state.model_type),
                                        guidance_scale=deepgen_guidance,
                                        src_image_path=src_path,
                                    )
                                else:
                                    # Use think_recaption for I2I when bot_task is "image"
                                    # to let the model analyze the input image properly
                                    i2i_task = config.bot_task
                                    if i2i_task == "image":
                                        i2i_task = "think_recaption"
                                    # Use quality preset from config
                                    i2i_steps = config.get_steps(state.model_type)
                                    result = model.generate_image(
                                        prompt=full_prompt,
                                        seed=current_seed,
                                        image=img_arg,
                                        image_size="auto",
                                        use_system_prompt="en_vanilla",
                                        bot_task=i2i_task,
                                        infer_align_image_size=True,
                                        diff_infer_steps=i2i_steps,
                                        diff_guidance_scale=config.guidance_scale,
                                    )
                                print(f"[I2I BATCH] ✓ generation returned successfully")
                            except Exception as gen_error:
                                print(f"[I2I BATCH] ✗ generate_image() threw exception: {gen_error}")
                                import traceback
                                print(traceback.format_exc())
                                raise

                            gen_time = time.time() - start_time
                            print(f"[I2I BATCH] Generation took {gen_time:.2f}s")

                            # Handle result for non-INT8/DeepGen paths
                            # (INT8 and DeepGen already set image/cot_text directly)
                            if image is None:
                                if isinstance(result, tuple):
                                    cot_text, outputs = result
                                    print(f"[I2I BATCH] Tuple result - CoT: {len(cot_text) if cot_text else 0} chars, Outputs: {type(outputs)}, Count: {len(outputs) if outputs else 0}")
                                    image = outputs[0] if outputs else None
                                    if outputs:
                                        print(f"[I2I BATCH] First output type: {type(outputs[0])}")
                                else:
                                    image = result
                                    print(f"[I2I BATCH] Direct image result: {type(image)}")

                            if cot_text:
                                print(f"[I2I BATCH] CoT text length: {len(cot_text)} chars")
                                print(f"[I2I BATCH] CoT preview: {cot_text[:200]}...")
                                print(f"[I2I BATCH] drop_think={config.drop_think}, will emit={not config.drop_think}")
                                if not config.drop_think:
                                    self.cot_received.emit(cot_text)
                                    print(f"[I2I BATCH] ✓ CoT text emitted to UI")
                            else:
                                print(f"[I2I BATCH] No CoT text generated (cot_text is None/empty)")

                            if image:
                                print(f"[I2I BATCH] ✓ Image generated successfully, saving...")
                            else:
                                print(f"[I2I BATCH] ✗ ERROR: image is None! Nothing to save.")

                            if image:
                                filename = (
                                    f"{total_count:04d}_{current_seed}_"
                                    f"{style.replace(' ', '_')[:20]}.png"
                                )
                                filepath = batch_dir / filename
                                image.save(str(filepath))

                                # Save JSON sidecar
                                img_config = {
                                    "prompt": prompt_text,
                                    "full_prompt": full_prompt,
                                    "style": style,
                                    "guidance_scale": config.guidance_scale,
                                    "seed": current_seed,
                                    "generation_time": gen_time,
                                    "batch_name": config.batch_name,
                                    "batch_iteration": iteration_idx + 1,
                                    "batch_total_iterations": config.variations_per_prompt,
                                    "model_type": state.model_type,
                                    "bot_task": config.bot_task,
                                    "drop_think": config.drop_think,
                                    "input_images": img_paths,
                                    "use_ollama": config.enhance,
                                    "ollama_model": (
                                        config.ollama_model if config.enhance
                                        else None
                                    ),
                                }
                                if cot_text:
                                    img_config["cot_text"] = cot_text

                                json_path = filepath.with_suffix(".json")
                                with open(json_path, "w") as f:
                                    json.dump(
                                        img_config, f, indent=2, default=str
                                    )

                                generated_images.append(str(filepath))
                                total_count += 1
                                self.image_ready.emit(str(filepath))

                                gc.collect()
                                torch.cuda.empty_cache()

                        except Exception as e:
                            import traceback
                            error_details = traceback.format_exc()
                            print(f"[I2I BATCH] ✗ EXCEPTION during generation:")
                            print(error_details)
                            self.progress.emit(
                                total_count, total_images, f"Error: {e}"
                            )

                        gen_index += 1

        # Write batch manifest
        manifest = {
            "batch_name": config.batch_name,
            "batch_type": "i2i",
            "total_images": total_count,
            "settings": config.to_dict(),
            "images": [str(p) for p in generated_images],
        }
        manifest_path = batch_dir / "batch_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        if self._stop_requested:
            self.stopped.emit(str(batch_dir), total_count)
        else:
            self.completed.emit(str(batch_dir), total_count)
