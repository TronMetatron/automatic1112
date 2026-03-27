"""QThread worker for batch image generation.

Mirrors the batch logic from hunyuan_ui_v2.py start_batch_generation():
- Round-robin iteration through themes/styles/variations
- 3-phase starred wildcard processing
- Ollama enhancement with protected tokens
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

from models.batch_config import BatchConfig


class BatchWorker(QThread):
    """Executes batch generation with round-robin iteration."""

    progress = Signal(int, int, str)       # current, total, status_text
    image_ready = Signal(str)               # image_path
    cot_received = Signal(str)              # chain-of-thought text
    completed = Signal(str, int)            # batch_dir, total_count
    stopped = Signal(str, int)              # batch_dir, count_so_far
    error = Signal(str)                     # error_msg
    config_updated = Signal()               # emitted when config changes are applied

    def __init__(self, config: BatchConfig):
        super().__init__()
        self.config = config
        self._stop_requested = False
        self._pending_themes = None  # Thread-safe pending theme update
        self._pending_negative = None  # Thread-safe pending negative prompt update

    def request_stop(self):
        self._stop_requested = True

    def update_config(self, themes: list, negative_prompt: str = None):
        """Thread-safe method to update themes/prompts for next iteration.
        Changes are picked up at the start of the next iteration loop."""
        self._pending_themes = themes
        if negative_prompt is not None:
            self._pending_negative = negative_prompt
        print(f"[BATCH] Config update queued: {len(themes)} themes")

    def run(self):
        try:
            self._run_batch()
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

    def _run_batch(self):
        from core.model_manager import get_model, is_model_loaded
        from ui.state import get_state
        from ui.constants import OUTPUT_DIR, DEFAULT_STYLE_PRESETS, is_int8_model

        import torch

        state = get_state()
        config = self.config

        if not is_model_loaded():
            # Model not ready yet — wait for it if a load is in progress
            if state.model_load_lock.locked():
                self.progress.emit(0, 0, "Waiting for model to finish loading...")
                print("[BATCH] Model loading in progress, waiting...")
                poll_count = 0
                while not is_model_loaded() and not self._stop_requested:
                    time.sleep(1)
                    poll_count += 1
                    if poll_count % 5 == 0:
                        self.progress.emit(0, 0, f"Waiting for model... ({poll_count}s)")
                    # Safety: if lock is released but model still not loaded, it failed
                    if not state.model_load_lock.locked() and not is_model_loaded():
                        self.error.emit("Model load failed. Check the log and try again.")
                        return
                if self._stop_requested:
                    self.stopped.emit("", 0)
                    return
                print(f"[BATCH] Model ready after {poll_count}s wait")
            else:
                self.error.emit("Model not loaded. Load the model first.")
                return

        # Create batch output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c for c in config.batch_name[:30] if c.isalnum() or c in " -_"
        ).strip().replace(" ", "_") or "batch"
        batch_dir = OUTPUT_DIR / "batches" / f"{safe_name}_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        model = get_model()
        if model is None:
            self.error.emit("Model not available")
            return

        # Get generation parameters (model-aware)
        from ui.constants import get_quality_presets, get_aspect_ratios
        presets = get_quality_presets(state.model_type)
        ratios = get_aspect_ratios(state.model_type)
        quality_info = presets.get(config.quality, {"steps": 50 if state.model_type == "deepgen" else 20})
        steps = quality_info["steps"]
        image_size = ratios.get(config.aspect_ratio, "512x512" if state.model_type == "deepgen" else "1024x1024")

        # Initialize Ollama once before the loop - with lazy initialization
        ollama_model_name = None
        print(f"\n{'='*60}")
        print(f"[BATCH OLLAMA] Enhancement Status Check:")
        print(f"[BATCH OLLAMA]   config.enhance: {config.enhance}")
        print(f"[BATCH OLLAMA]   ollama_available: {state.ollama_available}")
        print(f"[BATCH OLLAMA]   ollama_enhancer exists: {state.ollama_enhancer is not None}")
        print(f"[BATCH OLLAMA]   ollama_manager exists: {state.ollama_manager is not None}")

        # Lazy initialization: if enhancement is requested but Ollama isn't ready, initialize it
        if config.enhance and (not state.ollama_available or state.ollama_manager is None):
            print(f"[BATCH OLLAMA]   Ollama not initialized, attempting lazy init...")
            self.progress.emit(0, config.total_images(), "Initializing Ollama for batch...")
            try:
                from ollama_manager import OllamaManager
                from ollama_prompts import PromptEnhancer
                from core.settings import get_settings

                settings = get_settings()
                gpu_indices = settings.ollama_gpu_indices

                # Filter out GPU 1 (Blackwell)
                safe_gpus = [i for i in gpu_indices if i != 1] if isinstance(gpu_indices, list) else [0, 2]
                if not safe_gpus:
                    safe_gpus = [0]

                print(f"[BATCH OLLAMA]   Using GPUs: {safe_gpus}")

                if state.ollama_manager is None:
                    print(f"[BATCH OLLAMA]   Creating OllamaManager...")
                    state.ollama_manager = OllamaManager()
                    state.ollama_available = True

                if not state.ollama_manager.is_running():
                    print(f"[BATCH OLLAMA]   Starting Ollama server...")
                    success, msg = state.ollama_manager.start(gpu_indices=safe_gpus)
                    if not success:
                        print(f"[BATCH OLLAMA] ✗ FAILED to start Ollama: {msg}")
                    else:
                        print(f"[BATCH OLLAMA] ✓ Ollama server started")

                if state.ollama_enhancer is None and state.ollama_manager.is_running():
                    print(f"[BATCH OLLAMA]   Creating PromptEnhancer...")
                    state.ollama_enhancer = PromptEnhancer()

                print(f"[BATCH OLLAMA] ✓ Lazy initialization complete")
            except ImportError as e:
                print(f"[BATCH OLLAMA] ✗ FAILED - Import error: {e}")
            except Exception as e:
                print(f"[BATCH OLLAMA] ✗ FAILED to initialize: {e}")

        if config.enhance and state.ollama_available:
            ollama_model_name = config.ollama_model
            # Force small model for batch
            if any(size in ollama_model_name.lower() for size in ("30b", "24b", "20b")):
                ollama_model_name = "qwen2.5:7b-instruct"
                print(f"[BATCH OLLAMA]   Forced smaller model for batch: {ollama_model_name}")

            if state.ollama_enhancer is None:
                print(f"[BATCH OLLAMA]   Creating new PromptEnhancer...")
                from ollama_prompts import PromptEnhancer
                state.ollama_enhancer = PromptEnhancer()

            print(f"[BATCH OLLAMA] ✓ ENHANCEMENT ENABLED")
            print(f"[BATCH OLLAMA]   Model: {ollama_model_name}")
            print(f"[BATCH OLLAMA]   Length: {config.ollama_length}, Complexity: {config.ollama_complexity}")
        elif config.enhance:
            print(f"[BATCH OLLAMA] ✗ ENHANCEMENT REQUESTED BUT UNAVAILABLE")
            print(f"[BATCH OLLAMA]   Reason: ollama_available={state.ollama_available}")
            if not state.ollama_manager:
                print(f"[BATCH OLLAMA]   Hint: Ollama manager not initialized - need to click Enhance first")
        else:
            print(f"[BATCH OLLAMA] ✗ ENHANCEMENT DISABLED - config.enhance is False")
        print(f"{'='*60}\n")

        # Calculate total
        total_images = config.total_images()
        total_count = 0
        generated_images = []

        themes = config.themes
        styles = config.styles if config.styles else ["None"]

        print(f"\n{'='*60}")
        print(f"[BATCH] Starting batch generation")
        print(f"[BATCH]   Themes: {len(themes)}")
        print(f"[BATCH]   Styles: {len(styles)} - {styles}")
        print(f"[BATCH]   Variations per theme: {config.variations_per_theme}")
        print(f"[BATCH]   Images per combo: {config.images_per_combo}")
        print(f"[BATCH]   Starred reroll count: {config.starred_reroll_count}")
        print(f"[BATCH]   TOTAL EXPECTED: {total_images} images")
        print(f"[BATCH]   Quality: {config.quality} ({steps} steps)")
        print(f"[BATCH]   Size: {image_size}")
        print(f"[BATCH]   Bot task: {config.bot_task}")
        print(f"{'='*60}\n")

        # Round-robin: for iteration -> for theme -> for style -> for img_per
        for iteration_idx in range(config.variations_per_theme):
            if self._stop_requested:
                break

            # Check for pending config updates between iterations
            if self._pending_themes is not None:
                themes = self._pending_themes
                config.themes = themes  # Update config so total_images recalculates
                self._pending_themes = None
                total_images = config.total_images()
                print(f"[BATCH] ✓ Applied updated themes: {len(themes)} themes (new total: {total_images})")
                self.config_updated.emit()
            if self._pending_negative is not None:
                config.negative_prompt = self._pending_negative
                self._pending_negative = None
                print(f"[BATCH] ✓ Applied updated negative prompt")

            for theme_idx, theme in enumerate(themes):
                if self._stop_requested:
                    break

                for style in styles:
                    if self._stop_requested:
                        break

                    for img_idx in range(config.images_per_combo):
                        if self._stop_requested:
                            break

                        # Build prompt with style suffix
                        full_prompt = theme
                        style_suffix = DEFAULT_STYLE_PRESETS.get(style, "")
                        if style_suffix:
                            full_prompt += style_suffix

                        # Generate seed
                        if config.random_seeds:
                            current_seed = random.randint(0, 2**31 - 1)
                        else:
                            current_seed = (iteration_idx * 10000
                                          + theme_idx * 1000
                                          + img_idx)

                        # PHASE 1: Starred wildcard pre-processing
                        starred_placeholders = {}
                        has_starred = False

                        if state.wildcard_available and state.wildcard_manager:
                            try:
                                has_starred = state.wildcard_manager.has_starred_wildcards(full_prompt)
                                if has_starred and config.starred_reroll_count > 1:
                                    starred_pattern = r'\[\*([^\]]+)\]'
                                    starred_matches = re.findall(starred_pattern, full_prompt)

                                    # Protect starred wildcards
                                    temp_protected = full_prompt
                                    for key in starred_matches:
                                        temp_protected = temp_protected.replace(
                                            f"[*{key}]", f"__TEMP_{key.upper()}__"
                                        )

                                    # Process standard wildcards (pass generation_index for alternating)
                                    enhanced_base = state.wildcard_manager.process_prompt(
                                        temp_protected, generation_index=total_count
                                    )

                                    # Convert to VAR_XXX format
                                    full_prompt = enhanced_base
                                    for key in starred_matches:
                                        placeholder = f"VAR_{key.upper()}"
                                        starred_placeholders[placeholder] = key
                                        full_prompt = full_prompt.replace(
                                            f"__TEMP_{key.upper()}__", placeholder
                                        )
                                else:
                                    # Pass generation_index for alternating wildcards [a/b]
                                    full_prompt = state.wildcard_manager.process_prompt(
                                        full_prompt, generation_index=total_count
                                    )
                            except Exception:
                                pass

                        # PHASE 2: Ollama enhancement with protected tokens
                        if config.enhance and state.ollama_available and state.ollama_enhancer:
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

                                input_for_ollama = full_prompt
                                if has_starred and starred_placeholders:
                                    tokens_to_keep = ", ".join(starred_placeholders.keys())
                                    instruction = (
                                        f"Rewrite and enhance the following description. "
                                        f"CRITICAL: You MUST include these placeholders exactly: "
                                        f"{tokens_to_keep}. Do not change or describe them."
                                    )
                                    input_for_ollama = f"{instruction}\n\nORIGINAL PROMPT:\n{full_prompt}"

                                print(f"[BATCH ENHANCE] ▶ Item {total_count+1}: {input_for_ollama[:60]}...")
                                enhanced = state.ollama_enhancer.enhance(
                                    input_for_ollama,
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

                                print(f"[BATCH ENHANCE] ✓ Item {total_count+1}: {clean[:60]}...")
                                full_prompt = clean
                            except Exception as e:
                                print(f"[BATCH ENHANCE] ✗ Item {total_count+1} FAILED: {e}")
                        elif config.enhance:
                            # Enhancement requested but not available
                            if total_count == 0:  # Only log once
                                print(f"[BATCH ENHANCE] ✗ SKIPPING - Enhancement requested but Ollama not ready")

                        # PHASE 3: Generate starred variations
                        starred_variations = []
                        if has_starred and starred_placeholders and config.starred_reroll_count > 1:
                            for var_idx in range(config.starred_reroll_count):
                                var_prompt = full_prompt
                                var_values = {}

                                for placeholder, wildcard_key in starred_placeholders.items():
                                    val = state.wildcard_manager.get_random_value(wildcard_key)
                                    if not val:
                                        val = wildcard_key
                                    var_values[wildcard_key] = val

                                    if placeholder in var_prompt:
                                        var_prompt = var_prompt.replace(placeholder, val)
                                    else:
                                        var_prompt = f"{var_prompt}, {val}"

                                starred_variations.append({
                                    "prompt": var_prompt,
                                    "index": var_idx,
                                    "values": var_values,
                                })

                        if not starred_variations:
                            starred_variations = [{
                                "prompt": full_prompt,
                                "index": None,
                                "values": None,
                            }]

                        # Generate images for each starred variation
                        for star_var in starred_variations:
                            if self._stop_requested:
                                print(f"[BATCH] Stop requested, breaking out of starred variations loop")
                                break

                            var_prompt = star_var["prompt"]
                            var_index = star_var["index"]
                            var_values = star_var["values"]

                            status = (
                                f"Iter {iteration_idx+1}/{config.variations_per_theme} | "
                                f"#{total_count+1}/{total_images}: "
                                f"{theme[:25]}... ({style})"
                            )
                            self.progress.emit(total_count, total_images, status)
                            print(f"[BATCH] {status}")

                            try:
                                start_time = time.time()
                                print(f"[BATCH] Generating image {total_count+1}/{total_images}...")
                                print(f"[BATCH]   model_type: {state.model_type}")
                                print(f"[BATCH]   steps: {steps}, image_size: {image_size}")
                                print(f"[BATCH]   bot_task: {config.bot_task}, seed: {current_seed}")

                                # Use the appropriate generation path for the model
                                cot_text = None
                                image = None
                                if is_int8_model(state.model_type):
                                    from core.model_manager import generate_int8
                                    def _progress_cb(msg):
                                        self.progress.emit(total_count, total_images, msg)
                                    int8_result = generate_int8(
                                        model=model,
                                        prompt=var_prompt,
                                        seed=current_seed,
                                        image_size=image_size,
                                        bot_task=config.bot_task,
                                        steps=steps,
                                        guidance_scale=config.guidance_scale,
                                        progress_callback=_progress_cb,
                                    )
                                    # generate_int8 returns (image, seed) or (image, seed, cot_text)
                                    if len(int8_result) == 3:
                                        image, current_seed, cot_text = int8_result
                                    else:
                                        image, current_seed = int8_result
                                    result = image
                                    # Re-fetch model in case OOM recovery deleted it
                                    if not is_model_loaded():
                                        self.error.emit("Model was unloaded during VAE OOM recovery. Reload to continue.")
                                        self._stop_requested = True
                                        break
                                    model = get_model()
                                elif state.model_type == "firered":
                                    # FireRed T2I mode (no input images) for batch
                                    from core.model_manager import generate_firered
                                    from ui.constants import get_default_guidance
                                    firered_cfg = config.guidance_scale
                                    if firered_cfg == 5.0:
                                        firered_cfg = get_default_guidance("firered")
                                    fw, fh = None, None
                                    if image_size and image_size.lower() != "auto":
                                        try:
                                            fw, fh = map(int, image_size.lower().split('x'))
                                        except (ValueError, AttributeError):
                                            pass
                                    image, current_seed = generate_firered(
                                        model=model,
                                        prompt=var_prompt,
                                        seed=current_seed,
                                        width=fw,
                                        height=fh,
                                        steps=steps,
                                        true_cfg_scale=firered_cfg,
                                        src_image_path=None,
                                    )
                                    result = image
                                elif state.model_type == "deepgen":
                                    from core.model_manager import generate_deepgen
                                    try:
                                        w, h = map(int, image_size.lower().split('x'))
                                    except (ValueError, AttributeError):
                                        w, h = 1024, 1024
                                    image, current_seed = generate_deepgen(
                                        model=model,
                                        prompt=var_prompt,
                                        seed=current_seed,
                                        width=w,
                                        height=h,
                                        steps=steps,
                                        guidance_scale=config.guidance_scale,
                                    )
                                    result = image
                                elif state.model_type in ("instruct", "distil"):
                                    result = model.generate_image(
                                        prompt=var_prompt,
                                        seed=current_seed,
                                        image_size=image_size,
                                        use_system_prompt="en_vanilla",
                                        bot_task=config.bot_task,
                                        diff_infer_steps=steps,
                                        diff_guidance_scale=config.guidance_scale,
                                    )
                                else:
                                    result = model.generate_image(
                                        prompt=var_prompt,
                                        seed=current_seed,
                                        image_size=image_size,
                                        stream=True,
                                        diff_infer_steps=steps,
                                    )

                                print(f"[BATCH] Generation returned: {type(result)}")

                                gen_time = time.time() - start_time
                                print(f"[BATCH] Generation took {gen_time:.1f}s")

                                # Handle result for non-INT8/DeepGen paths
                                # (INT8 and DeepGen already set image/cot_text directly)
                                if image is None:
                                    if isinstance(result, tuple):
                                        cot_text, outputs = result
                                        print(f"[BATCH] Got tuple result: cot_text={len(cot_text) if cot_text else 0} chars, outputs={len(outputs) if outputs else 0} items")
                                        image = outputs[0] if outputs else None
                                    else:
                                        image = result
                                        print(f"[BATCH] Got direct result: {type(image)}")

                                if cot_text and not config.drop_think:
                                    self.cot_received.emit(cot_text)

                                if image:
                                    print(f"[BATCH] Image received, saving...")
                                    starred_suffix = ""
                                    if var_index is not None:
                                        if var_index == 0:
                                            starred_suffix = "_base"
                                        else:
                                            starred_suffix = f"_var{chr(ord('A') + var_index - 1)}"

                                    filename = (
                                        f"{total_count:04d}_{current_seed}_"
                                        f"{style.replace(' ', '_')[:20]}"
                                        f"{starred_suffix}.png"
                                    )
                                    filepath = batch_dir / filename
                                    image.save(str(filepath))

                                    # Save JSON sidecar
                                    img_config = {
                                        "prompt": theme,
                                        "full_prompt": var_prompt,
                                        "negative_prompt": config.negative_prompt,
                                        "style": style,
                                        "aspect_ratio": config.aspect_ratio,
                                        "quality": config.quality,
                                        "image_size": image_size,
                                        "steps": steps,
                                        "seed": current_seed,
                                        "use_ollama": config.enhance,
                                        "ollama_model": config.ollama_model if config.enhance else None,
                                        "ollama_length": config.ollama_length if config.enhance else None,
                                        "ollama_complexity": config.ollama_complexity if config.enhance else None,
                                        "generation_time": gen_time,
                                        "batch_name": config.batch_name,
                                        "batch_iteration": iteration_idx + 1,
                                        "batch_total_iterations": config.variations_per_theme,
                                        "starred_reroll_index": var_index,
                                        "starred_values": var_values,
                                        "is_starred_variation": var_index is not None,
                                        "model_type": state.model_type,
                                        "bot_task": config.bot_task,
                                        "drop_think": config.drop_think,
                                    }
                                    if cot_text:
                                        img_config["cot_text"] = cot_text
                                    json_path = filepath.with_suffix(".json")
                                    with open(json_path, "w") as f:
                                        json.dump(img_config, f, indent=2, default=str)

                                    generated_images.append(str(filepath))
                                    total_count += 1
                                    self.image_ready.emit(str(filepath))
                                    print(f"[BATCH] ✓ Saved: {filename} (total: {total_count})")

                                    gc.collect()
                                    torch.cuda.empty_cache()
                                else:
                                    print(f"[BATCH] ✗ No image returned from generation!")

                            except Exception as e:
                                import traceback
                                print(f"[BATCH] ✗ ERROR generating image {total_count+1}: {e}")
                                print(f"[BATCH] Traceback:\n{traceback.format_exc()}")
                                self.progress.emit(
                                    total_count, total_images, f"Error: {e}"
                                )

        # Write batch manifest
        manifest = {
            "batch_name": config.batch_name,
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
