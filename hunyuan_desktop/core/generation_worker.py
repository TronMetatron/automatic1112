"""QThread worker for single image generation."""

import time
import random
import json
import gc
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from models.generation_params import GenerationParams


class GenerationWorker(QThread):
    """Generates images on a background thread.

    Mirrors the generation logic from hunyuan_ui_v2.py generate_image().
    """

    progress = Signal(int, int, str)             # current, total, status
    image_generated = Signal(str, int, float)    # filepath, seed, gen_time
    cot_received = Signal(str)                    # chain-of-thought text
    completed = Signal(int)                       # total_count
    error = Signal(str)                           # error_msg

    def __init__(self, params: GenerationParams):
        super().__init__()
        self.params = params
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._generate()
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

    def _generate(self):
        from core.model_manager import get_model, is_model_loaded
        from ui.state import get_state
        from ui.constants import DEFAULT_STYLE_PRESETS, is_int8_model

        state = get_state()
        params = self.params

        # Validate
        if not params.prompt.strip():
            self.error.emit("Empty prompt")
            return

        # Wait for model if loading
        if state.model_load_lock.locked():
            self.progress.emit(0, params.batch_count, "Waiting for model to load...")
            with state.model_load_lock:
                pass

        if not is_model_loaded():
            self.error.emit("Model not loaded. Click 'Load Model' first.")
            return

        model = get_model()
        steps = params.get_steps(state.model_type)
        image_size = params.get_image_size(state.model_type)
        style_suffix = params.get_style_suffix()

        # Build the full prompt
        full_prompt = params.prompt + style_suffix

        # Ollama enhancement - with lazy initialization
        print(f"\n{'='*60}")
        print(f"[OLLAMA] Enhancement Status Check:")
        print(f"[OLLAMA]   use_ollama checkbox: {params.use_ollama}")
        print(f"[OLLAMA]   ollama_available: {state.ollama_available}")
        print(f"[OLLAMA]   ollama_enhancer exists: {state.ollama_enhancer is not None}")
        print(f"[OLLAMA]   ollama_manager exists: {state.ollama_manager is not None}")

        # Lazy initialization: if enhancement is requested but Ollama isn't ready, initialize it
        if params.use_ollama and (not state.ollama_available or state.ollama_enhancer is None):
            print(f"[OLLAMA]   Ollama not initialized, attempting lazy init...")
            self.progress.emit(0, params.batch_count, "Initializing Ollama...")
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

                print(f"[OLLAMA]   Using GPUs: {safe_gpus}")

                if state.ollama_manager is None:
                    print(f"[OLLAMA]   Creating OllamaManager...")
                    state.ollama_manager = OllamaManager()
                    state.ollama_available = True

                if not state.ollama_manager.is_running():
                    print(f"[OLLAMA]   Starting Ollama server...")
                    success, msg = state.ollama_manager.start(gpu_indices=safe_gpus)
                    if not success:
                        print(f"[OLLAMA] ✗ FAILED to start Ollama: {msg}")
                    else:
                        print(f"[OLLAMA] ✓ Ollama server started")

                if state.ollama_enhancer is None and state.ollama_manager.is_running():
                    print(f"[OLLAMA]   Creating PromptEnhancer...")
                    state.ollama_enhancer = PromptEnhancer()

                print(f"[OLLAMA] ✓ Lazy initialization complete")
            except ImportError as e:
                print(f"[OLLAMA] ✗ FAILED - Import error: {e}")
            except Exception as e:
                print(f"[OLLAMA] ✗ FAILED to initialize: {e}")

        if params.use_ollama and state.ollama_available and state.ollama_enhancer:
            print(f"[OLLAMA] ✓ ENHANCEMENT ENABLED - Will enhance prompt")
            print(f"[OLLAMA]   Model: {params.ollama_model}, Length: {params.ollama_length}, Complexity: {params.ollama_complexity}")
            print(f"{'='*60}")
            try:
                print(f"[OLLAMA ENHANCE] ▶ STARTING with {params.ollama_model}...")
                print(f"[OLLAMA ENHANCE]   Input ({len(full_prompt)} chars): {full_prompt[:80]}...")
                self.progress.emit(0, params.batch_count, "Enhancing prompt with Ollama...")
                enhanced = state.ollama_enhancer.enhance(
                    full_prompt,
                    length=params.ollama_length,
                    complexity=params.ollama_complexity,
                    model=params.ollama_model,
                )
                print(f"[OLLAMA ENHANCE] ✓ SUCCESS - Output ({len(enhanced)} chars): {enhanced[:80]}...")
                full_prompt = enhanced
            except Exception as e:
                print(f"[OLLAMA ENHANCE] ✗ FAILED: {e}")
                self.progress.emit(0, params.batch_count, f"Enhancement failed: {e}")
        else:
            print(f"[OLLAMA] ✗ ENHANCEMENT DISABLED - Reasons:")
            if not params.use_ollama:
                print(f"[OLLAMA]   - 'Enhance with Ollama' checkbox is OFF")
            if not state.ollama_available:
                print(f"[OLLAMA]   - Ollama modules not available/imported")
            if not state.ollama_enhancer:
                print(f"[OLLAMA]   - Ollama enhancer not initialized (click 'Enhance' button first to initialize)")
            if not state.ollama_manager:
                print(f"[OLLAMA]   - Ollama manager not created")
            print(f"{'='*60}")

        # Wildcard processing
        if state.wildcard_available and state.wildcard_manager:
            try:
                full_prompt = state.wildcard_manager.process(full_prompt)
            except Exception:
                pass

        # Create session directory
        from core.image_utils import get_session_dir
        from ui.constants import OUTPUT_DIR
        output_base = params.output_dir or str(OUTPUT_DIR)
        session_dir = get_session_dir(output_base)

        generated_count = 0

        for i in range(params.batch_count):
            if self._stop_requested:
                break

            current_seed = params.seed if params.seed >= 0 else random.randint(0, 2**31 - 1)
            if i > 0 and params.seed < 0:
                current_seed = random.randint(0, 2**31 - 1)

            self.progress.emit(i, params.batch_count, f"Generating {i+1}/{params.batch_count}...")

            try:
                start_time = time.time()

                # Handle I2I vs T2I, with instruct/distil bot_task support
                i2i_images = params.get_i2i_images()
                print(f"[GEN] Model type: {state.model_type}, I2I images: {i2i_images}, Steps: {steps}")

                cot_text = None
                image = None

                if is_int8_model(state.model_type):
                    # INT8 BnB models use exception-trick VAE decode
                    from core.model_manager import generate_int8
                    print(f"[GEN] Running INT8 mode: {image_size}, steps={steps}, bot_task={params.bot_task}")
                    def _progress_cb(msg):
                        self.progress.emit(i, params.batch_count, msg)
                    int8_result = generate_int8(
                        model=model,
                        prompt=full_prompt,
                        seed=current_seed,
                        image_size=image_size,
                        bot_task=params.bot_task,
                        steps=steps,
                        guidance_scale=params.guidance_scale,
                        i2i_images=i2i_images,
                        progress_callback=_progress_cb,
                    )
                    # generate_int8 returns (image, seed) or (image, seed, cot_text)
                    if len(int8_result) == 3:
                        image, current_seed, cot_text = int8_result
                    else:
                        image, current_seed = int8_result
                    result = image
                elif state.model_type == "firered":
                    # FireRed Image Edit 1.1 - diffusers pipeline
                    # Supports I2I (1-3 images) and T2I (no images)
                    # guidance_scale is fixed at 1.0 internally; true_cfg_scale is the real control
                    from core.model_manager import generate_firered
                    # Parse WxH - use None for "auto" to let pipeline match input image dims
                    w, h = None, None
                    if image_size and image_size.lower() != "auto":
                        try:
                            w, h = map(int, image_size.lower().split('x'))
                        except (ValueError, AttributeError):
                            pass
                    src_imgs = i2i_images if i2i_images else None
                    print(f"[GEN] Running FireRed mode: size={w}x{h}, steps={steps}, "
                          f"true_cfg={params.guidance_scale}, images={len(i2i_images) if i2i_images else 0}")
                    image, current_seed = generate_firered(
                        model=model,
                        prompt=full_prompt,
                        seed=current_seed,
                        width=w,
                        height=h,
                        steps=steps,
                        true_cfg_scale=params.guidance_scale,
                        src_image_path=src_imgs,
                    )
                    result = image
                elif state.model_type == "deepgen":
                    # DeepGen uses a completely different API
                    from core.model_manager import generate_deepgen
                    # Parse WxH from image_size string
                    try:
                        w, h = map(int, image_size.lower().split('x'))
                    except (ValueError, AttributeError):
                        w, h = 1024, 1024
                    src_img = i2i_images[0] if i2i_images else None
                    print(f"[GEN] Running DeepGen mode: {w}x{h}, steps={steps}, cfg={params.guidance_scale}")
                    image, current_seed = generate_deepgen(
                        model=model,
                        prompt=full_prompt,
                        seed=current_seed,
                        width=w,
                        height=h,
                        steps=steps,
                        guidance_scale=params.guidance_scale,
                        src_image_path=src_img,
                    )
                    result = image  # For type checking below
                elif i2i_images and state.model_type in ("instruct", "distil"):
                    # I2I with instruct/distil: file paths + bot_task
                    print(f"[GEN] Running I2I mode with {len(i2i_images)} image(s), bot_task={params.bot_task}")
                    img_arg = i2i_images if len(i2i_images) > 1 else i2i_images[0]
                    result = model.generate_image(
                        prompt=full_prompt,
                        seed=current_seed,
                        image=img_arg,
                        image_size="auto",
                        use_system_prompt="en_vanilla",
                        bot_task=params.bot_task,
                        infer_align_image_size=True,
                        diff_infer_steps=steps,
                        diff_guidance_scale=params.guidance_scale,
                    )
                elif state.model_type in ("instruct", "distil"):
                    # T2I with instruct/distil: supports bot_task/think
                    print(f"[GEN] Running T2I mode (instruct/distil) bot_task={params.bot_task}")
                    result = model.generate_image(
                        prompt=full_prompt,
                        seed=current_seed,
                        image_size=image_size,
                        use_system_prompt="en_vanilla",
                        bot_task=params.bot_task,
                        diff_infer_steps=steps,
                        diff_guidance_scale=params.guidance_scale,
                    )
                else:
                    # T2I with base model
                    print(f"[GEN] Running T2I mode (base model)")
                    result = model.generate_image(
                        prompt=full_prompt,
                        seed=current_seed,
                        image_size=image_size,
                        stream=True,
                        diff_infer_steps=steps,
                    )

                # Handle result types (DeepGen returns PIL directly, others may return tuple)
                if image is None:
                    print(f"[GEN] Result type: {type(result)}")
                    if isinstance(result, tuple):
                        cot_text, outputs = result
                        print(f"[GEN] Tuple result - CoT length: {len(cot_text) if cot_text else 0}, Outputs: {type(outputs)}, Count: {len(outputs) if outputs else 0}")
                        image = outputs[0] if outputs else None
                    else:
                        image = result
                        print(f"[GEN] Direct image result: {type(image)}")

                # Emit CoT text if available and not dropped
                if cot_text and not params.drop_think:
                    self.cot_received.emit(cot_text)

                if image is None:
                    print(f"[GEN] ERROR: No image returned from generate_image()!")
                    self.progress.emit(i + 1, params.batch_count, f"No image returned for {i+1}")
                    continue

                gen_time = time.time() - start_time

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_prompt = "".join(
                    c for c in params.prompt[:30] if c.isalnum() or c in " -_"
                ).strip().replace(" ", "_")
                filename = f"{timestamp}_{current_seed}_{safe_prompt}.png"
                filepath = session_dir / filename
                image.save(str(filepath))

                # Save JSON sidecar
                config = params.to_metadata()
                config.update({
                    "full_prompt": full_prompt,
                    "seed": current_seed,
                    "generation_time": gen_time,
                    "model_type": state.model_type,
                })
                if cot_text:
                    config["cot_text"] = cot_text

                json_path = filepath.with_suffix(".json")
                with open(json_path, "w") as f:
                    json.dump(config, f, indent=2, default=str)

                # Update state
                state.last_generated_image = str(filepath)
                state.last_seed_used = current_seed

                generated_count += 1
                self.image_generated.emit(str(filepath), current_seed, gen_time)

            except Exception as e:
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.progress.emit(i + 1, params.batch_count, f"Error: {e}")

        self.completed.emit(generated_count)
