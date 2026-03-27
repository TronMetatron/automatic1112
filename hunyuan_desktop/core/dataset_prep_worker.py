"""QThread worker for dataset preparation generation.

Processes source images through multiple I2I passes to generate
diverse training datasets for character LoRA training.
"""

import time
import random
import re
import gc
import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from models.dataset_prep_config import DatasetPrepConfig, _pass_name_to_folder


# Image extensions to scan for
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class DatasetPrepWorker(QThread):
    """Executes dataset preparation generation."""

    progress = Signal(int, int, str)       # current, total, status_text
    image_ready = Signal(str)               # image_path
    cot_received = Signal(str)              # chain-of-thought text
    completed = Signal(str, int)            # output_dir, total_count
    stopped = Signal(str, int)              # output_dir, count_so_far
    error = Signal(str)                     # error_msg

    def __init__(self, config: DatasetPrepConfig):
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

    def _resolve_wildcards(self, prompt: str) -> str:
        """Resolve [option1, option2, ...] wildcards by picking one at random."""
        def _pick_random(match):
            options = [o.strip() for o in match.group(1).split(",")]
            return random.choice(options) if options else match.group(0)

        return re.sub(r"\[([^\]]+)\]", _pick_random, prompt)

    def _scan_input_folder(self) -> list:
        """Find all image files in the input folder."""
        input_path = Path(self.config.input_folder)
        if not input_path.is_dir():
            return []

        images = []
        for f in sorted(input_path.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(str(f))
        return images

    def _run_batch(self):
        from core.model_manager import get_model, is_model_loaded
        from ui.state import get_state

        import torch

        state = get_state()
        config = self.config

        if not is_model_loaded():
            # Model not ready yet — wait for it if a load is in progress
            if state.model_load_lock.locked():
                self.progress.emit(0, 0, "Waiting for model to finish loading...")
                print("[DATASET PREP] Model loading in progress, waiting...")
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
                print(f"[DATASET PREP] Model ready after {poll_count}s wait")
            else:
                self.error.emit("Model not loaded. Load an Instruct or Distil model first.")
                return

        if state.model_type not in ("instruct", "distil", "instruct_int8", "distil_int8"):
            self.error.emit(
                f"Dataset prep requires Instruct or Distil model. "
                f"Current model: {state.model_type}"
            )
            return

        # Scan input folder
        source_images = self._scan_input_folder()
        if not source_images:
            self.error.emit(
                f"No images found in input folder: {config.input_folder}"
            )
            return

        if not config.enabled_passes:
            self.error.emit("No passes enabled. Enable at least one pass.")
            return

        # Create output directory
        output_base = Path(config.output_folder)
        output_base.mkdir(parents=True, exist_ok=True)

        model = get_model()
        if model is None:
            self.error.emit("Model not available")
            return

        total_images = config.total_images_for_sources(len(source_images))
        total_count = 0
        generated_images = []

        # Process: pass -> source image -> images_per_pass
        for pass_name, pass_prompt in config.enabled_passes.items():
            if self._stop_requested:
                break

            # Create pass subdirectory
            folder_name = _pass_name_to_folder(pass_name)
            pass_dir = output_base / folder_name
            pass_dir.mkdir(parents=True, exist_ok=True)

            for src_idx, source_path in enumerate(source_images):
                if self._stop_requested:
                    break

                source_stem = Path(source_path).stem

                for img_idx in range(config.images_per_pass):
                    if self._stop_requested:
                        break

                    # Resolve wildcards in the prompt
                    full_prompt = self._resolve_wildcards(pass_prompt)

                    # Also run through wildcard manager if available
                    if state.wildcard_available and state.wildcard_manager:
                        try:
                            full_prompt = state.wildcard_manager.process_prompt(
                                full_prompt
                            )
                        except Exception:
                            pass

                    # Generate seed
                    if config.random_seeds:
                        current_seed = random.randint(0, 2**31 - 1)
                    else:
                        current_seed = (
                            src_idx * 10000
                            + img_idx * 1000
                            + hash(pass_name) % 1000
                        )

                    status = (
                        f"{pass_name} | "
                        f"#{total_count+1}/{total_images}: "
                        f"{source_stem} (img {img_idx+1}/{config.images_per_pass})"
                    )
                    self.progress.emit(total_count, total_images, status)

                    try:
                        start_time = time.time()

                        print(f"[DATASET PREP] Generating {total_count+1}/{total_images}")
                        print(f"[DATASET PREP] Pass: {pass_name}")
                        print(f"[DATASET PREP] Source: {source_path}")
                        print(f"[DATASET PREP] Prompt: {full_prompt[:80]}...")

                        result = model.generate_image(
                            prompt=full_prompt,
                            seed=current_seed,
                            image=source_path,
                            image_size="auto",
                            use_system_prompt="en_vanilla",
                            bot_task=config.bot_task,
                            infer_align_image_size=True,
                            diff_infer_steps=8,
                            diff_guidance_scale=config.guidance_scale,
                        )

                        gen_time = time.time() - start_time
                        print(f"[DATASET PREP] Generation took {gen_time:.2f}s")

                        # Handle instruct return format
                        cot_text = None
                        if isinstance(result, tuple):
                            cot_text, outputs = result
                            image = outputs[0] if outputs else None
                        else:
                            image = result

                        if cot_text and not config.drop_think:
                            self.cot_received.emit(cot_text)

                        if image:
                            filename = (
                                f"{total_count:04d}_{source_stem}_"
                                f"seed{current_seed}.png"
                            )
                            filepath = pass_dir / filename
                            image.save(str(filepath))

                            # Save JSON sidecar
                            img_config = {
                                "source_image": source_path,
                                "pass_name": pass_name,
                                "prompt_template": pass_prompt,
                                "resolved_prompt": full_prompt,
                                "guidance_scale": config.guidance_scale,
                                "seed": current_seed,
                                "generation_time": gen_time,
                                "model_type": state.model_type,
                                "bot_task": config.bot_task,
                                "drop_think": config.drop_think,
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
                        print(f"[DATASET PREP] Error: {traceback.format_exc()}")
                        self.progress.emit(
                            total_count, total_images, f"Error: {e}"
                        )

        # Write batch manifest
        manifest = {
            "batch_type": "dataset_prep",
            "timestamp": datetime.now().isoformat(),
            "input_folder": config.input_folder,
            "source_images": source_images,
            "total_generated": total_count,
            "settings": config.to_dict(),
            "images": [str(p) for p in generated_images],
        }
        manifest_path = output_base / "batch_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        if self._stop_requested:
            self.stopped.emit(str(output_base), total_count)
        else:
            self.completed.emit(str(output_base), total_count)
