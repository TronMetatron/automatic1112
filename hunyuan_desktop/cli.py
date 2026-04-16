#!/usr/bin/env python3
"""Headless CLI for HunyuanImage desktop.

Runs the same model loader and I2I batch pipeline as the GUI but with no Qt
widgets / no display server required. Intended for batch experiments, tests,
and unattended generation (e.g. AI-driven prompt optimization loops where an
outer script mutates prompts between runs).

Usage examples:

    # Run an existing i2i_batch config JSON (same format the GUI saves)
    python -m hunyuan_desktop.cli i2i \\
        --model distil_nf4 \\
        --config outputs/configs/i2i_batch/my_config.json

    # Ad-hoc: prompts from a text file (one per line), global reference images
    python -m hunyuan_desktop.cli i2i \\
        --model distil \\
        --prompts prompts.txt \\
        --image ref1.png --image ref2.png \\
        --variations 4 \\
        --name experiment_A

    # Single prompt, single image
    python -m hunyuan_desktop.cli i2i \\
        --model distil \\
        --prompt "Turn this into a watercolor" \\
        --image ref.png

Exit codes:
    0 success, 1 argument / validation error, 2 model load failed,
    3 generation failed, 130 interrupted.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# --- Bootstrap sys.path / env BEFORE torch imports ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PACKAGE_DIR)
HUNYUAN_DIR = os.path.join(PROJECT_ROOT, "HunyuanImage-3.0")
if os.path.isdir(HUNYUAN_DIR):
    sys.path.insert(0, HUNYUAN_DIR)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no display needed


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="hunyuan_desktop.cli",
        description="Headless HunyuanImage generation runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── i2i ──
    i2i = sub.add_parser(
        "i2i", help="Image-to-image batch (same config as the GUI I2I tab)."
    )
    i2i.add_argument("--model", required=True,
                     help="Model type: base, instruct, distil, nf4, distil_nf4, firered")
    i2i.add_argument("--config", type=Path,
                     help="Path to an i2i_batch JSON config saved from the GUI.")
    i2i.add_argument("--prompts", type=Path,
                     help="Text file with one prompt per line (ignored if --config is set).")
    i2i.add_argument("--prompt", action="append", default=[],
                     help="Inline prompt (can be given multiple times).")
    i2i.add_argument("--image", action="append", default=[],
                     help="Reference image path (up to 3). Can be given multiple times.")
    i2i.add_argument("--image-folder", action="append", default=[],
                     help="Folder of reference images to cycle through per generation. "
                          "Repeat to supply up to 3 folders (aligned to slots 1..3).")
    i2i.add_argument("--name", default="cli_batch",
                     help="Batch name (used in the output directory).")
    i2i.add_argument("--variations", type=int, default=1,
                     help="Variations per prompt.")
    i2i.add_argument("--imgs-per-combo", type=int, default=1)
    i2i.add_argument("--quality", default="Standard",
                     help="Quality preset name (e.g. 'Standard', 'Maximum', 'Distil (8 steps)').")
    i2i.add_argument("--guidance", type=float, default=None,
                     help="Guidance scale. Default: model-specific.")
    i2i.add_argument("--style", action="append", default=[],
                     help="Style preset name (repeat for multiple).")
    i2i.add_argument("--bot-task", default="think_recaption",
                     choices=["image", "think", "think_recaption", "recaption"])
    i2i.add_argument("--drop-think", action="store_true",
                     help="Drop chain-of-thought output.")
    i2i.add_argument("--seed-mode", choices=["random", "deterministic"],
                     default="random")
    i2i.add_argument("--enhance", action="store_true",
                     help="Run prompt enhancement (LM Studio / Ollama).")
    i2i.add_argument("--ollama-model", default="qwen2.5:7b-instruct")
    i2i.add_argument("--ollama-length", default="medium")
    i2i.add_argument("--ollama-complexity", default="detailed")
    i2i.add_argument("--nf4-dual-gpu", action="store_true",
                     help="Enable NF4 dual-GPU split before loading.")
    i2i.add_argument("--output-json", type=Path,
                     help="If set, write a result summary JSON here on completion.")
    i2i.set_defaults(func=_cmd_i2i)

    # ── list-models ──
    lm = sub.add_parser("list-models", help="Print available model types.")
    lm.set_defaults(func=_cmd_list_models)

    return p.parse_args(argv)


def _cmd_list_models(args):
    from ui.constants import MODEL_INFO, MODEL_PATHS
    print(f"{'type':<14} {'steps':<6}  {'path (exists)'}")
    print("-" * 70)
    for key, info in MODEL_INFO.items():
        path = MODEL_PATHS.get(key)
        exists = "✓" if path and Path(path).exists() else "✗"
        print(f"{key:<14} {info.get('default_steps', '?'):<6}  {exists} {path}")
    return 0


def _build_config_from_args(args):
    """Build an I2IBatchConfig from CLI args (when --config isn't used)."""
    from models.i2i_batch_config import I2IBatchConfig

    prompts = list(args.prompt)
    if args.prompts:
        prompts.extend(
            line.strip() for line in args.prompts.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    if not prompts:
        raise SystemExit("error: no prompts given — use --prompt, --prompts, or --config")

    # Slot arrays (length 3)
    slots = ["", "", ""]
    folders = ["", "", ""]
    for i, p in enumerate(args.image[:3]):
        slots[i] = str(Path(p).resolve())
    for i, f in enumerate(args.image_folder[:3]):
        folders[i] = str(Path(f).resolve())

    global_images = [s for s in slots if s]

    guidance = args.guidance
    if guidance is None:
        from ui.constants import get_default_guidance
        guidance = get_default_guidance(args.model)

    return I2IBatchConfig(
        batch_name=args.name,
        prompts=prompts,
        global_images=global_images,
        global_image_slots=slots,
        global_image_folders=folders,
        prompt_image_overrides={},
        bot_task=args.bot_task,
        drop_think=args.drop_think,
        variations_per_prompt=args.variations,
        images_per_combo=args.imgs_per_combo,
        styles=args.style or ["None"],
        quality=args.quality,
        guidance_scale=float(guidance),
        random_seeds=(args.seed_mode == "random"),
        enhance=args.enhance,
        ollama_model=args.ollama_model,
        ollama_length=args.ollama_length,
        ollama_complexity=args.ollama_complexity,
    )


def _load_config_file(path: Path):
    import json
    from models.i2i_batch_config import I2IBatchConfig
    data = json.loads(path.read_text())
    return I2IBatchConfig.from_dict(data)


def _load_model_blocking(model_type: str) -> bool:
    """Run the model loader generator to completion. Returns True on success."""
    from core.model_manager import load_model, is_model_loaded
    for status in load_model(model_type):
        print(f"[LOAD] {status}")
        if "Error" in status or "ERROR" in status:
            return False
    return is_model_loaded()


def _cmd_i2i(args):
    # Apply NF4 dual-GPU toggle *before* load
    if args.nf4_dual_gpu:
        from core.settings import get_settings
        get_settings().nf4_dual_gpu = True
        print("[CLI] NF4 dual-GPU split enabled")

    # Build config
    if args.config:
        if not args.config.exists():
            print(f"error: config not found: {args.config}", file=sys.stderr)
            return 1
        print(f"[CLI] Loading batch config: {args.config}")
        config = _load_config_file(args.config)
    else:
        config = _build_config_from_args(args)

    # Resolve & validate reference images / folders
    from pathlib import Path as _P
    for p in config.global_images:
        if p and not _P(p).exists():
            print(f"error: reference image not found: {p}", file=sys.stderr)
            return 1
    for f in config.global_image_folders:
        if f and not _P(f).is_dir():
            print(f"error: reference folder not found: {f}", file=sys.stderr)
            return 1

    has_any_images = (
        bool(config.global_images)
        or any(config.global_image_folders)
        or bool(config.prompt_image_overrides)
    )
    if not has_any_images:
        print("error: i2i batch needs at least one reference image or folder",
              file=sys.stderr)
        return 1

    print(f"[CLI] Batch: {config.batch_name}")
    print(f"[CLI]   prompts: {len(config.prompts)}")
    print(f"[CLI]   global images: {[p for p in config.global_image_slots if p]}")
    print(f"[CLI]   folders: {[f for f in config.global_image_folders if f]}")
    print(f"[CLI]   total generations: {config.total_images()}")

    # Load model
    print(f"\n[CLI] Loading model: {args.model}")
    t0 = time.time()
    if not _load_model_blocking(args.model):
        print("error: model load failed", file=sys.stderr)
        return 2
    print(f"[CLI] Model loaded in {time.time() - t0:.1f}s")

    # Run the batch via I2IBatchWorker inside a QCoreApplication event loop
    from PySide6.QtCore import QCoreApplication
    from core.i2i_batch_worker import I2IBatchWorker

    qapp = QCoreApplication.instance() or QCoreApplication(sys.argv)

    worker = I2IBatchWorker(config)

    result = {"batch_dir": None, "count": 0, "error": None, "stopped": False}

    def _on_progress(cur, total, status):
        print(f"[{cur}/{total}] {status}")

    def _on_image(path):
        print(f"  ✓ {path}")

    def _on_cot(text):
        snippet = text.strip().splitlines()[0][:120] if text.strip() else ""
        if snippet:
            print(f"  [CoT] {snippet}")

    def _on_completed(batch_dir, count):
        result["batch_dir"] = batch_dir
        result["count"] = count
        qapp.quit()

    def _on_stopped(batch_dir, count):
        result["batch_dir"] = batch_dir
        result["count"] = count
        result["stopped"] = True
        qapp.quit()

    def _on_error(msg):
        result["error"] = msg
        print(f"[CLI] worker error: {msg[:400]}", file=sys.stderr)
        qapp.quit()

    worker.progress.connect(_on_progress)
    worker.image_ready.connect(_on_image)
    worker.cot_received.connect(_on_cot)
    worker.completed.connect(_on_completed)
    worker.stopped.connect(_on_stopped)
    worker.error.connect(_on_error)

    # SIGINT → request graceful stop
    import signal
    def _sigint(*_):
        print("\n[CLI] interrupt — requesting stop...", file=sys.stderr)
        worker.request_stop()
    signal.signal(signal.SIGINT, _sigint)

    print(f"\n[CLI] Starting batch...")
    t_start = time.time()
    worker.start()
    qapp.exec()
    worker.wait(5000)

    elapsed = time.time() - t_start
    print(f"\n[CLI] Finished in {elapsed:.1f}s")
    print(f"[CLI] Output dir: {result['batch_dir']}")
    print(f"[CLI] Images generated: {result['count']}")

    if args.output_json:
        import json
        args.output_json.write_text(json.dumps({
            "batch_name": config.batch_name,
            "batch_dir": result["batch_dir"],
            "count": result["count"],
            "elapsed_seconds": elapsed,
            "stopped": result["stopped"],
            "error": result["error"],
            "config": config.to_dict(),
        }, indent=2, default=str))
        print(f"[CLI] Summary: {args.output_json}")

    if result["error"]:
        return 3
    return 0


def main(argv=None):
    args = _parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
