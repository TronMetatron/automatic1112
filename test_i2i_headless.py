#!/usr/bin/env python3
"""
Headless I2I (image-to-image) test for HunyuanImage-3.0 dual-GPU.
Tests the instruct model's ability to edit an existing image with think_recaption mode.
"""

import os
import sys
import time

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from PIL import Image


def gpu_status():
    """Print current GPU memory status."""
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = total - reserved
        print(f"  GPU {i} ({name}): {alloc:.1f}GB alloc, {reserved:.1f}GB reserved, {free:.1f}GB free / {total:.1f}GB")


def main():
    # ========== CONFIG ==========
    from ui.constants import MODEL_DIR
    model_id = str(MODEL_DIR / "HunyuanImage3-Instruct-SDNQ")

    # Find a test image
    test_image = "outputs/test_headless.png"
    if not os.path.exists(test_image):
        print(f"ERROR: No test image found. Generate one first with test_headless.py")
        return

    prompt = "Transform the image. Leave the image untouched except turn the cyberpunk boy into a cyberpunk girl with similar outfit and pose"
    bot_task = "think_recaption"
    steps = 50
    image_size = (1024, 1024)
    seed = 42
    # ============================

    print("=" * 60)
    print("HunyuanImage-3.0 I2I (Image-to-Image) Dual-GPU Test")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    gpu_status()
    print(f"\nModel: {model_id}")
    print(f"Input image: {test_image}")
    print(f"Prompt: {prompt[:80]}...")
    print(f"Bot task: {bot_task}, Steps: {steps}")

    # Import SDNQ
    print("\n[1/5] Importing SDNQ...")
    from sdnq import SDNQConfig
    print("  SDNQ imported")

    # Import transformers
    print("[2/5] Importing transformers...")
    from transformers import AutoModelForCausalLM

    # Dual-GPU device map
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        gpu_sizes = [(i, torch.cuda.get_device_properties(i).total_memory / (1024**3))
                     for i in range(num_gpus)]
        gpu_sizes.sort(key=lambda x: x[1], reverse=True)
        primary_idx = gpu_sizes[0][0]
        secondary_idx = gpu_sizes[1][0]

        # With MoE chunking patch, the 35GB int64 spike is eliminated.
        # Peak MoE allocation is now ~4.5GB (bool tensors in 32K chunks).
        # Combined with aggressive KV cache clearing, both T2I and I2I fit
        # on dual-GPU without ANY CPU offloading.
        device_map = {
            "vae": secondary_idx,
            "vision_model": secondary_idx,
            "vision_aligner": secondary_idx,
            "model": primary_idx,
            "patch_embed": primary_idx,
            "timestep_emb": primary_idx,
            "time_embed": primary_idx,
            "time_embed_2": primary_idx,
            "final_layer": primary_idx,
            "lm_head": primary_idx,
        }
        print(f"\n[3/5] Loading model dual-GPU (no CPU offload — MoE chunking patch applied):")
        print(f"  GPU {primary_idx} (Blackwell): all 32 transformer layers + diffusion (~44GB)")
        print(f"  GPU {secondary_idx} (5090): VAE + vision (~3GB)")
    else:
        device_map = "cuda:0"
        print(f"\n[3/5] Loading on single GPU")

    t0 = time.time()
    print("  Loading model... (7-8 minutes)")
    sys.stdout.flush()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        moe_impl="eager",
        local_files_only=True,
    )

    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")
    if hasattr(model, 'hf_device_map'):
        device_counts = {}
        for d in model.hf_device_map.values():
            device_counts[d] = device_counts.get(d, 0) + 1
        print(f"  Device distribution: {device_counts}")
    print("\nGPU after load:")
    gpu_status()

    # Load tokenizer
    print("\n[4/5] Loading tokenizer...")
    try:
        model.load_tokenizer(model_id, local_files_only=True)
    except Exception:
        model.load_tokenizer(model_id)
    print("  Tokenizer loaded")

    # Patch generation config for instruct
    for attr, val in [
        ('use_system_prompt', 'en_vanilla'),
        ('bot_task', 'image'),
        ('drop_think', False),
        ('diff_infer_steps', 50),
        ('diff_guidance_scale', 5.0),
        ('flow_shift', 3.0),
    ]:
        if not hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, val)

    # Load input image
    print(f"\n[5/5] Loading input image: {test_image}")
    input_image = Image.open(test_image).convert("RGB")
    print(f"  Image size: {input_image.size}")

    # Generate
    print(f"\n{'=' * 60}")
    print(f"I2I Generation: {bot_task}, {steps} steps")
    print(f"Prompt: {prompt[:80]}...")
    print(f"{'=' * 60}")

    try:
        t0 = time.time()
        print("\nGenerating (I2I with input image)...")
        sys.stdout.flush()

        result = model.generate_image(
            prompt=prompt,
            image=input_image,
            seed=seed,
            image_size=image_size,
            bot_task=bot_task,
            diff_infer_steps=steps,
            diff_guidance_scale=5.0,
        )

        gen_time = time.time() - t0
        print(f"\nI2I Generation completed in {gen_time:.1f}s")

        # Save result
        outpath = "outputs/test_i2i_headless.png"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        if isinstance(result, tuple):
            print(f"  Result is tuple with {len(result)} elements")
            for item in result:
                if hasattr(item, 'save'):
                    item.save(outpath)
                    print(f"  Image saved to {outpath}")
                    break
                elif isinstance(item, list) and len(item) > 0 and hasattr(item[0], 'save'):
                    item[0].save(outpath)
                    print(f"  Image saved to {outpath}")
                    break
        elif hasattr(result, 'save'):
            result.save(outpath)
            print(f"  Image saved to {outpath}")

        print("\nFinal GPU status:")
        gpu_status()
        print("\nSUCCESS!")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nGPU status at OOM:")
        gpu_status()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nGPU status at error:")
        gpu_status()


if __name__ == "__main__":
    main()
