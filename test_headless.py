#!/usr/bin/env python3
"""
Headless test script for HunyuanImage-3.0 dual-GPU loading and generation.
No Qt/UI needed — runs directly from terminal.
"""

import os
import sys
import time

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"
# Only expose the Blackwell GPU (95GB) — matches working config on old machine.
# Both GPUs visible for dual-GPU split (patched model fixes device mismatch)
# CUDA device 0 = Blackwell (95GB), device 1 = 5090 (31GB)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch

def gpu_status():
    """Print current GPU memory status."""
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        alloc = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = total - reserved
        print(f"  GPU {i} ({name}): {alloc:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free / {total:.1f}GB total")


def main():
    print("=" * 60)
    print("HunyuanImage-3.0 Headless Dual-GPU Test")
    print("=" * 60)

    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    gpu_status()

    # Model path
    from ui.constants import MODEL_DIR
    model_id = str(MODEL_DIR / "HunyuanImage3-Instruct-SDNQ")
    print(f"\nModel: {model_id}")

    # Import SDNQ
    print("\n[1/4] Importing SDNQ...")
    from sdnq import SDNQConfig
    print("  SDNQ imported (eager mode)")

    # Import transformers
    print("[2/4] Importing transformers...")
    from transformers import AutoModelForCausalLM

    # Configure dual-GPU with custom device_map
    # Strategy: VAE + vision on 5090, all transformer layers on Blackwell
    # Bridge patch fixes the device mismatch in scatter_ operations
    # KV cache clear patch frees ~15GB between text gen and diffusion
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        # Find primary (largest) and secondary GPUs
        gpu_sizes = [(i, torch.cuda.get_device_properties(i).total_memory / (1024**3))
                     for i in range(num_gpus)]
        gpu_sizes.sort(key=lambda x: x[1], reverse=True)
        primary_idx = gpu_sizes[0][0]    # Blackwell (95GB)
        secondary_idx = gpu_sizes[1][0]  # 5090 (31GB)

        device_map = {
            # Non-MoE components → 5090 (no activation spike here)
            "vae": secondary_idx,
            "vision_model": secondary_idx,
            "vision_aligner": secondary_idx,
            # MoE transformer + diffusion components → Blackwell
            "model": primary_idx,         # All 32 transformer layers
            "patch_embed": primary_idx,
            "timestep_emb": primary_idx,
            "time_embed": primary_idx,
            "time_embed_2": primary_idx,
            "final_layer": primary_idx,
            "lm_head": primary_idx,
        }

        print(f"\n[3/4] Loading model with DUAL-GPU split (patched):")
        print(f"  GPU {primary_idx} (Blackwell): All 32 MoE layers + diffusion (~44GB)")
        print(f"  GPU {secondary_idx} (5090): VAE + vision (~4GB)")
        print(f"  KV cache will be freed before diffusion (patch applied)")
    else:
        device_map = "cuda:0"
        print(f"\n[3/4] Loading model on single GPU")

    t0 = time.time()
    print("  Loading... (this takes 2-5 minutes)")
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

    # Show device distribution
    if hasattr(model, 'hf_device_map'):
        device_counts = {}
        for name, d in model.hf_device_map.items():
            device_counts[d] = device_counts.get(d, 0) + 1
        print(f"  Device distribution: {device_counts}")

    print("\nGPU status after load:")
    gpu_status()

    # Load tokenizer
    print("\n[4/4] Loading tokenizer...")
    try:
        model.load_tokenizer(model_id, local_files_only=True)
        print("  Tokenizer loaded")
    except Exception as e:
        print(f"  Tokenizer warning: {e}")
        try:
            model.load_tokenizer(model_id)
            print("  Tokenizer loaded (fallback)")
        except Exception as e2:
            print(f"  Tokenizer failed: {e2}")

    # Patch generation config for instruct
    if not hasattr(model.generation_config, 'use_system_prompt'):
        model.generation_config.use_system_prompt = "en_vanilla"
    if not hasattr(model.generation_config, 'bot_task'):
        model.generation_config.bot_task = "image"
    if not hasattr(model.generation_config, 'drop_think'):
        model.generation_config.drop_think = False
    if not hasattr(model.generation_config, 'diff_infer_steps'):
        model.generation_config.diff_infer_steps = 50
    if not hasattr(model.generation_config, 'diff_guidance_scale'):
        model.generation_config.diff_guidance_scale = 5.0
    if not hasattr(model.generation_config, 'flow_shift'):
        model.generation_config.flow_shift = 3.0

    # Test generation
    prompt = "A young boy with steampunk goggles standing in a workshop filled with brass gears and copper pipes"
    print(f"\n{'=' * 60}")
    print(f"Test generation: think_recaption mode, 50 steps, 1024x1024")
    print(f"Prompt: {prompt[:80]}...")
    print(f"{'=' * 60}")

    try:
        t0 = time.time()
        print("\nGenerating...")
        sys.stdout.flush()

        result = model.generate_image(
            prompt=prompt,
            seed=42,
            image_size=(1024, 1024),
            bot_task="think_recaption",
            diff_infer_steps=50,
            diff_guidance_scale=5.0,
        )

        gen_time = time.time() - t0
        print(f"\nGeneration completed in {gen_time:.1f}s")

        # Save result
        if result is not None:
            if isinstance(result, tuple):
                print(f"  Result is tuple with {len(result)} elements")
                # Try to find the image in the result
                for i, item in enumerate(result):
                    if hasattr(item, 'save'):
                        outpath = "outputs/test_headless.png"
                        os.makedirs(os.path.dirname(outpath), exist_ok=True)
                        item.save(outpath)
                        print(f"  Image saved to {outpath}")
                        break
                    elif isinstance(item, list) and len(item) > 0 and hasattr(item[0], 'save'):
                        outpath = "outputs/test_headless.png"
                        os.makedirs(os.path.dirname(outpath), exist_ok=True)
                        item[0].save(outpath)
                        print(f"  Image saved to {outpath}")
                        break
            elif hasattr(result, 'save'):
                outpath = "outputs/test_headless.png"
                os.makedirs(os.path.dirname(outpath), exist_ok=True)
                result.save(outpath)
                print(f"  Image saved to {outpath}")
            else:
                print(f"  Result type: {type(result)}")

        print("\nFinal GPU status:")
        gpu_status()
        print("\nSUCCESS!")

    except torch.cuda.OutOfMemoryError as e:
        print(f"\nOOM ERROR: {e}")
        print("\nGPU status at OOM:")
        gpu_status()
        print("\nTrying with torch.cuda.empty_cache() and retry...")
        torch.cuda.empty_cache()
        gpu_status()

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nGPU status at error:")
        gpu_status()


if __name__ == "__main__":
    main()
