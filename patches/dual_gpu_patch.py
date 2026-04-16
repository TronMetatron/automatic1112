#!/usr/bin/env python3
"""
Dual-GPU Patch for HunyuanImage-3.0 Instruct SDNQ

Applies three patches to the model's trust_remote_code files:
1. Bridge Patch: Adds .to(target_device) to scatter_ calls so VAE can live on a different GPU
2. KV Cache Clear: Adds torch.cuda.empty_cache() between think/recaption and diffusion phases
3. (Optional) Chunked MoE: Could reduce the 35GB MoE gating spike

This script patches the model source files in-place (with backups).
Run BEFORE loading the model.

Usage:
    python patches/dual_gpu_patch.py [--model-dir /path/to/HunyuanImage3-Instruct-SDNQ]
"""

import os
import re
import sys
import shutil
import argparse


def patch_file(filepath, patches, backup=True):
    """Apply text patches to a file. Each patch is (old_text, new_text)."""
    if not os.path.exists(filepath):
        print(f"  SKIP: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    applied = []

    for i, (old_text, new_text) in enumerate(patches):
        if old_text in content:
            content = content.replace(old_text, new_text, 1)
            applied.append(i)
        elif new_text in content:
            print(f"  Patch {i}: already applied")
        else:
            print(f"  Patch {i}: target text not found (may need manual review)")

    if content != original:
        if backup:
            backup_path = filepath + '.orig'
            if not os.path.exists(backup_path):
                shutil.copy2(filepath, backup_path)
                print(f"  Backup: {backup_path}")

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Applied patches: {applied}")
        return True
    else:
        print(f"  No changes needed")
        return False


def get_bridge_patches():
    """Strategy 1: Fix device mismatch in scatter_ operations.

    The model's instantiate_vae_image_tokens() calls hidden_states.scatter_()
    with src from patch_embed (which may be on a different GPU).
    We add .to(hidden_states.device) to ensure device consistency.
    """
    patches = []

    # Patch 1a: Tensor path (line ~1847)
    patches.append((
        """            image_scatter_index = index.masked_select(image_mask.bool()).reshape(bsz, -1)   # (bsz, num_patches)
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd),
                src=image_seq,
            )""",
        """            image_scatter_index = index.masked_select(image_mask.bool()).reshape(bsz, -1)   # (bsz, num_patches)
            # DUAL-GPU PATCH: ensure all tensors on same device before scatter
            _target = hidden_states.device
            hidden_states.scatter_(
                dim=1,
                index=image_scatter_index.unsqueeze(-1).repeat(1, 1, n_embd).to(_target),
                src=image_seq.to(_target),
            )"""
    ))

    # Patch 1b: List path (line ~1875)
    patches.append((
        """                hidden_states[i:i + 1].scatter_(
                    dim=1,
                    index=image_i_index.unsqueeze(-1).repeat(1, 1, n_embd),
                    src=image_i_seq.reshape(1, -1, n_embd),  # (1, img_seqlen, n_embd)
                )""",
        """                # DUAL-GPU PATCH: ensure all tensors on same device before scatter
                _target = hidden_states.device
                hidden_states[i:i + 1].scatter_(
                    dim=1,
                    index=image_i_index.unsqueeze(-1).repeat(1, 1, n_embd).to(_target),
                    src=image_i_seq.reshape(1, -1, n_embd).to(_target),  # (1, img_seqlen, n_embd)
                )"""
    ))

    # Patch 1c: The non-first-step path (line ~1812-1833)
    # When hidden_states is None, the function constructs hidden_states from
    # patch_embed output. These should already be on the same device, but
    # the timestep_emb might be on a different GPU.
    patches.append((
        """            t_emb = self.time_embed(timesteps)     # (bsz, n_embd)
            image_seq, token_h, token_w = self.patch_embed(images, t_emb)   # (bsz, num_patches, n_embd)""",
        """            t_emb = self.time_embed(timesteps.to(self.time_embed.mlp[0].weight.device))     # (bsz, n_embd)
            image_seq, token_h, token_w = self.patch_embed(images.to(t_emb.device), t_emb)   # (bsz, num_patches, n_embd)"""
    ))

    return patches


def get_kv_cache_clear_patch():
    """Strategy 2: Clear KV cache between think/recaption and diffusion.

    After text generation (think/recaption), the KV cache holds ~15GB.
    We free it before starting the diffusion loop to reduce peak memory.
    """
    patches = []

    # Insert cache clear before the diffusion generate call
    patches.append((
        """        # Generate image
        self.use_taylor_cache = use_taylor_cache
        model_inputs = self.prepare_model_inputs(""",
        """        # Generate image
        # DUAL-GPU PATCH: Free KV cache from text generation before starting diffusion.
        # This reduces peak VRAM by ~15GB, preventing OOM during the MoE activation spike.
        import gc
        if 'outputs' in dir():
            del outputs
        gc.collect()
        torch.cuda.empty_cache()

        self.use_taylor_cache = use_taylor_cache
        model_inputs = self.prepare_model_inputs("""
    ))

    return patches


def patch_model_dir(model_dir):
    """Apply all patches to a model directory."""
    model_file = os.path.join(model_dir, 'modeling_hunyuan_image_3.py')

    print(f"\nPatching: {model_file}")
    print("=" * 60)

    # Combine all patches
    all_patches = []

    print("\n[Strategy 1] Bridge Patch (device mismatch fix):")
    bridge = get_bridge_patches()
    all_patches.extend(bridge)

    print(f"\n[Strategy 2] KV Cache Clear (free 15GB before diffusion):")
    kv_clear = get_kv_cache_clear_patch()
    all_patches.extend(kv_clear)

    result = patch_file(model_file, all_patches)

    if result:
        print(f"\n{'=' * 60}")
        print("Patches applied successfully!")
        print("Original backed up to: modeling_hunyuan_image_3.py.orig")
    else:
        print(f"\n{'=' * 60}")
        print("No patches were needed (already applied or file structure changed)")

    return result


def patch_hf_cache():
    """Also patch the HuggingFace cache copy if it exists."""
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/HunyuanImage3_hyphen_Instruct_hyphen_SDNQ"
    )
    if os.path.isdir(cache_dir):
        print(f"\nAlso patching HF cache: {cache_dir}")
        patch_model_dir(cache_dir)

    # Check desktop app cache too
    desktop_cache = os.path.expanduser(
        "~/hunyuan_desktop/.hf_cache/modules/transformers_modules/HunyuanImage3_hyphen_Instruct_hyphen_SDNQ"
    )
    if os.path.isdir(desktop_cache):
        print(f"\nAlso patching desktop cache: {desktop_cache}")
        patch_model_dir(desktop_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply dual-GPU patches to HunyuanImage-3.0")
    parser.add_argument("--model-dir", required=True,
                        help="Path to model directory")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only patch HF cache, not the model directory")
    args = parser.parse_args()

    print("HunyuanImage-3.0 Dual-GPU Patch")
    print("================================")

    if not args.cache_only:
        patch_model_dir(args.model_dir)

    patch_hf_cache()

    print("\nDone! Restart the application to use the patched model.")
    print("\nRecommended device_map for loading:")
    print("""
    device_map = {
        "vae": 1,           # 5090 — runs before/after diffusion
        "vision_model": 1,  # 5090
        "vision_aligner": 1,# 5090
        "model": 0,         # Blackwell — all 32 MoE transformer layers
        "patch_embed": 0,   # Blackwell — used during diffusion loop
        "timestep_emb": 0,  # Blackwell
        "time_embed": 0,    # Blackwell
        "time_embed_2": 0,  # Blackwell
        "final_layer": 0,   # Blackwell
        "lm_head": 0,       # Blackwell
        "model.wte": 0,     # Blackwell
        "model.ln_f": 0,    # Blackwell
    }
""")
