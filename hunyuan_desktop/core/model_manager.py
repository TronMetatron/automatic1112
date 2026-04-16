"""
Model management for HunyuanImage-3.0.
Handles loading, unloading, and status of the image generation model.
Supports both Base and Instruct model variants.
"""

import os
import gc
import torch
from typing import Generator

from ui.state import get_state
from ui.constants import MODEL_PATHS, MODEL_INFO, is_int8_model, is_firered_model

# Compat aliases
MODEL_PATH = MODEL_PATHS.get("base", "")
DEEPGEN_REPO = None
DEEPGEN_CHECKPOINTS = None
FIRERED_MODEL = MODEL_PATHS.get("firered", None)

# Prevent CUDA memory fragmentation (critical for large models)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def _patch_nf4_dual_gpu_device_alignment(model):
    """Wrap image-token instantiation methods so cross-device tensor ops
    don't blow up with 'tensors on different devices'.

    Under our dual-GPU NF4 device map, VAE / vision submodules live on the
    secondary GPU while the main transformer (and hidden_states) live on the
    primary. Many functions in the Hunyuan model do raw tensor ops (scatter,
    masked_select, arithmetic) between tensors produced on different devices.
    This patch wraps the relevant methods to move tensors to the right
    device at each boundary.
    """
    import torch

    # Device where vision_model's weights live (usually secondary GPU)
    try:
        vision_device = next(model.vision_model.parameters()).device
    except Exception:
        vision_device = None

    def _to(target, t):
        if t is None or target is None:
            return t
        if isinstance(t, torch.Tensor) and t.device != target:
            return t.to(target)
        return t

    # --- instantiate_vae_image_tokens ---
    orig_vae = model.instantiate_vae_image_tokens

    def patched_vae(hidden_states, timesteps, images, image_mask,
                    guidance=None, timesteps_r=None):
        tgt = hidden_states.device if hidden_states is not None else None
        image_mask = _to(tgt, image_mask)
        if isinstance(timesteps, torch.Tensor):
            timesteps = _to(tgt, timesteps)
        guidance = _to(tgt, guidance)
        timesteps_r = _to(tgt, timesteps_r)
        return orig_vae(hidden_states, timesteps, images, image_mask,
                        guidance=guidance, timesteps_r=timesteps_r)

    model.instantiate_vae_image_tokens = patched_vae

    # --- _forward_vision_encoder: move inputs to vision_device, output back ---
    # The caller (instantiate_vit_image_tokens) sets model._vit_target_device
    # so we know where to put the output tensor.
    orig_fve = model._forward_vision_encoder

    def patched_fve(images, **image_kwargs):
        if vision_device is not None:
            images = _to(vision_device, images)
            image_kwargs = {
                k: _to(vision_device, v) for k, v in image_kwargs.items()
            }
        out = orig_fve(images, **image_kwargs)
        tgt = getattr(model, "_vit_target_device", None)
        if isinstance(out, torch.Tensor):
            out = _to(tgt, out)
        return out

    model._forward_vision_encoder = patched_fve

    # --- instantiate_vit_image_tokens ---
    orig_vit = model.instantiate_vit_image_tokens

    def patched_vit(hidden_states, images, image_masks, image_kwargs):
        tgt = hidden_states.device if hidden_states is not None else None
        image_masks = _to(tgt, image_masks)
        # Stash target device for patched_fve to read
        model._vit_target_device = tgt
        try:
            return orig_vit(hidden_states, images, image_masks, image_kwargs)
        finally:
            model._vit_target_device = None

    model.instantiate_vit_image_tokens = patched_vit

    # --- ragged_final_layer ---
    orig_rfl = model.ragged_final_layer

    def patched_rfl(hidden_states, image_mask, timesteps,
                    token_h, token_w, first_step=None):
        tgt = hidden_states.device if hidden_states is not None else None
        image_mask = _to(tgt, image_mask)
        if isinstance(timesteps, torch.Tensor):
            timesteps = _to(tgt, timesteps)
        return orig_rfl(hidden_states, image_mask, timesteps,
                        token_h, token_w, first_step=first_step)

    model.ragged_final_layer = patched_rfl

    # --- instantiate_continuous_tokens / guidance_tokens / timestep_r_tokens ---
    # All three do scatter_(index=..., src=self.<emb>(...)) where hidden_states
    # and the embedding submodule live on the primary GPU but the *_index /
    # timesteps tensors may arrive from other devices.
    def _wrap_scatter_method(method_name):
        orig = getattr(model, method_name)

        def patched(hidden_states, values=None, indices=None):
            tgt = hidden_states.device if hidden_states is not None else None

            def _coerce(t):
                if isinstance(t, torch.Tensor):
                    return _to(tgt, t)
                if isinstance(t, list):
                    return [_coerce(x) for x in t]
                return t

            return orig(hidden_states, _coerce(values), _coerce(indices))

        setattr(model, method_name, patched)

    _wrap_scatter_method("instantiate_continuous_tokens")
    _wrap_scatter_method("instantiate_guidance_tokens")
    _wrap_scatter_method("instantiate_timestep_r_tokens")
    print(f"[LOAD] NF4 dual-GPU device-alignment patch applied "
          f"(vision_device={vision_device})")


def load_model(model_type: str = None) -> Generator[str, None, None]:
    """Load the quantized HunyuanImage-3.0 model.
    
    Args:
        model_type: "base" or "instruct". If None, uses state.model_type.

    Yields status messages during loading process.
    """
    state = get_state()
    
    # Determine which model to load
    if model_type is None:
        model_type = state.model_type
    
    # Validate model type — fall back to base for unavailable models
    if model_type not in MODEL_PATHS:
        print(f"[LOAD] Unknown model type '{model_type}', falling back to 'base'")
        model_type = "base"

    # DeepGen uses a completely different loading path
    if model_type == "deepgen":
        yield from _load_deepgen_model()
        return

    # FireRed uses a diffusers pipeline loading path
    if is_firered_model(model_type):
        yield from _load_firered_model()
        return

    # INT8 BnB models use a different loading path
    if is_int8_model(model_type):
        yield from _load_int8_model(model_type)
        return
    
    model_path = MODEL_PATHS[model_type]
    model_info = MODEL_INFO[model_type]

    # Quick check without lock
    if state.model_loaded and state.model is not None:
        if state.model_type == model_type:
            yield f"{model_info['name']} already loaded and ready!"
            return
        else:
            yield f"Different model loaded ({state.model_type}). Unload first to switch."
            return

    # Try to acquire lock (non-blocking to give feedback)
    if not state.model_load_lock.acquire(blocking=False):
        yield "Model is currently being loaded by another request..."
        # Wait for the other load to complete
        state.model_load_lock.acquire()
        state.model_load_lock.release()
        if state.model_loaded:
            yield "Model loaded by another request. Ready!"
        return

    try:
        # Double-check after acquiring lock
        if state.model_loaded and state.model is not None:
            yield f"{model_info['name']} already loaded and ready!"
            return

        # Check if GPU already has significant memory usage
        gpu_idx = state.selected_gpu
        if torch.cuda.is_available():
            # Remap GPU index for CUDA_VISIBLE_DEVICES filtering
            visible_device_count = torch.cuda.device_count()
            check_device_idx = 0 if visible_device_count == 1 else min(gpu_idx, visible_device_count - 1)

            # Use memory_reserved which includes cached memory
            mem_reserved = torch.cuda.memory_reserved(check_device_idx) / (1024**3)
            mem_allocated = torch.cuda.memory_allocated(check_device_idx) / (1024**3)
            mem_total = torch.cuda.get_device_properties(check_device_idx).total_memory / (1024**3)

            print(f"[LOAD] GPU {gpu_idx} memory: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved, {mem_total:.1f}GB total")

            if mem_reserved > 30:  # More than 30GB suggests model is already loaded
                # Check if this is a kept-alive model from a previous session
                from core.settings import get_settings
                settings = get_settings()
                if settings.keep_model_loaded and settings.last_model_type == model_type:
                    yield f"Detected {mem_reserved:.1f}GB on GPU — re-attaching to {model_info['name']}..."
                    print(f"[LOAD] Re-attaching to model kept in VRAM ({mem_reserved:.1f}GB reserved)")
                    # Reset the flag so next quit asks again
                    settings.keep_model_loaded = False
                else:
                    yield f"ERROR: GPU {gpu_idx} already has {mem_reserved:.1f}GB reserved!"
                    yield f"This usually means a model is loaded or a previous load failed."
                    yield f"Click 'Unload Model' to free memory, or restart the app if that doesn't work."
                    if not state.model_loaded:
                        state.model_loaded = False
                        state.model = None
                    return

        print(f"[LOAD] Loading {model_info['name']}...")
        yield f"Step 1: Importing libraries for {model_info['name']}..."
        from transformers import AutoModelForCausalLM

        is_nf4 = model_type in ("nf4", "distil_nf4")

        if not is_nf4:
            print("[LOAD] Importing SDNQ...")
            from sdnq import SDNQConfig  # Registers SDNQ into transformers

        model_id = str(model_path)

        # Get selected GPU from state
        gpu_index = state.selected_gpu

        # Handle CUDA_VISIBLE_DEVICES filtering:
        # When CUDA_VISIBLE_DEVICES is set (e.g., "1"), PyTorch renumbers GPUs.
        # GPU 1 becomes cuda:0. We must remap the index to avoid "invalid device ordinal".
        if torch.cuda.is_available():
            visible_device_count = torch.cuda.device_count()
            if visible_device_count == 1:
                # Only one GPU visible after filtering → always use cuda:0
                device_index = 0
                device = "cuda:0"
            elif gpu_index < visible_device_count:
                # Multiple GPUs and index is valid → use selected index
                device_index = gpu_index
                device = f"cuda:{gpu_index}"
            else:
                # Selected index out of range → use last available GPU
                device_index = visible_device_count - 1
                device = f"cuda:{device_index}"
        else:
            device = "cpu"
            device_index = -1

        print(f"[LOAD] Loading model from {model_id} to {device}...")
        print(f"[LOAD] Selected GPU {gpu_index}, mapped to {device} ({visible_device_count} visible)")

        num_gpus = torch.cuda.device_count()

        if is_nf4:
            # NF4 4-bit BitsAndBytes quantization. Default: single 48GB GPU.
            # Optional dual-GPU split: VAE + vision on secondary GPU to free
            # VRAM on the primary for multi-image I2I conditioning.
            gpu_sizes = [(i, torch.cuda.get_device_properties(i).total_memory / (1024**3))
                         for i in range(num_gpus)]
            gpu_sizes.sort(key=lambda x: x[1], reverse=True)
            largest_gpu = gpu_sizes[0][0]
            target_device = f"cuda:{largest_gpu}"
            device_index = largest_gpu

            from core.settings import get_settings
            use_nf4_dual = get_settings().nf4_dual_gpu and num_gpus >= 2

            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            if use_nf4_dual:
                secondary_idx = gpu_sizes[1][0]
                primary_name = torch.cuda.get_device_name(largest_gpu)
                secondary_name = torch.cuda.get_device_name(secondary_idx)
                # Mirror SDNQ split: keep transformer/diffusion bulk on primary,
                # offload smaller vision/VAE pieces to the secondary GPU.
                custom_device_map = {
                    "vae": secondary_idx,
                    "vision_model": secondary_idx,
                    "vision_aligner": secondary_idx,
                    "model": largest_gpu,
                    "patch_embed": largest_gpu,
                    "timestep_emb": largest_gpu,
                    "timestep_r_emb": largest_gpu,
                    "guidance_emb": largest_gpu,
                    "time_embed": largest_gpu,
                    "time_embed_2": largest_gpu,
                    "final_layer": largest_gpu,
                    "lm_head": largest_gpu,
                }
                yield (f"Step 2: Loading {model_info['name']} NF4 dual-GPU "
                       f"({primary_name} + {secondary_name})...")
                print(f"[LOAD] NF4 dual-GPU: transformer on GPU {largest_gpu}, "
                      f"VAE+vision on GPU {secondary_idx}")
                load_kwargs = dict(device_map=custom_device_map)
            else:
                yield f"Step 2: Loading {model_info['name']} NF4 on {torch.cuda.get_device_name(largest_gpu)}..."
                print(f"[LOAD] NF4 single-GPU: loading to GPU {largest_gpu} "
                      f"({torch.cuda.get_device_name(largest_gpu)})")
                load_kwargs = dict(device_map=target_device)

            state.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quant_config,
                attn_implementation="sdpa",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                low_cpu_mem_usage=True,
                **load_kwargs,
            )

            # Dual-GPU NF4: patch the model's multimodal token instantiation
            # methods so image_mask tensors get coerced to hidden_states.device
            # before bare (un-hooked) tensor ops like `index.masked_select(mask)`.
            # Without this, multi-image I2I fails with "tensors on different
            # devices" when the VAE / vision submodules live on a different GPU
            # than the main transformer. Safe no-op for single-GPU.
            if use_nf4_dual:
                _patch_nf4_dual_gpu_device_alignment(state.model)
        else:
            # SDNQ models — dual-GPU strategy with patched model (bridge patch fixes device mismatch):
            # VAE + vision on secondary GPU (small, no MoE spikes)
            # All 32 transformer layers + diffusion on primary GPU (handles MoE activation spikes)
            # KV cache cleared between text gen and diffusion phases (patched in model code)
            #
            # Tested: 11 min total, ~5.2s/step diffusion, peak 93.3GB/95GB on Blackwell
            if num_gpus >= 2:
                gpu_sizes = [(i, torch.cuda.get_device_properties(i).total_memory / (1024**3))
                             for i in range(num_gpus)]
                gpu_sizes.sort(key=lambda x: x[1], reverse=True)
                primary_idx = gpu_sizes[0][0]
                secondary_idx = gpu_sizes[1][0]

                custom_device_map = {
                    "vae": secondary_idx,
                    "vision_model": secondary_idx,
                    "vision_aligner": secondary_idx,
                    "model": primary_idx,
                    "patch_embed": primary_idx,
                    "timestep_emb": primary_idx,
                    "timestep_r_emb": primary_idx,   # Distil model only
                    "guidance_emb": primary_idx,      # Distil model only
                    "time_embed": primary_idx,
                    "time_embed_2": primary_idx,
                    "final_layer": primary_idx,
                    "lm_head": primary_idx,
                }

                primary_name = torch.cuda.get_device_name(primary_idx)
                secondary_name = torch.cuda.get_device_name(secondary_idx)
                yield f"Step 2: Loading {model_info['name']} dual-GPU ({primary_name} + {secondary_name})..."
                print(f"[LOAD] Dual-GPU: transformer layers on GPU {primary_idx}, VAE+vision on GPU {secondary_idx}")

                load_kwargs = dict(device_map=custom_device_map)
            else:
                yield f"Step 2: Loading {model_info['name']} on {device}..."
                load_kwargs = dict(device_map=device)

            state.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                attn_implementation="sdpa",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                moe_impl="eager",
                local_files_only=True,
                low_cpu_mem_usage=True,
                **load_kwargs,
            )

        # Log device distribution
        if hasattr(state.model, 'hf_device_map'):
            device_counts = {}
            for d in state.model.hf_device_map.values():
                device_counts[d] = device_counts.get(d, 0) + 1
            print(f"[LOAD] Device distribution: {device_counts}")

        print("[LOAD] Model weights loaded!")
        yield "Step 3: Model weights loaded, configuring..."

        # Instruct/Distil model specific setup
        if model_type in ("instruct", "distil", "nf4", "distil_nf4"):
            yield f"Step 4: Configuring {model_info['name']}..."

            # Patch generation_config for Instruct/Distil/NF4 model
            # CRITICAL: sequence_template must be "instruct" for instruct-tuned models.
            # SDNQ quantized models ship with "pretrain" which causes malformed input
            # (no system/user/assistant framing), breaking think/recaption and I2I quality.
            state.model.generation_config.sequence_template = "instruct"

            # use_system_prompt must be a valid type: en_vanilla, en_recaption, en_think_recaption, dynamic, custom
            # "None" causes get_system_prompt to return None which breaks .strip()
            if not hasattr(state.model.generation_config, 'use_system_prompt'):
                state.model.generation_config.use_system_prompt = "en_vanilla"
            if not hasattr(state.model.generation_config, 'bot_task'):
                state.model.generation_config.bot_task = "image"
            if not hasattr(state.model.generation_config, 'drop_think'):
                state.model.generation_config.drop_think = False

            # Set default steps based on model type
            default_steps = 8 if model_type in ("distil", "distil_nf4") else 50
            if not hasattr(state.model.generation_config, 'diff_infer_steps'):
                state.model.generation_config.diff_infer_steps = default_steps
            if not hasattr(state.model.generation_config, 'diff_guidance_scale'):
                state.model.generation_config.diff_guidance_scale = 5.0
            if not hasattr(state.model.generation_config, 'flow_shift'):
                state.model.generation_config.flow_shift = 3.0
            
            # Move vision model to same GPU only in single-GPU mode.
            # In dual-GPU mode, accelerate hooks manage device placement — calling .to()
            # moves weights but hooks still route inputs to the device_map device, causing
            # a cuda:0 vs cuda:1 mismatch in the vision encoder.
            if num_gpus < 2 and hasattr(state.model, 'vision_model') and state.model.vision_model is not None:
                print(f"[LOAD] Moving vision model to {device}...")
                state.model.vision_model = state.model.vision_model.to(device)
                if hasattr(state.model.vision_model, 'encoder'):
                    state.model.vision_model.encoder = state.model.vision_model.encoder.to(device)

        state.model_type = model_type

        # Try to load tokenizer (must complete before marking model as loaded)
        print("[LOAD] Loading tokenizer...")
        yield "Step 5: Loading tokenizer..."
        try:
            state.model.load_tokenizer(model_id)
            print("[LOAD] Tokenizer loaded successfully!")
        except Exception as tok_err:
            print(f"[LOAD] Tokenizer load failed: {tok_err}")
            print("[LOAD] WARNING: Model loaded without tokenizer - think/recaption tasks may fail")

        # Mark model as loaded only after tokenizer is ready
        state.model_loaded = True

        # Show GPU memory usage
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(device_index) / (1024**3)
            mem_total = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
            yield f"{model_info['name']} loaded! Using {mem_used:.1f}GB / {mem_total:.1f}GB GPU memory. Ready to generate."
        else:
            yield f"{model_info['name']} loaded successfully! Ready to generate images."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[LOAD ERROR] {str(e)}")
        print(f"[LOAD TRACEBACK]\n{error_details}")
        yield f"Error loading model: {str(e)}\n\nDetails:\n{error_details}"
    finally:
        state.model_load_lock.release()


def _load_deepgen_model() -> "Generator[str, None, None]":
    """Load DeepGen 1.0 model (separate path from HunyuanImage).

    DeepGen uses xtuner/mmengine config system with a Qwen2.5-VL + SD3.5M Kontext backbone.
    """
    import sys

    state = get_state()
    model_info = MODEL_INFO["deepgen"]
    deepgen_repo = str(DEEPGEN_REPO)

    # Quick check without lock
    if state.model_loaded and state.model is not None:
        if state.model_type == "deepgen":
            yield f"{model_info['name']} already loaded and ready!"
            return
        else:
            yield f"Different model loaded ({state.model_type}). Unload first to switch."
            return

    if not state.model_load_lock.acquire(blocking=False):
        yield "Model is currently being loaded by another request..."
        state.model_load_lock.acquire()
        state.model_load_lock.release()
        if state.model_loaded:
            yield "Model loaded by another request. Ready!"
        return

    try:
        if state.model_loaded and state.model is not None:
            yield f"{model_info['name']} already loaded and ready!"
            return

        gpu_idx = state.selected_gpu
        if torch.cuda.is_available():
            visible_count = torch.cuda.device_count()
            device_index = 0 if visible_count == 1 else min(gpu_idx, visible_count - 1)
            device = f"cuda:{device_index}"
        else:
            device = "cpu"
            device_index = -1

        yield "Step 1: Setting up DeepGen imports..."
        print(f"[DEEPGEN] Loading DeepGen 1.0 from {deepgen_repo}")

        # Add deepgen repo to sys.path for its imports
        if deepgen_repo not in sys.path:
            sys.path.insert(0, deepgen_repo)
        existing_pypath = os.environ.get("PYTHONPATH", "")
        if deepgen_repo not in existing_pypath:
            os.environ["PYTHONPATH"] = f"{deepgen_repo}:{existing_pypath}" if existing_pypath else deepgen_repo

        os.environ.setdefault("HF_HOME", "/media/james/BigDrive/AI/cache/huggingface")

        # Temporarily disable offline mode for DeepGen model build.
        # transformers 4.57+ has a bug where tokenizer loading calls model_info()
        # even for cached models. The base models (Qwen2.5-VL-3B, UniPic2-SD3.5M-Kontext)
        # are already cached locally, so this just reads from cache.
        _saved_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        _saved_tf_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)

        from xtuner.registry import BUILDER
        from mmengine.config import Config
        from xtuner.model.utils import guess_load_checkpoint

        yield "Step 2: Loading DeepGen config..."
        config_path = os.path.join(deepgen_repo, "configs/models/deepgen_scb.py")
        config = Config.fromfile(config_path)

        yield "Step 3: Building DeepGen model (3B VLM + 2B DiT)..."
        print(f"[DEEPGEN] Building model from config...")
        model = BUILDER.build(config.model)

        # Try to load checkpoint (RL > SFT > Pretrain)
        # Check both the central models dir and the repo's local checkpoints
        ckpt_base = str(DEEPGEN_CHECKPOINTS)
        checkpoint_candidates = [
            os.path.join(ckpt_base, "RL/MR-GDPO_final.pt"),
            os.path.join(ckpt_base, "SFT/iter_400000.pth"),
            os.path.join(ckpt_base, "Pretrain/iter_200000.pth"),
            os.path.join(deepgen_repo, "checkpoints/DeepGen_CKPT/RL/MR-GDPO_final.pt"),
            os.path.join(deepgen_repo, "checkpoints/DeepGen_CKPT/SFT/iter_400000.pth"),
            os.path.join(deepgen_repo, "checkpoints/DeepGen_CKPT/Pretrain/iter_200000.pth"),
        ]

        ckpt_path = None
        for candidate in checkpoint_candidates:
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

        if ckpt_path:
            yield f"Step 4: Loading checkpoint ({os.path.basename(ckpt_path)})..."
            print(f"[DEEPGEN] Loading checkpoint: {ckpt_path}")
            if ckpt_path.endswith('.pt'):
                ckpt_state_dict = torch.load(ckpt_path, map_location='cpu')
            else:
                ckpt_state_dict = guess_load_checkpoint(ckpt_path)
            missing, unexpected = model.load_state_dict(ckpt_state_dict, strict=False)
            if unexpected:
                print(f"[DEEPGEN] Unexpected parameters: {len(unexpected)}")
            del ckpt_state_dict
            gc.collect()
        else:
            yield "Step 4: No checkpoint found (using base weights)..."
            print("[DEEPGEN] WARNING: No checkpoint found. Download with:")
            print("  huggingface-cli download deepgenteam/DeepGen-1.0 --local-dir checkpoints --repo-type model")

        yield f"Step 5: Moving model to {device}..."
        print(f"[DEEPGEN] Moving model to {device}")
        model = model.to(device=device)
        model = model.to(model.dtype)
        model.eval()

        state.model = model
        state.model_loaded = True
        state.model_type = "deepgen"

        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(device_index) / (1024**3)
            mem_total = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
            yield f"{model_info['name']} loaded! Using {mem_used:.1f}GB / {mem_total:.1f}GB GPU memory. Ready to generate."
        else:
            yield f"{model_info['name']} loaded! Ready to generate."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[DEEPGEN LOAD ERROR] {str(e)}")
        print(f"[DEEPGEN LOAD TRACEBACK]\n{error_details}")
        yield f"Error loading DeepGen: {str(e)}\n\nDetails:\n{error_details}"
    finally:
        # Restore offline mode env vars
        if _saved_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = _saved_offline
        if _saved_tf_offline is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = _saved_tf_offline
        state.model_load_lock.release()


class _SaveLatentsAndBail(Exception):
    """Exception used to unwind the pipeline call stack after diffusion,
    saving latents for separate VAE decode."""
    def __init__(self, latents):
        import torch
        self.latents = latents.cpu().clone()


def _load_int8_model(model_type: str) -> "Generator[str, None, None]":
    """Load a HunyuanImage-3 model with BnB INT8 quantization.

    Uses bitsandbytes Linear8bitLt for transformer layers with device_map="auto"
    to distribute across GPU and CPU. Always requires the Blackwell GPU (96GB).
    """
    state = get_state()
    model_path = MODEL_PATHS[model_type]
    model_info = MODEL_INFO[model_type]
    is_distil = "distil" in model_type
    is_base = model_type == "base_int8"

    # Quick check without lock
    if state.model_loaded and state.model is not None:
        if state.model_type == model_type:
            yield f"{model_info['name']} already loaded and ready!"
            return
        else:
            yield f"Different model loaded ({state.model_type}). Unload first to switch."
            return

    if not state.model_load_lock.acquire(blocking=False):
        yield "Model is currently being loaded by another request..."
        state.model_load_lock.acquire()
        state.model_load_lock.release()
        if state.model_loaded:
            yield "Model loaded by another request. Ready!"
        return

    try:
        if state.model_loaded and state.model is not None:
            yield f"{model_info['name']} already loaded and ready!"
            return

        # Check GPU memory
        gpu_idx = state.selected_gpu
        if torch.cuda.is_available():
            visible_count = torch.cuda.device_count()
            device_index = 0 if visible_count == 1 else min(gpu_idx, visible_count - 1)
            mem_reserved = torch.cuda.memory_reserved(device_index) / (1024**3)

            if mem_reserved > 30:
                yield f"ERROR: GPU {gpu_idx} already has {mem_reserved:.1f}GB reserved!"
                yield "Click 'Unload Model' to free memory first."
                return
        else:
            device_index = 0

        model_id = str(model_path)
        print(f"[INT8 LOAD] Loading {model_info['name']} from {model_id}")

        yield f"Step 1: Importing libraries for {model_info['name']}..."
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # INT8 quantization config — skip modules incompatible with INT8
        skip_modules = [
            "vae", "vision_model", "vision_aligner",
            "patch_embed", "final_layer",
            "time_embed", "time_embed_2", "timestep_emb",
            "guidance_emb", "timestep_r_emb",
            "lm_head", "model.wte", "model.ln_f",
        ]
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=skip_modules,
        )

        # Memory budgets: Base and Distil can use more GPU (no CFG, batch=1)
        if is_base or is_distil:
            gpu_budget = f"{state.max_gpu_memory_gb}GiB" if state.cpu_offload_enabled else "55GiB"
            cpu_budget = f"{state.max_cpu_memory_gb}GiB" if state.cpu_offload_enabled else "60GiB"
        else:
            gpu_budget = f"{state.max_gpu_memory_gb}GiB" if state.cpu_offload_enabled else "40GiB"
            cpu_budget = f"{state.max_cpu_memory_gb}GiB" if state.cpu_offload_enabled else "70GiB"

        max_memory = {device_index: gpu_budget, "cpu": cpu_budget}

        yield f"Step 2: Loading {model_info['name']} with INT8 quantization (GPU≤{gpu_budget}, CPU≤{cpu_budget})..."
        print(f"[INT8 LOAD] max_memory: {max_memory}")

        t0 = __import__('time').time()
        state.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            moe_impl="eager",
            moe_drop_tokens=True,
            low_cpu_mem_usage=True,
        )
        load_time = __import__('time').time() - t0
        print(f"[INT8 LOAD] Model loaded in {load_time:.0f}s")

        # Log device distribution
        if hasattr(state.model, 'hf_device_map'):
            device_counts = {}
            for d in state.model.hf_device_map.values():
                device_counts[d] = device_counts.get(d, 0) + 1
            print(f"[INT8 LOAD] Device distribution: {device_counts}")

        yield "Step 3: Model weights loaded, configuring..."

        # Distil workaround: load_tokenizer references missing config.model_version
        if is_distil and not hasattr(state.model.config, 'model_version'):
            state.model.config.model_version = None

        # Configure generation_config
        yield f"Step 4: Configuring {model_info['name']}..."

        if is_base:
            # Base INT8: T2I only, no bot_task/use_system_prompt
            default_steps = 50
            if not hasattr(state.model.generation_config, 'diff_infer_steps'):
                state.model.generation_config.diff_infer_steps = default_steps
        else:
            # Instruct/Distil INT8: instruct-style features
            if not hasattr(state.model.generation_config, 'use_system_prompt'):
                state.model.generation_config.use_system_prompt = "en_vanilla"
            if not hasattr(state.model.generation_config, 'bot_task'):
                state.model.generation_config.bot_task = "image"
            if not hasattr(state.model.generation_config, 'drop_think'):
                state.model.generation_config.drop_think = False

            default_steps = 8 if is_distil else 50
            if not hasattr(state.model.generation_config, 'diff_infer_steps'):
                state.model.generation_config.diff_infer_steps = default_steps
            if not hasattr(state.model.generation_config, 'diff_guidance_scale'):
                state.model.generation_config.diff_guidance_scale = 5.0
            if not hasattr(state.model.generation_config, 'flow_shift'):
                state.model.generation_config.flow_shift = 3.0

        state.model_type = model_type

        # Load tokenizer (must complete before marking model as loaded)
        yield "Step 5: Loading tokenizer..."
        print("[INT8 LOAD] Loading tokenizer...")
        try:
            state.model.load_tokenizer(model_id)
            print("[INT8 LOAD] Tokenizer loaded successfully!")
        except Exception as tok_err:
            print(f"[INT8 LOAD] Tokenizer load failed: {tok_err}")
            print("[INT8 LOAD] WARNING: Model loaded without tokenizer - think/recaption tasks may fail")

        # Mark model as loaded only after tokenizer is ready
        state.model_loaded = True

        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(device_index) / (1024**3)
            mem_total = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
            yield (f"{model_info['name']} loaded in {load_time:.0f}s! "
                   f"Using {mem_used:.1f}GB / {mem_total:.1f}GB GPU. Ready to generate.")
        else:
            yield f"{model_info['name']} loaded in {load_time:.0f}s! Ready to generate."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[INT8 LOAD ERROR] {str(e)}")
        print(f"[INT8 LOAD TRACEBACK]\n{error_details}")
        yield f"Error loading INT8 model: {str(e)}\n\nDetails:\n{error_details}"
    finally:
        state.model_load_lock.release()


def generate_int8(model, prompt, seed, image_size, bot_task="image",
                  steps=None, guidance_scale=5.0, i2i_images=None,
                  progress_callback=None):
    """Generate an image using INT8 model with the exception-trick VAE decode.

    Pipeline phases:
      0. (think/recaption only) Generate CoT text, then free KV cache
      1. Monkey-patch VAE decode, run diffusion (exception saves latents)
      2. gc + empty_cache to free KV cache (model stays loaded)
      3. Move VAE to GPU, decode latents, move VAE back

    Returns (PIL.Image, seed) or (PIL.Image, seed, cot_text) tuple.
    """
    import time as _time

    state = get_state()
    is_distil = "distil" in state.model_type
    is_base = state.model_type == "base_int8"

    # Determine CUDA device
    gpu_idx = state.selected_gpu
    visible_count = torch.cuda.device_count()
    device_index = 0 if visible_count == 1 else min(gpu_idx, visible_count - 1)
    device = f"cuda:{device_index}"

    # Phase 0: For think/recaption modes, generate CoT text FIRST (before VAE trick)
    # This avoids KV cache from text gen competing with diffusion for VRAM
    cot_text = None
    if bot_task in ("think", "recaption") and not is_base and not i2i_images:
        if progress_callback:
            progress_callback("Phase 0: Generating thinking text...")
        print(f"[INT8 GEN] Phase 0: CoT text generation (bot_task={bot_task})")

        try:
            from hunyuan_image_3.system_prompt import get_system_prompt
            system_prompt = get_system_prompt("en_vanilla", bot_task, None)

            t_cot = _time.time()
            model_inputs = model.prepare_model_inputs(
                prompt=prompt, bot_task=bot_task, system_prompt=system_prompt,
                max_new_tokens=2048,
            )
            outputs = model._generate(**model_inputs, verbose=0)
            cot_text = model.get_cot_text(outputs[0])
            cot_time = _time.time() - t_cot

            if isinstance(cot_text, list):
                cot_text = cot_text[0]
            print(f"[INT8 GEN] CoT generated in {cot_time:.0f}s ({len(cot_text)} chars)")

            # Free CoT KV cache before diffusion
            del outputs, model_inputs
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            mem_after_cot = torch.cuda.memory_allocated(device_index) / (1024**3)
            print(f"[INT8 GEN] After CoT cleanup: {mem_after_cot:.1f}GB allocated")

            if progress_callback:
                progress_callback(f"Phase 0: Thinking done ({cot_time:.0f}s)")
        except Exception as e:
            print(f"[INT8 GEN] CoT generation failed: {e}, falling back to direct mode")
            cot_text = None
            gc.collect()
            torch.cuda.empty_cache()

    # Save reference to VAE before monkey-patching
    vae = model.vae
    original_vae_decode = vae.decode

    # Phase 1: Monkey-patch VAE and run diffusion
    def _intercept_vae_decode(latents_input, *args, **kwargs):
        raise _SaveLatentsAndBail(latents_input)
    vae.decode = _intercept_vae_decode

    if progress_callback:
        progress_callback("Phase 1: Running diffusion...")

    saved_latents = None
    t0 = _time.time()
    try:
        if is_base:
            # Base INT8: T2I only, uses stream=True, no bot_task/use_system_prompt
            result = model.generate_image(
                prompt=prompt,
                seed=seed,
                image_size=image_size,
                stream=True,
                diff_infer_steps=steps,
            )
            # Base model with stream=True returns a generator — consume it
            if hasattr(result, '__iter__'):
                for _ in result:
                    pass
        elif i2i_images:
            # I2I mode (instruct/distil INT8)
            img_arg = i2i_images if len(i2i_images) > 1 else i2i_images[0]
            model.generate_image(
                prompt=prompt,
                seed=seed,
                image=img_arg,
                image_size="auto",
                use_system_prompt="en_vanilla",
                bot_task=bot_task,
                infer_align_image_size=True,
                diff_infer_steps=steps,
                diff_guidance_scale=guidance_scale,
            )
        elif cot_text is not None:
            # Think/recaption with pre-generated CoT: call model internals directly
            # to skip redundant CoT generation and use already-freed memory
            from hunyuan_image_3.system_prompt import get_system_prompt
            system_prompt = get_system_prompt("en_vanilla", bot_task, None)

            # Handle drop_think: switch system prompt if enabled
            if model.generation_config.drop_think and system_prompt:
                from hunyuan_image_3.system_prompt import t2i_system_prompts
                system_prompt = t2i_system_prompts["en_recaption"][0]

            # Handle auto image size
            if image_size == "auto":
                ratio_inputs = model.prepare_model_inputs(
                    prompt=prompt, cot_text=cot_text, bot_task="img_ratio",
                    system_prompt=system_prompt, seed=seed,
                )
                ratio_outputs = model._generate(**ratio_inputs, verbose=0)
                ratio_index = ratio_outputs[0, -1].item() - model._tkwrapper.ratio_token_offset
                if ratio_index < 0 or ratio_index >= len(model.image_processor.reso_group):
                    ratio_index = 16
                reso = model.image_processor.reso_group[ratio_index]
                image_size = reso.height, reso.width
                del ratio_outputs, ratio_inputs
                gc.collect()
                torch.cuda.empty_cache()

            # Generate image with pre-computed cot_text
            model_inputs = model.prepare_model_inputs(
                prompt=prompt, cot_text=cot_text, system_prompt=system_prompt,
                mode="gen_image", seed=seed, image_size=image_size,
            )
            model._generate(
                **model_inputs, verbose=0,
                diff_infer_steps=steps,
                diff_guidance_scale=guidance_scale,
            )
        else:
            # T2I mode (instruct/distil INT8) — direct or fallback
            model.generate_image(
                prompt=prompt,
                seed=seed,
                image_size=image_size,
                use_system_prompt="en_vanilla",
                bot_task=bot_task,
                diff_infer_steps=steps,
                diff_guidance_scale=guidance_scale,
                max_new_tokens=2048,
            )
    except _SaveLatentsAndBail as e:
        saved_latents = e.latents
        diffusion_time = _time.time() - t0
        print(f"[INT8 GEN] Diffusion completed in {diffusion_time:.0f}s, latents shape={saved_latents.shape}")
    except Exception as e:
        # Restore VAE before re-raising
        vae.decode = original_vae_decode
        raise

    # Always restore original VAE decode
    vae.decode = original_vae_decode

    if saved_latents is None:
        raise RuntimeError("INT8 generation failed: no latents captured from diffusion")

    # Phase 2: Free KV cache and intermediate state
    if progress_callback:
        progress_callback("Phase 2: Freeing diffusion state...")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    mem_after_gc = torch.cuda.memory_allocated(device_index) / (1024**3)
    print(f"[INT8 GEN] After gc: {mem_after_gc:.1f}GB allocated")

    # Phase 3: VAE decode
    if progress_callback:
        progress_callback("Phase 3: VAE decode...")

    # Note original VAE device/dtype so we can restore after decode
    vae_original_device = next(vae.parameters()).device
    vae_original_dtype = next(vae.parameters()).dtype

    # Move VAE to CPU first, remove accelerate hooks, then clear GPU to avoid fragmentation
    vae = vae.cpu()
    try:
        from accelerate.hooks import remove_hook_from_module
        for module in vae.modules():
            remove_hook_from_module(module)
    except Exception:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    try:
        # Move VAE to GPU for decode (bfloat16 to halve memory)
        torch.backends.cudnn.benchmark = False
        vae = vae.to(device=device, dtype=torch.bfloat16)
        latents = saved_latents.to(device=device, dtype=torch.bfloat16)
        del saved_latents

        t0 = _time.time()
        with torch.inference_mode():
            image_tensor = vae.decode(latents, return_dict=False)[0]
        vae_time = _time.time() - t0
        print(f"[INT8 GEN] VAE decode completed in {vae_time:.1f}s")

        del latents

    except torch.cuda.OutOfMemoryError:
        # OOM during VAE — delete model and retry with full GPU
        print("[INT8 GEN] VAE OOM with model loaded — deleting model and retrying")
        del state.model
        state.model = None
        state.model_loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        vae = vae.to(device=device, dtype=torch.bfloat16)
        latents = saved_latents.to(device=device, dtype=torch.bfloat16) if saved_latents is not None else None
        if latents is None:
            raise RuntimeError("Lost latents during OOM recovery")

        with torch.inference_mode():
            image_tensor = vae.decode(latents, return_dict=False)[0]
        del latents
        print("[INT8 GEN] VAE decode succeeded after model deletion (model needs reload)")

    # Handle temporal dimension (video VAE compatibility)
    if hasattr(vae, "ffactor_temporal"):
        if image_tensor.shape[2] == 1:
            image_tensor = image_tensor.squeeze(2)

    # Convert to PIL
    from diffusers.image_processor import VaeImageProcessor
    processor = VaeImageProcessor()
    do_denormalize = [True] * image_tensor.shape[0]
    images = processor.postprocess(image_tensor, output_type="pil", do_denormalize=do_denormalize)

    del image_tensor
    gc.collect()
    torch.cuda.empty_cache()

    # Restore VAE to original device/dtype for next generation
    try:
        vae.to(device=vae_original_device, dtype=vae_original_dtype)
    except Exception:
        pass  # Best-effort restore

    gc.collect()
    torch.cuda.empty_cache()

    if cot_text:
        return images[0], seed, cot_text
    return images[0], seed


def generate_deepgen(model, prompt, seed, width, height, steps, guidance_scale, src_image_path=None):
    """Run DeepGen generation and return a PIL Image.

    Called from generation_worker.py for the 'deepgen' model type.
    """
    import random
    from einops import rearrange

    # Defensive parameter clamping for DeepGen
    width = max(256, min(1024, round(width / 64) * 64))
    height = max(256, min(1024, round(height / 64) * 64))
    if steps < 25:
        steps = 50
    if guidance_scale == 5.0:
        guidance_scale = 4.0

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    generator = torch.Generator(device=model.device).manual_seed(seed)

    prompt_list = [prompt.strip()]
    cfg_prompt = [""]

    # Handle img2img if source image provided
    pixel_values_src = None
    if src_image_path and os.path.exists(src_image_path):
        import numpy as np
        from PIL import Image as PILImage
        src_img = PILImage.open(src_image_path).convert('RGB')
        src_img = src_img.resize((width, height))
        pv = torch.from_numpy(np.array(src_img)).float()
        pv = pv / 255.0
        pv = 2.0 * pv - 1.0
        pv = rearrange(pv, 'h w c -> c h w')
        pixel_values_src = [[pv[None].to(dtype=model.dtype, device=model.device)]]

    output = model.generate(
        prompt=prompt_list,
        cfg_prompt=cfg_prompt,
        pixel_values_src=pixel_values_src,
        cfg_scale=guidance_scale,
        num_steps=steps,
        progress_bar=False,
        generator=generator,
        height=height,
        width=width,
    )

    # Post-process tensor -> PIL Image
    output = rearrange(output, 'b c h w -> b h w c')
    output = torch.clamp(127.5 * output + 128.0, 0, 255)
    output = output.to("cpu", dtype=torch.uint8).numpy()

    from PIL import Image as PILImage
    return PILImage.fromarray(output[0]), seed


def _load_firered_model() -> "Generator[str, None, None]":
    """Load FireRed Image Edit 1.1 model (diffusers QwenImageEditPlusPipeline).

    FireRed is a general-purpose image editing model based on the Qwen-Image backbone.
    Uses ~30GB+ VRAM with bfloat16.
    """
    state = get_state()
    model_info = MODEL_INFO["firered"]
    model_path = str(FIRERED_MODEL)

    # Quick check without lock
    if state.model_loaded and state.model is not None:
        if state.model_type == "firered":
            yield f"{model_info['name']} already loaded and ready!"
            return
        else:
            yield f"Different model loaded ({state.model_type}). Unload first to switch."
            return

    if not state.model_load_lock.acquire(blocking=False):
        yield "Model is currently being loaded by another request..."
        state.model_load_lock.acquire()
        state.model_load_lock.release()
        if state.model_loaded:
            yield "Model loaded by another request. Ready!"
        return

    _saved_offline = None
    try:
        if state.model_loaded and state.model is not None:
            yield f"{model_info['name']} already loaded and ready!"
            return

        gpu_idx = state.selected_gpu
        if torch.cuda.is_available():
            visible_count = torch.cuda.device_count()
            device_index = 0 if visible_count == 1 else min(gpu_idx, visible_count - 1)
            device = f"cuda:{device_index}"
        else:
            device = "cpu"
            device_index = -1

        yield "Step 1: Importing diffusers pipeline..."
        print(f"[FIRERED] Loading FireRed Image Edit 1.1 from {model_path}")

        # Temporarily disable offline mode - diffusers may need to resolve config
        _saved_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.setdefault("HF_HOME", "/media/james/BigDrive/AI/cache/huggingface")

        from diffusers import QwenImageEditPlusPipeline

        yield "Step 2: Loading pipeline (this may take a few minutes)..."
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )

        yield f"Step 3: Moving pipeline to {device}..."
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        # Enable VAE memory optimizations
        yield "Step 4: Enabling VAE tiling & slicing..."
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        state.model = pipe
        state.model_loaded = True
        state.model_type = "firered"

        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(device_index) / (1024**3)
            mem_total = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
            yield f"{model_info['name']} loaded! Using {mem_used:.1f}GB / {mem_total:.1f}GB GPU memory. Ready to edit."
        else:
            yield f"{model_info['name']} loaded! Ready to edit."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[FIRERED LOAD ERROR] {str(e)}")
        print(f"[FIRERED LOAD TRACEBACK]\n{error_details}")
        yield f"Error loading FireRed: {str(e)}\n\nDetails:\n{error_details}"
    finally:
        if _saved_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = _saved_offline
        state.model_load_lock.release()


def generate_firered(model, prompt, seed, width, height, steps, true_cfg_scale,
                     src_image_path=None, num_images=1):
    """Run FireRed Image Edit 1.1 generation and return a PIL Image.

    Called from generation_worker.py for the 'firered' model type.
    Supports both image editing (I2I with 1-3 input images) and text-to-image (no input).

    Key pipeline details:
      - guidance_scale is FIXED at 1.0 (internal requirement)
      - true_cfg_scale controls actual CFG strength (default 4.0, range 1.0-10.0)
      - negative_prompt is always a single space " "
      - height/width=None lets the model auto-detect from input images
      - Accepts up to 3 input images as PIL Image list
    """
    import random
    from PIL import Image as PILImage

    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    # Load input image(s) - max 3 supported
    images = []
    if src_image_path:
        if isinstance(src_image_path, (list, tuple)):
            for p in src_image_path[:3]:
                if os.path.exists(str(p)):
                    images.append(PILImage.open(str(p)).convert("RGB"))
        elif os.path.exists(str(src_image_path)):
            images.append(PILImage.open(str(src_image_path)).convert("RGB"))

    # Determine height/width: None = auto from input images, or explicit
    out_height = None
    out_width = None
    if width and height and width > 0 and height > 0:
        # Round to nearest multiple of 8 (pipeline requirement)
        out_width = round(width / 8) * 8
        out_height = round(height / 8) * 8

    # Pipeline requires an image — for T2I, create a blank canvas
    if not images:
        t2i_w = out_width or 1024
        t2i_h = out_height or 1024
        images = [PILImage.new("RGB", (t2i_w, t2i_h), (255, 255, 255))]
        print(f"[FIRERED] T2I mode: using blank {t2i_w}x{t2i_h} canvas")

    # Determine device for generator
    device = model.device if hasattr(model, 'device') else "cuda"

    inputs = {
        "image": images,
        "prompt": prompt,
        "height": out_height,
        "width": out_width,
        "generator": torch.Generator(device=device).manual_seed(seed),
        "guidance_scale": 1.0,          # Fixed at 1.0 - required by pipeline
        "true_cfg_scale": true_cfg_scale,  # Actual CFG control (default 4.0)
        "negative_prompt": " ",         # Single space - required by pipeline
        "num_inference_steps": steps,
        "num_images_per_prompt": num_images,
    }

    print(f"[FIRERED] Generating: {len(images)} input image(s), steps={steps}, "
          f"true_cfg={true_cfg_scale}, size={out_width}x{out_height}, seed={seed}")

    with torch.inference_mode():
        result = model(**inputs)

    output_image = result.images[0]
    return output_image, seed


def unload_model() -> str:
    """Unload the image generation model to free GPU memory."""
    state = get_state()

    model_info = MODEL_INFO.get(state.model_type, {})
    model_name = model_info.get('name', state.model_type)

    # Get memory before unload
    mem_before = 0
    if torch.cuda.is_available():
        gpu_idx = state.selected_gpu
        visible_count = torch.cuda.device_count()
        check_idx = 0 if visible_count == 1 else min(gpu_idx, visible_count - 1)
        mem_before = torch.cuda.memory_reserved(check_idx) / (1024**3)

    try:
        # Delete model if it exists
        if state.model is not None:
            print(f"[UNLOAD] Deleting model object...")
            del state.model
            state.model = None

        # Always reset state flags
        state.model_loaded = False

        # Force garbage collection
        print(f"[UNLOAD] Running garbage collection...")
        gc.collect()

        # Clear ALL CUDA memory (aggressive cleanup)
        if torch.cuda.is_available():
            print(f"[UNLOAD] Clearing CUDA cache...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Check if memory was freed
            mem_after = torch.cuda.memory_reserved(check_idx) / (1024**3)
            freed = mem_before - mem_after

            if freed > 1:
                return f"{model_name} unloaded! Freed {freed:.1f}GB. Current: {mem_after:.1f}GB"
            elif mem_after > 10:
                return f"State reset. Warning: {mem_after:.1f}GB still reserved (may need app restart)"
            else:
                return f"{model_name} unloaded! Current usage: {mem_after:.1f}GB"

        return f"{model_name} unloaded successfully!"

    except Exception as e:
        # Even on error, reset the state
        state.model = None
        state.model_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f"Error during unload: {str(e)} (state reset anyway)"


def force_cleanup_gpu() -> str:
    """Force cleanup GPU memory even if model state is inconsistent.

    Use this when model loading failed mid-way and memory is stuck.
    """
    state = get_state()

    print(f"[FORCE CLEANUP] Starting aggressive GPU cleanup...")

    # Get memory before
    mem_before = 0
    if torch.cuda.is_available():
        gpu_idx = state.selected_gpu
        visible_count = torch.cuda.device_count()
        check_idx = 0 if visible_count == 1 else min(gpu_idx, visible_count - 1)
        mem_before = torch.cuda.memory_reserved(check_idx) / (1024**3)
        print(f"[FORCE CLEANUP] Memory before: {mem_before:.1f}GB")

    # Clear any model reference
    if state.model is not None:
        try:
            del state.model
        except:
            pass
        state.model = None

    state.model_loaded = False

    # Aggressive garbage collection
    gc.collect()
    gc.collect()  # Run twice

    # Clear CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset memory stats
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except:
            pass

        mem_after = torch.cuda.memory_reserved(check_idx) / (1024**3)
        freed = mem_before - mem_after

        print(f"[FORCE CLEANUP] Memory after: {mem_after:.1f}GB (freed {freed:.1f}GB)")

        if mem_after > 10:
            return f"Cleanup done but {mem_after:.1f}GB still held. If stuck, restart the app."
        return f"Cleanup complete! Freed {freed:.1f}GB. Current: {mem_after:.1f}GB"

    return "Cleanup complete (no CUDA available)"


def get_gpu_memory_info() -> dict:
    """Get detailed GPU memory information for diagnostics."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    state = get_state()
    gpu_idx = state.selected_gpu
    visible_count = torch.cuda.device_count()
    check_idx = 0 if visible_count == 1 else min(gpu_idx, visible_count - 1)

    try:
        props = torch.cuda.get_device_properties(check_idx)
        return {
            "gpu_index": gpu_idx,
            "device_index": check_idx,
            "name": props.name,
            "total_gb": props.total_memory / (1024**3),
            "allocated_gb": torch.cuda.memory_allocated(check_idx) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(check_idx) / (1024**3),
            "model_loaded_flag": state.model_loaded,
            "model_object_exists": state.model is not None,
        }
    except Exception as e:
        return {"error": str(e)}


def get_model_status() -> str:
    """Get current model loading status and GPU memory info."""
    state = get_state()

    status_lines = []

    # Image model status
    gpu_idx = state.selected_gpu
    model_info = MODEL_INFO.get(state.model_type, {})
    model_name = model_info.get('name', state.model_type)

    if state.model_loaded and state.model is not None:
        status_lines.append(f"**{model_name}: LOADED** (GPU {gpu_idx})")
    else:
        status_lines.append(f"**Image Model: NOT LOADED** (will use GPU {gpu_idx})")

    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            mem_used = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
            mem_total = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)

            status_lines.append(f"GPU {gpu_idx}: {gpu_name}")
            status_lines.append(f"Memory: {mem_used:.1f}GB / {mem_total:.1f}GB")

            if not state.model_loaded:
                status_lines.append("*Ready to load*")
        except Exception as e:
            status_lines.append(f"GPU info error: {e}")
    else:
        status_lines.append("No CUDA GPU available")

    # LLM/Ollama status
    status_lines.append("")  # Separator
    llm_gpu_idx = state.selected_ollama_gpu

    if state.ollama_available and state.ollama_manager:
        try:
            if state.ollama_manager.is_running():
                models = state.ollama_manager.list_models()
                if models:
                    model_names = [m['name'] for m in models[:3]]  # Show first 3
                    models_str = ", ".join(model_names)
                    if len(models) > 3:
                        models_str += f" (+{len(models)-3} more)"
                    status_lines.append(f"**LLM: RUNNING** (GPU {llm_gpu_idx})")
                    status_lines.append(f"Models: {models_str}")
                else:
                    status_lines.append(f"**LLM: RUNNING** (GPU {llm_gpu_idx}, no models)")
            else:
                status_lines.append(f"**LLM: NOT RUNNING** (will use GPU {llm_gpu_idx})")
                status_lines.append("*Start with: ollama serve*")
        except Exception as e:
            status_lines.append(f"**LLM: ERROR** - {e}")
    else:
        status_lines.append("**LLM: NOT AVAILABLE**")

    return "\n".join(status_lines)


def get_model():
    """Get the loaded model instance."""
    state = get_state()
    return state.model


def is_model_loaded() -> bool:
    """Check if the model is currently loaded."""
    state = get_state()
    return state.model_loaded and state.model is not None


def get_model_type() -> str:
    """Get the current model type setting."""
    state = get_state()
    return state.model_type


def set_model_type(model_type: str) -> str:
    """Set the model type to load next time.
    
    Note: If a model is already loaded, you need to unload it first
    before loading a different model type.
    """
    state = get_state()
    
    if model_type not in MODEL_PATHS:
        print(f"[MODEL] Unknown model type '{model_type}', falling back to 'base'")
        model_type = "base"
    
    if state.model_loaded:
        if state.model_type == model_type:
            return f"Model type already set to {model_type} (and loaded)."
        else:
            return f"Model type set to {model_type}. Unload current model ({state.model_type}) first to switch."
    
    state.model_type = model_type
    model_info = MODEL_INFO[model_type]
    return f"Model type set to: {model_info['name']}"


def get_available_model_types() -> list:
    """Get list of available model types for UI dropdown."""
    return [
        ("HunyuanImage-3.0 Base (T2I)", "base"),
        ("HunyuanImage-3.0 Instruct (T2I + I2I)", "instruct"),
        ("HunyuanImage-3.0 Instruct-Distil (Fast T2I + I2I)", "distil"),
        ("DeepGen 1.0 Unified (T2I + I2I, 5B)", "deepgen"),
        ("FireRed 1.1 Image Edit (T2I + I2I)", "firered"),
        ("Base INT8 (T2I, 96GB GPU)", "base_int8"),
        ("Instruct INT8 (T2I + I2I, 96GB GPU)", "instruct_int8"),
        ("Distil INT8 (Fast T2I + I2I, 96GB GPU)", "distil_int8"),
    ]
