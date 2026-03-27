"""
Qt Signal-emitting wrapper around the existing AppState singleton.
Bridges the threading.Lock-based backend state with PySide6 signal/slot UI updates.
"""

from PySide6.QtCore import QObject, Signal, QTimer


class DesktopState(QObject):
    """Central state manager that wraps the existing AppState and adds Qt signals."""

    # Model signals
    model_loading_started = Signal(str)           # model_type
    model_loading_progress = Signal(str)          # status_message
    model_loaded = Signal(str, str)               # model_type, status_msg
    model_load_failed = Signal(str)               # error_msg
    model_unloaded = Signal()

    # Generation signals
    generation_started = Signal(str)              # gen_id
    generation_progress = Signal(str, int, int)   # gen_id, current, total
    generation_image_ready = Signal(str, str, int, float)  # gen_id, path, seed, time
    generation_completed = Signal(str, int)        # gen_id, total_count
    generation_failed = Signal(str, str)           # gen_id, error_msg

    # Batch signals
    batch_started = Signal(str)                   # batch_name
    batch_progress = Signal(int, int, str, str)   # current, total, status, batch_dir
    batch_image_ready = Signal(str)               # image_path
    batch_completed = Signal(str, int)            # batch_dir, total_count
    batch_stopped = Signal(str, int)              # batch_dir, count_so_far

    # Ollama signals
    ollama_status_changed = Signal(str)           # status_message
    enhancement_started = Signal()
    enhancement_completed = Signal(str, str)      # original, enhanced
    enhancement_failed = Signal(str)              # error

    # GPU/VRAM signals
    vram_updated = Signal(int, float, float)      # gpu_idx, used_gb, total_gb
    gpus_detected = Signal(list)                  # gpu_list

    def __init__(self):
        super().__init__()
        self._state = None
        self._vram_timer = None

    def initialize(self):
        """Initialize the underlying AppState and detect GPUs."""
        from ui.state import get_state, init_gpus

        self._state = get_state()
        init_gpus()

        # Load RAM offload settings from persistent storage
        self._init_offload_settings()

        # Initialize wildcard manager
        self._init_wildcards()

        # NOTE: Ollama is now initialized lazily on first use (when enhance is clicked)
        # This avoids loading the LLM until actually needed
        self.ollama_status_changed.emit("LLM: Not initialized")

        # Load style presets
        self._init_styles()

        # Start VRAM monitoring timer
        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._update_vram)
        self._vram_timer.start(5000)  # Update every 5 seconds

        # Emit initial GPU list
        self.gpus_detected.emit(self._state.available_gpus)

        # Check for other processes using the GPU
        self._check_gpu_processes()

    @property
    def state(self):
        """Access the underlying AppState singleton."""
        if self._state is None:
            from ui.state import get_state
            self._state = get_state()
        return self._state

    def _init_offload_settings(self):
        """Load RAM offload settings from persistent storage into runtime state."""
        from core.settings import get_settings

        settings = get_settings()
        self.state.cpu_offload_enabled = settings.cpu_offload_enabled
        self.state.max_gpu_memory_gb = settings.max_gpu_memory_gb
        self.state.max_cpu_memory_gb = settings.max_cpu_memory_gb

        # Load global generation mode settings
        self.state.global_bot_task = settings.global_bot_task
        self.state.global_drop_think = settings.global_drop_think

        if self.state.cpu_offload_enabled:
            print(f"[INIT] RAM offload enabled: GPU≤{self.state.max_gpu_memory_gb}GB, RAM≤{self.state.max_cpu_memory_gb}GB")
        else:
            print("[INIT] RAM offload disabled")

        mode_names = {"image": "Direct", "think": "Think", "think_recaption": "Think+Rewrite", "recaption": "Rewrite"}
        print(f"[INIT] Generation mode: {mode_names.get(self.state.global_bot_task, self.state.global_bot_task)}")

    def _init_wildcards(self):
        """Initialize the wildcard manager."""
        try:
            from wildcard_utils import WildcardManager
            from ui.constants import WILDCARDS_FILE

            wm = WildcardManager(str(WILDCARDS_FILE))
            self.state.wildcard_manager = wm
            self.state.wildcard_available = True
            count = len(wm.get_available_wildcards()) if hasattr(wm, 'get_available_wildcards') else 0
            print(f"[INIT] Wildcards loaded: {count} categories")
        except Exception as e:
            print(f"[INIT] Wildcards not available: {e}")
            self.state.wildcard_available = False

    def _init_ollama(self, gpu_indices: list = None):
        """Initialize Ollama manager and enhancer (called lazily on first use).

        Args:
            gpu_indices: List of GPU indices to use for Ollama, or None to use settings.
                         GPU 1 (Blackwell) is always excluded automatically.
        """
        print(f"\n{'='*60}")
        print(f"[OLLAMA INIT] Starting Ollama initialization...")
        try:
            from ollama_manager import OllamaManager
            from ollama_prompts import PromptEnhancer
            from core.settings import get_settings

            print(f"[OLLAMA INIT] ✓ Imports successful")

            self.state.ollama_manager = OllamaManager()
            self.state.ollama_available = True

            # Get GPU selection from settings if not provided
            if gpu_indices is None:
                settings = get_settings()
                gpu_indices = settings.ollama_gpu_indices

            # Defensive: ensure gpu_indices is a list
            if not isinstance(gpu_indices, list):
                print(f"[OLLAMA INIT]   WARNING: gpu_indices was {type(gpu_indices).__name__}, using default [0, 2]")
                gpu_indices = [0, 2]
            print(f"[OLLAMA INIT]   Requested GPU indices: {gpu_indices}")

            # Filter out GPU 1 (Blackwell) - always exclude it
            safe_gpus = [i for i in gpu_indices if i != 1]
            if not safe_gpus:
                safe_gpus = [0]  # Fallback to GPU 0
            print(f"[OLLAMA INIT]   Safe GPU indices (excluding Blackwell): {safe_gpus}")

            if self.state.ollama_manager.is_running():
                print(f"[OLLAMA INIT] ✓ Ollama server already running")
                self.state.ollama_enhancer = PromptEnhancer()
                self.ollama_status_changed.emit(f"LLM: Running (GPU {','.join(map(str, safe_gpus))})")
                print(f"[OLLAMA INIT] ✓ PromptEnhancer created")
                print(f"[OLLAMA INIT] ✓ READY - Ollama connected on GPU(s) {safe_gpus}")
            else:
                # Start Ollama on the selected GPUs
                print(f"[OLLAMA INIT]   Ollama not running, starting...")
                self.ollama_status_changed.emit("LLM: Starting...")
                success, msg = self.state.ollama_manager.start(gpu_indices=safe_gpus)
                if success:
                    self.state.ollama_enhancer = PromptEnhancer()
                    self.ollama_status_changed.emit(f"LLM: Running (GPU {','.join(map(str, safe_gpus))})")
                    print(f"[OLLAMA INIT] ✓ READY - Ollama started on GPU(s) {safe_gpus}")
                else:
                    self.ollama_status_changed.emit(f"LLM: Failed - {msg[:30]}")
                    print(f"[OLLAMA INIT] ✗ FAILED to start Ollama: {msg}")
        except ImportError as e:
            print(f"[OLLAMA INIT] ✗ FAILED - Import error: {e}")
            print(f"[OLLAMA INIT]   Make sure ollama_manager.py and ollama_prompts.py exist")
            self.state.ollama_available = False
            self.ollama_status_changed.emit("LLM: Not installed")
        except Exception as e:
            print(f"[OLLAMA INIT] ✗ FAILED - {type(e).__name__}: {e}")
            self.state.ollama_available = False
            self.ollama_status_changed.emit("LLM: Not available")
        print(f"{'='*60}\n")

    def _init_styles(self):
        """Load style presets."""
        from ui.constants import DEFAULT_STYLE_PRESETS, STYLE_PRESETS_FILE
        import json

        self.state.style_presets = dict(DEFAULT_STYLE_PRESETS)

        try:
            if STYLE_PRESETS_FILE.exists():
                with open(STYLE_PRESETS_FILE, "r") as f:
                    saved = json.load(f)
                self.state.style_presets.update(saved)
                print(f"[INIT] Loaded {len(saved)} custom style presets")
        except Exception as e:
            print(f"[INIT] Style presets load error: {e}")

    def _update_vram(self):
        """Periodic VRAM usage update."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_idx = self.state.selected_gpu
                used = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
                total = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
                self.vram_updated.emit(gpu_idx, used, total)
        except Exception:
            pass

    def get_gpu_list(self) -> list:
        """Get the list of detected GPUs."""
        return self.state.available_gpus

    def get_model_type(self) -> str:
        """Get the current model type."""
        return self.state.model_type

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.state.model_loaded and self.state.model is not None

    def get_wildcard_manager(self):
        """Get the WildcardManager instance."""
        return self.state.wildcard_manager if self.state.wildcard_available else None

    def get_ollama_enhancer(self):
        """Get the PromptEnhancer instance."""
        return self.state.ollama_enhancer

    def get_style_presets(self) -> dict:
        """Get all style presets."""
        return dict(self.state.style_presets)

    def refresh_ollama(self, gpu_indices: list = None):
        """Refresh Ollama connection status."""
        self._init_ollama(gpu_indices)

    def ensure_ollama_ready(self, gpu_indices: list = None) -> bool:
        """Ensure Ollama is initialized and ready. Called lazily on first enhance.

        Returns True if Ollama is ready, False otherwise.
        """
        if not self.state.ollama_available or self.state.ollama_manager is None:
            self._init_ollama(gpu_indices)
        return self.state.ollama_available and self.state.ollama_enhancer is not None

    def is_ollama_initialized(self) -> bool:
        """Check if Ollama has been initialized."""
        return self.state.ollama_manager is not None

    def _check_gpu_processes(self):
        """Check for other processes using the GPU and warn user."""
        import os
        import subprocess

        my_pid = os.getpid()
        gpu_idx = self.state.selected_gpu

        try:
            # Use nvidia-smi to get GPU processes
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode != 0:
                return

            other_processes = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    pid = int(parts[0])
                    gpu_name = parts[1]
                    mem_mb = int(parts[2])

                    # Check if it's another python/hunyuan process using significant memory
                    if pid != my_pid and mem_mb > 5000:  # More than 5GB
                        # Check if it's a hunyuan process
                        try:
                            cmdline_result = subprocess.run(
                                ["ps", "-p", str(pid), "-o", "comm="],
                                capture_output=True, text=True, timeout=2
                            )
                            cmd = cmdline_result.stdout.strip()
                            if 'python' in cmd.lower():
                                other_processes.append((pid, gpu_name, mem_mb))
                        except:
                            pass

            if other_processes:
                print(f"\n{'!'*60}")
                print(f"[WARNING] Other GPU processes detected:")
                for pid, gpu, mem in other_processes:
                    print(f"  PID {pid}: {mem}MB on {gpu}")
                print(f"[WARNING] These may be old instances. Kill them with:")
                print(f"  kill {' '.join(str(p[0]) for p in other_processes)}")
                print(f"{'!'*60}\n")

        except Exception as e:
            print(f"[INIT] GPU process check failed: {e}")
