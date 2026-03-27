"""Persistent application settings using QSettings."""

from PySide6.QtCore import QSettings


class AppSettings:
    """Wrapper around QSettings for persistent user preferences."""

    def __init__(self):
        self._settings = QSettings("HunyuanImage", "Desktop")

    # Window geometry
    def save_geometry(self, geometry):
        self._settings.setValue("window/geometry", geometry)

    def load_geometry(self):
        return self._settings.value("window/geometry")

    def save_window_state(self, state):
        self._settings.setValue("window/state", state)

    def load_window_state(self):
        return self._settings.value("window/state")

    # Last-used generation settings
    def save_gen_settings(self, settings: dict):
        self._settings.beginGroup("generation")
        for key, value in settings.items():
            self._settings.setValue(key, value)
        self._settings.endGroup()

    def load_gen_settings(self) -> dict:
        self._settings.beginGroup("generation")
        keys = self._settings.childKeys()
        result = {k: self._settings.value(k) for k in keys}
        self._settings.endGroup()
        return result

    # Model preferences
    @property
    def last_model_type(self) -> str:
        from ui.constants import MODEL_PATHS
        saved = self._settings.value("model/type", "base")
        if saved not in MODEL_PATHS:
            return "base"
        return saved

    @last_model_type.setter
    def last_model_type(self, value: str):
        self._settings.setValue("model/type", value)

    @property
    def last_gpu_index(self) -> int:
        # Default to GPU 0 (Blackwell 95GB on this machine)
        return int(self._settings.value("model/gpu_index", 0))

    @last_gpu_index.setter
    def last_gpu_index(self, value: int):
        self._settings.setValue("model/gpu_index", value)

    @property
    def last_ollama_gpu_index(self) -> int:
        """Legacy single GPU index (for backwards compatibility)."""
        return int(self._settings.value("ollama/gpu_index", 0))

    @last_ollama_gpu_index.setter
    def last_ollama_gpu_index(self, value: int):
        self._settings.setValue("ollama/gpu_index", value)

    @property
    def ollama_gpu_indices(self) -> list:
        """List of GPU indices to use for Ollama. GPU 1 (Blackwell) is excluded by default."""
        val = self._settings.value("ollama/gpu_indices", None)
        # Default: GPUs 0 and 2, excluding GPU 1 (Blackwell)
        default = [0, 2]
        if val is None or isinstance(val, bool):
            return default
        if isinstance(val, str):
            try:
                import json
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    return [int(i) for i in parsed]
                return default
            except:
                return default
        if isinstance(val, list):
            return [int(i) for i in val]
        if isinstance(val, int):
            return [val]
        return default

    @ollama_gpu_indices.setter
    def ollama_gpu_indices(self, value: list):
        import json
        self._settings.setValue("ollama/gpu_indices", json.dumps(value))

    # Ollama settings
    @property
    def last_ollama_model(self) -> str:
        return self._settings.value("ollama/model", "qwen2.5:7b-instruct")

    @last_ollama_model.setter
    def last_ollama_model(self, value: str):
        self._settings.setValue("ollama/model", value)

    @property
    def last_ollama_length(self) -> str:
        return self._settings.value("ollama/length", "medium")

    @last_ollama_length.setter
    def last_ollama_length(self, value: str):
        self._settings.setValue("ollama/length", value)

    @property
    def last_ollama_complexity(self) -> str:
        return self._settings.value("ollama/complexity", "detailed")

    @last_ollama_complexity.setter
    def last_ollama_complexity(self, value: str):
        self._settings.setValue("ollama/complexity", value)

    # Output directory
    @property
    def last_output_dir(self) -> str:
        from ui.constants import OUTPUT_DIR
        return self._settings.value("output/directory", str(OUTPUT_DIR))

    @last_output_dir.setter
    def last_output_dir(self, value: str):
        self._settings.setValue("output/directory", value)

    # CPU Offloading settings
    @property
    def cpu_offload_enabled(self) -> bool:
        val = self._settings.value("model/cpu_offload_enabled", False)
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    @cpu_offload_enabled.setter
    def cpu_offload_enabled(self, value: bool):
        self._settings.setValue("model/cpu_offload_enabled", value)

    @property
    def max_gpu_memory_gb(self) -> int:
        return int(self._settings.value("model/max_gpu_memory_gb", 80))

    @max_gpu_memory_gb.setter
    def max_gpu_memory_gb(self, value: int):
        self._settings.setValue("model/max_gpu_memory_gb", value)

    @property
    def max_cpu_memory_gb(self) -> int:
        return int(self._settings.value("model/max_cpu_memory_gb", 64))

    @max_cpu_memory_gb.setter
    def max_cpu_memory_gb(self, value: int):
        self._settings.setValue("model/max_cpu_memory_gb", value)

    # Global generation mode settings
    @property
    def global_bot_task(self) -> str:
        return self._settings.value("generation/global_bot_task", "image")

    @global_bot_task.setter
    def global_bot_task(self, value: str):
        self._settings.setValue("generation/global_bot_task", value)

    @property
    def global_drop_think(self) -> bool:
        val = self._settings.value("generation/global_drop_think", False)
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    @global_drop_think.setter
    def global_drop_think(self, value: bool):
        self._settings.setValue("generation/global_drop_think", value)

    # UI state
    @property
    def last_tab_index(self) -> int:
        return int(self._settings.value("ui/tab_index", 0))

    @last_tab_index.setter
    def last_tab_index(self, value: int):
        self._settings.setValue("ui/tab_index", value)

    @property
    def wildcard_sidebar_visible(self) -> bool:
        val = self._settings.value("ui/wildcard_sidebar", True)
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    @wildcard_sidebar_visible.setter
    def wildcard_sidebar_visible(self, value: bool):
        self._settings.setValue("ui/wildcard_sidebar", value)

    @property
    def gallery_visible(self) -> bool:
        val = self._settings.value("ui/gallery_visible", True)
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    @gallery_visible.setter
    def gallery_visible(self, value: bool):
        self._settings.setValue("ui/gallery_visible", value)


# Global singleton
_settings_instance = None


def get_settings() -> AppSettings:
    """Get the global AppSettings singleton."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
    return _settings_instance
