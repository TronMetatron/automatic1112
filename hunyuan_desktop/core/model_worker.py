"""QThread worker for model loading/unloading operations."""

from PySide6.QtCore import QThread, Signal


class ModelLoadWorker(QThread):
    """Loads the HunyuanImage model on a background thread."""

    progress = Signal(str)           # status message
    finished = Signal(bool, str)     # success, final_message

    def __init__(self, model_type: str = "base"):
        super().__init__()
        self.model_type = model_type

    def run(self):
        try:
            from core.model_manager import load_model, set_model_type

            # Set the model type first
            set_model_type(self.model_type)

            # Consume the generator, emitting progress for each step
            last_msg = ""
            for status_msg in load_model(self.model_type):
                last_msg = status_msg
                self.progress.emit(status_msg)

            # Check if load was successful
            from core.model_manager import is_model_loaded
            if is_model_loaded():
                self.finished.emit(True, last_msg)
            else:
                self.finished.emit(False, last_msg)

        except Exception as e:
            self.finished.emit(False, f"Error loading model: {str(e)}")


class ModelUnloadWorker(QThread):
    """Unloads the model on a background thread."""

    finished = Signal(str)  # status message

    def __init__(self, force: bool = False):
        super().__init__()
        self.force = force

    def run(self):
        try:
            if self.force:
                from core.model_manager import force_cleanup_gpu
                result = force_cleanup_gpu()
            else:
                from core.model_manager import unload_model
                result = unload_model()
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(f"Error unloading: {str(e)}")
