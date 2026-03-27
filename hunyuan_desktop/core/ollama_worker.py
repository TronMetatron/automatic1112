"""QThread worker for LM Studio prompt enhancement (remote, no local GPU)."""

from PySide6.QtCore import QThread, Signal


class OllamaEnhanceWorker(QThread):
    """Enhances a prompt using LM Studio on a background thread.

    Uses the remote LM Studio server for prompt enhancement,
    keeping local GPUs completely free.
    """

    completed = Signal(str, str)  # original, enhanced
    failed = Signal(str)          # error message

    def __init__(self, prompt: str, model: str, length: str, complexity: str):
        super().__init__()
        self.prompt = prompt
        self.model = model
        self.length = length
        self.complexity = complexity

    def run(self):
        try:
            print(f"\n{'='*60}")
            print(f"[LM STUDIO WORKER] Enhancement request received")
            print(f"[LM STUDIO WORKER]   Length: {self.length}, Complexity: {self.complexity}")
            print(f"[LM STUDIO WORKER]   Prompt: {self.prompt[:60]}...")

            from ui.constants import LMSTUDIO_URL
            from lmstudio_client import LMStudioClient
            from ollama_prompts import get_enhance_system_prompt, strip_thinking_tags

            client = LMStudioClient(base_url=LMSTUDIO_URL)

            if not client.is_available():
                print(f"[LM STUDIO WORKER] Server not reachable at {LMSTUDIO_URL}")
                self.failed.emit(f"LM Studio not reachable at {LMSTUDIO_URL}")
                return

            system_prompt = get_enhance_system_prompt(self.length, self.complexity)

            max_tokens_map = {
                "minimal": 512, "short": 768, "medium": 1024,
                "long": 1536, "detailed": 2048, "cinematic": 4096,
                "experimental": 6000,
            }
            max_tokens = max_tokens_map.get(self.length, 1024)

            print(f"[LM STUDIO WORKER] Calling LM Studio at {LMSTUDIO_URL}...")
            response = client.generate(
                prompt=self.prompt,
                system=system_prompt,
                temperature=0.7,
                max_tokens=max_tokens,
            )

            enhanced = strip_thinking_tags(response.text) if response.text else ""
            if not enhanced:
                self.failed.emit("LM Studio returned empty response")
                return

            print(f"[LM STUDIO WORKER] Enhancement complete: {enhanced[:60]}...")
            print(f"{'='*60}\n")
            self.completed.emit(self.prompt, enhanced)

        except ConnectionError:
            print(f"[LM STUDIO WORKER] Connection error - server not running")
            self.failed.emit("Cannot connect to LM Studio. Is the server running?")
        except Exception as e:
            print(f"[LM STUDIO WORKER] Exception: {e}")
            self.failed.emit(str(e))


class OllamaModelListWorker(QThread):
    """Fetches the list of available models from LM Studio."""

    completed = Signal(list)
    failed = Signal(str)

    def run(self):
        try:
            from ui.constants import LMSTUDIO_URL
            from lmstudio_client import LMStudioClient

            client = LMStudioClient(base_url=LMSTUDIO_URL)
            if client.is_available():
                models = client.list_models()
                if models:
                    self.completed.emit(models)
                    return
            self.completed.emit(["lmstudio"])
        except Exception as e:
            self.failed.emit(str(e))
