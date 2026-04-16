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

            from ui.constants import get_lmstudio_url
            from lmstudio_client import LMStudioClient
            from ollama_prompts import get_enhance_system_prompt, strip_thinking_tags

            client = LMStudioClient(base_url=get_lmstudio_url())

            if not client.is_available():
                print(f"[LM STUDIO WORKER] Server not reachable at {get_lmstudio_url()}")
                self.failed.emit(f"LM Studio not reachable at {get_lmstudio_url()}")
                return

            # Use custom system prompt if set, otherwise generate default
            from core.settings import get_settings
            custom_prompt = get_settings().enhance_system_prompt
            if custom_prompt:
                system_prompt = custom_prompt
                print(f"[LM STUDIO WORKER] Using custom system prompt ({len(custom_prompt)} chars)")
            else:
                system_prompt = get_enhance_system_prompt(self.length, self.complexity)

            # Thinking models need a much larger token budget because they use
            # tokens for reasoning before producing the actual answer.
            # These values are total budget (thinking + answer).
            max_tokens_map = {
                "minimal": 4096, "short": 6000, "medium": 8192,
                "long": 10000, "detailed": 12000, "cinematic": 16000,
                "experimental": 16000,
            }
            max_tokens = max_tokens_map.get(self.length, 8192)

            print(f"[LM STUDIO WORKER] Calling LM Studio at {get_lmstudio_url()} "
                  f"(max_tokens={max_tokens})...")
            response = client.generate(
                prompt=self.prompt,
                system=system_prompt,
                temperature=0.7,
                max_tokens=max_tokens,
            )

            raw_text = response.text
            print(f"[LM STUDIO WORKER] Raw response ({len(raw_text)} chars): "
                  f"{raw_text[:200]}...")

            enhanced = strip_thinking_tags(raw_text) if raw_text else ""

            if not enhanced:
                self.failed.emit(
                    "LM Studio returned empty response — the thinking model "
                    "may have used all tokens on reasoning. Check the console "
                    "log for details."
                )
                return

            print(f"[LM STUDIO WORKER] Enhancement complete: {enhanced[:80]}...")
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
            from ui.constants import get_lmstudio_url
            from lmstudio_client import LMStudioClient

            client = LMStudioClient(base_url=get_lmstudio_url())
            if client.is_available():
                models = client.list_models()
                if models:
                    self.completed.emit(models)
                    return
            self.completed.emit(["lmstudio"])
        except Exception as e:
            self.failed.emit(str(e))
