#!/usr/bin/env python3
"""
Ollama Prompt Enhancement System for HunyuanImage-3.0

Provides local LLM-based prompt enhancement, generation, and batch processing
without requiring external API keys.
"""

import json
import re
import requests
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def strip_thinking_tags(text: str) -> str:
    """
    Remove thinking/reasoning tags from model output and clean up encoding.

    Many newer models (Qwen3, DeepSeek R1, etc.) output chain-of-thought
    reasoning wrapped in <think>...</think> or similar tags.
    This function strips those out to get just the final answer.

    For models that wrap the answer in <recaption>...</recaption>, the
    recaption content is extracted as the final answer.
    """
    if not text:
        return text

    original = text

    # Decode unicode escapes (e.g., \u2014 -> —)
    try:
        # Handle double-encoded JSON strings
        if '\\u' in text:
            text = text.encode('utf-8').decode('unicode_escape')
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass  # Keep original if decoding fails

    # First: extract <recaption> content if present (HunyuanImage think models)
    recaption_match = re.search(
        r'<recaption>(.*?)</recaption>', text, flags=re.DOTALL | re.IGNORECASE
    )
    if recaption_match:
        logger.info("Extracted <recaption> content from thinking model output")
        text = recaption_match.group(1).strip()
        if text:
            return text

    # Remove <think>...</think> blocks (Qwen3, DeepSeek style)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove <reasoning>...</reasoning> blocks
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove <reflection>...</reflection> blocks
    text = re.sub(r'<reflection>.*?</reflection>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove ```thinking ... ``` blocks
    text = re.sub(r'```thinking.*?```', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Handle unclosed <think> tag (model started thinking but never closed it)
    # Remove everything from <think> to end of string
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Clean up any leftover whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double

    result = text.strip()

    # If stripping removed everything, the model output was *only* thinking.
    # Log it so we can diagnose.
    if not result and original.strip():
        logger.warning(
            f"strip_thinking_tags removed all content. "
            f"Original ({len(original)} chars): {original[:200]}..."
        )

    return result

# Default configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434"
_FALLBACK_LMSTUDIO_URL = "http://localhost:1234"


_cached_lmstudio_url = None


def _get_lmstudio_url():
    """Get the LM Studio URL from settings, with auto-discovery fallback."""
    global _cached_lmstudio_url

    # Try settings first
    try:
        from core.settings import get_settings
        url = get_settings().lmstudio_url
        if url:
            # Quick check if it's still reachable
            try:
                import requests
                r = requests.get(f"{url}/v1/models", timeout=2)
                if r.status_code == 200:
                    return url
            except Exception:
                pass
            # Saved URL unreachable — fall through to discovery
    except Exception:
        pass

    # Try ui.constants default
    try:
        from ui.constants import LMSTUDIO_URL
        try:
            import requests
            r = requests.get(f"{LMSTUDIO_URL}/v1/models", timeout=2)
            if r.status_code == 200:
                return LMSTUDIO_URL
        except Exception:
            pass
    except Exception:
        pass

    # Use cached discovery result if available
    if _cached_lmstudio_url:
        return _cached_lmstudio_url

    # Auto-discover on the network
    try:
        from lmstudio_client import discover_lmstudio
        found = discover_lmstudio(timeout=0.3)
        if found:
            _cached_lmstudio_url = found
            print(f"[LM STUDIO] Auto-discovered at {found}")
            # Save for next time
            try:
                from core.settings import get_settings
                get_settings().lmstudio_url = found
            except Exception:
                pass
            return found
    except Exception:
        pass

    return _FALLBACK_LMSTUDIO_URL


# Keep module-level name for backward compat but make it dynamic
DEFAULT_LMSTUDIO_URL = _FALLBACK_LMSTUDIO_URL
DEFAULT_MODEL = "qwen2.5:7b-instruct"

# Backend detection
BACKEND_OLLAMA = "ollama"
BACKEND_LMSTUDIO = "lmstudio"


def detect_backend() -> tuple:
    """
    Auto-detect which LLM backend is available.

    Returns (backend_type, base_url) — checks LM Studio first, then Ollama.
    """
    # Check LM Studio first (OpenAI-compatible API)
    lmstudio_url = _get_lmstudio_url()
    try:
        response = requests.get(f"{lmstudio_url}/v1/models", timeout=3)
        if response.status_code == 200:
            logger.info("Detected LM Studio backend")
            return BACKEND_LMSTUDIO, lmstudio_url
    except Exception:
        pass

    # Fall back to Ollama
    try:
        response = requests.get(f"{DEFAULT_OLLAMA_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            logger.info("Detected Ollama backend")
            return BACKEND_OLLAMA, DEFAULT_OLLAMA_URL
    except Exception:
        pass

    logger.warning("No LLM backend detected (tried LM Studio and Ollama)")
    return None, None


def get_loaded_ollama_model() -> Optional[str]:
    """
    Auto-discover the currently loaded model from any backend.

    Returns the model name if one is loaded, None otherwise.
    """
    # Check LM Studio first
    lmstudio_url = _get_lmstudio_url()
    try:
        response = requests.get(f"{lmstudio_url}/v1/models", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            if models:
                return models[0].get('id')
    except Exception:
        pass

    # Fall back to Ollama
    try:
        response = requests.get(f"{DEFAULT_OLLAMA_URL}/api/ps", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            if models:
                return models[0].get('name')
    except Exception:
        pass
    return None


def get_available_models() -> List[str]:
    """
    Get list of available models from whichever backend is running.

    Returns model names/IDs from LM Studio or Ollama.
    """
    models = []

    # Check LM Studio
    lmstudio_url = _get_lmstudio_url()
    try:
        response = requests.get(f"{lmstudio_url}/v1/models", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models.extend(m.get('id', '') for m in data.get('data', []) if m.get('id'))
    except Exception:
        pass

    # Check Ollama
    try:
        response = requests.get(f"{DEFAULT_OLLAMA_URL}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models.extend(m['name'] for m in data.get('models', []))
    except Exception:
        pass

    return models


def get_best_available_model() -> str:
    """
    Get the best available model to use.

    Priority:
    1. Currently loaded model (from LM Studio or Ollama)
    2. Default model if available
    3. Any available model
    """
    loaded = get_loaded_ollama_model()
    if loaded:
        logger.info(f"Auto-discovered loaded model: {loaded}")
        return loaded

    # Fall back to default
    return DEFAULT_MODEL

# Length presets (word counts)
LENGTH_PRESETS = {
    "minimal": {"words": "15-30", "desc": "Very short, key elements only"},
    "short": {"words": "30-50", "desc": "Brief but complete"},
    "medium": {"words": "50-100", "desc": "Balanced detail"},
    "long": {"words": "100-150", "desc": "Rich detail"},
    "detailed": {"words": "150-250", "desc": "Maximum detail"},
    "cinematic": {"words": "250-400", "desc": "Epic film-quality scene description"},
    "experimental": {"words": "300-500", "desc": "Wild, surreal, boundary-pushing"},
}

# Complexity presets
COMPLEXITY_PRESETS = {
    "simple": {
        "desc": "Basic subject description only",
        "elements": "Focus only on the main subject. No complex lighting, composition, or technical terms.",
    },
    "moderate": {
        "desc": "Subject, setting, and mood",
        "elements": "Include subject, environment, basic composition (close-up, wide shot), and mood/atmosphere.",
    },
    "detailed": {
        "desc": "Full cinematographic approach",
        "elements": "Include all 5 cinematographic elements: subject details, quality descriptors, camera angle/composition, lighting/atmosphere, and art style.",
    },
    "complex": {
        "desc": "Maximum artistic detail",
        "elements": "Include extensive details: materials, textures, specific lighting setups, advanced composition techniques, color palettes, artistic references, and rendering style.",
    },
    "cinematic": {
        "desc": "Epic film director's vision",
        "elements": """Create an immersive, film-quality scene description like a master cinematographer would envision:
- SUBJECT: Exhaustive detail on the main subject - physical characteristics, pose, expression, clothing/materials with textures
- ENVIRONMENT: Rich world-building - architectural details, environmental storytelling, background elements that add depth
- CAMERA: Specific shot type (extreme close-up, medium shot, wide establishing shot), camera movement suggestion, lens choice (35mm, 85mm portrait, anamorphic), depth of field
- LIGHTING: Dramatic lighting setup - key light direction, fill ratios, rim lights, practical lights in scene, time of day, weather affecting light
- COLOR: Specific color palette and color grading style (teal and orange, desaturated, vibrant, monochromatic accent)
- ATMOSPHERE: Mood, emotional tone, narrative tension, environmental effects (fog, dust motes, rain, volumetric light rays)
- STYLE: Art direction references, rendering technique, quality markers (8K, hyperdetailed, photorealistic, masterwork)
- COMPOSITION: Rule of thirds placement, leading lines, framing elements, foreground/midground/background layers""",
    },
    "experimental": {
        "desc": "Wild, surreal, boundary-pushing artistic vision",
        "elements": """Push creative boundaries and generate unexpected, visually striking scenes. Be bold and experimental:
- SURREAL ELEMENTS: Impossible geometry, dream logic, reality-bending physics, scale distortions, metamorphosis
- UNUSUAL JUXTAPOSITIONS: Combine unexpected elements - organic with mechanical, ancient with futuristic, microscopic with cosmic
- DRAMATIC EXTREMES: Push lighting, color, and atmosphere to extremes - bioluminescence, chromatic aberration, infrared, x-ray vision
- TEXTURE OBSESSION: Hyper-detailed materials - iridescent surfaces, subsurface scattering, crystalline structures, liquid metal, living stone
- ARTISTIC CHAOS: Controlled chaos in composition - fractals, impossible architecture, M.C. Escher perspectives, non-Euclidean spaces
- EMOTIONAL INTENSITY: Evoke strong feelings - awe, unease, wonder, vertigo, transcendence
- CROSS-GENRE MASHUPS: Blend genres freely - cosmic horror meets Art Nouveau, cyberpunk baroque, biopunk renaissance
- SYNESTHESIA: Describe sounds as colors, emotions as textures, time as space
- SCALE PLAY: Macro/micro worlds, giants and miniatures, infinite recursion, worlds within worlds
- UNEXPECTED BEAUTY: Find beauty in strange places - decay, glitches, entropy, emergence""",
    },
}


def get_enhance_system_prompt(length: str = "medium", complexity: str = "detailed") -> str:
    """Generate a dynamic system prompt based on length and complexity settings."""
    length_info = LENGTH_PRESETS.get(length, LENGTH_PRESETS["medium"])
    complexity_info = COMPLEXITY_PRESETS.get(complexity, COMPLEXITY_PRESETS["detailed"])

    return f"""You are an expert image prompt engineer for HunyuanImage-3.0, an advanced AI image generator.

Your task is to enhance the user's simple prompt into a description for image generation.

LENGTH REQUIREMENT: {length_info['words']} words ({length_info['desc']})
COMPLEXITY LEVEL: {complexity} - {complexity_info['desc']}

WHAT TO INCLUDE:
{complexity_info['elements']}

RULES:
- Keep the core intent and subject of the original prompt
- Output ONLY the enhanced prompt text, no explanations or labels
- Stay within the {length_info['words']} word limit
- Be creative but stay true to what was requested
- Don't add inappropriate or unsafe content
- Do NOT use <think> tags or any reasoning - respond directly with the prompt only
- No preamble, no explanation, just the enhanced prompt text"""


def get_generation_system_prompt(length: str = "medium", complexity: str = "detailed") -> str:
    """Generate a dynamic system prompt for prompt generation based on length and complexity."""
    length_info = LENGTH_PRESETS.get(length, LENGTH_PRESETS["medium"])
    complexity_info = COMPLEXITY_PRESETS.get(complexity, COMPLEXITY_PRESETS["detailed"])

    return f"""You are a creative prompt generator for HunyuanImage-3.0 image generation.

Given a theme or concept, generate unique image prompts.

LENGTH REQUIREMENT: Each prompt should be {length_info['words']} words ({length_info['desc']})
COMPLEXITY LEVEL: {complexity} - {complexity_info['desc']}

WHAT TO INCLUDE IN EACH PROMPT:
{complexity_info['elements']}

CRITICAL RULES:
- Output ONLY the prompts, one per line, no numbering or explanations
- Do NOT use <think> tags or any reasoning - respond directly with prompts only
- No preamble, no meta-commentary, just the prompt text"""


# Legacy system prompts (for backward compatibility)
ENHANCE_SYSTEM_PROMPT = """You are an expert image prompt engineer for HunyuanImage-3.0, an advanced AI image generator.

Your task is to enhance the user's simple prompt into a detailed, cinematic description that will generate stunning images. Follow this 5-part cinematographic formula:

1. **Main Subject & Scene**: Describe the primary subject with rich, specific details (materials, textures, expressions, poses, environment context)
2. **Image Quality & Style**: Add quality descriptors (8K resolution, photorealistic, hyperdetailed, masterpiece quality)
3. **Composition & Viewpoint**: Specify camera angle (low angle, bird's eye, close-up), framing, depth of field, perspective
4. **Lighting & Atmosphere**: Define lighting (golden hour, dramatic shadows, soft diffused light), mood, time of day, weather
5. **Technical Parameters**: Mention art style or rendering technique when appropriate (cinematic, oil painting, digital art, etc.)

RULES:
- Keep the core intent and subject of the original prompt
- Output ONLY the enhanced prompt text, no explanations or labels
- Keep it under 200 words for optimal results
- Be creative but stay true to what was requested
- Don't add inappropriate or unsafe content"""

GENERATE_PROMPTS_SYSTEM = """You are a creative prompt generator for HunyuanImage-3.0 image generation.

Given a theme or concept, generate unique, detailed image prompts. Each prompt should be:
- Self-contained and complete
- Visually interesting and specific
- Between 50-150 words
- Following cinematographic principles (subject, quality, composition, lighting, style)

Output ONLY the prompts, one per line, no numbering or explanations."""

VARIATION_SYSTEM_PROMPT = """You are a creative prompt variation generator for HunyuanImage-3.0.

Given a base prompt, create variations that:
- Maintain the core subject/concept
- Change aspects like: angle, lighting, time of day, style, mood, setting
- Are distinctly different from each other
- Each stands alone as a complete, detailed prompt

Output ONLY the variation prompts, one per line, no numbering or explanations."""

ANALYZE_PROMPT_SYSTEM = """You are an expert image prompt analyst for HunyuanImage-3.0.

Analyze the given prompt and provide:
1. **Strengths**: What's good about this prompt
2. **Missing Elements**: What cinematographic elements are missing
3. **Suggestions**: Specific improvements to make the image better
4. **Enhanced Version**: A rewritten, improved version

Be concise and actionable."""


@dataclass
class OllamaResponse:
    """Response from Ollama API"""
    text: str
    model: str
    total_duration: float  # in seconds
    prompt_eval_count: int
    eval_count: int

    @property
    def tokens_per_second(self) -> float:
        if self.total_duration > 0:
            return self.eval_count / self.total_duration
        return 0.0


def get_gpu_free_vram(gpu_index: int = 0) -> float:
    """Get free VRAM on specified GPU in GB."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', f'-i={gpu_index}'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            free_mb = float(result.stdout.strip())
            return free_mb / 1024  # Convert to GB
    except Exception as e:
        logger.warning(f"Could not get GPU VRAM: {e}")
    return 0.0


def estimate_model_vram(model_name: str) -> float:
    """Estimate VRAM needed for a model based on its name/size.

    Returns estimated VRAM in GB.
    """
    model_lower = model_name.lower()

    # Extract size from model name
    size_patterns = [
        ('70b', 45), ('80b', 50), ('72b', 45),  # Very large
        ('30b', 20), ('32b', 22), ('34b', 23),  # Large
        ('24b', 16), ('27b', 18),               # Medium-large
        ('13b', 10), ('14b', 11), ('15b', 12),  # Medium
        ('7b', 5), ('8b', 6), ('9b', 7),        # Small
        ('3b', 3), ('4b', 3.5),                 # Tiny
        ('1b', 1.5), ('0.5b', 1),               # Very tiny
    ]

    for pattern, vram in size_patterns:
        if pattern in model_lower:
            return vram

    # Default guess for unknown models
    return 8.0


def check_model_fits_gpu(model_name: str, gpu_index: int = 0) -> tuple[bool, str]:
    """Check if a model will fit in GPU VRAM.

    Returns (fits: bool, message: str).
    """
    free_vram = get_gpu_free_vram(gpu_index)
    estimated_vram = estimate_model_vram(model_name)

    if free_vram == 0:
        return True, "Could not check GPU VRAM"

    fits = free_vram >= estimated_vram

    if fits:
        return True, f"Model {model_name} (~{estimated_vram:.1f}GB) fits in {free_vram:.1f}GB free VRAM"
    else:
        return False, f"WARNING: Model {model_name} needs ~{estimated_vram:.1f}GB but only {free_vram:.1f}GB free!"


class OllamaClient:
    """Unified client for Ollama and LM Studio (OpenAI-compatible) APIs.

    Auto-detects which backend is available. LM Studio is preferred when both
    are running since it uses the OpenAI-compatible /v1/ API.
    """

    def __init__(self, base_url: str = None, model: str = DEFAULT_MODEL):
        if base_url:
            # Explicit URL — detect backend type from URL
            self.base_url = base_url.rstrip('/')
            if '/v1' in base_url or base_url == _get_lmstudio_url() or base_url == _FALLBACK_LMSTUDIO_URL:
                self.backend = BACKEND_LMSTUDIO
            else:
                self.backend = BACKEND_OLLAMA
        else:
            # Auto-detect
            self.backend, url = detect_backend()
            self.base_url = (url or DEFAULT_OLLAMA_URL).rstrip('/')

        self.model = model
        self._check_connection()
        logger.info(f"LLM backend: {self.backend or 'none'} at {self.base_url}")

    def _check_connection(self) -> bool:
        """Check if the LLM server is running."""
        try:
            if self.backend == BACKEND_LMSTUDIO:
                response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            else:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.warning(f"LLM server not running at {self.base_url}")
            return False

    def list_models(self) -> List[str]:
        """List available models from whichever backend is active."""
        return get_available_models()

    def get_loaded_model(self) -> Optional[Dict]:
        """Get the currently loaded model info.

        Returns dict with 'name' and 'size_vram' keys, or None if no model loaded.
        """
        if self.backend == BACKEND_LMSTUDIO:
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    models = [m for m in data.get('data', [])
                              if not m.get('id', '').startswith('text-embedding')]
                    if models:
                        return {
                            'name': models[0].get('id', 'unknown'),
                            'size_vram': 0,  # LM Studio doesn't report this
                        }
            except Exception as e:
                logger.warning(f"Could not get loaded model: {e}")
            return None
        else:
            try:
                response = requests.get(f"{self.base_url}/api/ps", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get('models', [])
                    if models:
                        return {
                            'name': models[0].get('name', 'unknown'),
                            'size_vram': models[0].get('size_vram', 0) / (1024**3),
                        }
            except Exception as e:
                logger.warning(f"Could not get loaded model: {e}")
            return None

    def wait_for_model(self, model_name: str, timeout: int = 600, callback=None) -> bool:
        """Wait for a model to be loaded.

        Args:
            model_name: Model name to wait for
            timeout: Maximum seconds to wait
            callback: Optional callback(message) for progress updates

        Returns True if model loaded, False on timeout.
        """
        import time
        start = time.time()

        if callback:
            callback(f"Loading model {model_name}...")

        # VRAM check only relevant for Ollama
        if self.backend == BACKEND_OLLAMA:
            fits, msg = check_model_fits_gpu(model_name)
            if not fits:
                logger.warning(msg)
                if callback:
                    callback(msg)

        while time.time() - start < timeout:
            loaded = self.get_loaded_model()
            if loaded and model_name in loaded['name']:
                if callback:
                    vram_info = f" ({loaded['size_vram']:.1f}GB VRAM)" if loaded['size_vram'] else ""
                    callback(f"Model {model_name} ready{vram_info}")
                return True

            elapsed = int(time.time() - start)
            if callback and elapsed % 10 == 0:
                callback(f"Loading {model_name}... ({elapsed}s)")

            time.sleep(2)

        return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> OllamaResponse:
        """Generate text using the active backend."""
        model = model or self.model

        if self.backend == BACKEND_LMSTUDIO:
            return self._generate_openai(prompt, system, model, temperature, max_tokens)
        else:
            return self._generate_ollama(prompt, system, model, temperature, max_tokens, stream)

    def _generate_openai(
        self, prompt, system, model, temperature, max_tokens
    ) -> OllamaResponse:
        """Generate via OpenAI-compatible API (LM Studio)."""
        import time
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=900
            )

            if response.status_code == 200:
                data = response.json()
                message = data['choices'][0].get('message', {})

                # Content is the actual answer
                raw_text = message.get('content') or ''

                # reasoning_content is the thinking — don't use as answer
                reasoning = (message.get('reasoning_content') or '').strip()
                if reasoning:
                    logger.info(
                        f"Thinking model: reasoning={len(reasoning)} chars, "
                        f"answer={len(raw_text)} chars"
                    )

                if not raw_text.strip() and reasoning:
                    logger.warning(
                        "Model ran out of tokens during thinking — no answer produced"
                    )

                clean_text = strip_thinking_tags(raw_text) if raw_text else ''

                usage = data.get('usage', {})
                duration = time.time() - start_time
                return OllamaResponse(
                    text=clean_text,
                    model=data.get('model', model),
                    total_duration=duration,
                    prompt_eval_count=usage.get('prompt_tokens', 0),
                    eval_count=usage.get('completion_tokens', 0)
                )
            else:
                raise Exception(f"LM Studio error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to LM Studio at {self.base_url}")

    def _generate_ollama(
        self, prompt, system, model, temperature, max_tokens, stream
    ) -> OllamaResponse:
        """Generate via Ollama API."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=900
            )

            if response.status_code == 200:
                data = response.json()
                raw_text = data.get('response', '')
                clean_text = strip_thinking_tags(raw_text)
                return OllamaResponse(
                    text=clean_text,
                    model=data.get('model', model),
                    total_duration=data.get('total_duration', 0) / 1e9,
                    prompt_eval_count=data.get('prompt_eval_count', 0),
                    eval_count=data.get('eval_count', 0)
                )
            else:
                raise Exception(f"Ollama error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to Ollama. Is the server running? (ollama serve)")

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Generator[str, None, None]:
        """Stream generation from the active backend."""
        model = model or self.model

        if self.backend == BACKEND_LMSTUDIO:
            yield from self._stream_openai(prompt, system, model, temperature, max_tokens)
        else:
            yield from self._stream_ollama(prompt, system, model, temperature, max_tokens)

    def _stream_openai(self, prompt, system, model, temperature, max_tokens):
        """Stream via OpenAI-compatible API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            with requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=900
            ) as response:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            data = json.loads(line[6:])
                            delta = data.get('choices', [{}])[0].get('delta', {})
                            # Try content first, fall back to reasoning_content
                            content = delta.get('content') or ''
                            if not content:
                                content = delta.get('reasoning_content') or ''
                            if content:
                                yield content
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to LM Studio at {self.base_url}")

    def _stream_ollama(self, prompt, system, model, temperature, max_tokens):
        """Stream via Ollama API."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=900
            ) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to Ollama. Is the server running?")


class PromptEnhancer:
    """Enhance and generate image prompts using Ollama or LM Studio"""

    def __init__(self, ollama_url: str = None, model: str = None):
        if model is None:
            model = get_best_available_model()
        self.client = OllamaClient(ollama_url, model)

    def enhance(
        self,
        prompt: str,
        style: Optional[str] = None,
        temperature: float = 0.7,
        model: Optional[str] = None,
        length: str = "medium",
        complexity: str = "detailed",
        max_length: int = 0
    ) -> str:
        """
        Enhance a simple prompt into a detailed, cinematic description.

        Args:
            prompt: The simple prompt to enhance
            style: Optional style to incorporate (e.g., "cinematic", "anime", "oil painting")
            temperature: Creativity level (0.0-1.0)
            model: Ollama model to use (default: qwen2.5:7b-instruct)
            length: Prompt length preset (minimal, short, medium, long, detailed)
            complexity: Prompt complexity preset (simple, basic, moderate, detailed, complex)
            max_length: Maximum character count for result (0 = no limit)

        Returns:
            Enhanced prompt string
        """
        user_prompt = prompt
        if style:
            user_prompt = f"[Style: {style}] {prompt}"

        # Use custom system prompt from settings if set, otherwise auto-generate
        system_prompt = None
        try:
            from core.settings import get_settings
            custom = get_settings().enhance_system_prompt
            if custom:
                system_prompt = custom
        except Exception:
            pass
        if not system_prompt:
            system_prompt = get_enhance_system_prompt(length, complexity)

        # Thinking models need a large token budget (reasoning + answer).
        max_tokens_map = {
            "minimal": 4096,
            "short": 6000,
            "medium": 8192,
            "long": 10000,
            "detailed": 12000,
            "cinematic": 16000,
            "experimental": 16000,
        }
        max_tokens = max_tokens_map.get(length, 8192)

        # If max_length is set, also limit max_tokens proportionally
        # Rough estimate: 1 token ≈ 4 characters
        if max_length > 0:
            estimated_tokens = max_length // 3  # Be conservative
            max_tokens = min(max_tokens, max(256, estimated_tokens))

        response = self.client.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens
        )

        result = response.text

        # Enforce max_length limit
        if max_length > 0 and len(result) > max_length:
            # Try to truncate at a sentence boundary
            truncated = result[:max_length]
            # Find last sentence ending
            for ending in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_end = truncated.rfind(ending)
                if last_end > max_length * 0.6:  # At least 60% of max length
                    truncated = truncated[:last_end + 1]
                    break
            else:
                # No good sentence boundary, find last comma or space
                last_comma = truncated.rfind(', ')
                last_space = truncated.rfind(' ')
                if last_comma > max_length * 0.8:
                    truncated = truncated[:last_comma]
                elif last_space > max_length * 0.9:
                    truncated = truncated[:last_space]

            result = truncated.strip()
            logger.info(f"Truncated prompt from {len(response.text)} to {len(result)} chars (max: {max_length})")

        logger.info(f"Enhanced prompt ({length}/{complexity}) in {response.total_duration:.1f}s ({response.tokens_per_second:.1f} tok/s)")
        return result

    def generate_prompts(
        self,
        theme: str,
        count: int = 10,
        style: Optional[str] = None,
        temperature: float = 0.8,
        model: Optional[str] = None,
        length: str = "medium",
        complexity: str = "detailed"
    ) -> List[str]:
        """
        Generate multiple creative prompts for a theme.

        Args:
            theme: The theme or concept to generate prompts for
            count: Number of prompts to generate
            style: Optional style preference
            temperature: Creativity level
            model: Ollama model to use
            length: Prompt length preset (minimal, short, medium, long, detailed)
            complexity: Prompt complexity preset (simple, basic, moderate, detailed, complex)

        Returns:
            List of generated prompts
        """
        user_prompt = f"Generate {count} unique image prompts for the theme: '{theme}'"
        if style:
            user_prompt += f"\nPreferred style: {style}"

        # Use dynamic system prompt based on length/complexity settings
        system_prompt = get_generation_system_prompt(length, complexity)

        # Adjust max tokens based on length setting
        # Add base overhead for thinking models, plus per-prompt tokens
        tokens_per_prompt = {
            "minimal": 100,
            "short": 150,
            "medium": 250,
            "long": 350,
            "detailed": 500,
        }
        # Base 1024 tokens for thinking overhead + tokens per prompt
        max_tokens = 1024 + (count * tokens_per_prompt.get(length, 250))

        response = self.client.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=temperature,
            model=model,
            max_tokens=max_tokens
        )

        # Parse prompts from response
        prompts = [p.strip() for p in response.text.split('\n') if p.strip()]
        logger.info(f"Generated {len(prompts)} prompts ({length}/{complexity}) in {response.total_duration:.1f}s")
        return prompts

    def create_variations(
        self,
        base_prompt: str,
        count: int = 5,
        temperature: float = 0.8,
        model: Optional[str] = None
    ) -> List[str]:
        """
        Create variations of a base prompt.

        Args:
            base_prompt: The original prompt to create variations of
            count: Number of variations to generate
            temperature: Creativity level
            model: Ollama model to use

        Returns:
            List of prompt variations
        """
        user_prompt = f"Create {count} distinct variations of this prompt:\n\n{base_prompt}"

        response = self.client.generate(
            prompt=user_prompt,
            system=VARIATION_SYSTEM_PROMPT,
            temperature=temperature,
            model=model,
            max_tokens=count * 200
        )

        variations = [p.strip() for p in response.text.split('\n') if p.strip()]
        logger.info(f"Created {len(variations)} variations in {response.total_duration:.1f}s")
        return variations

    def analyze(
        self,
        prompt: str,
        model: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Analyze a prompt and provide improvement suggestions.

        Args:
            prompt: The prompt to analyze
            model: Ollama model to use

        Returns:
            Dict with analysis sections
        """
        response = self.client.generate(
            prompt=f"Analyze this image prompt:\n\n{prompt}",
            system=ANALYZE_PROMPT_SYSTEM,
            temperature=0.5,
            model=model,
            max_tokens=800
        )

        # Return raw analysis text
        return {"analysis": response.text}

    def batch_enhance(
        self,
        prompts: List[str],
        style: Optional[str] = None,
        temperature: float = 0.7,
        model: Optional[str] = None,
        progress_callback=None
    ) -> List[Dict[str, str]]:
        """
        Enhance multiple prompts.

        Args:
            prompts: List of prompts to enhance
            style: Optional style for all prompts
            temperature: Creativity level
            model: Ollama model to use
            progress_callback: Optional callback(current, total, prompt)

        Returns:
            List of dicts with 'original' and 'enhanced' keys
        """
        results = []
        total = len(prompts)

        for i, prompt in enumerate(prompts):
            if progress_callback:
                progress_callback(i + 1, total, prompt)

            try:
                enhanced = self.enhance(prompt, style=style, temperature=temperature, model=model)
                results.append({
                    "original": prompt,
                    "enhanced": enhanced
                })
            except Exception as e:
                logger.error(f"Error enhancing prompt {i+1}: {e}")
                results.append({
                    "original": prompt,
                    "enhanced": prompt,  # Fall back to original
                    "error": str(e)
                })

        return results


def main():
    """CLI interface for prompt enhancement"""
    import argparse

    parser = argparse.ArgumentParser(description="Ollama Prompt Enhancement for HunyuanImage-3.0")
    parser.add_argument("prompt", nargs="?", help="Prompt to enhance")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Ollama model to use")
    parser.add_argument("--style", "-s", help="Style to apply (cinematic, anime, etc.)")
    parser.add_argument("--generate", "-g", type=int, metavar="N", help="Generate N prompts for a theme")
    parser.add_argument("--variations", "-v", type=int, metavar="N", help="Create N variations of prompt")
    parser.add_argument("--analyze", "-a", action="store_true", help="Analyze the prompt")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Temperature (0.0-1.0)")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL, help="Ollama server URL")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    enhancer = PromptEnhancer(ollama_url=args.url, model=args.model)

    if args.list_models:
        models = enhancer.client.list_models()
        if args.json:
            print(json.dumps(models))
        else:
            print("Available Ollama models:")
            for m in models:
                print(f"  - {m}")
        return

    if not args.prompt:
        parser.error("Prompt is required (unless using --list-models)")

    try:
        if args.generate:
            # Generate prompts for a theme
            prompts = enhancer.generate_prompts(
                args.prompt,
                count=args.generate,
                style=args.style,
                temperature=args.temperature
            )
            if args.json:
                print(json.dumps(prompts))
            else:
                for i, p in enumerate(prompts, 1):
                    print(f"\n[{i}] {p}")

        elif args.variations:
            # Create variations
            variations = enhancer.create_variations(
                args.prompt,
                count=args.variations,
                temperature=args.temperature
            )
            if args.json:
                print(json.dumps(variations))
            else:
                print(f"\nVariations of: {args.prompt}\n")
                for i, v in enumerate(variations, 1):
                    print(f"[{i}] {v}\n")

        elif args.analyze:
            # Analyze prompt
            analysis = enhancer.analyze(args.prompt)
            if args.json:
                print(json.dumps(analysis))
            else:
                print(analysis["analysis"])

        else:
            # Default: enhance prompt
            enhanced = enhancer.enhance(
                args.prompt,
                style=args.style,
                temperature=args.temperature
            )
            if args.json:
                print(json.dumps({"original": args.prompt, "enhanced": enhanced}))
            else:
                print(enhanced)

    except ConnectionError as e:
        logger.error(str(e))
        print("\nERROR: Cannot connect to Ollama server.")
        print("Start it with: ollama serve")
        exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
