#!/usr/bin/env python3
"""
LM Studio Client for HunyuanImage-3.0
Provides OpenAI-compatible API access to LM Studio for prompt enhancement.
"""

import requests
from typing import Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_LMSTUDIO_URL = "http://localhost:1234"


@dataclass
class LMStudioResponse:
    """Response from LM Studio API"""
    text: str
    model: str
    usage: dict


class LMStudioClient:
    """Client for interacting with LM Studio's OpenAI-compatible API"""

    def __init__(self, base_url: str = DEFAULT_LMSTUDIO_URL):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/v1"

    def is_available(self) -> bool:
        """Check if LM Studio server is running"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models in LM Studio"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                return [m.get('id', 'unknown') for m in models]
        except Exception as e:
            logger.error(f"Error listing LM Studio models: {e}")
        return []

    def get_loaded_model(self) -> Optional[str]:
        """Get the currently loaded model in LM Studio"""
        models = self.list_models()
        return models[0] if models else None

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> LMStudioResponse:
        """Generate text using LM Studio's chat completions API.

        For thinking models, max_tokens is the *total* budget including
        both reasoning and the visible answer. Set it high enough so the
        model has room to think AND produce a response.
        """

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            # Send both — LM Studio uses whichever it supports.
            # max_completion_tokens is the OpenAI standard for thinking models.
            "max_tokens": max_tokens,
            "max_completion_tokens": max_tokens,
            "stream": False
        }

        # Model is optional - LM Studio uses the loaded model by default
        if model:
            payload["model"] = model

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=300  # 5 min timeout for large models
            )

            if response.status_code == 200:
                data = response.json()
                choices = data.get('choices', [])
                text = ""
                if choices:
                    message = choices[0].get('message', {})
                    finish_reason = choices[0].get('finish_reason', '')

                    # Log raw structure for debugging
                    logger.info(f"LM Studio message keys: {list(message.keys())}")
                    logger.info(f"LM Studio finish_reason: {finish_reason}")

                    # Content is the actual answer (may be null for thinking models
                    # that exhausted their token budget on thinking)
                    text = (message.get('content') or '').strip()

                    # reasoning_content is the thinking — NOT the answer.
                    # Log it for debugging but don't use it as the response.
                    reasoning = (message.get('reasoning_content') or '').strip()
                    if reasoning:
                        logger.info(
                            f"LM Studio thinking ({len(reasoning)} chars, "
                            f"answer: {len(text)} chars)"
                        )

                    if not text and reasoning:
                        logger.warning(
                            "Content is empty but reasoning_content has text. "
                            "The model likely ran out of tokens during thinking. "
                            "Try increasing max_tokens."
                        )

                    if text:
                        logger.info(f"LM Studio response ({len(text)} chars): {text[:100]}...")
                    else:
                        logger.warning(f"LM Studio returned empty content.")

                return LMStudioResponse(
                    text=text,
                    model=data.get('model', model or 'lmstudio'),
                    usage=data.get('usage', {})
                )
            else:
                raise Exception(f"LM Studio error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to LM Studio. Is the server running?")

    def enhance_prompt(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Enhance a prompt using LM Studio"""
        try:
            response = self.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.text
        except Exception as e:
            logger.error(f"LM Studio enhancement failed: {e}")
            return prompt  # Return original on failure


# Singleton instance
_client: Optional[LMStudioClient] = None


def get_client(base_url: str = DEFAULT_LMSTUDIO_URL) -> LMStudioClient:
    """Get or create LM Studio client"""
    global _client
    if _client is None or _client.base_url != base_url.rstrip('/'):
        _client = LMStudioClient(base_url)
    return _client


def check_lmstudio_available() -> bool:
    """Quick check if LM Studio is running"""
    try:
        response = requests.get(f"{DEFAULT_LMSTUDIO_URL}/v1/models", timeout=3)
        return response.status_code == 200
    except:
        return False


def discover_lmstudio(subnet: str = None, port: int = 1234, timeout: float = 0.5) -> Optional[str]:
    """Scan the local network for an LM Studio server.

    Args:
        subnet: e.g. "192.168.50" — if None, auto-detects from local IPs
        port: LM Studio port (default 1234)
        timeout: connection timeout per host in seconds

    Returns:
        URL string like "http://192.168.50.27:1234" or None
    """
    import socket
    import concurrent.futures

    # Auto-detect subnet from local network interfaces
    if subnet is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            subnet = ".".join(local_ip.split(".")[:3])
        except Exception:
            subnet = "192.168.1"

    def _check(ip):
        url = f"http://{ip}:{port}"
        try:
            r = requests.get(f"{url}/v1/models", timeout=timeout)
            if r.status_code == 200:
                return url
        except Exception:
            pass
        return None

    # Also check localhost
    candidates = [f"{subnet}.{i}" for i in range(1, 255)] + ["127.0.0.1", "localhost"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(_check, ip): ip for ip in candidates}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                # Cancel remaining work
                for f in futures:
                    f.cancel()
                return result
    return None


def get_lmstudio_model() -> Optional[str]:
    """Get the currently loaded model in LM Studio"""
    try:
        client = get_client()
        return client.get_loaded_model()
    except:
        return None


if __name__ == "__main__":
    # Test LM Studio connection
    print("Testing LM Studio connection...")

    if check_lmstudio_available():
        print("LM Studio is running!")
        client = get_client()
        models = client.list_models()
        print(f"Available models: {models}")

        if models:
            print("\nTesting generation...")
            response = client.generate(
                prompt="Describe a sunset over mountains in 20 words.",
                system="You are a creative writing assistant.",
                temperature=0.7,
                max_tokens=100
            )
            print(f"Response: {response.text}")
    else:
        print("LM Studio is not running or not accessible on localhost:1234")
