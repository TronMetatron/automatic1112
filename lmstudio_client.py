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
        max_tokens: int = 1024,
    ) -> LMStudioResponse:
        """Generate text using LM Studio's chat completions API"""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
                    text = message.get('content', '').strip()

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
