"""
LLM Interface — Generation Layer, Shoryan Blood Donation Assistant.

Everything upstream (prompt builder, verifier, pipeline) depends only on
the `LLMProvider` interface below, never on a vendor SDK directly. This
keeps the business logic provider-agnostic: swapping OpenAI for Gemini,
Ollama, or an internal endpoint means adding one subclass here — nothing
else in the Generation Layer changes.

Vendor SDKs are imported lazily inside each provider's __init__, so this
module can be imported (and EchoProvider used) even in environments where
no vendor packages are installed.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResponse:
    """Normalized response shape, regardless of backend."""
    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Minimal contract every LLM backend must satisfy."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Send one system+user turn and return the model's reply.

        Implementations must not add, remove, or rewrite instructions —
        the Prompt Builder is the single source of truth for prompt
        content; providers are a pure transport layer.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# GeminiProvider (Google Gemini)
# ----------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """Uses Google's Gemini API via google-generativeai."""

    def __init__(self, model: str = "gemini-3.5-flash", api_key: Optional[str] = None) -> None:
        import google.generativeai as genai  # lazy import

        if api_key:
            genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt,
        )
        resp = model.generate_content(user_prompt)
        return LLMResponse(text=resp.text or "", raw=None)


# ----------------------------------------------------------------------
# OpenRouterProvider (unified API with Qwen3, etc.)
# ----------------------------------------------------------------------

class OpenRouterProvider(LLMProvider):
    """Uses OpenRouter API to call various models (Gemini, OpenAI, Qwen, etc.)."""

    def __init__(
        self,
        model: str = "qwen/qwen3-30b-a3b",  # OpenRouter model ID
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package not installed. Install with: pip install requests")

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY env var or pass it.")

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._requests = requests

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        resp = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        # Better error handling: print the response body on 4xx/5xx
        try:
            resp.raise_for_status()
        except self._requests.exceptions.HTTPError as e:
            error_detail = e.response.text if hasattr(
                e, 'response') else str(e)
            print(
                f"\n❌ OpenRouter API error (status {e.response.status_code if hasattr(e, 'response') else 'unknown'})")
            print(f"   Response: {error_detail}")
            raise  # re-raise after printing

        data = resp.json()
        return LLMResponse(
            text=data["choices"][0]["message"]["content"],
            raw=data,
        )


class AllProvidersFailedError(Exception):
    """Raised when both primary and fallback providers fail."""
    pass


def is_fallback_eligible(error: Exception) -> bool:
    """Return True if the exception is a network/API error eligible for fallback."""
    # Exclude programming errors
    if isinstance(error, (AttributeError, TypeError, ValueError, KeyError)):
        return False

    # Google API errors (Gemini)
    try:
        from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, ServiceUnavailable
        if isinstance(error, (GoogleAPIError, ResourceExhausted, ServiceUnavailable)):
            return True
    except ImportError:
        pass

    # Requests errors (OpenRouter, etc.)
    try:
        import requests
        if isinstance(error, requests.exceptions.RequestException):
            return True
    except ImportError:
        pass

    # Network/timeout errors
    if isinstance(error, (ConnectionError, TimeoutError, BrokenPipeError)):
        return True

    # Heuristic: check error message for common keywords
    msg = str(error).lower()
    keywords = ["timeout", "connection", "unavailable", "500",
                "502", "503", "504", "429", "quota", "rate limit"]
    if any(k in msg for k in keywords):
        return True

    return False


class FallbackProvider(LLMProvider):
    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self.primary = primary
        self.fallback = fallback

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            return self.primary.complete(system_prompt, user_prompt)
        except Exception as e:
            if not is_fallback_eligible(e):
                raise  # programming error – don't mask
            print(f"[LLM] Primary failed: {e.__class__.__name__}: {e}")
            try:
                response = self.fallback.complete(system_prompt, user_prompt)
                print("[LLM] Fallback succeeded")
                return response
            except Exception as e2:
                print(
                    f"[LLM] Fallback also failed: {e2.__class__.__name__}: {e2}")
                raise AllProvidersFailedError(
                    "All LLM providers failed") from e2
