# providers/llm.py

import os
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from huggingface_hub import InferenceClient
import logging


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict],
        model: str,
        **kwargs
    ) -> Tuple[str, Dict]:
        """
        Send a chat-style request and return (response_text, usage_info_dict).
        `messages` is a list of {"role":..., "content":...}.
        """
        pass

    def generate_stream(
        self,
        messages: List[Dict],
        model: str,
        **kwargs
    ):
        """
        Streaming version — yields text chunks as they arrive.
        Default fallback: calls generate() and yields the full response at once.
        Override in subclasses for true token-by-token streaming.
        """
        text, _ = self.generate(messages, model, **kwargs)
        yield text


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        import groq
        self.client = groq.Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def generate(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1,
        **kwargs
    ) -> Tuple[str, Dict]:
        chat_completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
            stop=None,
            **kwargs
        )
        response = chat_completion.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": chat_completion.usage.prompt_tokens,
            "completion_tokens": chat_completion.usage.completion_tokens,
            "total_tokens": chat_completion.usage.total_tokens,
            "model_used": model,
        }
        return response, usage

    def generate_stream(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1,
        **kwargs
    ):
        """Yield tokens as they arrive from Groq's streaming API."""
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            stop=None,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class HFProvider(LLMProvider):
    def __init__(
        self,
        token: str = None,
        model_name: str = None,
        provider: str = None
    ):
        # Read token and model from env if not passed
        self.token = token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not self.token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN not set! "
                "Please `export HUGGINGFACEHUB_API_TOKEN=hf_xxx` or add to your .env"
            )

        # Default to the exact repo you want
        self.model_name = model_name or os.getenv(
            "HUGGINGFACE_MODEL",
            "mistralai/Mistral-7B-Instruct-v0.3"
        )

        # The HF Hub client supports multiple “providers” (huggingface, together, etc.)
        self.provider = provider or os.getenv("HF_PROVIDER", "hf-inference")
        logging.info(f"HFProvider initializing InferenceClient(provider={self.provider})")
        self.client = InferenceClient(
            provider=self.provider,
            api_key=self.token
        )

    def generate(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        top_p: float = 0.5,
        **kwargs
    ) -> Tuple[str, Dict]:
        # Forward the messages straight into the HF Hub chat endpoint

        # 🔧 Clamp top_p into the valid (0.0, 1.0) range for HF Inference
        if not (0.0 < top_p < 1.0):
            top_p = 0.9

        target_model = model or self.model_name
        logging.info(
            f"HFProvider: calling {target_model} via provider={self.provider}"
        )
        completion = self.client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )

        # Extract the assistant’s reply
        text = completion.choices[0].message.content.strip()
        # HF Hub doesn’t currently return token usage, so we only note the model used
        usage = {"model_used": target_model}
        return text, usage

    def generate_stream(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        top_p: float = 0.5,
        **kwargs
    ):
        """Yield tokens as they arrive from HuggingFace’s streaming API."""
        if not (0.0 < top_p < 1.0):
            top_p = 0.9
        target_model = model or self.model_name
        stream = self.client.chat.completions.create(
            model=target_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token


class GeminiProvider(LLMProvider):
    """Google Gemini provider via google-genai SDK."""

    def __init__(self, api_key: str = None, model_name: str = None):
        try:
            from google import genai
            from google.genai import types as genai_types
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "google-genai is required. Install with `pip install google-genai`."
            ) from exc
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) for GeminiProvider.")
        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    def _build_prompt(self, messages: List[Dict]) -> str:
        """Convert chat messages to a single prompt string."""
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}\n\n")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}\n\n")
            else:
                parts.append(f"[User]\n{content}\n\n")
        return "".join(parts)

    def generate(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[str, Dict]:
        prompt = self._build_prompt(messages)
        response = self._client.models.generate_content(
            model=model or self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
            ),
        )
        text = (response.text or "").strip()
        usage = {"model_used": model or self.model_name}
        return text, usage

    def generate_stream(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """Yield tokens as they arrive from Gemini's streaming API."""
        prompt = self._build_prompt(messages)
        for chunk in self._client.models.generate_content_stream(
            model=model or self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
            ),
        ):
            try:
                if chunk.text:
                    yield chunk.text
            except Exception:
                continue
