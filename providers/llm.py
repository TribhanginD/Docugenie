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


class GeminiProvider(LLMProvider):
    """Google Gemini provider via google-generativeai."""

    def __init__(self, api_key: str = None, model_name: str = None):
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "google-generativeai is required. Install with `pip install google-generativeai`."
            ) from exc
        self._genai = genai
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) for GeminiProvider.")
        self._genai.configure(api_key=self.api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

    def generate(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[str, Dict]:
        # Gemini doesn't use chat roles the same way; concatenate messages into a single prompt.
        prompt_parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System]\n{content}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant]\n{content}\n\n")
            else:
                prompt_parts.append(f"[User]\n{content}\n\n")
        prompt = "".join(prompt_parts)

        mdl = self._genai.GenerativeModel(model or self.model_name)
        resp = mdl.generate_content(prompt, generation_config={
            "temperature": temperature,
            "top_p": top_p,
            **{k: v for k, v in kwargs.items() if k not in {"messages"}}
        })
        text = (resp.text or "").strip()
        usage = {"model_used": model or self.model_name}
        return text, usage
