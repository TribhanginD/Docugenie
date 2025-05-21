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
        model: str = None,    # ignored, we use self.model_name
        temperature: float = 0.7,
        top_p: float = 0.5,
        **kwargs
    ) -> Tuple[str, Dict]:
        # Forward the messages straight into the HF Hub chat endpoint

        # 🔧 Clamp top_p into the valid (0.0, 1.0) range for HF Inference
       if not (0.0 < top_p < 1.0):
        top_p = 0.9

        logging.info(
            f"HFProvider: calling {self.model_name} via provider={self.provider}"
        )
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
           top_p=top_p,
            **kwargs
        )

        # Extract the assistant’s reply
        text = completion.choices[0].message.content.strip()
        # HF Hub doesn’t currently return token usage, so we only note the model used
        usage = {"model_used": self.model_name}
        return text, usage
