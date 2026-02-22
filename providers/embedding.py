# providers/embedding.py
"""
API-based embedding providers — no local model downloads required.

Supported:
  • GeminiEmbeddingProvider  — Google gemini-embedding-001 (free tier)
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import List

import numpy as np

# Free-tier quota: 100 embed items per minute.
# Each batch_size=100 call saturates the window, so we wait > 60 s before the
# next batch.  The API also returns a retryDelay on 429; we respect that too.
_BATCH_SIZE = 100
_INTER_BATCH_SLEEP = 62  # seconds — ensures the per-minute window resets
_MAX_RETRIES = 3


def _parse_retry_delay(exc: Exception) -> float:
    """Extract the server-suggested retry delay (seconds) from a 429 error."""
    msg = str(exc)
    # 'retryDelay': '59s'  or  'retryDelay': '0s'
    m = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", msg)
    if m:
        return float(m.group(1)) + 2.0
    # "retry in 59.88s"
    m = re.search(r"retry in (\d+(?:\.\d+)?)", msg)
    if m:
        return float(m.group(1)) + 2.0
    return 62.0  # safe default


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    def encode(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        """
        Encode a list of texts into a float32 numpy array of shape (n, dim).
        task_type: "retrieval_document" for indexing, "retrieval_query" for queries.
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Uses Google's gemini-embedding-001 model via the Gemini API (google-genai SDK).
    Free tier: 100 embed items per minute.
    Dimension: 3072.
    """

    _DIM = 3072

    def __init__(self, api_key: str = None, model: str = "models/gemini-embedding-001"):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required. Install with `pip install google-genai`."
            ) from exc

        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing GOOGLE_API_KEY (or GEMINI_API_KEY) for GeminiEmbeddingProvider."
            )
        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.model = model

    @property
    def dim(self) -> int:
        return self._DIM

    def encode(self, texts: List[str], task_type: str = "retrieval_document") -> np.ndarray:
        """
        Embed texts in batches of up to 100 (the free-tier per-minute item quota).
        Retries up to _MAX_RETRIES times on 429 RESOURCE_EXHAUSTED, sleeping for
        the server-suggested delay between attempts.  After each successful batch,
        waits _INTER_BATCH_SLEEP seconds so the next batch starts with a fresh
        per-minute window.
        """
        if not texts:
            return np.empty((0, self._DIM), dtype="float32")

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            success = False

            for attempt in range(_MAX_RETRIES):
                try:
                    result = self._client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=self._types.EmbedContentConfig(task_type=task_type),
                    )
                    all_embeddings.extend([e.values for e in result.embeddings])
                    success = True
                    break

                except Exception as exc:
                    is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
                    if is_rate_limit and attempt < _MAX_RETRIES - 1:
                        delay = _parse_retry_delay(exc)
                        logging.warning(
                            f"[GeminiEmbed] Rate limited on batch {i//_BATCH_SIZE}, "
                            f"sleeping {delay:.0f}s (attempt {attempt+1}/{_MAX_RETRIES})"
                        )
                        time.sleep(delay)
                    else:
                        logging.error(
                            f"[GeminiEmbed] Batch {i//_BATCH_SIZE} failed "
                            f"(attempt {attempt+1}): {exc}"
                        )
                        break  # fall through to zero-vector fallback

            if not success:
                for _ in batch:
                    all_embeddings.append([0.0] * self._DIM)

            # After each batch, wait long enough for the per-minute quota to reset
            # before issuing the next batch.
            if i + _BATCH_SIZE < len(texts):
                logging.info(
                    f"[GeminiEmbed] Batch {i//_BATCH_SIZE} done; "
                    f"sleeping {_INTER_BATCH_SLEEP}s for quota reset…"
                )
                time.sleep(_INTER_BATCH_SLEEP)

        return np.array(all_embeddings, dtype="float32")
