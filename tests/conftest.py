import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from rank_bm25 import BM25Okapi

# ─── Mock the embedding provider at import time ─────────────────────────────
# This prevents any network calls when the test suite loads pages/Chatbot.py.

EMBED_DIM = 3072

class DummyEmbeddingProvider:
    """Returns deterministic random-ish embeddings based on text hash."""

    @property
    def dim(self):
        return EMBED_DIM

    def encode(self, texts, task_type="retrieval_document"):
        rng = np.random.default_rng(seed=42)
        embs = []
        for text in texts:
            seed = hash(text) % (2**31)
            r = np.random.default_rng(seed=seed)
            embs.append(r.standard_normal(EMBED_DIM).astype("float32"))
        return np.array(embs, dtype="float32")


# Patch before any pages.Chatbot import so module-level code uses the mock
_embed_patcher = patch(
    "providers.embedding.GeminiEmbeddingProvider",
    return_value=DummyEmbeddingProvider(),
)
_embed_patcher.start()

# Also patch _embedding_provider directly in Chatbot so lazy-init returns our dummy
import pages.Chatbot as _chatbot  # noqa: E402  (import after patch)
_chatbot._embedding_provider = DummyEmbeddingProvider()


# ─── Shared fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def dummy_corpus():
    return [
        "apple orange banana",
        "banana apple fruit",
        "lorem ipsum dolor",
        "foo bar baz",
    ]


@pytest.fixture
def bm25_and_idx(tmp_path, dummy_corpus):
    from pages.Chatbot import create_faiss_index
    idx = create_faiss_index(dummy_corpus, force_rebuild=True)
    bm25 = BM25Okapi([c.split() for c in dummy_corpus])
    return bm25, idx
