"""
Reranking tests — updated for LLM-based reranking (no local cross-encoder).
We mock llm_rerank to verify it's called and that the pipeline returns valid results.
"""
import pytest
from unittest.mock import patch, MagicMock
from pages.Chatbot import retrieve_relevant_chunks


def test_rerank_returns_list(dummy_corpus, bm25_and_idx):
    """retrieve_relevant_chunks should always return a list of dicts."""
    bm25, idx = bm25_and_idx
    metadata = [
        {"doc": f"doc_{i}.pdf", "page": 1, "chunk_size": 900, "chunk_overlap": 300}
        for i, _ in enumerate(dummy_corpus)
    ]
    result = retrieve_relevant_chunks(
        "ipsum",
        bm25,
        idx,
        dummy_corpus,
        metadata,
        top_k=3,
        alpha=0.5,
        provider=None,   # no provider → hybrid-score order, no LLM rerank
    )
    assert isinstance(result, list)
    assert len(result) <= len(dummy_corpus)


def test_llm_rerank_called_when_provider_given(dummy_corpus, bm25_and_idx):
    """When a provider is supplied, llm_rerank should be invoked."""
    bm25, idx = bm25_and_idx
    metadata = [
        {"doc": f"doc_{i}.pdf", "page": 1, "chunk_size": 900, "chunk_overlap": 300}
        for i, _ in enumerate(dummy_corpus)
    ]
    mock_provider = MagicMock()

    with patch("pages.Chatbot.llm_rerank", return_value=[]) as mock_rerank:
        retrieve_relevant_chunks(
            "apple fruit",
            bm25,
            idx,
            dummy_corpus,
            metadata,
            top_k=2,
            alpha=0.5,
            provider=mock_provider,
            use_cache=False,
        )
        assert mock_rerank.called, "llm_rerank should be called when provider is given"
