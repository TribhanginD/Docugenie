import numpy as np
import pytest
from rank_bm25 import BM25Okapi
from pages.Chatbot import retrieve_relevant_chunks, init_retrievers

@pytest.fixture
def dummy_corpus():
    return [
        "the quick brown fox",
        "jumps over the lazy dog",
        "lorem ipsum dolor sit amet"
    ]

@pytest.fixture
def bm25_idx_and_meta(dummy_corpus):
    bm25, faiss_idx = init_retrievers(dummy_corpus, force_rebuild=True)
    metadata = [
        {"doc": f"doc_{i}.pdf", "page": 1, "chunk_size": 1000, "chunk_overlap": 400}
        for i, _ in enumerate(dummy_corpus)
    ]
    return bm25, faiss_idx, metadata

def test_bm25_scores(dummy_corpus, bm25_idx_and_meta):
    bm25, idx, _ = bm25_idx_and_meta
    scores = bm25.get_scores("quick fox".split())
    # highest score for first doc
    assert np.argmax(scores) == 0

def test_hybrid_retrieval(dummy_corpus, bm25_idx_and_meta):
    bm25, idx, metadata = bm25_idx_and_meta
    chunks = retrieve_relevant_chunks(
        "lazy dog",
        bm25,
        idx,
        dummy_corpus,
        metadata,
        top_k=2,
        alpha=0.5
    )
    texts = [ch["text"] for ch in chunks]
    assert dummy_corpus[1] in texts
