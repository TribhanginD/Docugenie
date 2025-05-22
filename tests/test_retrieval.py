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
def bm25_and_idx(dummy_corpus):
    bm25, faiss_idx = init_retrievers(dummy_corpus, force_rebuild=True)
    return bm25, faiss_idx

def test_bm25_scores(dummy_corpus, bm25_and_idx):
    bm25, idx = bm25_and_idx
    scores = bm25.get_scores("quick fox".split())
    # highest score for first doc
    assert np.argmax(scores) == 0

def test_hybrid_retrieval(dummy_corpus, bm25_and_idx):
    bm25, idx = bm25_and_idx
    chunks = retrieve_relevant_chunks("lazy dog", bm25, idx, dummy_corpus, top_k=2, alpha=0.5)
    # “lazy dog” appears in the second doc
    assert dummy_corpus[1] in chunks
