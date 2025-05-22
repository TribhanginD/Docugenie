import pytest
import numpy as np
from rank_bm25 import BM25Okapi
from pages.Chatbot import create_faiss_index

@pytest.fixture
def dummy_corpus():
    # a tiny “corpus” of 4 mini‐chunks
    return [
        "apple orange banana",
        "banana apple fruit",
        "lorem ipsum dolor",
        "foo bar baz"
    ]

@pytest.fixture
def bm25_and_idx(tmp_path, dummy_corpus):
    # force rebuild so we don't pick up an old pickle
    idx = create_faiss_index(dummy_corpus, force_rebuild=True)
    bm25 = BM25Okapi([c.split() for c in dummy_corpus])
    return bm25, idx
