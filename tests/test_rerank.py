import pytest
from sentence_transformers import CrossEncoder
from pages.Chatbot import retrieve_relevant_chunks

@pytest.fixture(autouse=True)
def mock_cross_encoder(monkeypatch):
    class DummyCE:
        def predict(self, pairs):
            return [float(len(p[1])) for p in pairs]  # longer chunk → higher score
    monkeypatch.setattr("pages.Chatbot.cross_encoder", DummyCE())

def test_cross_encoder_rerank(dummy_corpus, bm25_and_idx):
    bm25, idx = bm25_and_idx
    # after monkeypatch, reranking picks longest passages
    top_chunks = retrieve_relevant_chunks("ipsum", bm25, idx, dummy_corpus, top_k=3, alpha=0.5)
    # ensure we get back exactly 5 or fewer unique items
    assert isinstance(top_chunks, list)
    assert len(top_chunks) <= 5
