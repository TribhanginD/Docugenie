import json
import os
import pytest
import tempfile
from unittest.mock import patch
from pages import Chatbot
from pages.Chatbot import update_cache, retrieve_from_cache, MODEL_NAME
import numpy as np


def test_cache_roundtrip(tmp_path):
    """Cache write + read should survive a round-trip."""
    cache_path = str(tmp_path / "test_semantic_cache.json")

    # Temporarily redirect the module-level cache to our temp file
    original_cache_file = Chatbot.cache_file
    original_cache = Chatbot.cache

    Chatbot.cache_file = cache_path
    Chatbot.cache = {
        "queries": [],
        "embeddings": [],
        "responses": [],
        "model_name": MODEL_NAME,
    }

    try:
        query = "hello world"
        vector = np.array([0.1] * 768)
        responses = [
            {"text": "resp1", "doc": "doc1.pdf", "page": 1, "score": 0.5, "index": 0},
            {"text": "resp2", "doc": "doc2.pdf", "page": 2, "score": 0.7, "index": 1},
        ]
        update_cache(query, vector, responses)

        with open(cache_path) as f:
            data = json.load(f)
        assert query in data["queries"]

        found = retrieve_from_cache(vector, thr=0.01)
        assert found == responses

    finally:
        # Restore module state
        Chatbot.cache_file = original_cache_file
        Chatbot.cache = original_cache
