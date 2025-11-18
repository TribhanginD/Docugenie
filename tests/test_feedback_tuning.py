import json
import os
import pytest
from pages import Chatbot
from pages.Chatbot import update_cache, retrieve_from_cache, cache_file, MODEL_NAME

def test_cache_roundtrip(tmp_path):
    # clear any existing cache
    if os.path.exists(cache_file):
        os.remove(cache_file)
    Chatbot.cache = {'queries': [], 'embeddings': [], 'responses': [], 'model_name': MODEL_NAME}

    query = "hello world"
    vector = [0.1] * 768
    responses = [
        {"text": "resp1", "doc": "doc1.pdf", "page": 1, "score": 0.5, "index": 0},
        {"text": "resp2", "doc": "doc2.pdf", "page": 2, "score": 0.7, "index": 1},
    ]
    update_cache(query, vector, responses)

    with open(cache_file) as f:
        data = json.load(f)
    assert query in data["queries"]
    # test retrieve_from_cache threshold behavior
    import numpy as np
    found = retrieve_from_cache(np.array(vector), thr=0.01)
    assert found == responses
