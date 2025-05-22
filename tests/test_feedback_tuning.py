import json
import os
import pytest
from pages.Chatbot import update_cache, retrieve_from_cache, cache_file

def test_cache_roundtrip(tmp_path):
    # clear any existing cache
    if os.path.exists(cache_file):
        os.remove(cache_file)

    query = "hello world"
    vector = [0.1] * 768
    responses = ["resp1", "resp2"]
    update_cache(query, vector, responses)

    with open(cache_file) as f:
        data = json.load(f)
    assert query in data["queries"]
    # test retrieve_from_cache threshold behavior
    import numpy as np
    found = retrieve_from_cache(np.array(vector), thr=0.01)
    assert found == responses
