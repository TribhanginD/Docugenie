DocuGenie
=========

An on-demand, interactive document Q&A assistant powered by Retrieval-Augmented Generation (RAG), hybrid retrieval, cross-encoder re-ranking, self-ask question decomposition, and an active feedback loop. Containerized for one-click startup and designed for production-grade deployment.

* * *

What Is DocuGenie?
------------------

DocuGenie transforms your static PDFs into a living knowledge base. Instead of manually searching through dozens of documents, simply upload them, ask your question, and let DocuGenie dig into the right passages and generate concise, accurate answers.

While many RAG demos stop at simple vector search + generation, DocuGenie goes further:

* **Hybrid Retrieval** (BM25 + FAISS): Combines lexical (BM25) and semantic (FAISS) search to surface a high-quality shortlist.
    
* **Cross-Encoder Re-Ranking**: Uses a lightweight cross-encoder (e.g. `msmarco-MiniLM-L-6-v2`) to reorder your shortlist and pick the top 5 most relevant passages.
    
* **Self-Ask Question Decomposition**: Breaks complex queries into independent sub-questions, retrieves context for each, and stitches the answers back together.
    
* **Active Feedback Loop**: Thumbs-up/thumbs-down buttons let you rate each answer. Disliked answers automatically trigger a re-generation with the same query. Feedback is logged for future tuning of the hybrid weight or cross-encoder fine-tuning.
    
* **Provider-Agnostic LLM Interface**: Swap between Groq’s API or any Hugging Face Inference endpoint (e.g. Mistral-7B-Instruct) via a pluggable `LLMProvider` abstraction.
    

* * *

Features
--------

1. **Multi-PDF Ingestion**
    
    * Drag-and-drop upload of multiple PDFs.
        
    * MD5 hashing + Streamlit’s caching to avoid reprocessing unchanged files.
        
2. **Advanced Retrieval Pipeline**
    
    * **BM25**: Lexical matching—perfect for proper names, dates, code snippets.
        
    * **FAISS**: Semantic similarity—captures meaning across paraphrases.
        
    * **Hybrid Scoring**: Weighted combination of BM25 & FAISS before re-ranking.
        
    * **Cross-Encoder**: Fine-grained reranking to pick the final top-5.
        
3. **Self-Ask / Query Decomposition**
    
    * Automatically splits complex questions into smaller sub-queries.
        
    * Retrieves and answers each sub-query before assembling a unified response.
        
4. **Generation & Transparency**
    
    * Answers crafted in Markdown.
        
    * **Do not** restate the question—only the answer.
        
    * Shows exactly which document passages were used (with expanders).
        
    * Tracks token usage and inference/retrieval timings.
        
5. **Active Feedback Loop**
    
    * Thumbs-up/thumbs-down on the latest answer only.
        
    * 👍 Stored for analytics.
        
    * 👎 Triggers an immediate re-generation with the same query + updated retrieval, then updates metrics.
        
    * Feedback logged to `feedback_log.json` for periodic offline analysis & weight adjustment.
        
6. **Provider Agnostic**
    
    * **Groq API** or any **Hugging Face Inference** provider.
        
    * Easily add new providers by extending the `LLMProvider` interface.
        
7. **Production-Ready Packaging**
    
    * **Dockerfile** (Python 3.11-slim) with model pre-warming at build time—instant startup.
        
    * **Requirements** pinned for reproducibility.
        

* * *

Why DocuGenie?
--------------

| Aspect | Run-of-the-Mill RAG | DocuGenie |
| --- | --- | --- |
| Retrieval | Vector-only | Hybrid (BM25 + FAISS) + Cross-Encoder |
| Complex Query Handling | N/A | Self-Ask decomposition + re-assembly |
| Answer Feedback | No feedback loop | Active thumbs-up/down → instant re-gen |
| Provider Flexibility | Single API only | Pluggable interface: Groq, HF Inference, etc. |
| Startup Latency | Downloads on start | Docker build-time cache of models |
| Transparency | “Black-box” | Shows passages, docs, usage, and timings |

* * *

Getting Started
---------------

### Prerequisites

* Docker
    
* `GROQ_API_KEY` (if using Groq)
    
* `HUGGINGFACEHUB_API_TOKEN` (if using Hugging Face Inference)
    

### Local (without Docker)

```bash
git clone https://github.com/your-org/cutting-edge-ragify.git
cd cutting-edge-ragify

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

export GROQ_API_KEY="…"
export HUGGINGFACEHUB_API_TOKEN="…"
streamlit run welcome.py
```

### Containerized

```bash
docker build -t docugenie .
docker run -p 8501:8501 \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  -e HUGGINGFACEHUB_API_TOKEN="$HUGGINGFACEHUB_API_TOKEN" \
  docugenie
```

Visit [http://localhost:8501](http://localhost:8501).

* * *

Configuration
-------------

All tunable parameters live in **`retrieval_config.json`**. For example:

```json
{
  "bm25_weight": 0.4,
  "faiss_weight": 0.6,
  "bm25_top_k": 20,
  "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "self_ask_model": "gemma2-9b-it"
}
```

Adjust the hybrid weight or shortlist sizes without touching code.

* * *

Testing
-------

* **Unit tests** for PDF extraction, chunking, retrieval, reranking, and feedback loop.
    
* **Integration tests** via `pytest` against a toy corpus and a mock LLM provider.
    

_Run tests locally:_

```bash
pytest --maxfail=1 --disable-warnings -q
```

* * *

Next Steps
----------

1. Add CI/CD with GitHub Actions: build, test, publish Docker image, and smoke-test.
    
2. Deploy to Fly.io / Heroku / AWS App Runner for a public demo.
    
3. Create an analytics dashboard to visualize feedback trends and retrieval performance.
    
4. Fine-tune your cross-encoder on logged feedback to personalize retrieval over time.
    

* * *

License
-------

MIT License

DocuGenie isn’t just another RAG demo—it’s a full-featured, production-grade platform for turning your documents into an intelligent knowledge base. Give it a spin and watch your PDFs come alive!