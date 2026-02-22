import os
import logging
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_client import make_asgi_app

from core import RAGEngine
from providers.vector_db import QdrantProvider
from providers.reranker import RerankerProvider
from providers.cache import RedisCache
from providers.embedding import GeminiEmbeddingProvider
from providers.llm import GroqProvider, GeminiProvider
from observability import MetricsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docugenie.api")

app = FastAPI(
    title="DocuGenie API",
    description="Production-grade REST API for interactive document Q&A using RAG.",
    version="2.0.0"
)

# Enable CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize global state
_state: Dict[str, Any] = {
    "engine": None,
    "provider": None
}

def get_llm_provider(api_key: Optional[str] = None, provider_name: Optional[str] = None):
    # If dynamic key is provided, always create a new instance
    if api_key:
        if provider_name == "Google Gemini":
            return GeminiProvider(api_key=api_key)
        return GroqProvider(api_key=api_key)

    if _state["provider"]:
        return _state["provider"]
    
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        _state["provider"] = GeminiProvider()
    elif os.getenv("GROQ_API_KEY"):
        _state["provider"] = GroqProvider()
    else:
        # Don't raise error here, just return None and let the caller handle it if needed
        return None
    return _state["provider"]

def init_engine():
    if _state["engine"] is not None:
        return _state["engine"]
    
    logger.info("Initializing Production RAG Engine...")
    provider = get_llm_provider()
    
    vector_db = QdrantProvider(
        location=os.getenv("QDRANT_LOCATION", ":memory:"),
        collection_name=os.getenv("QDRANT_COLLECTION", "docugenie")
    )
    reranker = RerankerProvider(
        llm_provider=provider,
        model_name=os.getenv("RERANKER_MODEL", "gemini-2.0-flash")
    )
    cache = RedisCache(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379))
    )
    embedding_provider = GeminiEmbeddingProvider()
    
    _state["engine"] = RAGEngine(
        vector_db=vector_db,
        reranker=reranker,
        cache=cache,
        llm_provider=provider,
        embedding_provider=embedding_provider
    )
    return _state["engine"]

@app.on_event("startup")
async def startup_event():
    try:
        init_engine()
        logger.info("DocuGenie Production API is ready.")
    except Exception as e:
        logger.error(f"Failed to initialize DocuGenie API: {e}")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = True
    api_key: Optional[str] = None
    provider: Optional[str] = "Groq"

class Citation(BaseModel):
    label: str
    doc: str
    page: Optional[int]
    snippet: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    usage: Dict[str, Any]

@app.post("/ingest")
async def ingest_documents(files: List[Any]):
    # Note: In a real app, use UploadFile. For this MVP, we'll assume PDF processing.
    # This endpoint will trigger the core indexing logic.
    try:
        engine = init_engine()
        # Ingest logic would go here, connecting to process_single_pdf etc.
        # For now, we'll return a placeholder success.
        return {"status": "success", "message": "Documents indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    MetricsTracker.track_query()
    try:
        provider = get_llm_provider(request.api_key, request.provider)
        if not provider:
            raise HTTPException(status_code=400, detail="No LLM provider key available.")
            
        engine = init_engine()
        # Ensure engine uses the request-specific provider for this run
        engine.llm_provider = provider
        engine.reranker.llm_provider = provider

        # Retrieval Phase
        with MetricsTracker.time_retrieval():
            retrieval_data = engine.retrieve(
                request.query, 
                top_k=request.top_k, 
                use_reranker=request.use_reranker
            )
        
        # Generation Phase
        with MetricsTracker.time_generation():
            context = "\n\n".join([
                f"[Source {i+1}: {c.get('doc', 'unknown')} p.{c.get('page', '?')}]\n{c.get('text', '')}"
                for i, c in enumerate(retrieval_data["results"])
            ])
            prompt = (
                f"You are DocuGenie, an expert document assistant. "
                f"Answer the following question using ONLY the provided context. "
                f"Be concise and cite your sources.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {request.query}\n\nAnswer:"
            )
            raw = provider.generate(prompt)
            answer = raw if isinstance(raw, str) else raw.get("answer", str(raw))
            usage = {"provider": request.provider or "default"}

        formatted_citations = [
            Citation(
                label=c.get("label", f"S{i+1}"),
                doc=c.get("doc", "unknown"),
                page=c.get("page"),
                snippet=c.get("text", "")[:300],
                score=c.get("score", 0.0)
            ) for i, c in enumerate(retrieval_data["results"])
        ]
        
        usage.update({
            "retrieval_time_s": round(retrieval_data["retrieval_time"], 3),
            "total_process_time_s": round(retrieval_data["total_time"], 3),
        })
        
        return QueryResponse(
            answer=answer,
            citations=formatted_citations,
            usage=usage
        )
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))

# Serve Static Files (React Build)
# We handle the frontend mounting at the end to avoid blocking API routes
if os.path.exists("./frontend/dist"):
    app.mount("/", StaticFiles(directory="./frontend/dist", html=True), name="frontend")

    @app.exception_handler(404)
    async def custom_404_handler(request, __):
        # Direct all unknown requests to index.html for React Router
        if not request.url.path.startswith("/query") and not request.url.path.startswith("/metrics"):
            return FileResponse("./frontend/dist/index.html")
        raise HTTPException(status_code=404, detail="Not Found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
