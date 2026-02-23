import os
import logging
import time
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()  # Load .env file for local development

from fastapi import FastAPI, HTTPException, UploadFile, File
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
import httpx

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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Delegation Config
ML_SERVICE_URL = os.getenv("ML_PIPELINE_URL", "http://localhost:8000")
ORDER_SERVICE_URL = os.getenv("ORDER_SYSTEM_URL", "http://localhost:8001")

async def platform_governance_check(content: str) -> bool:
    """Delegates compliance scanning to the Real-Time ML Pipeline."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(f"{ML_SERVICE_URL}/scan", json={"text": content})
            if response.status_code == 200:
                result = response.json()
                if result["status"] == "FLAGGED":
                    logger.warning(f"ML Pipeline BLOCKED ingestion: {result['reason']}")
                    return False
    except Exception as e:
        logger.warning(f"Governance Delegation failed: {e}. Falling back to local scanner.")
        from governance import ComplianceScanner
        local_result = ComplianceScanner.scan_content(content)
        return local_result["status"] != "FLAGGED"
    return True

async def platform_reliable_ingest(task_id: str, doc_name: str) -> bool:
    """Delegates ingestion infrastructure to the Order Processing System."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            payload = {"task_id": task_id, "target_service": "docugenie", "payload": {"file": doc_name}}
            response = await client.post(f"{ORDER_SERVICE_URL}/tasks/ingest", json=payload)
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"Reliability Delegation failed: {e}. Falling back to local ingestion flow.")
        return True # Fallback: let DocuGenie handle it locally

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

    # Embedding provider: requires GOOGLE_API_KEY. Gracefully skip if absent.
    embedding_provider = None
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            embedding_provider = GeminiEmbeddingProvider()
        except Exception as e:
            logger.warning(f"Embedding provider unavailable: {e}")
    else:
        logger.warning("No GOOGLE_API_KEY found — document embedding disabled. Upload & retrieval will not work.")

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
async def ingest_documents(files: List[UploadFile], idempotency_key: Optional[str] = None):
    """
    Accept one or more PDFs, extract text, chunk by page, embed, and index into Qdrant.
    Implements idempotency and resilience patterns.
    """
    try:
        engine = init_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="RAG engine not initialized. Check GOOGLE_API_KEY.")
        if engine.embedding_provider is None:
            raise HTTPException(status_code=503, detail="Embedding provider not available. Set GOOGLE_API_KEY.")

        import pdfplumber, io
        all_texts, all_meta = [], []

        for file in files:
            raw = await file.read()
            doc_name = file.filename or "unknown.pdf"
            
            # 1. Delegate Governance (Compliance/PII)
            if not await platform_governance_check(raw.decode(errors='ignore')):
                raise HTTPException(status_code=403, detail="Governance Violation: PII detected by Compliance Service.")

            # 2. Delegate Reliability (Idempotency/DLQ)
            # Use filename as a simple task_id for this demo
            if not await platform_reliable_ingest(doc_name, doc_name):
                 logger.error(f"Reliability Check failed for {doc_name}")

            try:
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        text = text.strip()
                        if not text:
                            continue
                        # Chunk long pages into ~500-char segments
                        for chunk_start in range(0, len(text), 500):
                            chunk = text[chunk_start:chunk_start + 500].strip()
                            if chunk:
                                all_texts.append(chunk)
                                all_meta.append({
                                    "text": chunk,
                                    "doc": doc_name,
                                    "page": page_num,
                                })
            except Exception as e:
                logger.warning(f"Failed to parse {doc_name}: {e}")
                continue

        if not all_texts:
            raise HTTPException(status_code=400, detail="No extractable text found in uploaded PDFs.")

        # Embed all chunks
        logger.info(f"Embedding {len(all_texts)} chunks from {len(files)} file(s)...")
        vectors = engine.embedding_provider.encode(all_texts, task_type="retrieval_document")

        # Upsert into Qdrant
        engine.vector_db.upsert_chunks(vectors.tolist(), all_meta)

        return {
            "status": "success",
            "chunks_indexed": len(all_texts),
            "files": [f.filename for f in files],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during ingestion")
        # DLQ Simulation: Log failure for the AI SRE to pick up
        logger.error(f"DLQ_EVENT: Ingestion failed for files {[f.filename for f in files]}. Reason: {str(e)}")
        raise HTTPException(status_code=500, detail="Ingestion failed. Logged to DLQ.")


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
            messages = [
                {"role": "system", "content": (
                    "You are DocuGenie, an expert document assistant. "
                    "Answer the question using ONLY the provided context. Be concise and cite your sources."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.query}"},
            ]
            result = provider.generate(
                messages,
                model="gemini-2.0-flash" if request.provider == "Google Gemini" else "llama-3.3-70b-versatile",
            )
            # Both GroqProvider and GeminiProvider return (answer_str, usage_dict)
            if isinstance(result, tuple):
                answer, provider_usage = result
            else:
                answer, provider_usage = str(result), {}
            usage = {"provider": request.provider or "default", **provider_usage}

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
    except HTTPException:
        raise
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
