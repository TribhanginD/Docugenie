
import os
import json
import logging
import hashlib
import time
import pickle
import base64
import re
from pathlib import Path
import html
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import RateLimitError
from rank_bm25 import BM25Okapi
from providers.llm import LLMProvider, GroqProvider, HFProvider
try:
    from providers.llm import GeminiProvider  # type: ignore
except Exception:
    GeminiProvider = None  # type: ignore

# ─── Load config & logging ─────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}

CHUNK_STATS_FILE = "chunking_stats.json"
RERANK_STATS_FILE = "rerank_stats.json"
MANUAL_CORRECTIONS_FILE = "manual_corrections.json"

# ─── Persistence Helpers ────────────────────────────────────────────────────────
def _load_json_file(path: str, default: Any) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        logging.warning(f"Failed to decode JSON from {path}, resetting to default.")
        return default

def _save_json_file(path: str, data: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_chunk_stats() -> Dict[str, Any]:
    return _load_json_file(CHUNK_STATS_FILE, {})

def save_chunk_stats(stats: Dict[str, Any]) -> None:
    _save_json_file(CHUNK_STATS_FILE, stats)

def load_rerank_stats() -> Dict[str, Any]:
    return _load_json_file(RERANK_STATS_FILE, {})

def save_rerank_stats(stats: Dict[str, Any]) -> None:
    _save_json_file(RERANK_STATS_FILE, stats)

def load_manual_corrections() -> Dict[str, List[str]]:
    toggle = os.getenv("DOCUGENIE_ENABLE_MANUAL_CORRECTIONS", "").lower()
    if toggle not in {"1", "true", "yes", "on"}:
        return {}
    return _load_json_file(MANUAL_CORRECTIONS_FILE, {})

# ─── New: FAISS-only & BM25-only retrievers ────────────────────────────────────
def retrieve_faiss_only(
    query: str,
    idx: faiss.Index,
    all_chunks: List[str],
    top_k: int = 5
) -> List[str]:
    """Return top_k passages using FAISS vector search only."""
    qv = model.encode([query])[0].astype('float32')
    _, ids = idx.search(np.array([qv]), min(top_k, len(all_chunks)))
    return [all_chunks[i] for i in ids[0]]

def retrieve_bm25_only(
    query: str,
    bm25: BM25Okapi,
    all_chunks: List[str],
    top_k: int = 5
) -> List[str]:
    """Return top_k passages using BM25 lexical matching only."""
    scores = bm25.get_scores(query.split())
    top_idxs = np.argsort(scores)[::-1][:min(top_k, len(all_chunks))]
    return [all_chunks[i] for i in top_idxs]

# ─── Adaptive Chunking ─────────────────────────────────────────────────────────
def select_chunk_params(doc_name: str, doc_text: str, stats: Dict[str, Any]) -> Tuple[int, int]:
    entry = stats.get(doc_name, {})
    if entry:
        return entry.get("chunk_size", 1000), entry.get("chunk_overlap", 400)

    length = len(doc_text or "")
    if length < 5000:
        return 600, 200
    if length < 20000:
        return 900, 300
    return 1200, 400

def record_chunk_performance(
    doc_name: str,
    chunk_size: int,
    chunk_overlap: int,
    avg_rerank_score: float,
    liked: bool
) -> None:
    stats = load_chunk_stats()
    entry = stats.get(doc_name, {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "samples": 0,
        "avg_score": 0.0,
        "likes": 0,
        "dislikes": 0,
    })

    entry["samples"] = entry.get("samples", 0) + 1
    entry["avg_score"] = (
        (entry.get("avg_score", 0.0) * (entry["samples"] - 1) + avg_rerank_score) / entry["samples"]
    )
    if liked:
        entry["likes"] = entry.get("likes", 0) + 1
    else:
        entry["dislikes"] = entry.get("dislikes", 0) + 1

    # Simple auto-tuning: adjust chunk size based on running score
    if entry["samples"] >= 5:
        score = entry["avg_score"]
        if score < 0.4 and entry["chunk_size"] > 500:
            entry["chunk_size"] = max(500, entry["chunk_size"] - 100)
            entry["chunk_overlap"] = max(150, entry["chunk_overlap"] - 50)
        elif score > 0.75 and entry["chunk_size"] < 1500:
            entry["chunk_size"] = min(1500, entry["chunk_size"] + 100)
            entry["chunk_overlap"] = min(600, entry["chunk_overlap"] + 50)

    stats[doc_name] = entry
    save_chunk_stats(stats)

# ─── Conversational Memory Helpers ────────────────────────────────────────────
def build_memory_prompt(memory: List[Dict[str, Any]], limit: int = 3) -> str:
    if not memory:
        return ""
    tail = memory[-limit:]
    formatted = []
    for item in tail:
        q = item.get("question")
        a = item.get("answer")
        if not q or not a:
            continue
        formatted.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(formatted)

def attach_manual_corrections(
    query: str,
    chunks: List[Dict[str, Any]],
    corrections: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    manual = corrections.get(query, [])
    if not manual:
        return chunks
    augmented = list(chunks)
    for text in manual:
        augmented.append({
            "doc": "Manual Correction",
            "page": None,
            "chunk_size": None,
            "chunk_overlap": None,
            "text": text,
            "index": -1,
            "score": 1.0,
            "adjusted_score": 1.0,
        })
    return augmented

def average_rerank_score(chunks: List[Dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    return float(np.mean([c.get("score", 0.0) for c in chunks if c.get("index", -1) != -1]))

# ─── Reranker Personalization ──────────────────────────────────────────────────
def get_rerank_bias(doc_name: str) -> float:
    stats = load_rerank_stats()
    return stats.get(doc_name, {}).get("bias", 0.0)

def update_rerank_bias(doc_name: str, delta: float) -> None:
    stats = load_rerank_stats()
    entry = stats.get(doc_name, {"bias": 0.0})
    entry["bias"] = float(max(-1.0, min(1.0, entry.get("bias", 0.0) + delta)))
    stats[doc_name] = entry
    save_rerank_stats(stats)

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuGenie",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Retrieval config & auto‐tuning ────────────────────────────────────────────
CONFIG_FILE = "retrieval_config.json"
DEFAULT_RETRIEVAL_CONFIG: Dict[str, float] = {
    "alpha": 0.7,                # heavier semantic weighting
    "top_k": 14,                 # wider initial shortlist
    "candidate_multiplier": 4,   # expand shortlist before rerank
    "rerank_weight": 0.8,        # emphasize cross-encoder score
}

def load_config() -> Dict[str, float]:
    defaults = DEFAULT_RETRIEVAL_CONFIG.copy()
    try:
        with open(CONFIG_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        save_config(defaults)
        return defaults
    merged = {**defaults, **data}
    if merged != data:
        save_config(merged)
    return merged

def save_config(conf: Dict[str, float]) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(conf, f)

def tune_alpha() -> None:
    try:
        fb = json.load(open("feedback_log.json"))
    except FileNotFoundError:
        return
    total = len(fb)
    if total == 0:
        return
    s = sum(fb.values())  # +1 for like, -1 for dislike
    new_alpha = (total + s) / (2 * total)
    new_alpha = max(0.0, min(1.0, new_alpha))
    cfg = load_config()
    cfg["alpha"] = new_alpha
    save_config(cfg)
    logging.info(f"[Auto-Tune] Set new α = {new_alpha:.3f}")

# ─── Embedding & Cross-Encoder Models ───────────────────────────────────────────
MODEL_NAME = 'all-mpnet-base-v2'
model = SentenceTransformer(MODEL_NAME)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

PRIMARY_MODEL = os.getenv("DOCUGENIE_PRIMARY_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = os.getenv("DOCUGENIE_FALLBACK_MODEL", "llama-3.1-8b-instant")
DECOMPOSER_MODEL = os.getenv("DOCUGENIE_DECOMPOSER_MODEL", PRIMARY_MODEL)
FACT_CHECK_MODEL = os.getenv("DOCUGENIE_FACT_CHECK_MODEL", PRIMARY_MODEL)

# ─── PDF → Text Chunks Pipeline ────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    text = ''
    with open(path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def create_chunks(text: str, chunk_size=1000, chunk_overlap=400) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def get_files_hash(directory: str) -> str:
    h = hashlib.md5()
    for fn in sorted(os.listdir(directory)):
        if fn.lower().endswith('.pdf'):
            with open(os.path.join(directory, fn), 'rb') as f:
                for piece in iter(lambda: f.read(4096), b''):
                    h.update(piece)
    return h.hexdigest()

@st.cache_data
def process_pdfs(prev_hash: str = None) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, Dict[str, int]], str]:
    pdf_dir = './input_files/'
    cur_hash = get_files_hash(pdf_dir)
    if prev_hash is not None and cur_hash != prev_hash:
        st.cache_data.clear()
    stats = load_chunk_stats()
    all_chunks: List[str] = []
    chunk_metadata: List[Dict[str, Any]] = []
    doc_params: Dict[str, Dict[str, int]] = {}
    for fn in sorted(os.listdir(pdf_dir)):
        if fn.lower().endswith('.pdf'):
            path = os.path.join(pdf_dir, fn)
            reader = PdfReader(path)
            page_texts = [(page.extract_text() or "") + '\n' for page in reader.pages]
            full_text = "".join(page_texts)
            chunk_size, chunk_overlap = select_chunk_params(fn, full_text, stats)
            doc_params[fn] = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            for page_num, page_text in enumerate(page_texts, start=1):
                for c in create_chunks(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                    if not c.strip():
                        continue
                    all_chunks.append(c)
                    chunk_metadata.append({
                        "doc": fn,
                        "page": page_num,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    })

    logging.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks, chunk_metadata, doc_params, cur_hash

# ─── FAISS index & retriever init ───────────────────────────────────────────────
@st.cache_resource
def create_faiss_index(all_chunks: List[str], force_rebuild: bool = False):
    idx_file = 'faiss_index.pkl'
    if os.path.exists(idx_file) and not force_rebuild:
        with open(idx_file, 'rb') as f:
            return pickle.load(f)
    embs = model.encode(all_chunks).astype('float32')
    d, n = embs.shape[1], embs.shape[0]
    # Keep the flat index for moderate corpora to avoid FAISS training crashes on macOS.
    if n < 400:
        idx = faiss.IndexFlatL2(d)
    else:
        quant = faiss.IndexFlatL2(d)
        # FAISS recommends at least ~40 training points per centroid.
        # Clamp the number of clusters to avoid training warnings/crashes when n is small.
        suggested = int(np.sqrt(n))
        max_clusters = max(1, n // 40)
        n_clusters = max(1, min(suggested, max_clusters if max_clusters else 1, 100))
        if n_clusters < 1:
            n_clusters = 1
        idx = faiss.IndexIVFFlat(quant, d, n_clusters)
        try:
            idx.train(embs)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.warning("FAISS IVF training failed (%s); falling back to IndexFlatL2.", exc)
            idx = faiss.IndexFlatL2(d)
    idx.add(embs)
    with open(idx_file, 'wb') as f:
        pickle.dump(idx, f)
    logging.info("Created new FAISS index.")
    return idx

@st.cache_resource
def init_retrievers(all_chunks: List[str], force_rebuild: bool = False):
    faiss_idx = create_faiss_index(all_chunks, force_rebuild=force_rebuild)
    bm25 = BM25Okapi([c.split() for c in all_chunks])
    return bm25, faiss_idx

# ─── Semantic Cache ─────────────────────────────────────────────────────────────
cache_file = 'semantic_cache.json'
def load_cache():
    try:
        with open(cache_file) as f:
            data = json.load(f)
            if data.get('model_name') != MODEL_NAME:
                return {'queries': [], 'embeddings': [], 'responses': [], 'model_name': MODEL_NAME}
            return data
    except FileNotFoundError:
        return {'queries': [], 'embeddings': [], 'responses': [], 'model_name': MODEL_NAME}

def save_cache(c):
    with open(cache_file, 'w') as f:
        json.dump(c, f)

cache = load_cache()

def retrieve_from_cache(v: np.ndarray, thr: float = 0.5):
    for i, emb in enumerate(cache['embeddings']):
        if len(emb) != len(v):
            continue
        if np.linalg.norm(v - np.array(emb)) < thr:
            cached_resp = cache['responses'][i]
            if isinstance(cached_resp, list) and cached_resp and isinstance(cached_resp[0], str):
                cached_resp = [
                    {"text": text, "doc": "cached", "page": None, "score": 0.0, "index": -1}
                    for text in cached_resp
                ]
                cache['responses'][i] = cached_resp
                save_cache(cache)
            return cached_resp
    return None

def update_cache(q: str, v: np.ndarray, resps: List[str]):
    cache['queries'].append(q)
    emb_list = v.tolist() if hasattr(v, "tolist") else list(v)
    cache['embeddings'].append(emb_list)
    cache['responses'].append(resps)
    cache['model_name'] = MODEL_NAME
    save_cache(cache)

# ─── Query Reformulation (Self-Ask) ────────────────────────────────────────────
def self_ask_split(query: str, provider: LLMProvider) -> List[str]:
    prompt = (
        "You decompose complex information-seeking questions for a retrieval system."
        "\nReturn a JSON array of concise sub-questions that can be answered independently."
        "\nRules:"
        "\n- If the question is simple, return an array containing the original question."
        "\n- Keep each sub-question specific and non-overlapping."
        f"\n\nQuestion: \"{query}\""
    )
    text, _ = provider.generate(
        messages=[{"role":"user","content":prompt}],
        model=DECOMPOSER_MODEL,
        temperature=0.0,
        max_tokens=256,
        top_p=0.9
    )
    try:
        subs = json.loads(text)
        if isinstance(subs, list) and all(isinstance(s, str) for s in subs):
            return subs
    except json.JSONDecodeError:
        pass
    return [query]

# ─── Hybrid + Cross-Encoder Re-Ranking Retrieval ───────────────────────────────
def retrieve_relevant_chunks(
    query: str,
    bm25: BM25Okapi,
    idx,
    all_chunks: List[str],
    chunk_metadata: List[Dict[str, Any]],
    top_k: int = 5,
    alpha: float = None,   # load from config if None
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    cfg = load_config()
    if alpha is None:
        alpha = float(cfg.get("alpha", DEFAULT_RETRIEVAL_CONFIG["alpha"]))
    alpha = max(0.35, min(0.95, alpha))

    chunk_count = len(all_chunks)
    requested_top_k = max(1, top_k)
    cfg_top_k = int(cfg.get("top_k", requested_top_k))
    base_top_k = max(requested_top_k, cfg_top_k)
    adaptive_top_k = max(base_top_k, int(np.sqrt(chunk_count) + 6) if chunk_count else base_top_k)
    requested_top_k = min(chunk_count, max(6, adaptive_top_k))

    candidate_multiplier = max(1, int(cfg.get("candidate_multiplier", DEFAULT_RETRIEVAL_CONFIG["candidate_multiplier"])))
    env_mult = os.getenv("DOCUGENIE_RETRIEVAL_CANDIDATE_MULTIPLIER")
    if env_mult:
        try:
            candidate_multiplier = max(1, int(env_mult))
        except ValueError:
            logging.warning("Invalid DOCUGENIE_RETRIEVAL_CANDIDATE_MULTIPLIER=%s", env_mult)
    auto_multiplier = max(2, min(6, (chunk_count // 40) + 2))
    candidate_multiplier = max(candidate_multiplier, auto_multiplier)

    rerank_weight = float(cfg.get("rerank_weight", DEFAULT_RETRIEVAL_CONFIG["rerank_weight"]))
    env_weight = os.getenv("DOCUGENIE_RERANK_WEIGHT")
    if env_weight:
        try:
            rerank_weight = float(env_weight)
        except ValueError:
            logging.warning("Invalid DOCUGENIE_RERANK_WEIGHT=%s", env_weight)
    rerank_weight = max(0.0, min(1.0, rerank_weight))

    v = model.encode([query])[0].astype('float32')
    cached = retrieve_from_cache(v) if use_cache else None
    if cached:
        return cached

    # FAISS shortlist
    shortlist_k = min(len(all_chunks), requested_top_k * candidate_multiplier)
    if shortlist_k == 0:
        return []
    K = shortlist_k
    D, I = idx.search(np.array([v]), K)
    faiss_rank = []
    for j, idx_ in enumerate(I[0]):
        if idx_ < 0:
            continue
        dist = float(D[0][j])
        sim = 1.0 / (1.0 + max(dist, 0.0))
        faiss_rank.append((int(idx_), sim))

    # BM25 shortlist
    bm25_scores = bm25.get_scores(query.split())
    bm25_rank = sorted(
        [(i, float(bm25_scores[i])) for i in range(len(all_chunks))],
        key=lambda x: x[1],
        reverse=True
    )[:K]

    # Normalize & hybrid score
    fs = np.array([s for _, s in faiss_rank]) if faiss_rank else np.array([])
    bs = np.array([s for _, s in bm25_rank]) if bm25_rank else np.array([])
    fs = (fs - fs.min())/(np.ptp(fs)+1e-8) if np.ptp(fs)>0 else fs
    bs = (bs - bs.min())/(np.ptp(bs)+1e-8) if np.ptp(bs)>0 else bs

    hybrid_scores: Dict[int, float] = {}
    for i,(chunk_idx,_) in enumerate(faiss_rank):
        hybrid_scores[chunk_idx] = alpha * (fs[i] if len(fs) else 0.0)
    for i,(chunk_idx,_) in enumerate(bm25_rank):
        hybrid_scores[chunk_idx] = hybrid_scores.get(chunk_idx, 0.0) + (1-alpha) * (bs[i] if len(bs) else 0.0)

    ranked_candidates = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:shortlist_k]
    if not ranked_candidates:
        return []

    shortlist_indices = [idx_ for idx_, _ in ranked_candidates]

    # Cross-encoder re-ranking → requested_top_k
    pairs = [(query, all_chunks[idx_]) for idx_ in shortlist_indices]
    rerank_scores = cross_encoder.predict(pairs)
    adjusted = []
    for (chunk_idx, base_score), raw_score in zip(ranked_candidates, rerank_scores):
        doc_name = chunk_metadata[chunk_idx]["doc"]
        bias = get_rerank_bias(doc_name)
        cross_score = float(raw_score)
        adjusted_score = (rerank_weight * cross_score) + ((1 - rerank_weight) * float(base_score))
        adjusted.append((chunk_idx, adjusted_score + bias, cross_score, float(base_score)))
    reranked = sorted(adjusted, key=lambda x: x[1], reverse=True)
    final: List[Dict[str, Any]] = []
    for chunk_idx, adj_score, raw_score, base_score in reranked[:requested_top_k]:
        meta = chunk_metadata[chunk_idx].copy()
        meta.update({
            "text": all_chunks[chunk_idx],
            "index": chunk_idx,
            "score": raw_score,
            "hybrid_score": base_score,
            "adjusted_score": adj_score,
        })
        final.append(meta)

    if use_cache:
        update_cache(query, v, final)
    return final

# ─── Fact Checking & Citations ────────────────────────────────────────────────
def fact_check_answer(
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    provider: LLMProvider,
    model: str = FACT_CHECK_MODEL
) -> Dict[str, Any]:
    if not chunks:
        return {"verdict": "unknown", "explanation": "No supporting context available."}

    context = "\n\n".join(
        f"[{chunk.get('doc', 'source')}]\n{chunk.get('text', '')}" for chunk in chunks
    )
    prompt = (
        "You are verifying whether an answer is supported by the supplied passages."
        "\nRespond with JSON of the form:"
        " {\"overall_verdict\": str, \"overall_explanation\": str, \"issues\": ["
        "{\"sentence\": str, \"verdict\": str, \"explanation\": str, \"labels\": [str]}...]}"
        "\n`overall_verdict` must be one of `confirmed`, `flagged`, `uncertain`."
        "\nEach issue should reference the exact sentence (or clause) from the answer,"
        " assign `verdict` values `supported`, `contradicted`, or `missing_evidence`,"
        " explain the reasoning, and list any cited source labels (e.g., `S1`)."
        "\nOnly rely on the provided sources; ignore outside knowledge."
        f"\n\nSources:\n{context}\n\nAnswer:\n{answer}\n\nQuestion:\n{query}"
    )
    try:
        text, _ = provider.generate(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=256,
            top_p=0.9
        )
        # Try to sanitize common LLM wrappers (code fences, prose) before JSON parsing
        raw = (text or "").strip()
        if raw.startswith("```"):
            # strip ```json ... ``` fences
            raw = raw.lstrip("`")
            # remove language tag if present
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
        # Attempt bracketed extraction if needed
        if raw and not raw.lstrip().startswith("{"):
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end+1]
        data = json.loads(raw)
        if {"overall_verdict", "issues"} <= data.keys():
            return {
                "verdict": data.get("overall_verdict", "uncertain"),
                "explanation": data.get("overall_explanation", ""),
                "issues": data.get("issues", []),
            }
        if {"verdict", "explanation"} <= data.keys():
            data.setdefault("issues", [])
            return data
    except Exception as exc:
        logging.debug(f"Fact check parse failed: {exc}")
    return {
        "verdict": "uncertain",
        "explanation": "Verification model could not confirm the answer.",
        "issues": []
    }

def build_citations(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    keywords = {word for word in re.findall(r'\w+', query.lower()) if len(word) > 3}
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        highlighted = []
        for sentence in sentences:
            norm = sentence.lower()
            if any(word in norm for word in keywords):
                highlighted.append(sentence)
        if not highlighted and sentences:
            highlighted.append(sentences[0])
        snippet = " … ".join(
            f"<mark>{html.escape(sentence)}</mark>" for sentence in highlighted
        )
        if not snippet:
            snippet = html.escape(text[:160]) + ("…" if len(text) > 160 else "")
        label = chunk.get("label", f"S{idx}")
        citations.append({
            "label": f"[{label}]",
            "doc": chunk.get("doc", "Unknown source"),
            "page": chunk.get("page"),
            "snippet": snippet,
            "score": round(chunk.get("score", 0.0), 3),
            "index": chunk.get("index"),
        })
    return citations

def critique_requires_retry(fact: Dict[str, Any]) -> bool:
    if not fact:
        return False
    verdict = (fact.get("verdict") or "").lower()
    issues = fact.get("issues") or []
    if verdict == "flagged" and issues:
        for issue in issues:
            if (issue.get("verdict") or "").lower() in {"contradicted", "missing_evidence"}:
                return True
    else:
        for issue in issues:
            if (issue.get("verdict") or "").lower() in {"contradicted", "missing_evidence"}:
                return True
    return False

def adjust_biases_from_critique(fact: Dict[str, Any], chunk_results: List[Dict[str, Any]]) -> None:
    issues = fact.get("issues") or []
    if not issues:
        return
    label_map = {
        chunk.get("label"): chunk
        for chunk in chunk_results
        if chunk.get("label") and chunk.get("doc") not in {None, "Manual Correction"}
    }
    for issue in issues:
        verdict = (issue.get("verdict") or "").lower()
        labels = issue.get("labels") or []
        if verdict not in {"supported", "contradicted", "missing_evidence"}:
            continue
        delta = 0.0
        if verdict == "supported":
            delta = 0.02
        elif verdict in {"contradicted", "missing_evidence"}:
            delta = -0.05
        if not delta:
            continue
        for label in labels:
            chunk = label_map.get(label)
            if not chunk:
                continue
            doc = chunk.get("doc")
            if not doc:
                continue
            update_rerank_bias(doc, delta)

def embed_pdf_viewer(path: Path, page: int, height: int = 600) -> None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        st.warning(f"Unable to find {path}.")
        return
    b64_pdf = base64.b64encode(data).decode("utf-8")
    src = f"data:application/pdf;base64,{b64_pdf}#page={page}"
    st.components.v1.html(
        f'<iframe src="{src}" width="100%" height="{height}" type="application/pdf"></iframe>',
        height=height,
    )

def answer_query(
    query: str,
    provider: LLMProvider,
    bm25: BM25Okapi,
    faiss_idx,
    all_chunks: List[str],
    chunk_metadata: List[Dict[str, Any]],
    memory: List[Dict[str, Any]],
    corrections: Dict[str, List[str]],
    corrections_key: str = None,
    alpha: float = None
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    memory_prompt = build_memory_prompt(memory)
    correction_target = corrections_key or query
    attempt = 0
    max_attempts = 1
    cfg = load_config()
    default_top_k = int(cfg.get("top_k", DEFAULT_RETRIEVAL_CONFIG["top_k"]))
    top_k_env = os.getenv("DOCUGENIE_EVAL_TOP_K")
    if top_k_env:
        try:
            top_k = max(1, int(top_k_env))
        except ValueError:
            logging.warning("Invalid DOCUGENIE_EVAL_TOP_K=%s", top_k_env)
            top_k = default_top_k
    else:
        top_k = default_top_k
    final_answer = ""
    final_usage: Dict[str, Any] = {}
    final_chunks: List[Dict[str, Any]] = []
    final_citations: List[Dict[str, Any]] = []
    final_fact: Dict[str, Any] = {}

    while True:
        use_cache = attempt == 0
        t0 = time.time()
        chunk_results = retrieve_relevant_chunks(
            query,
            bm25,
            faiss_idx,
            all_chunks,
            chunk_metadata,
            top_k=top_k,
            alpha=alpha,
            use_cache=use_cache,
        )
        rec_t = time.time() - t0

        augmented_chunks = attach_manual_corrections(correction_target, chunk_results, corrections)

        t1 = time.time()
        answer, usage, _ = generate_response(
            query=query,
            chunks=augmented_chunks,
            provider=provider,
            memory_context=memory_prompt,
        )
        inf_t = time.time() - t1

        fact = fact_check_answer(query, answer, chunk_results, provider)
        adjust_biases_from_critique(fact, chunk_results)

        citations = build_citations(query, chunk_results)

        usage = usage or {}
        usage.update({
            "retrieval_time_s": round(rec_t, 3),
            "inference_time_s": round(inf_t, 3),
            "fact_check_verdict": fact.get("verdict"),
            "self_critique_attempts": attempt + 1,
            "top_k_used": top_k,
        })

        final_answer = answer
        final_usage = usage
        final_chunks = chunk_results
        final_citations = citations
        final_fact = fact

        if fact.get("verdict") == "flagged":
            explanation = fact.get("explanation", "Potential issue detected.")
            final_answer += f"\n\n> ⚠️ Fact-check flagged: {explanation}"

        if attempt >= max_attempts or not critique_requires_retry(fact):
            break

        attempt += 1
        top_k = min(len(all_chunks), top_k + 3)

    return final_answer, final_usage, final_chunks, final_citations, final_fact
# ─── Generation ────────────────────────────────────────────────────────────────
def _enforce_answer_style(text: str) -> str:
    """Clamp answers to two sentences with trimmed whitespace."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    kept = sentences[:2]
    joined = " ".join(s.strip() for s in kept if s.strip())
    return joined or cleaned


def _ensure_citation_presence(text: str, chunks: List[Dict[str, Any]]) -> str:
    """Guarantee at least one citation tag if context was provided."""
    if "[S" in text or not chunks:
        return text
    first_label = chunks[0].get("label")
    if first_label:
        suffix = f" [{first_label}]"
        return text.rstrip() + suffix
    return text


def generate_response(
    query: str,
    chunks: List[Dict[str, Any]],
    provider: LLMProvider,
    primary_model: str = PRIMARY_MODEL,
    fallback_model: str = FALLBACK_MODEL,
    max_retries: int = 3,
    memory_context: str = "",
) -> Tuple[str, Dict, List[Dict[str, Any]]]:
    has_manual = any(chunk.get("doc") == "Manual Correction" for chunk in chunks)
    context_sections = []
    if memory_context:
        context_sections.append(f"Prior conversation snapshots:\n{memory_context}")

    if chunks:
        formatted_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            label = f"S{idx}"
            chunk["label"] = label
            doc = chunk.get("doc", "Unknown source")
            page = chunk.get("page")
            header = f"[{label}] {doc}"
            if page is not None:
                header += f" :: page {page}"
            formatted_chunks.append(f"{header}\n{chunk.get('text', '')}")
        context_sections.append("Source passages:\n" + "\n\n".join(formatted_chunks))

    ctx = "\n\n".join(context_sections) if context_sections else "No context provided."
    prompt = (
        "Using only the supplied context, answer in no more than two sentences."
        "\nSentence 1 must contain a direct, minimal answer (e.g., entity, quantity, yes/no) and end with the supporting citation `[S#]`."
        "\nSentence 2 is optional and should only appear if needed for short support, also citing sources."
        "\nIf the context truly lacks the answer, respond with `I couldn't find that in the supplied documents.`"
        "\nDo not restate the question."
        "\n\nContext:\n" + ctx + f"\n\nQuestion: {query}\n\nAnswer:"
    )

    system_prompt = (
        "You are DocuGenie, a retrieval-grounded assistant."
        "\nFollow these rules strictly:"
        "\n1. Use only the provided context (and manual corrections, if any)."
        "\n2. Provide a direct answer in the first sentence, ending with the source label (e.g., `[S1]`)."
        "\n3. Add at most one supporting sentence, also fully cited."
        "\n4. If the context does not contain the answer, state that clearly."
        "\n5. Never invent facts, citations, or leverage external knowledge."
    )
    if has_manual:
        system_prompt += (
            "\nManual corrections override conflicting context; reproduce them verbatim when applicable."
        )

    provider_name = provider.__class__.__name__
    ordered_models: List[Optional[str]] = []
    if provider_name == "GeminiProvider":
        ordered_models = [None]
    else:
        for candidate in (primary_model, fallback_model):
            if not candidate:
                continue
            if candidate not in ordered_models:
                ordered_models.append(candidate)

    for m in ordered_models:
        for _ in range(max_retries):
            try:
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                gen_kwargs = {
                    "messages": msgs,
                    "model": m,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
                if provider_name == "GeminiProvider":
                    gen_kwargs["max_output_tokens"] = 1024
                else:
                    gen_kwargs["max_tokens"] = 1024
                out, usage = provider.generate(**gen_kwargs)
                post = _ensure_citation_presence(_enforce_answer_style(out), chunks)
                return post, usage, chunks
            except RateLimitError:
                time.sleep(5)
            except Exception as exc:
                logging.warning("Generation error with model %s: %s", m, exc)
                break
    raise RuntimeError("All LLM requests failed.")

# ─── Streamlit App ─────────────────────────────────────────────────────────────
def main():
    # Sidebar: provider & file upload
    st.sidebar.header("LLM Provider")
    options = ["Groq", "Hugging Face", "Google Gemini"]
    default_idx = 0
    if GeminiProvider is not None and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        default_idx = options.index("Google Gemini")
    choice = st.sidebar.selectbox("Model endpoint:", options, index=default_idx)

    if choice.startswith("Hugging Face"):
        llm_provider = HFProvider()
    elif choice.startswith("Google") and GeminiProvider is not None:
        llm_provider = GeminiProvider()
    else:
        llm_provider = GroqProvider()

    st.sidebar.subheader("Upload your own PDFs")
    uploads = st.sidebar.file_uploader(
        "Upload PDFs (optional)", type="pdf", accept_multiple_files=True
    )

    # Ingest docs (uploaded or disk)
    if uploads:
        all_chunks: List[str] = []
        chunk_metadata: List[Dict[str, Any]] = []
        doc_params: Dict[str, Dict[str, int]] = {}
        for up in uploads:
            reader = PdfReader(up)
            page_texts = [(p.extract_text() or "") + "\n" for p in reader.pages]
            full_text = "".join(page_texts)
            chunk_size, chunk_overlap = select_chunk_params(up.name, full_text, load_chunk_stats())
            doc_params[up.name] = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            for page_num, page_text in enumerate(page_texts, start=1):
                for c in create_chunks(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                    if not c.strip():
                        continue
                    all_chunks.append(c)
                    chunk_metadata.append({
                        "doc": up.name,
                        "page": page_num,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    })
        force_idx = True
    else:
        all_chunks, chunk_metadata, doc_params, cur_hash = process_pdfs()
        force_idx = False

    # Build retrievers
    bm25, faiss_idx = init_retrievers(all_chunks, force_rebuild=force_idx)

    # Initialize session_state
    state = st.session_state
    if "messages" not in state:
        state.messages = [
            {"role": "assistant", "content": "👋 Hi there! Ask me anything."}
        ]
    state.setdefault("last_usage", None)
    state.setdefault("feedback", {})
    state.setdefault("memory", [])
    preload_corrections = load_manual_corrections()
    state.setdefault("manual_corrections", preload_corrections)
    state.setdefault("last_context", None)
    state.setdefault("last_citations", [])
    state.setdefault("fact_check", None)
    state.setdefault("active_viewer", None)

    def persist_feedback():
        try:
            with open("feedback_log.json", "w") as f:
                json.dump(state.feedback, f)
            tune_alpha()
        except Exception as exc:
            logging.warning(f"Feedback/tune error: {exc}")

    def handle_feedback(msg_index: int, liked: bool) -> None:
        state.feedback[msg_index] = 1 if liked else -1
        ctx = state.get("last_context")
        if ctx:
            chunks = ctx.get("chunks", [])
            doc_entries: Dict[str, Dict[str, Any]] = {}
            for chunk in chunks:
                doc = chunk.get("doc")
                if not doc or chunk.get("index", -1) == -1:
                    continue
                doc_entries.setdefault(doc, chunk)
            avg_score = average_rerank_score(chunks)
            for doc, meta in doc_entries.items():
                record_chunk_performance(
                    doc,
                    meta.get("chunk_size", 1000) or 1000,
                    meta.get("chunk_overlap", 400) or 400,
                    avg_score,
                    liked
                )
                update_rerank_bias(doc, 0.05 if liked else -0.07)
        persist_feedback()

    # Sidebar controls for manual corrections
    with st.sidebar.expander("Manual Corrections", expanded=False):
        st.caption(
            "Override answer details by adding a correction. "
            "Format suggestions: `Original statement -> Corrected fact`."
        )
        last_question = next(
            (m["content"] for m in reversed(state.messages) if m["role"] == "user"),
            None
        )
        correction_text = st.text_area(
            "Apply correction to the most recent user question",
            value="",
            key="manual_correction_text"
        )
        if st.button("Apply correction", key="manual_correction_btn"):
            if not correction_text.strip():
                st.warning("Enter a correction before applying.")
            elif not last_question:
                st.warning("Ask a question first so we know what to correct.")
            else:
                corrections = dict(state.manual_corrections)
                corrections.setdefault(last_question, []).append(correction_text.strip())
                state.manual_corrections = corrections
                st.success("Saved correction for future answers to that question.")

    # Optional PDF viewer
    if state.active_viewer:
        viewer_info = state.active_viewer
        path = Path(viewer_info.get("path", ""))
        page = viewer_info.get("page") or 1
        if path.exists():
            with st.sidebar.expander(f"Viewer: {path.name}", expanded=True):
                embed_pdf_viewer(path, page)
                if st.button("Close viewer", key="close_viewer_btn"):
                    state.active_viewer = None
        else:
            state.active_viewer = None

    # Render chat history + feedback + citations
    for idx, msg in enumerate(state.messages):
        is_last = idx == len(state.messages) - 1
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(msg["content"])
                if msg.get("citations"):
                    st.markdown("**Citations**")
                    for cite in msg["citations"]:
                        doc_line = cite["doc"]
                        page = cite.get("page")
                        if page:
                            doc_line += f" · page {page}"
                        snippet = cite.get("snippet", "")
                        st.markdown(
                            f"- {cite['label']} {doc_line}<br/>{snippet}",
                            unsafe_allow_html=True
                        )
                        doc_path = Path("input_files") / cite["doc"]
                        if page and doc_path.exists():
                            if st.button(
                                f"Open {cite['label']} ({cite['doc']})",
                                key=f"open_{idx}_{cite['label']}"
                            ):
                                state.active_viewer = {
                                    "path": str(doc_path),
                                    "page": page,
                                    "label": cite["label"],
                                }
                if msg.get("fact"):
                    verdict = msg["fact"].get("verdict", "unknown")
                    explanation = msg["fact"].get("explanation", "")
                    st.caption(f"Fact-check: {verdict} — {explanation}")
            else:
                st.write(msg["content"])

        if (
            msg["role"] == "assistant"
            and is_last
            and idx > 0
            and state.messages[idx-1]["role"] == "user"
        ):
            col_like, col_dislike = st.columns(2)
            last_user_query = state.messages[idx-1]["content"]
            with col_like:
                if st.button("👍", key=f"like_{idx}"):
                    handle_feedback(idx, True)
            with col_dislike:
                if st.button("👎", key=f"dislike_{idx}"):
                    handle_feedback(idx, False)
                    answer, usage, chunk_results, citations, fact = answer_query(
                        query=last_user_query,
                        provider=llm_provider,
                        bm25=bm25,
                        faiss_idx=faiss_idx,
                        all_chunks=all_chunks,
                        chunk_metadata=chunk_metadata,
                        memory=state.memory,
                        corrections=state.manual_corrections,
                        corrections_key=last_user_query,
                    )
                    state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "fact": fact,
                    })
                    state.last_usage = usage
                    state.last_context = {
                        "query": last_user_query,
                        "chunks": chunk_results,
                        "citations": citations,
                        "fact": fact,
                    }
                    state.last_citations = citations
                    state.fact_check = fact
                    state.memory.append({"question": last_user_query, "answer": answer})
                    state.memory = state.memory[-6:]
                    persist_feedback()
                    st.rerun()

    # Chat input
    user_input = st.chat_input("Type your question…")
    if user_input:
        state.messages.append({"role": "user", "content": user_input})

        subqs = self_ask_split(user_input, llm_provider)
        collected_answers: List[str] = []
        combined_chunks: List[Dict[str, Any]] = []
        combined_citations: List[Dict[str, Any]] = []
        fact_results: List[Dict[str, Any]] = []
        total_retrieval = 0.0
        total_inference = 0.0
        model_used = None

        for sq in subqs:
            answer, usage, chunk_results, citations, fact = answer_query(
                query=sq,
                provider=llm_provider,
                bm25=bm25,
                faiss_idx=faiss_idx,
                all_chunks=all_chunks,
                chunk_metadata=chunk_metadata,
                memory=state.memory,
                corrections=state.manual_corrections,
                corrections_key=user_input,
            )
            formatted_answer = answer
            if len(subqs) > 1:
                formatted_answer = f"**Sub-question:** {sq}\n\n{answer}"
            collected_answers.append(formatted_answer)
            combined_chunks.extend(chunk_results)
            combined_citations.extend(citations)
            fact_results.append(fact)
            total_retrieval += usage.get("retrieval_time_s", 0.0)
            total_inference += usage.get("inference_time_s", 0.0)
            if usage.get("model_used"):
                model_used = usage["model_used"]

        if any(fr.get("verdict") == "flagged" for fr in fact_results):
            aggregated_fact = next((fr for fr in fact_results if fr.get("verdict") == "flagged"), fact_results[-1])
        elif fact_results and all(fr.get("verdict") == "confirmed" for fr in fact_results):
            aggregated_fact = {"verdict": "confirmed", "explanation": "All sub-answers confirmed."}
        else:
            aggregated_fact = {"verdict": "uncertain", "explanation": "Some sub-answers could not be confirmed."}

        combined_answer = "\n\n".join(collected_answers)
        state.messages.append({
            "role": "assistant",
            "content": combined_answer,
            "citations": combined_citations,
            "fact": aggregated_fact,
        })

        state.last_usage = {
            "model_used": model_used,
            "retrieval_time_s": round(total_retrieval, 3),
            "inference_time_s": round(total_inference, 3),
            "subquestions": len(subqs),
            "fact_check_verdict": aggregated_fact.get("verdict"),
        }
        state.last_context = {
            "query": user_input,
            "chunks": combined_chunks,
            "citations": combined_citations,
            "fact": aggregated_fact,
        }
        state.last_citations = combined_citations
        state.fact_check = aggregated_fact
        state.memory.append({"question": user_input, "answer": combined_answer})
        state.memory = state.memory[-6:]
        persist_feedback()
        st.rerun()

    # Sidebar metrics
    last = state.get("last_usage")
    if last:
        with st.sidebar.expander("Last Query Metrics"):
            if last.get("retrieval_time_s") is not None:
                st.metric("Retrieval Time", f"{last['retrieval_time_s']} s")
            if last.get("inference_time_s") is not None:
                st.metric("Inference Time", f"{last['inference_time_s']} s")
            st.json(last)

if __name__ == "__main__":
    main()
