
import os
import json
import logging
import hashlib
import time
import pickle
from typing import List, Tuple, Dict

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

# ─── Load config & logging ─────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}

# ─── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuGenie",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Retrieval config & auto‐tuning ────────────────────────────────────────────
CONFIG_FILE = "retrieval_config.json"

def load_config() -> Dict[str, float]:
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        default = {"alpha": 0.6}
        with open(CONFIG_FILE, "w") as f:
            json.dump(default, f)
        return default

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

# ─── PDF → Text Chunks Pipeline ────────────────────────────────────────────────
def extract_text_from_pdf(path: str) -> str:
    text = ''
    with open(path, 'rb') as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def create_chunks(text: str, chunk_size=1000, chunk_overlap=200) -> List[str]:
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
def process_pdfs(prev_hash: str = None) -> Tuple[List[str], Dict[str, str], str]:
    pdf_dir = './input_files/'
    cur_hash = get_files_hash(pdf_dir)
    if prev_hash is not None and cur_hash != prev_hash:
        st.cache_data.clear()
    all_chunks, chunk_to_doc = [], {}
    for fn in sorted(os.listdir(pdf_dir)):
        if fn.lower().endswith('.pdf'):
            path = os.path.join(pdf_dir, fn)
            text = extract_text_from_pdf(path)
            for c in create_chunks(text):
                all_chunks.append(c)
                chunk_to_doc[c] = fn
    logging.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks, chunk_to_doc, cur_hash

# ─── FAISS index & retriever init ───────────────────────────────────────────────
@st.cache_resource
def create_faiss_index(all_chunks: List[str], force_rebuild: bool = False):
    idx_file = 'faiss_index.pkl'
    if os.path.exists(idx_file) and not force_rebuild:
        with open(idx_file, 'rb') as f:
            return pickle.load(f)
    embs = model.encode(all_chunks).astype('float32')
    d, n = embs.shape[1], embs.shape[0]
    if n < 100:
        idx = faiss.IndexFlatL2(d)
    else:
        quant = faiss.IndexFlatL2(d)
        n_clusters = min(int(np.sqrt(n)), 100)
        idx = faiss.IndexIVFFlat(quant, d, n_clusters)
        idx.train(embs)
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
            return cache['responses'][i]
    return None

def update_cache(q: str, v: np.ndarray, resps: List[str]):
    cache['queries'].append(q)
    cache['embeddings'].append(v.tolist())
    cache['responses'].append(resps)
    cache['model_name'] = MODEL_NAME
    save_cache(cache)

# ─── Query Reformulation (Self-Ask) ────────────────────────────────────────────
def self_ask_split(query: str, provider: LLMProvider) -> List[str]:
    prompt = f"""
Break the following user question into independent sub-questions.
Return a JSON array of strings.

Question: "{query}"
"""
    text, _ = provider.generate(
        messages=[{"role":"user","content":prompt}],
        model="gemma2-9b-it",
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
    top_k: int = 20,
    alpha: float = None   # load from config if None
) -> List[str]:
    if alpha is None:
        alpha = load_config().get("alpha", 0.6)

    v = model.encode([query])[0].astype('float32')
    cached = retrieve_from_cache(v)
    if cached:
        return cached

    # FAISS shortlist
    K = min(top_k, len(all_chunks))
    D, I = idx.search(np.array([v]), K)
    faiss_rank = [(all_chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]

    # BM25 shortlist
    bm25_scores = BM25Okapi([c.split() for c in all_chunks]).get_scores(query.split())
    bm25_rank = sorted(
        [(all_chunks[i], bm25_scores[i]) for i in range(len(all_chunks))],
        key=lambda x: x[1],
        reverse=True
    )[:K]

    # Normalize & hybrid score
    fs = np.array([s for _, s in faiss_rank])
    bs = np.array([s for _, s in bm25_rank])
    fs = (fs - fs.min())/(np.ptp(fs)+1e-8) if np.ptp(fs)>0 else fs
    bs = (bs - bs.min())/(np.ptp(bs)+1e-8) if np.ptp(bs)>0 else bs

    hybrid = {}
    for i,(c,_) in enumerate(faiss_rank):
        hybrid[c] = alpha * fs[i]
    for i,(c,_) in enumerate(bm25_rank):
        hybrid[c] = hybrid.get(c, 0.0) + (1-alpha) * bs[i]

    shortlist = [c for c,_ in sorted(hybrid.items(), key=lambda x: x[1], reverse=True)][:top_k]

    # Cross-encoder re-ranking → final top 5
    pairs = [(query, passage) for passage in shortlist]
    rerank_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(shortlist, rerank_scores), key=lambda x: x[1], reverse=True)
    final = [c for c,_ in reranked[:5]]

    update_cache(query, v, final)
    return final

# ─── Generation ────────────────────────────────────────────────────────────────
def generate_response(
    query: str,
    chunks: List[str],
    provider: LLMProvider,
    primary_model: str = "llama-3.1-8b-instant",
    fallback_model: str = "gemma2-9b-it",
    max_retries: int = 3
) -> Tuple[str, Dict, List[str]]:
    ctx = "\n".join(chunks)
    prompt = (
        "Answer **in Markdown** based on the context below.  "
        "**Do not** restate the question—only provide the answer.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )
    for m in (primary_model, fallback_model):
        for _ in range(max_retries):
            try:
                msgs = [
                    {"role":"system","content":"You are a helpful assistant."},
                    {"role":"user",  "content":prompt},
                ]
                out, usage = provider.generate(
                    messages=msgs,
                    model=m,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9
                )
                return out, usage, chunks
            except RateLimitError:
                time.sleep(5)
            except Exception:
                break
    raise RuntimeError("All LLM requests failed.")

# ─── Streamlit App ─────────────────────────────────────────────────────────────
def main():
    # Sidebar: provider & file upload
    st.sidebar.header("LLM Provider")
    choice = st.sidebar.selectbox("Model endpoint:", ["Groq", "Hugging Face"])
    llm_provider = HFProvider() if choice.startswith("Hugging Face") else GroqProvider()

    st.sidebar.subheader("Upload your own PDFs")
    uploads = st.sidebar.file_uploader(
        "Upload PDFs (optional)", type="pdf", accept_multiple_files=True
    )

    # Ingest docs (uploaded or disk)
    if uploads:
        all_chunks, chunk_to_doc = [], {}
        for up in uploads:
            reader = PdfReader(up)
            text = "".join(p.extract_text()+"\n" for p in reader.pages)
            for c in create_chunks(text):
                all_chunks.append(c)
                chunk_to_doc[c] = up.name
        force_idx = True
    else:
        all_chunks, chunk_to_doc, cur_hash = process_pdfs()
        force_idx = False

    # Build retrievers
    bm25, faiss_idx = init_retrievers(all_chunks, force_rebuild=force_idx)

    # Initialize session_state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi there! Ask me anything."}
        ]
    if "last_usage" not in st.session_state:
        st.session_state.last_usage = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    # Render chat history + feedback
    for idx, msg in enumerate(st.session_state.messages):
        is_last = idx == len(st.session_state.messages) - 1
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(msg["content"])
            else:
                st.write(msg["content"])

        # feedback only on last assistant answer after a user
        if (
            msg["role"] == "assistant"
            and is_last
            and idx > 0
            and st.session_state.messages[idx-1]["role"] == "user"
        ):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"like_{idx}"):
                    st.session_state.feedback[idx] = 1
            with col2:
                if st.button("👎", key=f"dislike_{idx}"):
                    st.session_state.feedback[idx] = -1

                    # regenerate & re-time
                    last_user = st.session_state.messages[idx-1]["content"]
                    t0 = time.time()
                    chunks = retrieve_relevant_chunks(last_user, bm25, faiss_idx, all_chunks)
                    rec_t = time.time() - t0

                    t1 = time.time()
                    answer, usage, _ = generate_response(last_user, chunks, llm_provider)
                    inf_t = time.time() - t1

                    usage["retrieval_time_s"] = round(rec_t, 3)
                    usage["inference_time_s"] = round(inf_t, 3)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    st.session_state.last_usage = usage

                    # persist feedback & tune α
                    try:
                        with open("feedback_log.json", "w") as f:
                            json.dump(st.session_state.feedback, f)
                        tune_alpha()
                    except Exception as e:
                        logging.warning(f"Feedback/tune error: {e}")

                    st.rerun()

    # Chat input
    user_input = st.chat_input("Type your question…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # self-ask split
        subqs = self_ask_split(user_input, llm_provider)

        # answer each sub-question
        all_ans = []
        total_rec, total_inf = 0.0, 0.0
        for sq in subqs:
            t0 = time.time()
            chunks = retrieve_relevant_chunks(sq, bm25, faiss_idx, all_chunks)
            rec_t = time.time() - t0

            t1 = time.time()
            ans, usage, _ = generate_response(sq, chunks, llm_provider)
            inf_t = time.time() - t1

            total_rec += rec_t
            total_inf += inf_t
            all_ans.append(ans)

        combined = "\n\n".join(all_ans)
        st.session_state.last_usage = {
            "model_used":       usage.get("model_used"),
            "retrieval_time_s": round(total_rec, 3),
            "inference_time_s": round(total_inf, 3),
        }
        st.session_state.messages.append({"role": "assistant", "content": combined})

        # persist feedback & tune α
        try:
            with open("feedback_log.json", "w") as f:
                json.dump(st.session_state.feedback, f)
            tune_alpha()
        except Exception as e:
            logging.warning(f"Feedback/tune error: {e}")

        st.rerun()

    # Sidebar metrics
    last = st.session_state.get("last_usage")
    if last:
        with st.sidebar.expander("Last Query Metrics"):
            if last.get("retrieval_time_s") is not None:
                st.metric("Retrieval Time", f"{last['retrieval_time_s']} s")
            if last.get("inference_time_s") is not None:
                st.metric("Inference Time",  f"{last['inference_time_s']} s")
            st.json(last)

if __name__ == "__main__":
    main()
