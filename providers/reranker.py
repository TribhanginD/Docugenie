import logging
import json
from typing import List, Dict, Any, Optional
from providers.llm import LLMProvider

logger = logging.getLogger("docugenie.providers.reranker")

class RerankerProvider:
    def __init__(self, llm_provider: LLMProvider, model_name: Optional[str] = None):
        self.provider = llm_provider
        self.model_name = model_name or "gemini-2.0-flash"
        logger.info(f"Initialized API-based Reranker with model: {self.model_name}")

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        # Prepare passages for listwise reranking
        passages = "\n".join(
            f"[{i}] {c.get('text', '')[:500]}" for i, c in enumerate(chunks)
        )
        
        prompt = (
            f"Query: {query}\n\n"
            f"Passages:\n{passages}\n\n"
            "Return a JSON array of passage indices sorted from most to least relevant "
            f"to the query. Only include the top {min(top_k, len(chunks))} indices. "
            "Example: [2, 0, 4]. Output only the JSON array."
        )

        try:
            response_text, _ = self.provider.generate(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0,
                max_tokens=256
            )
            
            # Extract JSON from response
            raw = response_text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            
            indices = json.loads(raw)
            if isinstance(indices, list):
                reranked = []
                for rank, idx in enumerate(indices):
                    if isinstance(idx, int) and 0 <= idx < len(chunks):
                        chunk = chunks[idx].copy()
                        chunk["rerank_score"] = 1.0 - (rank / len(indices))
                        chunk["score"] = chunk["rerank_score"]
                        reranked.append(chunk)
                return reranked[:top_k]
        except Exception as e:
            logger.error(f"API reranking failed: {e}. Falling back to original order.")
        
        return chunks[:top_k]

