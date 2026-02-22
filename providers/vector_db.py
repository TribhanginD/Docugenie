import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger("docugenie.providers.vector_db")

class QdrantProvider:
    def __init__(self, location: str = ":memory:", collection_name: str = "docugenie", vector_size: int = 3072):
        if location == ":memory:":
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(path=location)
        self.collection_name = collection_name
        self._ensure_collection(vector_size=vector_size)

    def _ensure_collection(self, vector_size: int = 3072):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name} (size={vector_size})")

    def upsert_chunks(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]):
        points = [
            PointStruct(
                id=i,
                vector=vector,
                payload=meta
            ) for i, (vector, meta) in enumerate(zip(vectors, metadata))
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} points to Qdrant.")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        ).points
        return [
            {
                "text": hit.payload.get("text", ""),
                "doc": hit.payload.get("doc", "unknown"),
                "page": hit.payload.get("page"),
                "score": hit.score,
                "metadata": hit.payload
            } for hit in search_result
        ]

    def hybrid_search(self, query_vector: List[float], query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Qdrant supports hybrid search via Prefetch and Rerank if setup with sparse vectors
        # For now, we'll implement a simplified version or use dense search if sparse not enabled
        # In a real production system, we'd configure sparse vectors for BM25-like search
        return self.search(query_vector, top_k)
