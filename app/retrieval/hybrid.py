from collections import defaultdict

from app.models.document import LegalChunk
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector import VectorRetriever


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[tuple[LegalChunk, float]]:
        vector_results = self.vector_retriever.search(
            query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.search(
            query,
            top_k=candidate_k,
        )

        scores: dict[str, float] = defaultdict(float)
        chunks_by_id: dict[str, LegalChunk] = {}

        for results in (vector_results, bm25_results):
            for rank, (chunk, _) in enumerate(results, start=1):
                scores[chunk.id] += 1 / (self.rrf_k + rank)
                chunks_by_id[chunk.id] = chunk

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )[:top_k]

        return [
            (chunks_by_id[chunk_id], scores[chunk_id])
            for chunk_id in ranked_ids
        ]