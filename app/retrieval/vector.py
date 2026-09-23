import numpy as np

from app.models.document import LegalChunk
from app.retrieval.embeddings import EmbeddingService


class VectorRetriever:
    def __init__(
        self,
        chunks: list[LegalChunk],
        embedding_service: EmbeddingService,
    ) -> None:
        self.chunks = chunks
        self.embedding_service = embedding_service

        texts = [chunk.text for chunk in chunks]

        self.embeddings = embedding_service.embed_documents(texts)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[LegalChunk, float]]:
        query_embedding = self.embedding_service.embed_query(query)

        scores = np.dot(self.embeddings, query_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.chunks[index], float(scores[index]))
            for index in top_indices
        ]