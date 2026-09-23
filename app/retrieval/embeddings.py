from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"


class EmbeddingService:
    def __init__(self) -> None:
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_documents(self, texts: list[str]):
        prepared = [f"passage: {text}" for text in texts]

        return self.model.encode(
            prepared,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str):
        return self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )