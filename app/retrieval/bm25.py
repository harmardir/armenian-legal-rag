import re

from rank_bm25 import BM25Okapi

from app.models.document import LegalChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, chunks: list[LegalChunk]) -> None:
        self.chunks = chunks
        tokenized_corpus = [
    tokenize(chunk.retrieval_text)
    for chunk in chunks
]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[LegalChunk, float]]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
    (
        index
        for index, score in enumerate(scores)
        if score > 0
    ),
    key=lambda index: scores[index],
    reverse=True,
)[:top_k]

        return [
            (self.chunks[index], float(scores[index]))
            for index in ranked_indices
        ]