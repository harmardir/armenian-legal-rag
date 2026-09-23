from dataclasses import dataclass

from app.models.document import LegalChunk
from app.models.rag import RAGAnswer


@dataclass
class CitationValidation:
    valid_citations: list[int]
    invalid_citations: list[int]

    @property
    def is_valid(self) -> bool:
        return len(self.invalid_citations) == 0


def validate_citations(
    answer: RAGAnswer,
    retrieved_chunks: list[LegalChunk],
) -> CitationValidation:
    retrieved_articles = {
        chunk.article_number
        for chunk in retrieved_chunks
    }

    valid = []
    invalid = []

    for citation in answer.citations:
        if citation in retrieved_articles:
            valid.append(citation)
        else:
            invalid.append(citation)

    return CitationValidation(
        valid_citations=valid,
        invalid_citations=invalid,
    )