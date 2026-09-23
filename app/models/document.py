from dataclasses import dataclass


@dataclass
class LegalChunk:
    id: str
    article_number: int
    article_title: str
    chunk_index: int
    text: str
    source_url: str

    @property
    def retrieval_text(self) -> str:
        return (
            f"Հոդված {self.article_number}. "
            f"{self.article_title}\n"
            f"{self.text}"
        )