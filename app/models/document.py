from dataclasses import dataclass


@dataclass
class LegalChunk:
    id: str
    article_number: int
    article_title: str
    chunk_index: int
    text: str
    source_url: str