from pydantic import BaseModel


class RAGAnswer(BaseModel):
    answer: str
    citations: list[int]
    insufficient_context: bool