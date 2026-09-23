from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.models.rag import RAGAnswer

@dataclass
class LLMResult:
    answer: RAGAnswer
    model: str
    provider: str
    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_ms: float | None
    total_latency_ms: float


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
    ) -> LLMResult:
        pass