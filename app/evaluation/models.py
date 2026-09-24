from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class BenchmarkResult:
    question_id: str
    language: str
    category: str
    provider: str
    model: str

    question: str
    expected_answer: str
    expected_articles: list[int]
    expected_answerable: bool

    retrieved_articles: list[int]

    generated_answer: str | None
    generated_citations: list[int] | None
    insufficient_context: bool | None

    citations_valid_in_context: bool | None

    prompt_tokens: int | None
    completion_tokens: int | None
    ttft_ms: float | None
    total_latency_ms: float | None

    success: bool
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)