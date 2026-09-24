import time

from dotenv import load_dotenv
from openai import OpenAI

from app.llm.base import LLMProvider, LLMResult
from app.models.rag import RAGAnswer
from app.rag.prompt import SYSTEM_PROMPT, build_user_prompt


load_dotenv()


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
    ) -> None:
        self.client = OpenAI()
        self.model = model

    def generate(
        self,
        question: str,
        context: str,
    ) -> LLMResult:
        start = time.perf_counter()

        response = self.client.responses.parse(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=build_user_prompt(question, context),
            text_format=RAGAnswer,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        answer = response.output_parsed

        return LLMResult(
            answer=answer,
            model=self.model,
            provider="openai",
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            ttft_ms=None,
            total_latency_ms=latency_ms,
        )