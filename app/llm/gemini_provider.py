import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.llm.base import LLMProvider, LLMResult
from app.models.rag import RAGAnswer
from app.rag.prompt import SYSTEM_PROMPT, build_user_prompt


load_dotenv()


class GeminiProvider(LLMProvider):
    def __init__(
        self,
         model: str = "gemini-3.6-flash",
    ) -> None:
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )
        self.model = model

    def generate(
        self,
        question: str,
        context: str,
    ) -> LLMResult:
        start = time.perf_counter()

        response = self.client.models.generate_content(
            model=self.model,
            contents=build_user_prompt(question, context),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RAGAnswer,
                temperature=0,
            ),
        )

        latency_ms = (time.perf_counter() - start) * 1000

        answer = RAGAnswer.model_validate_json(response.text)

        usage = response.usage_metadata

        return LLMResult(
            answer=answer,
            model=self.model,
            provider="gemini",
            prompt_tokens=usage.prompt_token_count,
            completion_tokens=usage.candidates_token_count,
            ttft_ms=None,
            total_latency_ms=latency_ms,
        )