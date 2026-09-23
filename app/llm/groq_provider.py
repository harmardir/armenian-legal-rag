import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

from app.llm.base import LLMProvider, LLMResult
from app.models.rag import RAGAnswer
from app.rag.prompt import SYSTEM_PROMPT, build_user_prompt

load_dotenv()


class GroqProvider(LLMProvider):
    def __init__(self, model: str = "openai/gpt-oss-120b") -> None:
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def generate(self, question: str, context: str) -> LLMResult:
        start = time.perf_counter()

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                        + "\n\nReturn ONLY a JSON object with exactly this structure:\n"
                        '{'
                        '"answer": "your answer", '
                        '"citations": [2, 15], '
                        '"insufficient_context": false'
                        '}\n'
                        "The citations field MUST be an array of integer article "
                        'numbers only. Do not write strings such as "Article 2".'
                    ),
                },
                {
                    "role": "user",
                    "content": build_user_prompt(question, context),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
            stream=True,
        )

        ttft_ms = None
        content_parts = []

        for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000

                content_parts.append(delta)

        total_latency_ms = (time.perf_counter() - start) * 1000

        content = "".join(content_parts)
        answer = RAGAnswer.model_validate(json.loads(content))

        return LLMResult(
            answer=answer,
            model=self.model,
            provider="groq",
            prompt_tokens=None,
            completion_tokens=None,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
        )