import json
import time
from pathlib import Path

from app.ingestion.fetch import fetch_law_html, parse_articles
from app.ingestion.chunk import chunk_articles

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector import VectorRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever

from app.rag.context import assemble_context
from app.rag.validation import validate_citations

from app.llm.openai_provider import OpenAIProvider
from app.llm.groq_provider import GroqProvider
from app.llm.gemini_provider import GeminiProvider

from app.evaluation.models import BenchmarkResult


DATASET_PATH = Path("data/evaluation_questions.json")
OUTPUT_PATH = Path("data/benchmark_results.json")


def load_questions() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_retriever() -> HybridRetriever:
    print("Building retrieval index...")

    html = fetch_law_html()
    articles = parse_articles(html)
    chunks = chunk_articles(articles)

    embedding_service = EmbeddingService()

    vector_retriever = VectorRetriever(
        chunks=chunks,
        embedding_service=embedding_service,
    )

    bm25_retriever = BM25Retriever(chunks)

    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )


def run_provider(
    question_data: dict,
    provider,
    retrieved_chunks,
    retrieved_articles: list[int],
    context: str,
) -> BenchmarkResult:
    """Run one LLM against already-fixed retrieval context."""



    try:
        result = None
        last_error = None

        for attempt in range(3):
            try:
                result = provider.generate(
                    question=question_data["question"],
                    context=context,
                )
                break

            except Exception as exc:
                last_error = exc

                if attempt < 2:
                    wait_seconds = 2 ** attempt
                    print(
                        f"    Attempt {attempt + 1} failed: "
                        f"{type(exc).__name__}. "
                        f"Retrying in {wait_seconds}s..."
                    )
                    time.sleep(wait_seconds)

        if result is None:
            raise last_error

        citation_validation = validate_citations(
            result.answer,
            retrieved_chunks,
        )

        return BenchmarkResult(
            question_id=question_data["id"],
            language=question_data["language"],
            category=question_data["category"],
            provider=result.provider,
            model=result.model,

            question=question_data["question"],
            expected_answer=question_data["expected_answer"],
            expected_articles=question_data["expected_articles"],
            expected_answerable=question_data["answerable"],

            retrieved_articles=retrieved_articles,

            generated_answer=result.answer.answer,
            generated_citations=result.answer.citations,
            insufficient_context=result.answer.insufficient_context,

            citations_valid_in_context=citation_validation.is_valid,

            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            ttft_ms=result.ttft_ms,
            total_latency_ms=result.total_latency_ms,

            success=True,
        )

    except Exception as exc:
        return BenchmarkResult(
            question_id=question_data["id"],
            language=question_data["language"],
            category=question_data["category"],
            provider=provider.__class__.__name__.replace("Provider", "").lower(),
            model=getattr(provider, "model", "unknown"),

            question=question_data["question"],
            expected_answer=question_data["expected_answer"],
            expected_articles=question_data["expected_articles"],
            expected_answerable=question_data["answerable"],

            retrieved_articles=retrieved_articles,

            generated_answer=None,
            generated_citations=None,
            insufficient_context=None,

            citations_valid_in_context=None,

            prompt_tokens=None,
            completion_tokens=None,
            ttft_ms=None,
            total_latency_ms=None,

            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def main() -> None:
    questions = load_questions()
    retriever = build_retriever()

    providers = [
        OpenAIProvider(),
        GroqProvider(),
        GeminiProvider(),
    ]

    results = []

    # SMOKE TEST ONLY
    for question_data in questions:
        print(
            f"\n[{question_data['id']}] "
            f"{question_data['question']}"
        )

        # -------------------------------
        # RETRIEVE EXACTLY ONCE
        # -------------------------------
        retrieved = retriever.search(
            question_data["question"],
            top_k=5,
            candidate_k=10,
        )

        retrieved_chunks = [
            chunk for chunk, _score in retrieved
        ]

        retrieved_articles = list(
            dict.fromkeys(
                chunk.article_number
                for chunk in retrieved_chunks
            )
        )

        context = assemble_context(retrieved_chunks)

        print(
            f"Retrieved articles: {retrieved_articles}"
        )

        # -------------------------------
        # SAME CONTEXT FOR EVERY MODEL
        # -------------------------------
        for provider in providers:
            print(
                f"  Running {provider.__class__.__name__} "
                f"({provider.model})..."
            )

            result = run_provider(
                question_data=question_data,
                provider=provider,
                retrieved_chunks=retrieved_chunks,
                retrieved_articles=retrieved_articles,
                context=context,
            )

            results.append(result.to_dict())

            if result.success:
                print(
                    f"    OK"
                    f" | citations={result.generated_citations}"
                    f" | ttft={result.ttft_ms}"
                    f" | total={result.total_latency_ms:.2f}ms"
                )
            else:
                print(
                    f"    FAILED"
                    f" | {result.error_type}: "
                    f"{result.error_message}"
                )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()