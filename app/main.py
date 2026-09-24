import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingestion.fetch import fetch_law_html, parse_articles
from app.ingestion.chunk import chunk_articles

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector import VectorRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever

from app.rag.context import assemble_context
from app.rag.validation import validate_citations

from app.llm.openai_provider import OpenAIProvider


STATIC_DIR = Path("static")
BENCHMARK_SUMMARY_PATH = Path("data/benchmark_summary.json")
BENCHMARK_RESULTS_PATH = Path("data/benchmark_scored.json")


# --------------------------------------------------
# API models
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    citations: list[int]
    insufficient_context: bool
    retrieved_articles: list[int]
    model: str
    latency_ms: float


# --------------------------------------------------
# Application state
# --------------------------------------------------

retriever: HybridRetriever | None = None
llm_provider: OpenAIProvider | None = None


# --------------------------------------------------
# Startup
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    global llm_provider

    print("Initializing legal RAG system...")

    # Fetch and parse the law.
    html = fetch_law_html()
    articles = parse_articles(html)

    # Convert articles into retrieval chunks.
    chunks = chunk_articles(articles)

    print(
        f"Loaded {len(articles)} articles "
        f"and {len(chunks)} chunks."
    )

    # Load multilingual embedding model once.
    embedding_service = EmbeddingService()

    # Build vector index once.
    vector_retriever = VectorRetriever(
        chunks=chunks,
        embedding_service=embedding_service,
    )

    # Build BM25 index once.
    bm25_retriever = BM25Retriever(chunks)

    # Hybrid retrieval using RRF.
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    # Default model for interactive Q&A.
    llm_provider = OpenAIProvider()

    print("Legal RAG system ready.")

    yield

    print("Application shutting down.")


app = FastAPI(
    title="Armenian Legal RAG Assistant",
    description=(
        "Grounded Q&A over the Armenian Law "
        "on Electronic Communications"
    ),
    lifespan=lifespan,
)


# --------------------------------------------------
# Static frontend
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


@app.get("/")
def index():
    return FileResponse(
        STATIC_DIR / "index.html"
    )


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "rag_ready": retriever is not None,
    }


# --------------------------------------------------
# Q&A
# --------------------------------------------------

@app.post(
    "/api/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):
    if retriever is None or llm_provider is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # ----------------------------------------------
    # 1. Retrieve legal evidence
    # ----------------------------------------------

    retrieved = retriever.search(
        question,
        top_k=5,
        candidate_k=10,
    )

    retrieved_chunks = [
        chunk
        for chunk, _score in retrieved
    ]

    retrieved_articles = list(
        dict.fromkeys(
            chunk.article_number
            for chunk in retrieved_chunks
        )
    )

    # ----------------------------------------------
    # 2. Assemble controlled context
    # ----------------------------------------------

    context = assemble_context(
        retrieved_chunks
    )

    # ----------------------------------------------
    # 3. Generate grounded answer
    # ----------------------------------------------

    try:
        result = llm_provider.generate(
            question=question,
            context=context,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {exc}",
        ) from exc

    # ----------------------------------------------
    # 4. Deterministic citation validation
    # ----------------------------------------------

    validation = validate_citations(
        result.answer,
        retrieved_chunks,
    )

    if not validation.is_valid:
        raise HTTPException(
            status_code=502,
            detail=(
                "Model generated citations that were "
                "not present in retrieved context."
            ),
        )

    # ----------------------------------------------
    # 5. Return answer
    # ----------------------------------------------

    return AskResponse(
        answer=result.answer.answer,
        citations=validation.valid_citations,
        insufficient_context=(
            result.answer.insufficient_context
        ),
        retrieved_articles=retrieved_articles,
        model=result.model,
        latency_ms=result.total_latency_ms,
    )


# --------------------------------------------------
# Benchmark summary
# --------------------------------------------------

@app.get("/api/benchmark")
def benchmark_summary():
    if not BENCHMARK_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Benchmark summary has not "
                "been generated."
            ),
        )

    with BENCHMARK_SUMMARY_PATH.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)


# --------------------------------------------------
# Raw/scored benchmark results
# --------------------------------------------------

@app.get("/api/benchmark/raw")
def benchmark_raw():
    if not BENCHMARK_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Benchmark results have not "
                "been generated."
            ),
        )

    with BENCHMARK_RESULTS_PATH.open(
        encoding="utf-8"
    ) as file:
        return json.load(file)