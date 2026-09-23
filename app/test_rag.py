from app.ingestion.fetch import fetch_law_html, parse_articles
from app.ingestion.chunk import chunk_articles
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector import VectorRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.rag.context import assemble_context
#from app.llm.openai_provider import OpenAIProvider
#from app.llm.gemini_provider import GeminiProvider
from app.llm.groq_provider import GroqProvider
from app.rag.validation import validate_citations


def main() -> None:
    html = fetch_law_html()
    articles = parse_articles(html)
    chunks = chunk_articles(articles)

    embedding_service = EmbeddingService()

    vector_retriever = VectorRetriever(
        chunks=chunks,
        embedding_service=embedding_service,
    )

    bm25_retriever = BM25Retriever(chunks)

    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    question = "What is an electronic communications network?"

    retrieval_results = retriever.search(
        question,
        top_k=5,
        candidate_k=10,
    )

    retrieved_chunks = [
        chunk for chunk, _ in retrieval_results
    ]

    print("\nRETRIEVED ARTICLES")
    for chunk in retrieved_chunks:
        print(
            f"Article {chunk.article_number}, "
            f"chunk {chunk.chunk_index}"
        )

    context = assemble_context(retrieved_chunks)

    #provider = OpenAIProvider()
    #provider = GeminiProvider()
    provider = GroqProvider()


    result = provider.generate(
        question=question,
        context=context,
    )


    citation_validation = validate_citations(
    answer=result.answer,
    retrieved_chunks=retrieved_chunks,
)

    print("\nANSWER")
    print(result.answer.answer)

    print("\nCITATIONS")
    print(result.answer.citations)

    print("\nINSUFFICIENT CONTEXT")
    print(result.answer.insufficient_context)

    print("\nMETRICS")
    print("Provider:", result.provider)
    print("Model:", result.model)
    print("Prompt tokens:", result.prompt_tokens)
    print("Completion tokens:", result.completion_tokens)
    print("Total latency ms:", round(result.total_latency_ms, 2))
    print(f"TTFT ms: {result.ttft_ms:.2f}")

    print("\nCITATION VALIDATION")
    print("Valid:", citation_validation.valid_citations)
    print("Invalid:", citation_validation.invalid_citations)
    print("All citations valid:", citation_validation.is_valid)


if __name__ == "__main__":
    main()