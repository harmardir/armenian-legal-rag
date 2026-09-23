from app.ingestion.chunk import chunk_articles
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector import VectorRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever

import re
import requests
from bs4 import BeautifulSoup


ARLIS_URL = "https://www.arlis.am/hy/acts/1869"

ARTICLE_PATTERN = re.compile(
    r"Հոդված\s+(\d+)\.\s*([^\n]*)\n"
)


def fetch_law_html() -> str:
    response = requests.get(ARLIS_URL, timeout=30)
    response.raise_for_status()
    return response.text


def parse_articles(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")

    # Convert HTML into normalized document text while preserving
    # element boundaries as newlines.
    text = soup.get_text("\n", strip=True)

    matches = list(ARTICLE_PATTERN.finditer(text))

    articles = []

    for index, match in enumerate(matches):
        article_number = int(match.group(1))
        article_title = match.group(2).strip()

        body_start = match.end()

        if index + 1 < len(matches):
            body_end = matches[index + 1].start()
        else:
            body_end = len(text)

        body = text[body_start:body_end].strip()

        articles.append(
            {
                "article_number": article_number,
                "title": article_title,
                "text": body,
                "source_url": ARLIS_URL,
            }
        )

    return articles

if __name__ == "__main__":
    html = fetch_law_html()
    articles = parse_articles(html)
    chunks = chunk_articles(articles)

    embedding_service = EmbeddingService()

    vector_retriever = VectorRetriever(
        chunks=chunks,
        embedding_service=embedding_service,
    )

    bm25_retriever = BM25Retriever(chunks)

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    queries = [
        "What is an electronic communications network?",
        "Ի՞նչ է էլեկտրոնային հաղորդակցության ցանցը։",
    ]

    for query in queries:
        print("\n" + "#" * 100)
        print("QUERY:", query)

        methods = {
            "VECTOR": vector_retriever,
            "BM25": bm25_retriever,
            "HYBRID": hybrid_retriever,
        }

        for name, retriever in methods.items():
            print(f"\n--- {name} ---")

            results = retriever.search(query, top_k=5)

            for rank, (chunk, score) in enumerate(results, start=1):
                print(
                    f"{rank}. "
                    f"Article {chunk.article_number} | "
                    f"chunk {chunk.chunk_index} | "
                    f"score={score:.4f} | "
                    f"{chunk.article_title}"
                )