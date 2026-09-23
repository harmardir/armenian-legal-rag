from app.ingestion.chunk import chunk_articles
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

    print(f"Articles: {len(articles)}")
    print(f"Chunks: {len(chunks)}")

    article_2_chunks = [
        chunk
        for chunk in chunks
        if chunk.article_number == 2
    ]

    for chunk in article_2_chunks:
        print(
            chunk.id,
            len(chunk.text),
            repr(chunk.text[:100]),
        )