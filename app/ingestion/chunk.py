from app.models.document import LegalChunk


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def chunk_articles(articles: list[dict]) -> list[LegalChunk]:
    chunks = []

    for article in articles:
        article_chunks = split_text(
            article["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk_index, text in enumerate(article_chunks):
            chunks.append(
                LegalChunk(
                    id=f"article-{article['article_number']}-chunk-{chunk_index}",
                    article_number=article["article_number"],
                    article_title=article["title"],
                    chunk_index=chunk_index,
                    text=text,
                    source_url=article["source_url"],
                )
            )

    return chunks


def split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end

        # Prefer a natural paragraph boundary.
        if target_end < len(text):
            paragraph_break = text.rfind("\n", start, target_end)

            if paragraph_break > start:
                end = paragraph_break

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Preserve some previous context in the next chunk.
        next_start = max(0, end - overlap)

        # Protection against a pathological boundary causing no progress.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks