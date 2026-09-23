from app.models.document import LegalChunk


def assemble_context(chunks: list[LegalChunk]) -> str:
    sections = []

    for chunk in chunks:
        section = (
            f"[Article {chunk.article_number}]\n"
            f"Title: {chunk.article_title}\n"
            f"{chunk.text}"
        )
        sections.append(section)

    return "\n\n---\n\n".join(sections)