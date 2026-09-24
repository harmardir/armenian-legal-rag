import re


SYSTEM_PROMPT = """
You are an internal legal and regulatory assistant for an Armenian
telecommunications company.

Answer questions using ONLY the legal context provided to you.

Rules:
1. Do not use outside knowledge.
2. Do not invent legal provisions.
3. Cite only article numbers that appear in the provided context.
4. If the context does not contain enough information to answer the
   question, say so and set insufficient_context to true.
5. Follow the explicitly provided RESPONSE LANGUAGE.
6. Keep the answer concise and factual.
"""


ARMENIAN_PATTERN = re.compile(r"[\u0530-\u058F]")


def detect_question_language(question: str) -> str:
    """
    Detect whether the question contains Armenian characters.

    For this application we support Armenian and English questions.
    """

    if ARMENIAN_PATTERN.search(question):
        return "Armenian"

    return "English"


def build_user_prompt(
    question: str,
    context: str,
) -> str:
    language = detect_question_language(question)

    return f"""
RESPONSE LANGUAGE: {language}

You MUST write the answer in {language}.

LEGAL CONTEXT:

{context}

USER QUESTION:

{question}
"""