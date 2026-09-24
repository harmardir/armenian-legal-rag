import json
from pathlib import Path

import numpy as np

from app.retrieval.embeddings import EmbeddingService


INPUT_PATH = Path("data/benchmark_results.json")
OUTPUT_PATH = Path("data/benchmark_scored.json")

# Heuristic threshold for semantic answer similarity.
ANSWER_SIMILARITY_THRESHOLD = 0.75


def citation_metrics(
    expected: list[int],
    generated: list[int] | None,
) -> tuple[float, float, float]:
    """
    Calculate citation precision, recall and F1.

    Example:
        expected  = [45, 55]
        generated = [45]

        precision = 1.0
        recall    = 0.5
        f1        = 0.667
    """
    expected_set = set(expected)
    generated_set = set(generated or [])

    # Correct abstention:
    # no citations expected and no citations generated.
    if not expected_set and not generated_set:
        return 1.0, 1.0, 1.0

    # Expected citations exist but model returned none.
    if not generated_set:
        return 0.0, 0.0, 0.0

    correct = expected_set & generated_set

    precision = len(correct) / len(generated_set)

    recall = (
        len(correct) / len(expected_set)
        if expected_set
        else 0.0
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return precision, recall, f1


def answer_similarity(
    expected: str,
    generated: str,
    embedding_service: EmbeddingService,
) -> float:
    """
    Compare expected and generated answers semantically.

    The embedding service returns normalized vectors,
    therefore dot product acts as cosine similarity.
    """
    embeddings = embedding_service.embed_documents(
        [expected, generated]
    )

    return float(
        np.dot(
            embeddings[0],
            embeddings[1],
        )
    )


def score_record(
    record: dict,
    embedding_service: EmbeddingService,
) -> dict:
    """
    Add evaluation metrics to one benchmark result.
    """
    scored = dict(record)

    # --------------------------------------------------
    # 1. Provider/API failure
    # --------------------------------------------------

    if not record["success"]:
        scored["answer_similarity"] = 0.0
        scored["answer_correct"] = False

        scored["citation_precision"] = 0.0
        scored["citation_recall"] = 0.0
        scored["citation_f1"] = 0.0

        scored["abstention_correct"] = False

        # API failure is tracked separately by failure rate.
        # It is not itself a hallucination.
        scored["hallucination_proxy"] = False

        return scored

    # --------------------------------------------------
    # 2. Answer accuracy
    # --------------------------------------------------

    if not record["expected_answerable"]:
        # For an out-of-scope question we care about
        # whether the model correctly abstained.
        scored["answer_similarity"] = None

        scored["answer_correct"] = (
            record["insufficient_context"] is True
        )

    else:
        similarity = answer_similarity(
            expected=record["expected_answer"],
            generated=record["generated_answer"],
            embedding_service=embedding_service,
        )

        scored["answer_similarity"] = similarity

        scored["answer_correct"] = (
            similarity >= ANSWER_SIMILARITY_THRESHOLD
        )

    # --------------------------------------------------
    # 3. Citation accuracy
    # --------------------------------------------------

    precision, recall, f1 = citation_metrics(
        expected=record["expected_articles"],
        generated=record["generated_citations"],
    )

    scored["citation_precision"] = precision
    scored["citation_recall"] = recall
    scored["citation_f1"] = f1

    # --------------------------------------------------
    # 4. Abstention correctness
    # --------------------------------------------------

    if not record["expected_answerable"]:
        # Out-of-scope question:
        # model should explicitly say context is insufficient
        # and should not invent citations.
        abstention_correct = (
            record["insufficient_context"] is True
            and not record["generated_citations"]
        )

    else:
        # Answerable question:
        # model should not claim context is insufficient.
        abstention_correct = (
            record["insufficient_context"] is False
        )

    scored["abstention_correct"] = abstention_correct

    # --------------------------------------------------
    # 5. Hallucination proxy
    # --------------------------------------------------
    #
    # This is deliberately conservative.
    #
    # We flag:
    #
    # A) citations that do not exist in retrieved context
    #
    # OR
    #
    # B) answering an out-of-scope question instead of
    #    abstaining.
    #
    # This is NOT a complete factual-entailment detector.
    # We will document that limitation.

    invalid_citation = (
        record["citations_valid_in_context"] is False
    )

    unsupported_out_of_scope_answer = (
        not record["expected_answerable"]
        and record["insufficient_context"] is not True
    )

    scored["hallucination_proxy"] = (
        invalid_citation
        or unsupported_out_of_scope_answer
    )

    return scored


def main() -> None:
    # Load the 45 raw benchmark records.
    with INPUT_PATH.open(
        encoding="utf-8"
    ) as file:
        records = json.load(file)

    print(
        f"Loaded {len(records)} benchmark records."
    )

    # Load multilingual embedding model once.
    print("Loading embedding model...")

    embedding_service = EmbeddingService()

    # Score every result.
    print("Calculating evaluation metrics...")

    scored = [
        score_record(
            record=record,
            embedding_service=embedding_service,
        )
        for record in records
    ]

    # Save separately so the original raw benchmark
    # remains untouched.
    OUTPUT_PATH.write_text(
        json.dumps(
            scored,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Scored {len(scored)} records "
        f"→ {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()