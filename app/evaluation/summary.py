import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


INPUT_PATH = Path("data/benchmark_scored.json")
OUTPUT_PATH = Path("data/benchmark_summary.json")


# Paid API pricing in USD per 1 million tokens.
# Pricing snapshot used for this benchmark.
PRICING_PER_MILLION = {
    "openai": {
        "input": 0.40,
        "output": 1.60,
    },
    "groq": {
        "input": 0.15,
        "output": 0.60,
    },
    "gemini": {
        "input": 0.75,
        "output": 3.75,
    },
}


def safe_mean(
    values: list[float | int | None],
) -> float | None:
    """Return the mean while ignoring unavailable values."""

    valid = [
        value
        for value in values
        if value is not None
    ]

    if not valid:
        return None

    return mean(valid)


def calculate_cost(
    provider: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """
    Calculate paid-rate cost for one request.

    Returns None when token usage is unavailable.
    """

    if (
        prompt_tokens is None
        or completion_tokens is None
    ):
        return None

    pricing = PRICING_PER_MILLION.get(provider)

    if pricing is None:
        return None

    input_cost = (
        prompt_tokens
        / 1_000_000
        * pricing["input"]
    )

    output_cost = (
        completion_tokens
        / 1_000_000
        * pricing["output"]
    )

    return input_cost + output_cost


def summarize_provider(
    provider: str,
    records: list[dict],
) -> dict:
    """Aggregate benchmark metrics for one provider."""

    total = len(records)

    successful = [
        record
        for record in records
        if record["success"]
    ]

    failures = [
        record
        for record in records
        if not record["success"]
    ]

    # --------------------------------------------------
    # 1. End-to-end answer accuracy
    # --------------------------------------------------
    #
    # Failures stay in the denominator.
    #
    # Example:
    # 12 correct / 15 total = 80%

    correct_answers = sum(
        1
        for record in records
        if record["answer_correct"]
    )

    answer_accuracy = (
        correct_answers / total
        if total
        else 0.0
    )

    # --------------------------------------------------
    # 2. Citation accuracy
    # --------------------------------------------------
    #
    # Failed requests already have citation scores of 0,
    # therefore failures affect the end-to-end metric.

    citation_precision = safe_mean([
        record["citation_precision"]
        for record in records
    ])

    citation_recall = safe_mean([
        record["citation_recall"]
        for record in records
    ])

    citation_f1 = safe_mean([
        record["citation_f1"]
        for record in records
    ])

    # --------------------------------------------------
    # 3. Hallucination proxy
    # --------------------------------------------------

    hallucination_count = sum(
        1
        for record in records
        if record["hallucination_proxy"]
    )

    hallucination_rate = (
        hallucination_count / total
        if total
        else 0.0
    )

    # --------------------------------------------------
    # 4. Failure rate
    # --------------------------------------------------

    failure_rate = (
        len(failures) / total
        if total
        else 0.0
    )

    # --------------------------------------------------
    # 5. Latency
    # --------------------------------------------------
    #
    # Latency is calculated only for successful requests.

    average_total_latency_ms = safe_mean([
        record["total_latency_ms"]
        for record in successful
    ])

    average_ttft_ms = safe_mean([
        record["ttft_ms"]
        for record in successful
    ])

    # --------------------------------------------------
    # 6. Token usage
    # --------------------------------------------------

    average_prompt_tokens = safe_mean([
        record["prompt_tokens"]
        for record in successful
    ])

    average_completion_tokens = safe_mean([
        record["completion_tokens"]
        for record in successful
    ])

    total_prompt_tokens = sum(
        record["prompt_tokens"]
        for record in successful
        if record["prompt_tokens"] is not None
    )

    total_completion_tokens = sum(
        record["completion_tokens"]
        for record in successful
        if record["completion_tokens"] is not None
    )

    # --------------------------------------------------
    # 7. Paid-rate cost
    # --------------------------------------------------

    request_costs = [
        calculate_cost(
            provider=provider,
            prompt_tokens=record["prompt_tokens"],
            completion_tokens=record["completion_tokens"],
        )
        for record in successful
    ]

    valid_costs = [
        cost
        for cost in request_costs
        if cost is not None
    ]

    average_cost_usd = (
        mean(valid_costs)
        if valid_costs
        else None
    )

    total_measured_cost_usd = (
        sum(valid_costs)
        if valid_costs
        else None
    )

    return {
        "total_questions": total,
        "successful_requests": len(successful),
        "failed_requests": len(failures),

        "answer_correct": correct_answers,
        "answer_accuracy": answer_accuracy,

        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_f1": citation_f1,

        "hallucination_count": hallucination_count,
        "hallucination_rate": hallucination_rate,

        "failure_rate": failure_rate,

        "average_total_latency_ms": average_total_latency_ms,
        "average_ttft_ms": average_ttft_ms,

        "average_prompt_tokens": average_prompt_tokens,
        "average_completion_tokens": average_completion_tokens,

        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,

        "average_cost_usd": average_cost_usd,
        "total_measured_cost_usd": total_measured_cost_usd,
    }


def main() -> None:
    # --------------------------------------------------
    # Load scored benchmark
    # --------------------------------------------------

    with INPUT_PATH.open(
        encoding="utf-8"
    ) as file:
        records = json.load(file)

    # --------------------------------------------------
    # Group records by provider
    # --------------------------------------------------

    grouped = defaultdict(list)

    for record in records:
        grouped[record["provider"]].append(record)

    # --------------------------------------------------
    # Calculate provider summaries
    # --------------------------------------------------

    summary = {}

    for provider, provider_records in grouped.items():
        summary[provider] = summarize_provider(
            provider=provider,
            records=provider_records,
        )

    # --------------------------------------------------
    # Save machine-readable summary
    # --------------------------------------------------

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Print human-readable summary
    # --------------------------------------------------

    print("\n=== BENCHMARK SUMMARY ===\n")

    for provider, metrics in summary.items():
        print(provider.upper())
        print("-" * 50)

        print(
            f"Questions: "
            f"{metrics['total_questions']}"
        )

        print(
            f"Successful requests: "
            f"{metrics['successful_requests']}"
        )

        print(
            f"Failed requests: "
            f"{metrics['failed_requests']}"
        )

        print(
            f"Answer accuracy: "
            f"{metrics['answer_accuracy'] * 100:.1f}%"
        )

        print(
            f"Citation precision: "
            f"{metrics['citation_precision'] * 100:.1f}%"
        )

        print(
            f"Citation recall: "
            f"{metrics['citation_recall'] * 100:.1f}%"
        )

        print(
            f"Citation F1: "
            f"{metrics['citation_f1'] * 100:.1f}%"
        )

        print(
            f"Hallucination rate: "
            f"{metrics['hallucination_rate'] * 100:.1f}%"
        )

        print(
            f"Failure rate: "
            f"{metrics['failure_rate'] * 100:.1f}%"
        )

        # ------------------------------
        # Latency
        # ------------------------------

        latency = metrics[
            "average_total_latency_ms"
        ]

        if latency is not None:
            print(
                f"Average total latency: "
                f"{latency:.0f} ms"
            )
        else:
            print(
                "Average total latency: unavailable"
            )

        ttft = metrics["average_ttft_ms"]

        if ttft is not None:
            print(
                f"Average TTFT: "
                f"{ttft:.0f} ms"
            )
        else:
            print(
                "Average TTFT: unavailable"
            )

        # ------------------------------
        # Tokens
        # ------------------------------

        prompt_tokens = metrics[
            "average_prompt_tokens"
        ]

        completion_tokens = metrics[
            "average_completion_tokens"
        ]

        if prompt_tokens is not None:
            print(
                f"Average prompt tokens: "
                f"{prompt_tokens:.1f}"
            )
        else:
            print(
                "Average prompt tokens: unavailable"
            )

        if completion_tokens is not None:
            print(
                f"Average completion tokens: "
                f"{completion_tokens:.1f}"
            )
        else:
            print(
                "Average completion tokens: unavailable"
            )

        # ------------------------------
        # Cost
        # ------------------------------

        average_cost = metrics[
            "average_cost_usd"
        ]

        if average_cost is not None:
            print(
                f"Average paid-rate cost/request: "
                f"${average_cost:.6f}"
            )
        else:
            print(
                "Average paid-rate cost/request: "
                "unavailable (token usage unavailable)"
            )

        total_cost = metrics[
            "total_measured_cost_usd"
        ]

        if total_cost is not None:
            print(
                f"Total measured benchmark cost: "
                f"${total_cost:.6f}"
            )
        else:
            print(
                "Total measured benchmark cost: unavailable"
            )

        print()

    print(
        f"Saved summary → {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()