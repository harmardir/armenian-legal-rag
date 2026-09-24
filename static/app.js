const qaTab = document.getElementById("qa-tab");
const benchmarkTab = document.getElementById("benchmark-tab");

const qaSection = document.getElementById("qa-section");
const benchmarkSection = document.getElementById("benchmark-section");

const askButton = document.getElementById("ask-button");
const questionInput = document.getElementById("question");

const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const answerCard = document.getElementById("answer-card");


// --------------------------------------------------
// Tabs
// --------------------------------------------------

qaTab.addEventListener("click", () => {
    qaTab.classList.add("active");
    benchmarkTab.classList.remove("active");

    qaSection.classList.remove("hidden");
    benchmarkSection.classList.add("hidden");
});


benchmarkTab.addEventListener("click", async () => {
    benchmarkTab.classList.add("active");
    qaTab.classList.remove("active");

    benchmarkSection.classList.remove("hidden");
    qaSection.classList.add("hidden");

    await loadBenchmark();
});


// --------------------------------------------------
// Q&A
// --------------------------------------------------

askButton.addEventListener("click", async () => {
    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    askButton.disabled = true;

    loading.classList.remove("hidden");
    errorBox.classList.add("hidden");
    answerCard.classList.add("hidden");

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Request failed"
            );
        }

        document.getElementById("answer").textContent =
            data.answer;

        document.getElementById("citations").textContent =
            data.citations.length
                ? data.citations
                    .map(article => `Article ${article}`)
                    .join(", ")
                : "None";

        document.getElementById("retrieved").textContent =
            data.retrieved_articles
                .map(article => `Article ${article}`)
                .join(", ");

        document.getElementById("model").textContent =
            data.model;

        document.getElementById("latency").textContent =
            `${Math.round(data.latency_ms)} ms`;

        answerCard.classList.remove("hidden");

    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");

    } finally {
        loading.classList.add("hidden");
        askButton.disabled = false;
    }
});


// --------------------------------------------------
// Benchmark
// --------------------------------------------------

function percent(value) {
    if (value === null || value === undefined) {
        return "N/A";
    }

    return `${(value * 100).toFixed(1)}%`;
}


function milliseconds(value) {
    if (value === null || value === undefined) {
        return "N/A";
    }

    return `${Math.round(value)} ms`;
}


function cost(value) {
    if (value === null || value === undefined) {
        return "N/A";
    }

    return `$${value.toFixed(6)}`;
}


async function loadBenchmark() {
    const body = document.getElementById(
        "benchmark-body"
    );

    body.innerHTML =
        "<tr><td colspan='4'>Loading...</td></tr>";

    try {
        const response = await fetch(
            "/api/benchmark"
        );

        if (!response.ok) {
            throw new Error(
                "Unable to load benchmark"
            );
        }

        const data = await response.json();

        const providers = [
            data.openai,
            data.groq,
            data.gemini
        ];

        const rows = [
            [
                "Answer accuracy",
                ...providers.map(
                    p => percent(p.answer_accuracy)
                )
            ],
            [
                "Citation F1",
                ...providers.map(
                    p => percent(p.citation_f1)
                )
            ],
            [
                "Hallucination rate",
                ...providers.map(
                    p => percent(p.hallucination_rate)
                )
            ],
            [
                "Failure rate",
                ...providers.map(
                    p => percent(p.failure_rate)
                )
            ],
            [
                "Average latency",
                ...providers.map(
                    p => milliseconds(
                        p.average_total_latency_ms
                    )
                )
            ],
            [
                "Average TTFT",
                ...providers.map(
                    p => milliseconds(
                        p.average_ttft_ms
                    )
                )
            ],
            [
                "Avg prompt tokens",
                ...providers.map(
                    p =>
                        p.average_prompt_tokens === null
                            ? "N/A"
                            : Math.round(
                                p.average_prompt_tokens
                            )
                )
            ],
            [
                "Avg completion tokens",
                ...providers.map(
                    p =>
                        p.average_completion_tokens === null
                            ? "N/A"
                            : Math.round(
                                p.average_completion_tokens
                            )
                )
            ],
            [
                "Paid-rate cost / request",
                ...providers.map(
                    p => cost(
                        p.average_cost_usd
                    )
                )
            ]
        ];

        body.innerHTML = rows
            .map(row => `
                <tr>
                    <td>${row[0]}</td>
                    <td>${row[1]}</td>
                    <td>${row[2]}</td>
                    <td>${row[3]}</td>
                </tr>
            `)
            .join("");

    } catch (error) {
        body.innerHTML = `
            <tr>
                <td colspan="4">
                    ${error.message}
                </td>
            </tr>
        `;
    }
}