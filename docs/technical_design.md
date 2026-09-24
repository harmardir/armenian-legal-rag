# Technical Design

## 1. Overview

This project implements a Retrieval-Augmented Generation (RAG) assistant over the Law of the Republic of Armenia on Electronic Communications.

The system supports questions in Armenian and English and generates answers grounded in retrieved legal text with citations to specific articles.

The same retrieval pipeline is also used to benchmark multiple LLM providers, allowing generation quality and operational characteristics to be compared under approximately equivalent retrieval conditions.

The implemented pipeline is:

```text
ARLIS law
→ article extraction
→ article-aware chunking
→ multilingual embeddings + BM25
→ Reciprocal Rank Fusion
→ context assembly
→ LLM generation
→ structured response
→ citation validation
```

## 2. Source and Ingestion

The legal source is the Armenian version of the Law on Electronic Communications published by ARLIS:

<https://www.arlis.am/hy/acts/1869>

The application downloads the HTML using `requests` and parses it with BeautifulSoup.

The ARLIS page is legacy HTML and its DOM structure is not sufficiently consistent for reliable extraction based purely on sibling elements. Therefore, BeautifulSoup is first used to normalize the document into text, after which article boundaries are identified using the textual structure:

```text
Հոդված <number>.
```

This produced 67 articles from the source document.

Each extracted article retains:

- article number
- article title
- article text
- source URL

**Implementation:** `app/ingestion/fetch.py`

## 3. Chunking Strategy

Legal documents have meaningful structural boundaries. A citation such as "Article 45" is more useful than a citation to an arbitrary character range.

For this reason, chunking is article-aware.

Short articles remain as a single chunk. Long articles are split into smaller chunks using approximately:

- 1,500 characters per chunk
- 200 characters overlap

Each chunk retains its parent article metadata:

- article number
- article title
- chunk index
- source URL

The current law produces 126 chunks from 67 articles.

The overlap reduces the risk that information close to a chunk boundary is lost during retrieval.

A limitation of the current character-based splitting is that some boundaries may occur in linguistically imperfect positions. A production version would use paragraph-, clause-, or section-aware splitting.

**Implementation:** `app/ingestion/chunk.py`

## 4. Multilingual Embeddings

The embedding model is `intfloat/multilingual-e5-small`.

A multilingual model was selected because:

1. the authoritative legal corpus is Armenian;
2. users may ask questions in Armenian or English;
3. cross-lingual semantic retrieval is therefore required.

Documents are embedded with the E5 `passage:` prefix and queries with the `query:` prefix.

Embeddings are normalized. As a result, the dot product used by the vector retriever is equivalent to cosine similarity.

For this small corpus, embeddings are stored in memory and exact nearest neighbor search is performed with NumPy.

This avoids introducing a vector database that would add operational complexity without providing meaningful benefit for only 126 chunks.

For a larger production corpus, the retrieval interface could be backed by pgvector, OpenSearch, Qdrant, or another persistent vector search system.

**Implementation:**

- `app/retrieval/embeddings.py`
- `app/retrieval/vector.py`

## 5. Lexical Retrieval with BM25

Semantic retrieval is useful for paraphrases and English-to-Armenian retrieval, but legal questions may also contain exact terminology or article references.

The system therefore includes BM25 lexical retrieval using `rank-bm25`.

BM25 indexes each chunk together with its article metadata.

A simple Unicode-aware tokenizer is currently used.

An important implementation detail is that zero-score BM25 results are discarded. This prevents unrelated chunks from being treated as ranked evidence when, for example, an English query has no lexical overlap with the Armenian corpus.

A limitation is that the tokenizer does not perform Armenian stemming or morphological analysis.

**Implementation:** `app/retrieval/bm25.py`

## 6. Hybrid Retrieval

Vector similarity and BM25 produce scores on different scales, so their raw scores are not directly added.

Instead, the system combines their rankings using Reciprocal Rank Fusion (RRF).

Each retriever first produces candidate results. RRF assigns rank-based scores and combines the two ranked lists.

For interactive Q&A:

- candidate retrieval: top 10
- final context: top 5

This gives the system both:

- semantic retrieval for paraphrases and cross-language questions;
- lexical retrieval for exact legal terminology.

**Implementation:** `app/retrieval/hybrid.py`

## 7. Context Assembly

Retrieval and context assembly are intentionally separate.

- **Retrieval** decides which chunks are relevant.
- **Context assembly** converts those chunks into a controlled representation for the LLM.

Each chunk is labelled with its article number and title:

```text
[Article 45]
Title: ...
<legal text>
```

Chunks are separated clearly before being passed to the model.

This structure makes the evidence easier for the model to interpret and provides explicit article identifiers for citation generation.

**Implementation:** `app/rag/context.py`

## 8. Answer Generation

The system instructs the LLM to:

- use only the provided legal context;
- avoid outside knowledge;
- avoid inventing legal provisions;
- cite only articles present in the context;
- abstain when the evidence is insufficient;
- answer concisely;
- use the language requested by the application.

Responses follow a structured schema:

```json
{
  "answer": "...",
  "citations": [45],
  "insufficient_context": false
}
```

This separates probabilistic language generation from deterministic application logic.

**Implementation:**

- `app/models/rag.py`
- `app/rag/prompt.py`

## 9. Language Handling

The application supports Armenian and English questions.

Initially, language selection was left entirely to the LLM with an instruction to answer in the same language as the question.

Testing showed that an English question could still produce an Armenian answer.

Because language detection for the supported languages is simple, it was moved into deterministic application logic.

The application checks for Armenian Unicode characters. If present, the requested response language is Armenian; otherwise it is English.

The selected language is then explicitly included in the LLM prompt.

This illustrates a broader design principle: deterministic application logic should handle simple deterministic decisions rather than relying on the LLM when unnecessary.

**Implementation:** `app/rag/prompt.py`

## 10. Citation Validation

After generation, citations are validated against the articles actually present in the retrieved context.

For example, if the retrieved articles are:

```text
[12, 18, 25]
```

and the model returns:

```text
[18, 37]
```

Article 37 is rejected as an invalid citation because it was not available in the supplied evidence.

This validation prevents the application from silently accepting fabricated article references.

This check only verifies that a citation exists in the retrieved context. It does not prove that the cited article semantically supports every statement in the generated answer.

**Implementation:** `app/rag/validation.py`

## 11. LLM Provider Abstraction

LLM access is implemented behind a common provider interface.

The benchmark uses:

- OpenAI
- Groq
- Gemini

Each provider returns a common `LLMResult` containing the structured answer and available operational metadata such as latency and token usage.

This allows the RAG pipeline and benchmark runner to remain mostly independent of provider-specific APIs.

**Implementation:**

- `app/llm/base.py`
- `app/llm/openai_provider.py`
- `app/llm/groq_provider.py`
- `app/llm/gemini_provider.py`

## 12. Fair Benchmark Design

For each benchmark question, retrieval is performed once.

The retrieved chunks and assembled context are then reused for every LLM provider.

Therefore:

```text
question
→ one retrieval operation
→ one context
→ OpenAI
→ Groq
→ Gemini
```

This is important because independently retrieving evidence for every model could introduce retrieval variation into what is intended to be a model comparison.

Using identical evidence makes differences more attributable to generation behavior and provider reliability.

## 13. Evaluation Dataset

The benchmark contains 15 ground-truth questions:

- 5 answerable Armenian questions
- 5 answerable English questions
- 3 out-of-scope/adversarial questions
- 2 synthesis questions requiring evidence across articles

Each case contains:

- question
- language
- expected answer
- expected article citations
- whether the question is answerable
- category

**Implementation/data:** `data/evaluation_questions.json`

## 14. Evaluation Metrics

The benchmark records the following metrics.

### Answer accuracy

For answerable questions, the generated answer is compared with the expected answer using multilingual embedding similarity.

A threshold of 0.75 is currently used.

For intentionally unanswerable questions, correctness is determined from the structured `insufficient_context` field rather than semantic similarity.

API failures remain in the denominator of end-to-end accuracy.

### Citation accuracy

Expected and generated article citations are compared using:

- precision
- recall
- F1

### Hallucination proxy

The automated hallucination metric flags:

- citations not present in retrieved context;
- failure to abstain on known out-of-scope questions.

This is intentionally described as a proxy. It is not a claim-level factual entailment detector.

### Latency

Total request latency is measured for successful requests.

TTFT is recorded where the provider integration exposes a usable streaming path.

### Token usage

Prompt and completion tokens are recorded when exposed by the provider API.

### Cost

Paid-rate cost is calculated from measured token usage even when the actual API usage is covered by credits or a free tier.

If token counts are unavailable, cost is reported as unavailable rather than estimated and presented as measured.

### Failure rate

Provider failures such as rate limits, service-unavailable responses, timeouts, or malformed responses are recorded rather than excluded.

**Implementation:**

- `app/evaluation/benchmark.py`
- `app/evaluation/scoring.py`
- `app/evaluation/summary.py`

## 15. API and UI

FastAPI provides:

| Method | Endpoint             | Description                  |
|--------|----------------------|------------------------------|
| GET    | `/`                  | Web interface                |
| GET    | `/api/health`        | Health status                |
| POST   | `/api/ask`           | Interactive RAG Q&A          |
| GET    | `/api/benchmark`     | Aggregated benchmark metrics |
| GET    | `/api/benchmark/raw` | Scored benchmark records     |

The frontend is intentionally implemented with plain HTML, CSS, and JavaScript because visual design is not the focus of the task.

The interface provides:

- Q&A tab
- Benchmark tab

The interactive Q&A currently uses OpenAI as its default generation provider.

**Implementation:**

- `app/main.py`
- `static/index.html`
- `static/app.js`
- `static/style.css`

## 16. Runtime Architecture

For this take-home implementation, the law is downloaded and the in-memory retrieval indexes are built once during FastAPI startup.

They are then reused for all incoming questions.

This avoids rebuilding embeddings for every request.

The trade-off is that application restart currently causes ingestion and embedding computation to run again.

A production architecture would separate ingestion from serving:

```text
ARLIS
→ scheduled/versioned ingestion
→ persistent document + vector index

API
→ query embedding
→ persistent retrieval system
→ context
→ LLM
```

## 17. Why No External Vector Database

The corpus currently contains only 126 chunks.

At this scale, an in-memory exact search is simple, transparent, and fast enough.

Introducing Elasticsearch/OpenSearch, pgvector, or a dedicated vector database would add deployment and operational complexity without solving a current scale problem.

The retrieval layer is separated from the rest of the RAG pipeline so that a persistent vector store can replace the current implementation if corpus size or production requirements justify it.

## 18. Current Limitations

The current implementation has several known limitations:

- Document embeddings are rebuilt on application restart.
- Chunk splitting is character-based inside long articles rather than clause-aware.
- BM25 uses simple tokenization without Armenian morphological processing.
- Retrieval quality is not yet evaluated independently with metrics such as Recall@K or MRR.
- Automated answer accuracy uses semantic similarity and should not be interpreted as human-reviewed legal correctness.
- The hallucination metric is a proxy rather than claim-level entailment evaluation.
- TTFT is unavailable for provider adapters currently using non-streaming structured-output calls.
- Groq token usage is unavailable in the current streaming integration, so measured paid-rate cost cannot be reported for Groq.
- Provider free-tier rate limits can affect benchmark reliability.
- The application is a legal-information assistant, not a substitute for professional legal review.

## 19. Production Improvements

With additional time, the next improvements would be:

- separate ingestion from the API runtime;
- persist chunks and embeddings;
- add document versioning and source update detection;
- use pgvector or OpenSearch for larger corpora;
- improve Armenian lexical normalization;
- use paragraph/clause-aware legal chunking;
- evaluate retrieval independently;
- add a dedicated reranking stage;
- add claim-level groundedness evaluation;
- add caching;
- add authentication and authorization;
- add structured logging and observability;
- add automated tests and CI/CD;
- add retries, provider fallback, and circuit-breaking policies;
- expand the benchmark dataset and perform human legal review.