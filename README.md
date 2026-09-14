# RAG Patterns Lab

A hands-on curriculum for understanding retrieval-augmented generation with Python, LangChain,
LangGraph, Azure OpenAI deployments in Microsoft Foundry, and local Chroma indexes.

The repository favors visible, small implementations over framework magic. All labs use the same
fictional Northstar engineering knowledge base and labeled questions so comparisons are meaningful.

New to Python, LangChain, LangGraph, or RAG? Start with the
[guided code walkthrough](docs/START_HERE.md). It explains the vocabulary, Python syntax used here,
the ingestion/question paths, a recommended reading order, and a runnable multi-turn example.

For a paced, hands-on route, follow the [14-day learning checklist](docs/DAILY_PLAN.md)
(60–90 minutes per day, with tasks and completion checks).

## Quickstart

Prerequisites: Python 3.12, [`uv`](https://docs.astral.sh/uv/), and Foundry deployments for a chat
model and `text-embedding-3-small`.

```bash
cp .env.example .env
# Fill in the Azure values in .env
uv sync --extra dev
uv run rag-lab ingest --lab basic-rag
uv run rag-lab ask --lab basic-rag "What does ORD-4097 mean?"
uv run rag-lab evaluate --lab basic-rag --output artifacts/basic-rag.json
```

`AZURE_OPENAI_*_DEPLOYMENT` values are deployment names. The endpoint is the Azure OpenAI endpoint
associated with the Foundry project. Credentials never belong in Git. See Microsoft's
[LangChain and LangGraph guidance](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/langchain).

## Implemented labs

| Order | Lab | Central question |
|---|---|---|
| 1 | [Basic RAG](src/rag_lab/labs/basic_rag/README.md) | How do chunking and top-k shape an answer? |
| 2 | [Vector dimensions](src/rag_lab/labs/vector_dimensions/README.md) | What quality, storage, and latency change with vector width? |
| 3 | [Hybrid search](src/rag_lab/labs/hybrid_search/README.md) | When should lexical and semantic retrieval cooperate? |
| 4 | [Reranking](src/rag_lab/labs/reranking/README.md) | Can a slower relevance model improve the candidate order? |
| 5 | [Multi-query](src/rag_lab/labs/multi_query/README.md) | Do alternate phrasings improve recall without drifting? |
| 6 | [Conversational RAG](src/rag_lab/labs/conversational_rag/README.md) | How should prior turns influence retrieval? |
| 7 | [Corrective RAG](src/rag_lab/labs/corrective_rag/README.md) | When should a workflow retry or abstain? |

Each guide contains conceptual and execution diagrams, a worked trace, commands, controls, failure
modes, tradeoffs, and three exercises. Run `uv run rag-lab --help` for the shared CLI.

## Shared experiment design

```mermaid
flowchart LR
    D[20 source documents] --> C[Deterministic chunks]
    C --> L[Selected lab]
    Q[30 labeled questions] --> L
    L --> R[Answer + cited evidence + trace]
    R --> E[Recall@k, MRR, citation validity, latency]
```

Indexes live under `.rag_indexes/`. A manifest records corpus fingerprint, chunking, model, and
dimensions. A mismatch fails explicitly; use `--rebuild` when a deliberate change requires a fresh
index. Evaluation reports go to `artifacts/`. Add `--judge` for optional model-estimated answer
correctness and grounding. Retrieval and citation metrics are reproducible; LLM-judge scores are
estimates and should be calibrated against human labels.

Standalone evaluation runs 27 questions and lists the three excluded conversational case IDs.
Conversational evaluation runs all 30, replaying each case's history in an isolated session.
The dimensions experiment uses only the 25 supported, standalone questions.

Recall and reciprocal rank are `null` for questions with no relevant documents, and those rows
are excluded from retrieval averages. Citation validity checks complete document/chunk pairs;
answers without citations have `null` validity. Each summary includes `sample_counts` so metric
denominators are visible. `unanswerable_abstention_rate` is reported separately: it uses explicit
corrective-graph abstention when available and a phrase-matching heuristic for other pipelines.
This heuristic is not a substitute for answer correctness or grounding evaluation.

Result `usage` sums available chat-model usage for expansion, rewriting, grading, and generation
within that invocation, including corrective attempts that end in abstention. It excludes prior
conversation turns, embeddings, and optional evaluation-judge calls; missing provider usage is
unavailable, not evidence of a free call.

## Full curriculum

### Stage 1 — Retrieval foundations (implemented)

Complete labs 1–3. Be able to explain chunk boundaries, cosine search, dimensions, exact-token
failure, BM25, reciprocal-rank fusion, recall@k, and reciprocal rank. Completion: run the common
evaluation set and write down three queries where the methods differ.

### Stage 2 — Retrieval quality (implemented)

Complete labs 4–5. Understand candidate generation versus reranking, cross-encoder cost, query
expansion, deduplication, and query drift. Completion: improve at least one weak baseline query and
measure its latency cost.

### Stage 3 — Stateful workflows (implemented)

Complete labs 6–7. Trace LangGraph state, nodes, conditional edges, bounded retry, conversation
isolation, and abstention. Completion: reproduce a bad rewrite and a bad relevance grade, then add
a guardrail for each.

### Stage 4 — Planned retrieval patterns

1. Metadata filtering and query routing: structured filters, self-querying, and heterogeneous stores.
2. Parent-child retrieval and contextual compression: retrieve small, answer with coherent context.
3. HyDE and contextual retrieval: generate hypothetical evidence and enrich chunks before indexing.
4. Tool-driven agentic and adaptive RAG: choose whether, where, and how often to retrieve.
5. Multi-hop decomposition: plan dependent searches and join evidence across documents.
6. Knowledge-graph RAG: entity extraction, graph traversal, and vector-plus-graph retrieval.
7. Multimodal and layout-aware retrieval: pages, images, tables, and document structure.

### Stage 5 — Azure and production engineering

1. Replace local Chroma with Azure AI Search and compare vector, hybrid, semantic-ranker, and filter behavior.
2. Add versioned ingestion, incremental updates, deletion, provenance, tenant isolation, and ACL filtering.
3. Add grounded-answer evaluation, a calibrated LLM judge, tracing, cost budgets, caching, and load tests.
4. Package an API, deploy it, monitor retrieval drift, and define rollback criteria for index/model changes.

## Useful commands

```bash
uv run rag-lab ingest --lab hybrid-search
uv run rag-lab ask --lab hybrid-search "What does PAY-2031 require?"
uv run rag-lab ask --lab conversational-rag --session-id demo "How long does it last?"
uv run rag-lab compare-dimensions --dimensions 256,512,1024,1536
uv run pytest
uv run ruff check .
```

Before running lab 4, install its optional runtime with `uv sync --extra dev --extra reranker`.
The reranking lab downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` on first use. Live tests are
opt-in and need Azure credentials; regular tests use deterministic fakes and make no network calls.
