# Your 14-day RAG practice checklist

Start with Day 1. Budget **60–90 minutes per day**: 15 minutes reading, 35–55 minutes
running/changing code, and 10–20 minutes explaining what you learned. These are learning
sessions, not deadlines. Repeat a day if its completion check is still unclear.

Run commands from `/home/rajes/engineer/rag-patterns-lab`. Keep daily notes in a file of your
choice using the template at the end. Check boxes here as you finish tasks.

## Day 1 — Get comfortable with the project

- [ ] Read the tool/concept table and ingestion diagram in [Start Here](START_HERE.md).
- [ ] Open [03_orders.md](../data/documents/03_orders.md). This is one document your RAG system will search.
- [ ] Prepare the Python environment and run the offline tests:

```bash
cd /home/rajes/engineer/rag-patterns-lab
uv sync --extra dev
uv run rag-lab --help
uv run pytest -q
```

**Done when:** you can explain the roles of Python, LangChain, LangGraph, Azure, and Chroma
in one sentence each. Skipped live Azure tests are expected today; no credentials are needed.

## Day 2 — Learn just enough Python to read this code

- [ ] Read the Python syntax table in [Start Here](START_HERE.md#python-syntax-you-will-encounter).
- [ ] Read [BasicRag](../src/rag_lab/labs/basic_rag/pipeline.py) and [result types](../src/rag_lab/shared/types.py).
- [ ] Identify a class, constructor, method, list comprehension, tuple unpacking, and type hint.
- [ ] Write the steps of `BasicRag.ask()` in plain English without copying the code.

**Done when:** you can follow `retrieve → collect documents → generate → return result`.
You do not need to learn all of Python before continuing.

## Day 3 — Connect Azure and ask your first question

- [ ] Follow [Quickstart](../README.md#quickstart). If `.env` does not exist, copy `.env.example`
  to `.env`; preserve any configuration you already entered.
- [ ] Configure your Azure endpoint, key, chat deployment, and embedding deployment. Ask for help
  here if you do not yet have the two deployments; do not guess deployment names.
- [ ] Read [model factories](../src/rag_lab/shared/models.py), then run:

```bash
uv run rag-lab ingest --lab basic-rag
uv run rag-lab ask --lab basic-rag "What does ORD-4097 mean?"
```

**Done when:** the answer cites a retrieved chunk and agrees with `03_orders.md`.
From today onward, live model calls use your Azure resources; offline tests remain available.
If Azure setup takes longer, you can do Day 4's reading while resolving it.

## Day 4 — Trace one question end to end

- [ ] Read the [basic lab guide](../src/rag_lab/labs/basic_rag/README.md) and both diagrams.
- [ ] Follow [document loading/splitting](../src/rag_lab/shared/documents.py),
  [storage](../src/rag_lab/shared/store.py), and [generation](../src/rag_lab/shared/generation.py).
- [ ] Find the exact point where the question is embedded, the evidence is retrieved, and the chat model is called.
- [ ] Inspect the last answer's `evidence`, `usage`, and `latency_ms` fields.

**Done when:** you can draw ingestion and question answering as two separate paths and explain
why a chunk ID is different from an embedding vector.

## Day 5 — Change chunking and top-k

- [ ] Run the same question with top-k 1 and 4:

```bash
RAG_TOP_K=1 uv run rag-lab ask --lab basic-rag "Which services participate in checkout before an order is confirmed?"
RAG_TOP_K=4 uv run rag-lab ask --lab basic-rag "Which services participate in checkout before an order is confirmed?"
```

- [ ] Compare retrieved evidence before comparing answer wording.
- [ ] Try a separate smaller-chunk index so you retain the baseline:

```bash
RAG_INDEX_DIR=.rag_indexes/small-chunks RAG_CHUNK_SIZE=250 RAG_CHUNK_OVERLAP=40 uv run rag-lab ask --lab basic-rag "Which services participate in checkout before an order is confirmed?"
```

**Done when:** you can explain chunk size, overlap, and top-k. Prefixing a command with these
environment variables changes only that command; subsequent commands return to your `.env` values.

## Day 6 — Establish a measured baseline

- [ ] Read [evaluation.py](../src/rag_lab/shared/evaluation.py) and three entries in the
  [labeled question set](../data/evaluation/questions.json).
- [ ] Run the baseline evaluation and save it:

```bash
uv run rag-lab evaluate --lab basic-rag --output artifacts/day06-baseline.json
```

- [ ] Explain recall@k and reciprocal rank with one actual result. Check `sample_counts`.
- [ ] Ask an unsupported question and inspect whether the answer abstains:

```bash
uv run rag-lab ask --lab basic-rag "What is the office Wi-Fi password?"
```

**Done when:** you can distinguish retrieval quality, citation validity, and answer correctness.
Record three useful test questions: one exact code, one semantic question, and one unsupported question.

## Day 7 — Experiment with vector dimensions

- [ ] Read the [dimensions guide](../src/rag_lab/labs/vector_dimensions/README.md).
- [ ] First compare two widths with one timed repetition to keep the initial run small:

```bash
uv run rag-lab compare-dimensions --dimensions 256,1536 --repetitions 1 --output-dir artifacts/day07-dimensions
```

- [ ] Open the generated PNG and CSV. Compare quality, raw vector bytes, persisted bytes, and query latency.
- [ ] Check `index_created` and the separate embedding/build/load timings before comparing runs.

**Done when:** you can explain what 256 dimensions means and why fewer dimensions do not
automatically imply a cheaper embedding API call. Optional later: all four widths and five repetitions.

## Day 8 — Combine lexical and semantic retrieval

- [ ] Read [hybrid search](../src/rag_lab/labs/hybrid_search/README.md) and its `pipeline.py`.
- [ ] Compare the same code-specific question with basic and hybrid RAG:

```bash
uv run rag-lab ask --lab basic-rag "What does PAY-2031 require?"
uv run rag-lab ask --lab hybrid-search "What does PAY-2031 require?"
uv run rag-lab evaluate --lab hybrid-search --output artifacts/day08-hybrid.json
```

- [ ] Read `reciprocal_rank_fusion()` in [retrieval.py](../src/rag_lab/shared/retrieval.py).
  Calculate `1 / (60 + rank)` for ranks 1 and 2 yourself.

**Done when:** you can explain why lexical matches help error codes and why RRF combines ranks.
An equal or worse result on some questions is a valid observation.

## Day 9 — Add a reranker

- [ ] Read [reranking](../src/rag_lab/labs/reranking/README.md), especially candidate-k versus top-k.
- [ ] Install the optional runtime; allow extra time and disk space for the model stack:

```bash
uv sync --extra dev --extra reranker
uv run --extra reranker rag-lab ask --lab reranking "How should a cache failure affect writes?"
```

- [ ] Run the same query again after model loading, then compare the baseline evidence order.
- [ ] Read `retrieve()` and identify the candidate pool, pair scoring, sorting, and final selection.

**Done when:** you can explain why reranking cannot recover a document outside the candidate pool.
If installation is blocked, study `test_reranker_changes_candidate_order` in
[test_retrieval.py](../tests/test_retrieval.py), then return to the live run later.

## Day 10 — Search multiple question formulations

- [ ] Read [multi-query](../src/rag_lab/labs/multi_query/README.md) and its expansion prompt.
- [ ] Run:

```bash
uv run rag-lab ask --lab multi-query "How can I trace a failing request across services?"
uv run rag-lab evaluate --lab multi-query --output artifacts/day10-multi-query.json
```

- [ ] Inspect `trace.queries`. Decide whether each variant preserves the original intent.
- [ ] Compare recall, reported usage, and latency with Day 6. Explain why expansion costs an extra call.

**Done when:** you can name one case where extra phrasings help and one where they might drift.

## Day 11 — Learn LangGraph with a straight workflow

- [ ] Read [How a LangGraph node works](START_HERE.md#how-a-langgraph-node-works).
- [ ] Read [conversational graph.py](../src/rag_lab/labs/conversational_rag/graph.py).
- [ ] Trace `StateGraph`, `add_node`, `add_edge`, `compile`, and `invoke`.
- [ ] Write down which state keys each node reads and updates.
- [ ] Run the complete multi-turn example in [Start Here](START_HERE.md#try-a-real-conversation-in-one-process).

**Done when:** you can explain state, nodes, and edges, and show the rewritten follow-up question.

## Day 12 — Understand conversation memory

- [ ] Repeat a follow-up in the same Python object with two different session IDs.
- [ ] Compare an established session with a new session that has no history.
- [ ] Read and run the session-isolation test:

```bash
uv run pytest tests/test_graphs.py::test_conversation_sessions_are_isolated -v
```

- [ ] Explain why running two separate CLI `ask` commands does not preserve conversation history.

**Done when:** you can distinguish stored documents, per-session history, and per-invocation graph state.
Optional: run conversational evaluation, inspecting its follow-up cases separately from standalone cases.

## Day 13 — Add branching, correction, and abstention

- [ ] Read [corrective RAG](../src/rag_lab/labs/corrective_rag/README.md) and trace each graph branch.
- [ ] Try one supported and one unsupported question:

```bash
uv run rag-lab ask --lab corrective-rag "What does ORD-4097 mean?"
uv run rag-lab ask --lab corrective-rag "What is the office Wi-Fi password?"
uv run pytest tests/test_graphs.py -v
```

- [ ] Inspect `trace.attempts`, `trace.rewrites`, `trace.abstained`, and usage.
- [ ] Explain why the retry limit matters and why the JSON string `"false"` must not count as approval.

**Done when:** you can trace retrieve → grade → answer, or retrieve → grade → rewrite → retrieve → abstain.
Live grades can be imperfect; the offline tests give deterministic examples of the branches.

## Day 14 — Review and explain your engineering choices

- [ ] Revisit your three test questions across the applicable labs. Use the same settings for comparisons.
- [ ] Fill a small table: pattern, evidence quality, latency, available token usage, failure observed.
- [ ] Explain one pattern aloud for five minutes: problem, data flow, tradeoff, failure mode, and test.
- [ ] Choose one exercise from a lab guide, implement it yourself, and rerun the relevant offline tests.
- [ ] Write what you would select for a documentation assistant and what evidence supports that choice.

**Done when:** you can choose a pattern for a concrete problem and justify it using observations.
Compare conversational follow-ups separately; their available context differs from standalone questions.
The broader [curriculum](../README.md#full-curriculum) is your next step after these foundations.

## Daily notes template

```text
Day:
What I expected:
Command and configuration used:
What the retrieved evidence showed:
What the answer/metrics showed:
One concept I can now explain:
One question I still have:
Next experiment:
```

If you get stuck, share the command, the error with credentials removed, and your own explanation
of what you expected. We can work through one day at a time.
