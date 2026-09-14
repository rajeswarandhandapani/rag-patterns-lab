"""Lab 5: search several phrasings of one question to improve evidence recall."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from rag_lab.shared.generation import answer_from_documents, invoke_text, merge_usage
from rag_lab.shared.retrieval import reciprocal_rank_fusion, to_evidence
from rag_lab.shared.types import RagResult

QUERY_SYSTEM = """Generate three distinct search queries for the user's question.
Preserve identifiers. Return one query per line, without numbering or commentary."""


class MultiQueryRag:
    """Expand -> retrieve each query -> fuse chunks -> answer the original question."""

    def __init__(self, store: Any, chat_model: Any, top_k: int = 4) -> None:
        self.store = store
        self.chat_model = chat_model
        self.top_k = top_k

    def expand(self, question: str) -> tuple[list[str], dict[str, int]]:
        """Return original plus up to three generated queries, and expansion usage.

        Keep the original so an imperfect rewrite cannot replace the user's query.
        *variants unpacks a list inside another list. Inspect trace.queries for drift.
        """
        text, usage = invoke_text(self.chat_model, QUERY_SYSTEM, question)
        variants = [line.strip(" -0123456789.\t") for line in text.splitlines() if line.strip()]
        return [question, *variants[:3]], usage

    def retrieve(self, queries: list[str]):
        """Search each formulation sequentially and fuse by stable chunk ID."""
        rankings = [
            self.store.similarity_search(query, k=max(self.top_k * 2, 8)) for query in queries
        ]
        return reciprocal_rank_fusion(rankings, limit=self.top_k)

    def ask(self, question: str) -> RagResult:
        """Include both expansion and answer calls in the reported token totals."""
        started = perf_counter()
        queries, expansion_usage = self.expand(question)
        retrieved = self.retrieve(queries)
        answer, usage = answer_from_documents(
            self.chat_model, question, [document for document, _ in retrieved]
        )
        return RagResult(
            question=question,
            answer=answer,
            evidence=to_evidence(retrieved),
            latency_ms=(perf_counter() - started) * 1000,
            usage=merge_usage(expansion_usage, usage),
            trace={"queries": queries},
        )
