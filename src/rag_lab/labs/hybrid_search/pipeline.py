"""Lab 3: combine meaning-based vector search with exact-term lexical search."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_lab.shared.generation import answer_from_documents
from rag_lab.shared.retrieval import reciprocal_rank_fusion, to_evidence
from rag_lab.shared.types import RagResult


def tokenize(text: str) -> list[str]:
    """Extract lowercase search terms, preserving dots/hyphens in error codes.

    These BM25 terms are not the subword tokens used for chat-model billing.
    """
    return re.findall(r"[a-z0-9][a-z0-9_.-]*", text.lower())


class HybridRag:
    """Search the same chunks two ways, then fuse their ranks before generation."""

    def __init__(self, store: Any, chunks: list[Document], chat_model: Any, top_k: int = 4) -> None:
        self.store = store
        self.chunks = chunks
        self.chat_model = chat_model
        self.top_k = top_k
        self.bm25 = BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])

    def lexical_search(self, question: str, k: int) -> list[Document]:
        """Rank term matches using BM25 (rarity, frequency, and length adjustment)."""
        scores = self.bm25.get_scores(tokenize(question))
        # lambda is a small anonymous function: sort each index by its score.
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [self.chunks[index] for index in ranked[:k] if scores[index] > 0]

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Gather a larger candidate pool from each search before taking final top-k."""
        candidate_k = max(self.top_k * 2, 8)
        dense = self.store.similarity_search(question, k=candidate_k)
        lexical = self.lexical_search(question, candidate_k)
        return reciprocal_rank_fusion([dense, lexical], limit=self.top_k)

    def ask(self, question: str) -> RagResult:
        """Generate from fused evidence; returned scores are RRF scores."""
        started = perf_counter()
        retrieved = self.retrieve(question)
        answer, usage = answer_from_documents(
            self.chat_model, question, [document for document, _ in retrieved]
        )
        return RagResult(
            question=question,
            answer=answer,
            evidence=to_evidence(retrieved),
            latency_ms=(perf_counter() - started) * 1000,
            usage=usage,
            trace={"fusion": "reciprocal_rank", "sources": ["dense", "bm25"]},
        )
