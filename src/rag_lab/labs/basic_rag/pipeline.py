"""Start here: the smallest complete question -> retrieval -> answer pipeline.

Ingestion happens before this class is created (see cli._resources). The store
already contains embedded document chunks; ask() handles a new user question.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from rag_lab.shared.generation import answer_from_documents
from rag_lab.shared.retrieval import to_evidence
from rag_lab.shared.types import RagResult


class BasicRag:
    """Keep a vector store and chat client together for repeated questions.

    Passing these objects in is dependency injection: callers can supply Azure
    clients during practice or deterministic fakes during offline tests.
    """

    def __init__(self, store: Any, chat_model: Any, top_k: int = 4) -> None:
        # __init__ runs when BasicRag(...) is constructed. self is this instance.
        self.store = store
        self.chat_model = chat_model
        self.top_k = top_k

    def retrieve(self, question: str):
        """Return up to top_k (Document, relevance_score) pairs.

        Chroma uses its embedding client to turn the question into a vector,
        then finds similar stored vectors. No answer is generated at this step.
        A relevance score is a ranking signal, not a correctness probability.
        """
        return self.store.similarity_search_with_relevance_scores(question, k=self.top_k)

    def ask(self, question: str) -> RagResult:
        """Retrieve evidence, request an answer, and package both for inspection."""
        # perf_counter measures elapsed time; it is not a wall-clock timestamp.
        started = perf_counter()
        retrieved = self.retrieve(question)
        # This list comprehension unpacks each pair and keeps only its Document.
        # '_' means we intentionally do not use the score in the model prompt.
        documents = [document for document, _ in retrieved]
        answer, usage = answer_from_documents(self.chat_model, question, documents)
        # Keep evidence alongside the answer so callers can check its citations.
        return RagResult(
            question=question,
            answer=answer,
            evidence=to_evidence(retrieved),
            latency_ms=(perf_counter() - started) * 1000,
            usage=usage,
        )
