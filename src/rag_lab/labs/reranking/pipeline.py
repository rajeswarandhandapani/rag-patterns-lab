"""Lab 4: retrieve broadly, then score question/passage pairs more carefully."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol

from langchain_core.documents import Document

from rag_lab.shared.generation import answer_from_documents
from rag_lab.shared.retrieval import to_evidence
from rag_lab.shared.types import RagResult


class Reranker(Protocol):
    """An interface: any object with this score method can serve as a reranker.

    Protocol supports structural typing; the fake used in tests need not inherit
    this class. The ellipsis declares the method shape, not an implementation.
    """

    def score(self, question: str, documents: list[Document]) -> list[float]: ...


class CrossEncoderReranker:
    """Loads the model lazily so ingestion and offline tests do not download it."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def score(self, question: str, documents: list[Document]) -> list[float]:
        """Score each passage jointly with the question using a local model.

        A cross-encoder reads both texts together. Unlike stored document vectors,
        these scores must be recomputed for each question and are not probabilities.
        """
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    "Install the local reranker with: uv sync --extra reranker"
                ) from error

            self._model = CrossEncoder(self.model_name)
        pairs = [(question, document.page_content) for document in documents]
        return [float(score) for score in self._model.predict(pairs)]


class RerankedRag:
    """Keep candidate_k (retrieved) separate from top_k (sent to the generator)."""

    def __init__(
        self,
        store: Any,
        chat_model: Any,
        reranker: Reranker,
        top_k: int = 4,
        candidate_k: int = 12,
    ) -> None:
        self.store = store
        self.chat_model = chat_model
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = candidate_k

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        """Reorder only retrieved candidates; missing evidence cannot be recovered."""
        candidates = self.store.similarity_search(question, k=self.candidate_k)
        scores = self.reranker.score(question, candidates)
        # zip pairs each document with its score; strict=True rejects length mismatch.
        # Sort by the pair's second item, then slice [:top_k] to keep the best few.
        return sorted(zip(candidates, scores, strict=True), key=lambda item: item[1], reverse=True)[
            : self.top_k
        ]

    def ask(self, question: str) -> RagResult:
        """Answer from reranked chunks; first-call latency can include model loading."""
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
            trace={"candidate_k": self.candidate_k, "reranker": type(self.reranker).__name__},
        )
