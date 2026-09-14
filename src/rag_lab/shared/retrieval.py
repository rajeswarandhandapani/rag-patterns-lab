"""Utilities for combining search rankings and exposing their source evidence."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document

from rag_lab.shared.types import Evidence


def document_key(document: Document) -> str:
    """Use chunk identity, so separate chunks of one document stay distinct."""
    return str(document.metadata["chunk_id"])


def reciprocal_rank_fusion(
    rankings: list[list[Document]], limit: int, constant: int = 60
) -> list[tuple[Document, float]]:
    """Fuse ranked lists with reciprocal rank fusion (RRF).

    Each occurrence earns 1/(constant + rank), with rank starting at 1.
    A chunk found near the top by multiple searches accumulates a higher score.
    Scores from BM25 and cosine search are not directly comparable; ranks are.
    The returned float is an RRF score, not a probability or cosine similarity.
    """
    # defaultdict(float) gives unseen IDs a starting score of 0.0.
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, Document] = {}
    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            key = document_key(document)
            by_id[key] = document
            scores[key] += 1 / (constant + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [(by_id[key], scores[key]) for key in ordered]


def to_evidence(items: list[tuple[Document, float | None]]) -> list[Evidence]:
    """Convert LangChain objects to the repo's serializable result representation."""
    return [
        Evidence(
            chunk_id=str(document.metadata["chunk_id"]),
            document_id=str(document.metadata["document_id"]),
            content=document.page_content,
            score=score,
            metadata=dict(document.metadata),
        )
        for document, score in items
    ]
