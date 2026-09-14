"""Offline stand-ins with the same methods as real model and vector-store clients.

These make control-flow tests deterministic and free of network calls. Hash-based
vectors and scripted answers are test fixtures, not demonstrations of RAG quality.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage


class FakeEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        values = [float(digest[index % len(digest)]) / 255 for index in range(self.dimensions)]
        magnitude = sum(value * value for value in values) ** 0.5 or 1
        return [value / magnitude for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def doc(document_id: str, content: str | None = None) -> Document:
    return Document(
        page_content=content or document_id,
        metadata={"document_id": document_id, "chunk_id": f"{document_id}-chunk"},
    )


class FakeStore:
    def __init__(self, rankings: dict[str, list[Document]] | list[Document]) -> None:
        self.rankings = rankings
        self.queries: list[str] = []

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        self.queries.append(query)
        if isinstance(self.rankings, dict):
            candidates = self.rankings.get(query, next(iter(self.rankings.values())))
        else:
            candidates = self.rankings
        return candidates[:k]

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
        return [
            (item, 1 - index / 10) for index, item in enumerate(self.similarity_search(query, k))
        ]


class RoutingChat:
    def __init__(self, grades: list[bool] | None = None) -> None:
        self.grades = list(grades or [])

    def invoke(self, messages: list[Any]) -> AIMessage:
        system = str(messages[0].content)
        user = str(messages[-1].content)
        if "strict JSON" in system:
            value = self.grades.pop(0) if self.grades else True
            return AIMessage(content=f'{{"relevant": {str(value).lower()}}}')
        if "Rewrite the latest question" in system:
            if "How long does it last?" in user:
                return AIMessage(content="How long does an inventory reservation last?")
            return AIMessage(content="standalone rewritten query")
        if "precise engineering documentation" in system:
            return AIMessage(content="rewritten search query")
        if "Generate three distinct" in system:
            return AIMessage(content="variant alpha\nvariant beta\nvariant gamma")
        return AIMessage(
            content="Grounded answer [doc:doc-chunk]",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
