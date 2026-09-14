"""Persist and reopen local Chroma collections with compatibility checks.

A collection stores vectors alongside source text and labels. Its JSON manifest
records how those vectors were produced so a changed corpus is not silently reused.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class TimedEmbeddings(Embeddings):
    """Wrap another embedding client and delegate its work while measuring time.

    Inheriting Embeddings supplies the interface Chroma expects. finally executes
    even if the wrapped call raises an exception; failures still propagate.
    """

    def __init__(self, delegate: Embeddings) -> None:
        self.delegate = delegate
        self.document_ms = 0.0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = perf_counter()
        try:
            return self.delegate.embed_documents(texts)
        finally:
            self.document_ms += (perf_counter() - started) * 1000

    def embed_query(self, text: str) -> list[float]:
        return self.delegate.embed_query(text)


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """Immutable description of an index's inputs (frozen=True prevents assignment)."""

    lab: str
    corpus_fingerprint: str
    embedding_model: str
    dimensions: int | None
    chunk_size: int
    chunk_overlap: int

    @property
    def collection_name(self) -> str:
        """A @property is read as spec.collection_name, without calling parentheses."""
        dimension = self.dimensions or "default"
        return f"{self.lab}-{dimension}".replace("_", "-")


def open_or_create_store(
    *,
    embeddings: Embeddings,
    chunks: list[Document],
    index_root: Path,
    spec: IndexSpec,
    rebuild: bool = False,
    timings: dict[str, Any] | None = None,
) -> tuple[Chroma, bool]:
    """Return (store, created), where created=False means an existing index was reused.

    The leading * makes arguments keyword-only for readable call sites.
    Rebuild replaces this collection's directory and regenerates its vectors.
    Optional timings is a caller-owned dictionary filled with measurements.
    Reuse avoids document embedding, but future search questions still need vectors.
    """
    path = index_root / spec.collection_name
    manifest_path = path / "manifest.json"
    # Convert the dataclass to a dictionary so it can be compared with JSON data.
    wanted = asdict(spec)
    existing = None
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing and existing != wanted and not rebuild:
        raise ValueError(
            f"Index configuration mismatch at {path}. Pass --rebuild or use another collection."
        )
    created = rebuild or existing is None
    if created and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    measured_embeddings = TimedEmbeddings(embeddings)
    started = perf_counter()
    store = Chroma(
        collection_name=spec.collection_name,
        embedding_function=measured_embeddings,
        persist_directory=str(path),
        # Cosine compares vector direction. HNSW is Chroma's vector search index.
        collection_metadata={"hnsw:space": "cosine"},
    )
    if created:
        ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
        # Chroma calls embed_documents internally, then stores vectors and text.
        store.add_documents(chunks, ids=ids)
        manifest_path.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    elapsed_ms = (perf_counter() - started) * 1000
    if timings is not None:
        timings.update(
            {
                "index_created": created,
                "document_embedding_ms": measured_embeddings.document_ms if created else None,
                "index_build_ms": max(0.0, elapsed_ms - measured_embeddings.document_ms)
                if created
                else None,
                "index_load_ms": elapsed_ms if not created else None,
            }
        )
    return store, created
