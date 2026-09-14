from pathlib import Path

import pytest

from rag_lab.shared.store import IndexSpec, open_or_create_store
from tests.fakes import FakeEmbeddings, doc


def spec(dimensions: int, fingerprint: str = "corpus") -> IndexSpec:
    return IndexSpec("test", fingerprint, "fake", dimensions, 100, 10)


def test_index_is_reused_only_for_matching_configuration(tmp_path: Path) -> None:
    chunks = [doc("a"), doc("b")]
    _, created = open_or_create_store(
        embeddings=FakeEmbeddings(8), chunks=chunks, index_root=tmp_path, spec=spec(8)
    )
    _, reused_created = open_or_create_store(
        embeddings=FakeEmbeddings(8), chunks=chunks, index_root=tmp_path, spec=spec(8)
    )
    assert created is True
    assert reused_created is False


def test_dimension_or_corpus_mismatch_is_rejected(tmp_path: Path) -> None:
    chunks = [doc("a")]
    open_or_create_store(
        embeddings=FakeEmbeddings(8), chunks=chunks, index_root=tmp_path, spec=spec(8)
    )
    # Same collection path, deliberately incompatible manifest.
    incompatible = IndexSpec("test", "changed-corpus", "other-model", 8, 100, 10)
    with pytest.raises(ValueError, match="configuration mismatch"):
        open_or_create_store(
            embeddings=FakeEmbeddings(8),
            chunks=chunks,
            index_root=tmp_path,
            spec=incompatible,
        )


def test_query_vector_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    store, _ = open_or_create_store(
        embeddings=FakeEmbeddings(8),
        chunks=[doc("a")],
        index_root=tmp_path,
        spec=spec(8),
    )
    with pytest.raises(Exception, match="dimension"):
        store.similarity_search_by_vector([0.0] * 4, k=1)
