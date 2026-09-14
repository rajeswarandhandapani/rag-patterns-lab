"""Ingestion: turn Markdown files into smaller, traceable LangChain Documents."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_dir: Path) -> list[Document]:
    """Read .md files; Document holds text in page_content and labels in metadata.

    Sorted file order keeps repeated ingestion reproducible. The filename stem
    (for example '03_orders') becomes the human-readable document identifier.
    """
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        document_id = path.stem
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"document_id": document_id, "source": str(path)},
            )
        )
    if not documents:
        raise FileNotFoundError(f"No Markdown documents found in {data_dir}")
    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split text by character count while preserving source labels.

    Recursive means try headings, then paragraphs, then progressively smaller
    separators when a section is too large. Overlap repeats nearby text between
    chunks to reduce the chance of splitting a fact from its explanation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        # A stable hash identifies the exact text and position for citations.
        # Changing a chunk changes its ID; this hash is not an embedding vector.
        identity = (
            f"{chunk.metadata['document_id']}:{chunk.metadata.get('start_index', 0)}:"
            f"{chunk.page_content}"
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return chunks


def corpus_fingerprint(documents: list[Document], chunk_size: int, chunk_overlap: int) -> str:
    """Hash content and split settings so incompatible stored indexes are rejected."""
    digest = hashlib.sha256(f"{chunk_size}:{chunk_overlap}".encode())
    for document in documents:
        digest.update(str(document.metadata["document_id"]).encode())
        digest.update(document.page_content.encode())
    return digest.hexdigest()
