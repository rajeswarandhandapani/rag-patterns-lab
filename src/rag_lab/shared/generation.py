"""Generation: place retrieved text in a prompt and ask the chat model to answer.

Context is ordinary text supplied at request time, not new model training.
Citations are requested in the prompt; evaluation checks their identifiers later.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


def merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    """Sum reported token counters; *usages accepts any number of dictionaries."""
    totals: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            totals[key] = totals.get(key, 0) + value
    return totals


SYSTEM_PROMPT = """You answer questions only from the supplied engineering documents.
Cite claims using [document_id:chunk_id]. If the evidence is insufficient, say that you do not
have enough information. Keep the answer concise and do not invent operational details."""


def document_context(documents: list[Document]) -> str:
    """Label every passage with the exact citation the answer should reference."""
    return "\n\n".join(
        f"[{doc.metadata['document_id']}:{doc.metadata['chunk_id']}]\n{doc.page_content}"
        for doc in documents
    )


def invoke_text(model: Any, system: str, user: str) -> tuple[str, dict[str, int]]:
    """Call the model synchronously and return (response_text, available_usage).

    SystemMessage carries instructions; HumanMessage carries the question/data.
    invoke() is the network boundary for Azure clients. getattr(..., None)
    safely handles clients that do not provide optional token usage metadata.
    """
    response = model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = response.content if hasattr(response, "content") else str(response)
    usage = getattr(response, "usage_metadata", None) or {}
    return str(content), {key: int(value) for key, value in usage.items() if isinstance(value, int)}


def answer_from_documents(
    model: Any, question: str, documents: list[Document]
) -> tuple[str, dict[str, int]]:
    """Combine question and evidence; prompt instructions alone cannot ensure truth."""
    prompt = f"Context:\n{document_context(documents)}\n\nQuestion: {question}"
    return invoke_text(model, SYSTEM_PROMPT, prompt)
