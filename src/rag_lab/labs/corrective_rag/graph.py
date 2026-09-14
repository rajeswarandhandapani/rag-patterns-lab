"""Lab 7: branch on evidence quality and allow at most one repair attempt.

Unlike the conversational graph's straight path, this workflow has a conditional
edge and a loop. The attempt counter guarantees termination even if grading fails.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from rag_lab.shared.generation import (
    answer_from_documents,
    document_context,
    invoke_text,
    merge_usage,
)
from rag_lab.shared.retrieval import to_evidence
from rag_lab.shared.types import RagResult


class EvidenceGrade(BaseModel):
    """Validate model output at runtime; the string 'false' is not a boolean.

    StrictBool forbids coercion; extra='forbid' rejects unexpected JSON fields.
    This checks output structure, not the accuracy of the model's judgment.
    """

    model_config = ConfigDict(extra="forbid")
    relevant: StrictBool


class CorrectiveState(TypedDict, total=False):
    """Values passed between nodes; attempts counts retrievals, not model calls."""

    question: str
    search_query: str
    documents: list[Document]
    relevant: bool
    attempts: int
    answer: str
    usage: dict[str, int]
    rewrites: list[str]


class CorrectiveRag:
    """Retrieve -> grade -> answer, rewrite/retrieve, or abstain."""

    max_attempts = 2

    def __init__(self, store: Any, chat_model: Any, top_k: int = 4) -> None:
        self.store = store
        self.chat_model = chat_model
        self.top_k = top_k
        builder = StateGraph(CorrectiveState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade", self._grade)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("answer", self._answer)
        builder.add_node("abstain", self._abstain)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")
        # _route returns a label; this mapping selects the node for that label.
        builder.add_conditional_edges(
            "grade", self._route, {"answer": "answer", "rewrite": "rewrite", "abstain": "abstain"}
        )
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("answer", END)
        builder.add_edge("abstain", END)
        self.graph = builder.compile()

    def _retrieve(self, state: CorrectiveState) -> dict[str, Any]:
        """Use the current search query and increment the bounded attempt count."""
        query = state.get("search_query", state["question"])
        return {
            "documents": self.store.similarity_search(query, k=self.top_k),
            "attempts": state.get("attempts", 0) + 1,
        }

    def _grade(self, state: CorrectiveState) -> dict[str, Any]:
        """Ask whether evidence is sufficient; invalid output fails closed.

        'Fail closed' here means treat uncertain formatting as insufficient evidence
        and take the repair/abstain path, rather than authorizing an answer.
        """
        system = """Decide whether the context contains evidence sufficient to answer the question.
Return strict JSON with one boolean field: {\"relevant\": true}."""
        text, usage = invoke_text(
            self.chat_model,
            system,
            f"Question: {state['question']}\n\nContext:\n{document_context(state['documents'])}",
        )
        try:
            relevant = EvidenceGrade.model_validate_json(text).relevant
        except ValidationError:
            relevant = False
        return {"relevant": relevant, "usage": merge_usage(state.get("usage", {}), usage)}

    def _route(self, state: CorrectiveState) -> Literal["answer", "rewrite", "abstain"]:
        """Choose the next node; Literal documents the permitted return strings."""
        if state["relevant"]:
            return "answer"
        if state["attempts"] < self.max_attempts:
            return "rewrite"
        return "abstain"

    def _rewrite(self, state: CorrectiveState) -> dict[str, Any]:
        """Change only the search wording; preserve the original question for answering."""
        query, usage = invoke_text(
            self.chat_model,
            "Rewrite the question into a precise engineering documentation search query. "
            "Return only the query.",
            state["question"],
        )
        query = query.strip()
        return {
            "search_query": query,
            # Create an updated list rather than mutating the incoming state's list.
            "rewrites": [*state.get("rewrites", []), query],
            "usage": merge_usage(state.get("usage", {}), usage),
        }

    def _answer(self, state: CorrectiveState) -> dict[str, Any]:
        """Generate only after approval; usage includes earlier grade/repair calls."""
        answer, usage = answer_from_documents(
            self.chat_model, state["question"], state["documents"]
        )
        return {"answer": answer, "usage": merge_usage(state.get("usage", {}), usage)}

    def _abstain(self, _state: CorrectiveState) -> dict[str, str]:
        """Return a fixed refusal to guess; this node makes no additional model call."""
        return {"answer": "I do not have enough information in the indexed documents to answer."}

    def ask(self, question: str) -> RagResult:
        """Start attempts at zero for every question and expose the final routing outcome."""
        started = perf_counter()
        state = self.graph.invoke({"question": question, "search_query": question, "attempts": 0})
        return RagResult(
            question=question,
            answer=state["answer"],
            evidence=to_evidence([(document, None) for document in state["documents"]]),
            latency_ms=(perf_counter() - started) * 1000,
            usage=state.get("usage", {}),
            trace={
                "attempts": state["attempts"],
                "rewrites": state.get("rewrites", []),
                "abstained": not state["relevant"],
            },
        )
