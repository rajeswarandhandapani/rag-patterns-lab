"""Lab 6: use prior turns to resolve a question before retrieval and generation.

LangGraph orchestrates Python functions (nodes) over a shared dictionary (state).
This example has a fixed path: rewrite -> retrieve -> answer.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from rag_lab.shared.generation import answer_from_documents, invoke_text, merge_usage
from rag_lab.shared.retrieval import to_evidence
from rag_lab.shared.types import RagResult


class ConversationState(TypedDict, total=False):
    """Describe state keys for readers/type checkers; this is still a dict at runtime.

    total=False means keys may be absent initially: nodes fill them as work proceeds.
    TypedDict itself does not validate values at runtime like a Pydantic model does.
    """

    question: str
    history: list[tuple[str, str]]
    standalone_question: str
    documents: list[Document]
    answer: str
    usage: dict[str, int]


REWRITE_SYSTEM = """Rewrite the latest question as a self-contained search query using the
conversation. Preserve exact identifiers. Return only the rewritten query. If it is already
self-contained, return it unchanged."""


class ConversationalRag:
    """Run the graph and retain history in memory, separately for each session ID.

    History survives repeated calls on this object, not a new CLI process. This
    educational store is not durable storage or a production authentication system.
    """

    def __init__(self, store: Any, chat_model: Any, top_k: int = 4) -> None:
        self.store = store
        self.chat_model = chat_model
        self.top_k = top_k
        self.sessions: dict[str, list[tuple[str, str]]] = {}
        # Register functions first; edges describe the order in which to call them.
        builder = StateGraph(ConversationState)
        builder.add_node("rewrite", self._rewrite)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("answer", self._answer)
        builder.add_edge(START, "rewrite")
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("retrieve", "answer")
        builder.add_edge("answer", END)
        # compile() creates the runnable workflow; it does not call Azure yet.
        self.graph = builder.compile()

    def _rewrite(self, state: ConversationState) -> dict[str, Any]:
        """Resolve references such as 'it'; avoid an LLM call on the first turn.

        A leading underscore marks an internal helper by Python convention.
        Nodes return partial state updates. Keys without a reducer are replaced;
        returning this dictionary does not discard the other existing state keys.
        """
        history = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in state.get("history", []))
        if not history:
            return {"standalone_question": state["question"]}
        rewritten, usage = invoke_text(
            self.chat_model,
            REWRITE_SYSTEM,
            f"Conversation:\n{history}\n\nLatest question: {state['question']}",
        )
        return {"standalone_question": rewritten.strip(), "usage": usage}

    def _retrieve(self, state: ConversationState) -> dict[str, list[Document]]:
        """Search with the resolved query, not the ambiguous latest turn."""
        documents = self.store.similarity_search(state["standalone_question"], k=self.top_k)
        return {"documents": documents}

    def _answer(self, state: ConversationState) -> dict[str, Any]:
        """Keep references resolved in generation and add this call's token usage."""
        answer, usage = answer_from_documents(
            self.chat_model, state["standalone_question"], state["documents"]
        )
        return {"answer": answer, "usage": merge_usage(state.get("usage", {}), usage)}

    def ask(self, question: str, session_id: str = "default") -> RagResult:
        """Execute a fresh graph run using this session's history, then append its turn."""
        started = perf_counter()
        history = list(self.sessions.get(session_id, []))
        # invoke() runs START through END and returns the final state dictionary.
        state = self.graph.invoke({"question": question, "history": history})
        self.sessions.setdefault(session_id, []).append((question, state["answer"]))
        return RagResult(
            question=question,
            answer=state["answer"],
            evidence=to_evidence([(document, None) for document in state["documents"]]),
            latency_ms=(perf_counter() - started) * 1000,
            usage=state.get("usage", {}),
            trace={"session_id": session_id, "standalone_question": state["standalone_question"]},
        )

    def clear(self, session_id: str) -> None:
        """Forget one session without affecting other sessions."""
        self.sessions.pop(session_id, None)
