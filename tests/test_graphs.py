from rag_lab.labs.conversational_rag import ConversationalRag
from rag_lab.labs.corrective_rag import CorrectiveRag
from tests.fakes import FakeStore, RoutingChat, doc


def test_conversation_sessions_are_isolated() -> None:
    store = FakeStore([doc("05_inventory")])
    rag = ConversationalRag(store, RoutingChat())
    rag.ask("Tell me about inventory reservations", session_id="alice")
    result = rag.ask("How long does it last?", session_id="alice")
    other = rag.ask("How long does it last?", session_id="bob")
    assert "inventory reservation" in result.trace["standalone_question"]
    assert other.trace["standalone_question"] == "How long does it last?"
    assert len(rag.sessions["alice"]) == 2
    assert len(rag.sessions["bob"]) == 1


def test_corrective_graph_stops_after_two_failed_attempts() -> None:
    store = FakeStore([doc("irrelevant")])
    rag = CorrectiveRag(store, RoutingChat(grades=[False, False]))
    result = rag.ask("unsupported question")
    assert result.trace["attempts"] == 2
    assert len(store.queries) == 2
    assert "not have enough information" in result.answer


def test_corrective_graph_answers_after_first_good_retrieval() -> None:
    store = FakeStore([doc("doc")])
    rag = CorrectiveRag(store, RoutingChat(grades=[True]))
    result = rag.ask("supported question")
    assert result.trace["attempts"] == 1
    assert "Grounded answer" in result.answer
