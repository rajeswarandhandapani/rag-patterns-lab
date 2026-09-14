from langchain_core.documents import Document

from rag_lab.labs.reranking import RerankedRag
from rag_lab.shared.retrieval import reciprocal_rank_fusion
from tests.fakes import FakeStore, RoutingChat, doc


def test_rrf_fuses_and_deduplicates() -> None:
    first = [doc("a"), doc("b")]
    second = [doc("b"), doc("c")]
    fused = reciprocal_rank_fusion([first, second], limit=3)
    assert [item.metadata["document_id"] for item, _ in fused] == ["b", "a", "c"]


class FixedReranker:
    def score(self, _question: str, documents: list[Document]) -> list[float]:
        return [0.1 if item.metadata["document_id"] == "a" else 0.9 for item in documents]


def test_reranker_changes_candidate_order() -> None:
    rag = RerankedRag(FakeStore([doc("a"), doc("b")]), RoutingChat(), FixedReranker(), top_k=1)
    retrieved = rag.retrieve("question")
    assert retrieved[0][0].metadata["document_id"] == "b"
    assert retrieved[0][1] == 0.9
