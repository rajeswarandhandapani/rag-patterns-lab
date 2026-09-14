import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from rag_lab.cli import app
from rag_lab.labs.conversational_rag import ConversationalRag
from rag_lab.labs.corrective_rag import CorrectiveRag
from rag_lab.labs.multi_query import MultiQueryRag
from rag_lab.labs.vector_dimensions import DimensionExperiment
from rag_lab.shared.config import Settings
from rag_lab.shared.evaluation import EvaluationCase, evaluate_cases, retrieval_metrics
from rag_lab.shared.store import IndexSpec, open_or_create_store
from rag_lab.shared.types import Evidence, RagResult
from tests.fakes import FakeEmbeddings, FakeStore, doc


class ScriptedChat:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return AIMessage(
            content=next(self.responses),
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def test_followup_generation_receives_resolved_question_and_sums_usage():
    model = ScriptedChat(
        [
            "Inventory reservations hold stock.",
            "How long does an inventory reservation last?",
            "20 minutes.",
            "An unrelated answer.",
        ]
    )
    rag = ConversationalRag(FakeStore([doc("inventory")]), model)
    rag.ask("Tell me about inventory reservations.", session_id="a")
    result = rag.ask("How long does it last?", session_id="a")
    assert "Question: How long does an inventory reservation last?" in model.prompts[-1][-1].content
    assert result.usage["total_tokens"] == 30
    # Usage is per invocation, not accumulated over the session or another user.
    other = rag.ask("Another question", session_id="b")
    assert other.usage["total_tokens"] == 15


@pytest.mark.parametrize(
    "grade",
    [
        '{"relevant": "false"}',
        '{"relevant": "true"}',
        '{"relevant": 1}',
        '{"relevant": null}',
        "{}",
        "yes",
        "not json",
    ],
)
def test_invalid_grades_cannot_approve_evidence(grade):
    model = ScriptedChat([grade, "repaired query", grade])
    result = CorrectiveRag(FakeStore([doc("irrelevant")]), model).ask("Unknown?")
    assert result.trace["abstained"] is True
    assert result.trace["attempts"] == 2
    assert result.usage["total_tokens"] == 45


def test_corrective_repair_and_generation_usage_are_both_counted():
    model = ScriptedChat(['{"relevant": false}', "repair", '{"relevant": true}', "Answer"])
    result = CorrectiveRag(FakeStore([doc("evidence")]), model).ask("Question")
    assert result.usage["total_tokens"] == 60
    assert result.trace["abstained"] is False


def test_multi_query_counts_expansion_and_generation():
    model = ScriptedChat(["one\ntwo\nthree", "Answer"])
    result = MultiQueryRag(FakeStore([doc("evidence")]), model).ask("Question")
    assert result.usage == {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}


def make_case(identifier="q", expected=None, conversation=None):
    return EvaluationCase(identifier, "Question", expected or [], "Reference", "fact", conversation)


def test_citations_must_match_complete_evidence_pairs():
    evidence = [Evidence("real", "doc", "content")]
    result = RagResult("q", "Claims [doc:real] [doc:invented] [other:real]", evidence, 1)
    assert retrieval_metrics(result, make_case(expected=["doc"]))["citation_validity"] == 1 / 3


def test_abstention_does_not_inflate_or_depress_retrieval_metrics():
    cases = [make_case("supported", ["doc"]), make_case("unsupported")]
    results = iter(
        [
            RagResult("q", "Claim [doc:real]", [Evidence("real", "doc", "text")], 1),
            RagResult(
                "q", "I do not have enough information.", [Evidence("x", "other", "text")], 1
            ),
        ]
    )
    report = evaluate_cases(lambda _: next(results), cases)
    summary = report["summary"]
    assert summary["recall_at_k"] == summary["reciprocal_rank"] == 1
    assert summary["unanswerable_abstention_rate"] == 1
    assert summary["sample_counts"]["recall_at_k"] == 1
    assert summary["sample_counts"]["citation_validity"] == 1
    assert report["cases"][1]["metrics"]["reciprocal_rank"] is None
    only_unknown = evaluate_cases(lambda _: RagResult("q", "Cannot answer", [], 1), [make_case()])
    assert only_unknown["summary"]["recall_at_k"] is None


def test_standalone_evaluation_excludes_followups():
    report = evaluate_cases(
        lambda _: RagResult("q", "answer", [], 1),
        [make_case("normal", ["doc"]), make_case("followup", ["doc"], ["Prior turn"])],
    )
    assert report["excluded_case_ids"] == ["followup"]
    assert len(report["cases"]) == 1


def test_dimension_cli_excludes_followups():
    settings = Settings()
    with (
        patch("rag_lab.cli.get_settings", return_value=settings),
        patch("rag_lab.cli.DimensionExperiment") as experiment,
    ):
        experiment.return_value.run.return_value = {}
        result = CliRunner().invoke(app, ["compare-dimensions"])
    assert result.exit_code == 0, result.output
    queries = experiment.call_args.kwargs["queries"]
    assert len(queries) == 25
    assert "How long does it last?" not in [query for query, _ in queries]


def test_dimension_reports_distinguish_cold_and_warm_indexes(tmp_path: Path):
    def factory(dimension, embeddings_client):
        timings = {}
        spec = IndexSpec("benchmark", "corpus", "fake", dimension, 100, 10)
        store, _ = open_or_create_store(
            embeddings_client=embeddings_client,
            chunks=[doc("doc")],
            index_root=tmp_path,
            spec=spec,
            timings=timings,
        )
        return store, tmp_path / spec.collection_name, timings

    # Prime only one dimension to exercise a mixed cold/warm experiment.
    factory(8, FakeEmbeddings(8))
    experiment = DimensionExperiment(
        [8, 16], FakeEmbeddings, factory, [("query", ["doc"])], repetitions=1
    )
    report = experiment.run()
    warm, cold = report["results"]
    assert warm["index_created"] is False
    assert warm["index_load_ms"] >= 0
    assert warm["document_embedding_ms"] is warm["index_build_ms"] is None
    assert cold["index_created"] is True
    assert cold["index_load_ms"] is None
    assert cold["document_embedding_ms"] > 0
    assert cold["index_build_ms"] >= 0
    with patch.object(DimensionExperiment, "_plot"):
        experiment.export(report, tmp_path / "reports")
    saved = json.loads((tmp_path / "reports/dimension-results.json").read_text())
    assert saved == report
    assert "index_load_ms" in (tmp_path / "reports/dimension-results.csv").read_text()
