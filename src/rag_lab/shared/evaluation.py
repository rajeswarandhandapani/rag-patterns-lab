"""Measure retrieval and citations against manually labeled questions.

Finding evidence, answering correctly, and abstaining are different behaviors.
These metrics keep them separate; none alone proves the answer is trustworthy.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rag_lab.shared.types import RagResult


@dataclass(slots=True)
class EvaluationCase:
    """One labeled question; conversation holds prior user turns when needed."""

    id: str
    question: str
    expected_documents: list[str]
    reference_answer: str
    kind: str
    conversation: list[str] | None = None


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load JSON; **item expands a dictionary into named constructor arguments."""
    return [EvaluationCase(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def standalone_cases(cases: list[EvaluationCase]) -> list[EvaluationCase]:
    """Only include questions that can be interpreted without preceding turns."""
    return [case for case in cases if not case.conversation and case.kind != "conversation"]


def retrieval_metrics(result: RagResult, case: EvaluationCase) -> dict[str, float | None]:
    """Compute metrics for one answer; None means not applicable, not a zero score.

    Recall: fraction of expected source documents retrieved among the top chunks.
    Reciprocal rank: 1 / position of the first relevant chunk (third means 1/3).
    Citation validity: fraction of unique citation pairs present in evidence.
    It checks references, not whether a passage actually supports the claim.
    """
    returned = [item.document_id for item in result.evidence]
    # Sets count a document only once even if several of its chunks were retrieved.
    expected = set(case.expected_documents)
    if not expected:
        recall = None
        reciprocal_rank = None
    else:
        recall = len(expected.intersection(returned)) / len(expected)
        ranks = [returned.index(doc) + 1 for doc in expected if doc in returned]
        reciprocal_rank = 1 / min(ranks) if ranks else 0.0
    cited = set(re.findall(r"\[([^:\]]+):([^\]]+)\]", result.answer))
    evidence_pairs = {(item.document_id, item.chunk_id) for item in result.evidence}
    citation_validity = len(cited & evidence_pairs) / len(cited) if cited else None
    # Graph abstention is explicit; text-only pipelines use a documented heuristic.
    abstained = result.trace.get("abstained")
    if abstained is None:
        abstained = bool(
            re.search(
                r"(?:do not|don't) have enough (?:information|evidence)|"
                r"insufficient (?:information|evidence)|cannot answer",
                result.answer,
                re.IGNORECASE,
            )
        )
    return {
        "recall_at_k": recall,
        "reciprocal_rank": reciprocal_rank,
        "citation_validity": citation_validity,
        "latency_ms": result.latency_ms,
        "unanswerable_abstention_rate": float(abstained) if not expected else None,
    }


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average applicable values and expose how many samples each metric used.

    Averaging reciprocal rank over the applicable questions produces MRR.
    Excluding None avoids penalizing unanswerable questions for missing evidence.
    """
    names = (
        "recall_at_k",
        "reciprocal_rank",
        "citation_validity",
        "latency_ms",
        "unanswerable_abstention_rate",
    )
    summary: dict[str, Any] = {}
    counts = {}
    for name in names:
        values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
        summary[name] = mean(values) if values else None
        counts[name] = len(values)
    summary["sample_counts"] = counts
    return summary


def evaluate_cases(
    runner: Callable[[str], RagResult],
    cases: list[EvaluationCase],
    judge: Callable[[RagResult, EvaluationCase], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run standalone cases once each, optionally passing results to an LLM judge.

    runner is a callable such as rag.ask. Receiving a function lets evaluation
    work with any lab that returns RagResult, without knowing its implementation.
    """
    rows = []
    selected = standalone_cases(cases)
    for case in selected:
        result = runner(case.question)
        row = {
            "case": asdict(case),
            "result": result.to_dict(),
            "metrics": retrieval_metrics(result, case),
        }
        if judge:
            row["judge_estimate"] = judge(result, case)
        rows.append(row)
    return {
        "summary": summarize_metrics(rows),
        "excluded_case_ids": [case.id for case in cases if case not in selected],
        "cases": rows,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    """Create the destination directory and write a human-readable JSON report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def judge_result(model: Any, result: RagResult, case: EvaluationCase) -> dict[str, Any]:
    """Optional, non-deterministic judge. Treat scores as estimates, not ground truth."""
    from rag_lab.shared.generation import invoke_text

    system = """Score a RAG answer against the reference and evidence. Return strict JSON:
{\"correctness\": 0.0, \"grounding\": 0.0, \"reason\": \"brief reason\"}.
Scores range from 0 to 1. Grounding means every material claim has retrieved support."""
    evidence = "\n\n".join(item.content for item in result.evidence)
    text, _ = invoke_text(
        model,
        system,
        f"Question: {case.question}\nReference: {case.reference_answer}\n"
        f"Answer: {result.answer}\nEvidence:\n{evidence}",
    )
    try:
        return json.loads(text.strip().strip("`json\n "))
    except json.JSONDecodeError:
        return {"correctness": None, "grounding": None, "reason": "Judge returned invalid JSON"}
