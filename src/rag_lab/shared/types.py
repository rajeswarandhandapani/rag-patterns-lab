"""Plain Python result containers, independent of Azure or LangGraph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Evidence:
    """One retrieved chunk and its source; scores depend on the selected lab.

    @dataclass generates the constructor. slots=True limits attributes to these
    declared fields. A score of None means unavailable, not zero relevance.
    """

    chunk_id: str
    document_id: str
    content: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RagResult:
    """An answer plus the evidence and measurements needed to inspect it.

    default_factory creates a NEW dictionary for every result, avoiding mutable
    data shared accidentally between requests. trace stores lab-specific steps.
    """

    question: str
    answer: str
    evidence: list[Evidence]
    latency_ms: float
    usage: dict[str, int] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert dataclasses into dictionaries for JSON output."""
        return asdict(self)
