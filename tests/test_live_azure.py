import os
from importlib.util import find_spec

import pytest

from rag_lab.cli import _pipeline
from rag_lab.labs.conversational_rag import ConversationalRag
from rag_lab.shared.config import Settings

pytestmark = pytest.mark.live


@pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1")
@pytest.mark.parametrize(
    "lab",
    [
        "basic-rag",
        "vector-dimensions",
        "hybrid-search",
        "reranking",
        "multi-query",
        "conversational-rag",
        "corrective-rag",
    ],
)
def test_live_question_for_every_lab(lab: str) -> None:
    if lab == "reranking" and find_spec("sentence_transformers") is None:
        pytest.skip("install --extra reranker")
    settings = Settings()
    settings.require_azure()
    dimensions = 256 if lab == "vector-dimensions" else None
    runner = _pipeline(settings, lab, dimensions, rebuild=False)
    result = (
        runner.ask("What does ORD-4097 mean?", session_id="live-smoke")
        if isinstance(runner, ConversationalRag)
        else runner.ask("What does ORD-4097 mean?")
    )
    assert result.answer
    assert result.evidence
