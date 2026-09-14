import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_each_lab_has_two_mermaid_diagrams_and_exercises() -> None:
    guides = sorted((ROOT / "src/rag_lab/labs").glob("*/README.md"))
    assert len(guides) == 7
    for guide in guides:
        text = guide.read_text(encoding="utf-8")
        assert text.count("```mermaid") >= 2, guide
        assert "## Exercises" in text, guide


def test_dataset_has_planned_size_and_valid_references() -> None:
    document_ids = {path.stem for path in (ROOT / "data/documents").glob("*.md")}
    questions = json.loads((ROOT / "data/evaluation/questions.json").read_text(encoding="utf-8"))
    assert len(document_ids) == 20
    assert len(questions) == 30
    assert all(set(question["expected_documents"]) <= document_ids for question in questions)
