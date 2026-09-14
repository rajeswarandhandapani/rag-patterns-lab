import json
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_each_lab_has_two_mermaid_diagrams_and_exercises() -> None:
    guides = sorted((ROOT / "src/rag_lab/labs").glob("*/README.md"))
    assert len(guides) == 7
    for guide in guides:
        text = guide.read_text(encoding="utf-8")
        assert text.count("```mermaid") >= 2, guide
        assert "## Exercises" in text, guide


def test_all_diagrams_use_codex_compatible_vertical_flowcharts() -> None:
    """Keep diagrams aligned with the simple form rendered by the Codex preview."""
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    markdown_files.extend(sorted((ROOT / "src/rag_lab/labs").glob("*/README.md")))
    diagrams = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        for diagram in re.findall(r"```mermaid\n(.*?)```", text, flags=re.DOTALL):
            diagrams.append((path, diagram))

    assert len(diagrams) == 16
    for path, diagram in diagrams:
        lines = [line.strip() for line in diagram.splitlines() if line.strip()]
        assert lines[0] == "flowchart TD", path
        assert all(" & " not in line for line in lines), path
        assert all(line.count("-->") <= 1 for line in lines), path
        # Quote every rectangular/cylindrical label. This also protects reserved
        # characters such as @ from being interpreted as Mermaid syntax.
        node_definitions = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\[(.*?)\]", diagram)
        assert node_definitions, path
        assert all(label.startswith('"') or label.startswith('("') for label in node_definitions), (
            path
        )


def test_dataset_has_planned_size_and_valid_references() -> None:
    document_ids = {path.stem for path in (ROOT / "data/documents").glob("*.md")}
    questions = json.loads((ROOT / "data/evaluation/questions.json").read_text(encoding="utf-8"))
    assert len(document_ids) == 20
    assert len(questions) == 30
    assert all(set(question["expected_documents"]) <= document_ids for question in questions)
