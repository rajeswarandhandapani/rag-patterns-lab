"""Connect terminal commands to the lab objects.

pyproject.toml maps 'rag-lab' to this module's app. Typer turns decorated functions
into commands; Annotated attaches CLI help/options to ordinary Python type hints.
Start reading in basic_rag/pipeline.py, then return here to see object construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from rag_lab.labs.basic_rag import BasicRag
from rag_lab.labs.conversational_rag import ConversationalRag
from rag_lab.labs.corrective_rag import CorrectiveRag
from rag_lab.labs.hybrid_search import HybridRag
from rag_lab.labs.multi_query import MultiQueryRag
from rag_lab.labs.reranking import CrossEncoderReranker, RerankedRag
from rag_lab.labs.vector_dimensions import DimensionExperiment
from rag_lab.shared.config import Settings, get_settings
from rag_lab.shared.documents import corpus_fingerprint, load_documents, split_documents
from rag_lab.shared.evaluation import (
    evaluate_cases,
    judge_result,
    load_cases,
    retrieval_metrics,
    standalone_cases,
    summarize_metrics,
    write_report,
)
from rag_lab.shared.models import create_chat_model, create_embeddings
from rag_lab.shared.store import IndexSpec, open_or_create_store

app = typer.Typer(help="Run and compare the RAG pattern labs.", no_args_is_help=True)
LABS = {
    "basic-rag",
    "vector-dimensions",
    "hybrid-search",
    "reranking",
    "multi-query",
    "conversational-rag",
    "corrective-rag",
}


def _validate_lab(lab: str) -> str:
    """Reject unknown slugs before creating clients or indexes."""
    if lab not in LABS:
        raise typer.BadParameter(f"Choose one of: {', '.join(sorted(LABS))}")
    return lab


def _resources(settings: Settings, lab: str, dimensions: int | None, rebuild: bool):
    """Prepare text chunks and a compatible vector store for the chosen lab.

    This is the ingestion path used by both ingest and ask. ask can therefore
    embed/index the corpus on first use; an existing compatible index is reused.
    """
    documents = load_documents(settings.data_dir)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    # This object calls Azure to create vectors; it is not the vectors themselves.
    embeddings_client = create_embeddings(settings, dimensions)
    spec = IndexSpec(
        lab=lab,
        corpus_fingerprint=corpus_fingerprint(
            documents, settings.chunk_size, settings.chunk_overlap
        ),
        embedding_model=settings.azure_openai_embedding_model,
        dimensions=dimensions,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    store, created = open_or_create_store(
        embeddings_client=embeddings_client,
        chunks=chunks,
        index_root=settings.index_dir,
        spec=spec,
        rebuild=rebuild,
    )
    return chunks, store, created


def _pipeline(settings: Settings, lab: str, dimensions: int | None, rebuild: bool):
    """Factory: choose a lab class and inject its store, model, and settings.

    The same interface (ask -> RagResult) lets CLI output work across patterns.
    The dimensions lab's ask command uses BasicRag; compare-dimensions runs its
    retrieval-only experiment instead.
    """
    chunks, store, _ = _resources(settings, lab, dimensions, rebuild)
    chat = create_chat_model(settings)
    if lab in {"basic-rag", "vector-dimensions"}:
        return BasicRag(store, chat, settings.top_k)
    if lab == "hybrid-search":
        return HybridRag(store, chunks, chat, settings.top_k)
    if lab == "reranking":
        return RerankedRag(store, chat, CrossEncoderReranker(), settings.top_k)
    if lab == "multi-query":
        return MultiQueryRag(store, chat, settings.top_k)
    if lab == "conversational-rag":
        return ConversationalRag(store, chat, settings.top_k)
    return CorrectiveRag(store, chat, settings.top_k)


@app.command()
def ingest(
    lab: Annotated[str, typer.Option(help="Lab slug.")] = "basic-rag",
    dimensions: Annotated[int | None, typer.Option(help="Embedding vector dimensions.")] = None,
    rebuild: Annotated[bool, typer.Option(help="Replace an existing compatible index.")] = False,
) -> None:
    """Chunk the corpus and create or reuse a lab-specific Chroma index."""
    lab = _validate_lab(lab)
    settings = get_settings()
    chunks, store, created = _resources(settings, lab, dimensions, rebuild)
    typer.echo(
        json.dumps(
            {
                "lab": lab,
                "chunks": len(chunks),
                "vectors": store._collection.count(),
                "created": created,
            },
            indent=2,
        )
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question about the sample engineering system.")],
    lab: Annotated[str, typer.Option(help="Lab slug.")] = "basic-rag",
    dimensions: Annotated[int | None, typer.Option(help="Embedding vector dimensions.")] = None,
    session_id: Annotated[str, typer.Option(help="Conversation session identifier.")] = "default",
) -> None:
    """Run one question through a selected lab.

    Each CLI invocation creates a new object. session_id does not persist history
    across shell commands; use one ConversationalRag object for multiple turns.
    """
    lab = _validate_lab(lab)
    runner = _pipeline(get_settings(), lab, dimensions, False)
    result = (
        runner.ask(question, session_id=session_id)
        if isinstance(runner, ConversationalRag)
        else runner.ask(question)
    )
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command()
def evaluate(
    lab: Annotated[str, typer.Option(help="Lab slug.")] = "basic-rag",
    output: Annotated[Path, typer.Option(help="JSON report path.")] = Path(
        "artifacts/evaluation.json"
    ),
    dimensions: Annotated[int | None, typer.Option(help="Embedding vector dimensions.")] = None,
    judge: Annotated[
        bool, typer.Option(help="Add estimated LLM correctness/grounding scores.")
    ] = False,
) -> None:
    """Evaluate retrieval, citations, and latency on the labeled questions.

    Conversation cases replay prior turns in isolated sessions. Other labs omit
    those cases because a follow-up alone does not identify the intended question.
    The optional judge makes additional Azure calls outside the lab usage totals.
    """
    lab = _validate_lab(lab)
    settings = get_settings()
    runner = _pipeline(settings, lab, dimensions, False)
    cases = load_cases(settings.eval_file)
    judge_model = create_chat_model(settings) if judge else None
    if isinstance(runner, ConversationalRag):
        # isinstance checks which Python class this runner belongs to.
        rows = []
        for case in cases:
            session_id = f"evaluation-{case.id}"
            for prior_turn in case.conversation or []:
                runner.ask(prior_turn, session_id=session_id)
            result = runner.ask(case.question, session_id=session_id)
            row = {"case": case, "result": result, "metrics": retrieval_metrics(result, case)}
            if judge_model:
                row["judge_estimate"] = judge_result(judge_model, result, case)
            rows.append(row)
        report = {
            "summary": summarize_metrics(rows),
            "cases": [
                {
                    "case": asdict(row["case"]),
                    "result": row["result"].to_dict(),
                    "metrics": row["metrics"],
                    **(
                        {"judge_estimate": row["judge_estimate"]} if "judge_estimate" in row else {}
                    ),
                }
                for row in rows
            ],
        }
    else:
        report = evaluate_cases(
            runner.ask,
            cases,
            judge=(
                lambda result, case: judge_result(judge_model, result, case) if judge_model else {}
            )
            if judge
            else None,
        )
    write_report(report, output)
    typer.echo(json.dumps(report["summary"], indent=2))
    typer.echo(f"Full report: {output}")


@app.command("compare-dimensions")
def compare_dimensions(
    dimensions: Annotated[
        str, typer.Option(help="Comma-separated dimensions.")
    ] = "256,512,1024,1536",
    repetitions: Annotated[int, typer.Option(min=1, help="Timed searches per question.")] = 5,
    output_dir: Annotated[Path, typer.Option(help="Report and chart directory.")] = Path(
        "artifacts/dimensions"
    ),
    rebuild: Annotated[bool, typer.Option(help="Recreate every dimension index.")] = False,
) -> None:
    """Hold the corpus and retrieval settings fixed while changing vector dimensions."""
    settings = get_settings()
    values = [int(value.strip()) for value in dimensions.split(",")]
    documents = load_documents(settings.data_dir)
    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)
    fingerprint = corpus_fingerprint(documents, settings.chunk_size, settings.chunk_overlap)
    cases = [
        case for case in standalone_cases(load_cases(settings.eval_file)) if case.expected_documents
    ]

    def store_factory(dimension: int, embeddings_client: Any):
        """Closure: use the enclosing function's corpus/settings for every dimension."""
        spec = IndexSpec(
            lab="vector-dimensions",
            corpus_fingerprint=fingerprint,
            embedding_model=settings.azure_openai_embedding_model,
            dimensions=dimension,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        timings: dict[str, Any] = {}
        store, _ = open_or_create_store(
            embeddings_client=embeddings_client,
            chunks=chunks,
            index_root=settings.index_dir,
            spec=spec,
            rebuild=rebuild,
            timings=timings,
        )
        return store, settings.index_dir / spec.collection_name, timings

    experiment = DimensionExperiment(
        dimensions=values,
        embedding_factory=lambda dimension: create_embeddings(settings, dimension),
        store_factory=store_factory,
        queries=[(case.question, case.expected_documents) for case in cases],
        top_k=settings.top_k,
        repetitions=repetitions,
    )
    report = experiment.run()
    experiment.export(report, output_dir)
    typer.echo(json.dumps(report, indent=2))
    typer.echo(f"Reports and chart: {output_dir}")


# This guard runs the CLI for 'python -m rag_lab.cli', but not when importing it.
if __name__ == "__main__":
    app()
