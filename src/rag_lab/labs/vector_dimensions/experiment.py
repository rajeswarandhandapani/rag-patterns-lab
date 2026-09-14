"""Lab 2: vary embedding width while holding the retrieval experiment fixed.

This benchmark retrieves evidence only; it does not generate chat answers.
Factories are functions supplied by the CLI to create clients/indexes for each width.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class DimensionExperiment:
    """A controlled comparison: only embedding output dimensions change."""

    def __init__(
        self,
        dimensions: list[int],
        embedding_factory: Callable[[int], Embeddings],
        store_factory: Callable[[int, Embeddings], tuple[Any, Path, dict[str, Any]]],
        queries: list[tuple[str, list[str]]],
        top_k: int = 4,
        repetitions: int = 5,
    ) -> None:
        self.dimensions = dimensions
        self.embedding_factory = embedding_factory
        self.store_factory = store_factory
        self.queries = queries
        self.top_k = top_k
        self.repetitions = repetitions

    @staticmethod
    def _directory_size(path: Path) -> int:
        """Sum actual file sizes, which include text, metadata, and index overhead."""
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def run(self) -> dict[str, Any]:
        """Return one row per dimension using the same labeled questions.

        Warmup reduces first-query initialization effects. Repeated measurements
        still include network variability: small timing differences may be noise.
        Each query vector must match the dimension of its own collection.
        """
        results = []
        for dimensions in self.dimensions:
            embeddings = self.embedding_factory(dimensions)
            store, index_path, index_timings = self.store_factory(dimensions, embeddings)
            embed_latencies: list[float] = []
            search_latencies: list[float] = []
            recalls: list[float] = []
            reciprocal_ranks: list[float] = []
            if self.queries:
                warm_vector = embeddings.embed_query(self.queries[0][0])
                store.similarity_search_by_vector(warm_vector, k=self.top_k)
            for query, expected_documents in self.queries:
                returned: list[Document] = []
                for _ in range(self.repetitions):
                    started = perf_counter()
                    vector = embeddings.embed_query(query)
                    embed_latencies.append((perf_counter() - started) * 1000)
                    started = perf_counter()
                    # Supply an already embedded vector so this interval excludes
                    # the Azure embedding request measured immediately above.
                    returned = store.similarity_search_by_vector(vector, k=self.top_k)
                    search_latencies.append((perf_counter() - started) * 1000)
                ids = [str(document.metadata["document_id"]) for document in returned]
                expected = set(expected_documents)
                recalls.append(len(expected.intersection(ids)) / len(expected) if expected else 1.0)
                ranks = [ids.index(doc) + 1 for doc in expected if doc in ids]
                reciprocal_ranks.append(1 / min(ranks) if ranks else 0.0)
            results.append(
                {
                    "dimensions": dimensions,
                    "recall_at_k": mean(recalls),
                    "reciprocal_rank": mean(reciprocal_ranks),
                    **index_timings,
                    "query_embedding_ms": mean(embed_latencies),
                    "vector_search_ms": mean(search_latencies),
                    # Estimate float32 vector payload: count * width * 4 bytes.
                    # This excludes all overhead, unlike persisted_index_bytes.
                    "estimated_raw_vector_bytes": store._collection.count() * dimensions * 4,
                    "persisted_index_bytes": self._directory_size(index_path),
                }
            )
        return {"top_k": self.top_k, "repetitions": self.repetitions, "results": results}

    @staticmethod
    def export(report: dict[str, Any], output_dir: Path) -> None:
        """Write structured JSON, spreadsheet-friendly CSV, and a PNG chart.

        @staticmethod needs neither self nor instance state; all inputs are explicit.
        A 'with' block closes the CSV file even when writing raises an exception.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dimension-results.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        rows = report["results"]
        with (output_dir / "dimension-results.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        DimensionExperiment._plot(rows, output_dir)

    @staticmethod
    def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
        """Plot retrieval quality and total on-disk storage against vector width."""
        import matplotlib.pyplot as plt

        dimensions = [row["dimensions"] for row in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(dimensions, [row["recall_at_k"] for row in rows], marker="o", label="Recall@k")
        axes[0].plot(dimensions, [row["reciprocal_rank"] for row in rows], marker="o", label="MRR")
        axes[0].set(xlabel="Dimensions", ylabel="Score", title="Retrieval quality")
        axes[0].legend()
        axes[1].plot(
            dimensions,
            [row["persisted_index_bytes"] / 1024 for row in rows],
            marker="o",
        )
        axes[1].set(xlabel="Dimensions", ylabel="KiB", title="Persisted index size")
        fig.tight_layout()
        fig.savefig(output_dir / "dimension-comparison.png", dpi=160)
        plt.close(fig)
