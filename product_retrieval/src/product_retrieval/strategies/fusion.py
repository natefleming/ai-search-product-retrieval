"""Weighted Reciprocal Rank Fusion over several retrievers (e.g. ANN + FULL_TEXT, or product index + pseudo-query index)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextvars

import mlflow
from mlflow.entities import SpanType

from product_retrieval.documents import Product, RetrievalResult
from product_retrieval.strategies.base import Retriever


def rrf(rankings: list[tuple[list[Product], float]], rrf_k: int = 60) -> list[tuple[Product, float]]:
    """score(d) = Σ_lists weight / (rrf_k + rank). Products are identified by `id`; first occurrence wins."""
    scores: dict[str, float] = {}
    first: dict[str, Product] = {}
    for products, weight in rankings:
        for rank, p in enumerate(products, start=1):
            scores[p.id] = scores.get(p.id, 0.0) + weight / (rrf_k + rank)
            first.setdefault(p.id, p)
    return sorted(((first[i], s) for i, s in scores.items()), key=lambda t: -t[1])


class FusionRetriever:
    def __init__(self, retrievers: list[tuple[Retriever, float]], k: int = 10, rrf_k: int = 60, name: str | None = None) -> None:
        self.retrievers, self.k, self.rrf_k = retrievers, k, rrf_k
        self.name = name or "rrf(" + ",".join(f"{r.name}:{w:g}" for r, w in retrievers) + ")"

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as pool:  # copy_context keeps child spans in this trace
            futures = [pool.submit(contextvars.copy_context().run, r, query) for r, _ in self.retrievers]
            results = [f.result() for f in futures]
        fused = rrf([(res.products, w) for res, (_, w) in zip(results, self.retrievers)], self.rrf_k)
        return RetrievalResult(products=[p for p, _ in fused[: self.k]], strategy=self.name,
                               scores={p.id: round(s, 6) for p, s in fused[: self.k]})
