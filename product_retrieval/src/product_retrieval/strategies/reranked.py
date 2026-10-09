"""Instructed retrieval as composition: any candidate retriever + any post-retrieval reranker."""

from __future__ import annotations

import mlflow
from mlflow.entities import SpanType

from product_retrieval.documents import RetrievalResult
from product_retrieval.rerankers import Reranker, rerank_safely
from product_retrieval.strategies.base import Retriever


class RerankedRetriever:
    """Retrieve a candidate pool (e.g. HYBRID top 25, or guarded-filter results), rerank it, return the top k."""

    def __init__(self, candidates: Retriever, reranker: Reranker, k: int = 10, name: str | None = None) -> None:
        self.candidates, self.reranker, self.k = candidates, reranker, k
        self.name = name or f"{candidates.name}+{reranker.name}"

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        pool = self.candidates(query)
        outcome = rerank_safely(self.reranker, query, pool.products)
        return pool.model_copy(update={
            "products": outcome.products[: self.k], "scores": outcome.scores, "rerank_error": outcome.error, "strategy": self.name,
        })
