"""Plain retrieval: the request goes straight to AI Search (ANN, FULL_TEXT or HYBRID), optionally server-side reranked."""

from __future__ import annotations

from typing import Callable

import mlflow
from mlflow.entities import SpanType

from product_retrieval.backends import QueryType, SearchBackend, SearchRequest
from product_retrieval.documents import RetrievalResult


class PlainRetriever:
    def __init__(
        self,
        backend: SearchBackend,
        query_type: QueryType = "HYBRID",
        k: int = 10,
        rerank_columns: list[str] | None = None,
        candidates: int | None = None,
        query_columns: list[str] | None = None,
        name: str | None = None,
        column_selector: Callable[[str], list[str]] | None = None,
    ) -> None:
        """`rerank_columns` enables the Databricks reranker (column order matters: first ~2,000 chars are used);
        `candidates` (> k) retrieves a deeper pool to rerank before truncating to k; `column_selector` picks
        `rerank_columns` per request (see `product_retrieval.column_selection.RerankColumnSelector`)."""
        self.backend, self.query_type, self.k = backend, query_type, k
        self.rerank_columns, self.candidates, self.query_columns = rerank_columns, candidates or k, query_columns
        self.column_selector = column_selector
        self.name = name or (f"{query_type.lower()}_rerank" if rerank_columns or column_selector else query_type.lower())

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        columns = self.column_selector(query) if self.column_selector else self.rerank_columns
        resp = self.backend.search(
            SearchRequest(text=query, k=self.candidates, query_type=self.query_type,
                          rerank_columns=columns or None, query_columns=self.query_columns)
        )
        return RetrievalResult(products=resp.products[: self.k], strategy=self.name, diagnostics={"rerank_columns": columns or []})
