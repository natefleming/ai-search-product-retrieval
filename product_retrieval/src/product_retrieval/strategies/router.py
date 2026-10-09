"""Query router: route by query shape, retrieve per route, optionally rerank (fast router = no reranker)."""

from __future__ import annotations

import mlflow
from mlflow.entities import SpanType

from product_retrieval.backends import SearchBackend, SearchRequest
from product_retrieval.documents import Filter, RetrievalResult
from product_retrieval.rerankers import Reranker, rerank_safely
from product_retrieval.understanding import CatalogVocabulary, analyze


class QueryRouter:
    """identifier → exact SKU/UPC filter; "not <brand>" → `brand NOT` filter; else cleaned HYBRID; then optional rerank.

    Query understanding is deterministic (regex + catalog vocabulary), so routing adds no model call.
    """

    def __init__(
        self,
        backend: SearchBackend,
        vocabulary: CatalogVocabulary,
        reranker: Reranker | None = None,
        candidates: int = 12,
        k: int = 10,
        name: str | None = None,
        exclusions: bool = True,
        clean_queries: bool = True,
    ) -> None:
        """`exclusions=False` / `clean_queries=False` switch off those stages (ablations); identifier routing is always on."""
        self.backend, self.vocab, self.reranker, self.candidates, self.k = backend, vocabulary, reranker, candidates, k
        self.exclusions, self.clean_queries = exclusions, clean_queries
        self.name = name or ("router_quality" if reranker else "router_fast")

    def _exact(self, code: str) -> list:
        catalog = self.backend.catalog
        column = catalog.id if len(code) == 8 or not catalog.upc else catalog.upc
        products = self.backend.search(SearchRequest(text=code, k=self.k, filters=[Filter(column=column, value=code)])).products
        return products or self.backend.search(SearchRequest(text=code, k=self.k, query_type="FULL_TEXT")).products

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        analysis = analyze(query, self.vocab)
        if analysis.route == "identifier":
            return RetrievalResult(products=self._exact(analysis.code), strategy=self.name, route=analysis.route)

        exclusion = self.exclusions and analysis.route == "exclusion"
        filters = [Filter(column=self.backend.catalog.brand, op="ne", value=analysis.excluded_brand)] if exclusion else []
        text = analysis.search_text if self.clean_queries or exclusion else query
        products = self.backend.search(SearchRequest(text=text, k=self.candidates, filters=filters)).products
        outcome = rerank_safely(self.reranker, query, products)  # reranker sees the original request, constraints included
        return RetrievalResult(
            products=outcome.products[: self.k], strategy=self.name, route="exclusion" if exclusion else "general",
            filters=filters, scores=outcome.scores, rerank_error=outcome.error, diagnostics={"search_text": text},
        )
