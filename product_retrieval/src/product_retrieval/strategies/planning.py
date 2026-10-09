"""LLM structured query plans: hard constraints become filters, soft constraints guide the reranker.

Industry pattern (e.g. Instacart's query-understanding engine, Amazon hint-augmented reranking): parse the request into a typed
plan, validate it against the catalog, cache it for head queries, and never let an *inferred* preference become a hard filter.
"""

from __future__ import annotations

import functools
import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from product_retrieval._llm import chat
from product_retrieval._retry import with_backoff
from product_retrieval.backends import SearchBackend, SearchRequest
from product_retrieval.documents import Filter, Product, RetrievalResult
from product_retrieval.rerankers import Reranker, rerank_safely
from product_retrieval.understanding import CODE_RE, CatalogVocabulary


class QueryPlan(BaseModel):
    """Typed interpretation of a shopper request (filled by the LLM through tool calling)."""

    semantic_query: str = Field(description="The product being sought: product type plus key attributes, no brand-exclusion words")
    product_type: str = Field(description="Short product type, e.g. 'impact driver', 'LED bulb'")
    required_brands: list[str] = Field(default_factory=list, description="Brands the shopper explicitly requires")
    excluded_brands: list[str] = Field(default_factory=list, description="Brands the shopper explicitly does NOT want")
    attributes: list[str] = Field(default_factory=list, description="Explicit specs: sizes, voltage, wattage, color, pack count, kit/tool-only")
    preferences: list[str] = Field(default_factory=list, description="Soft, inferred preferences (never hard requirements)")


PLANNER_PROMPT = """You parse hardware-store shopper requests into a search plan by calling `QueryPlan` exactly once.
Only put a brand in required_brands / excluded_brands when the shopper explicitly names it. Copy specs (numbers with units,
colors, pack counts, kit vs tool-only) into attributes verbatim. Put anything inferred or vague into preferences."""


class QueryPlanner:
    def __init__(self, llm_endpoint: str, vocabulary: CatalogVocabulary, cache_size: int = 10_000) -> None:
        self.vocab = vocabulary
        self.llm = chat(llm_endpoint).bind_tools([QueryPlan], tool_choice="required")
        self._cached = functools.lru_cache(maxsize=cache_size)(self._plan)

    def _plan(self, query: str) -> QueryPlan:
        msg = with_backoff(lambda: self.llm.invoke([SystemMessage(PLANNER_PROMPT), HumanMessage(query)]))
        plan = QueryPlan(**msg.tool_calls[0]["args"]) if msg.tool_calls else QueryPlan(semantic_query=query, product_type="")
        # Validate brands against the catalog: unknown brands are dropped, never turned into filters
        plan.required_brands = [b for v in plan.required_brands if (b := self.vocab.match_brand(v))]
        plan.excluded_brands = [b for v in plan.excluded_brands if (b := self.vocab.match_brand(v))]
        return plan

    @mlflow.trace(name="query_plan", span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> QueryPlan:
        return self._cached(query.strip().lower())


def rerank_request(query: str, plan: QueryPlan) -> str:
    """The original request plus the plan's explicit requirements, so the reranker enforces them."""
    notes = []
    if plan.required_brands:
        notes.append(f"required brand: {', '.join(plan.required_brands)}")
    if plan.attributes:
        notes.append(f"required specs: {'; '.join(plan.attributes)}")
    return f"{query}\n[{' | '.join(notes)}]" if notes else query


class PlanRouter:
    """identifier → exact lookup (no LLM); otherwise LLM plan → hard filters (excluded brands only) → HYBRID → plan-aware rerank.

    Required brands and attributes are passed to the reranker as explicit requirements instead of filters: earlier evaluation
    showed hard brand/category filters remove the right product too often when the guess is wrong.
    """

    def __init__(
        self,
        backend: SearchBackend,
        planner: QueryPlanner,
        reranker: Reranker | None = None,
        candidates: int = 12,
        k: int = 10,
        name: str = "plan_router",
    ) -> None:
        self.backend, self.planner, self.reranker = backend, planner, reranker
        self.candidates, self.k, self.name = candidates, k, name

    def _exact(self, code: str) -> list[Product]:
        catalog = self.backend.catalog
        column = catalog.id if len(code) == 8 or not catalog.upc else catalog.upc
        products = self.backend.search(SearchRequest(text=code, k=self.k, filters=[Filter(column=column, value=code)])).products
        return products or self.backend.search(SearchRequest(text=code, k=self.k, query_type="FULL_TEXT")).products

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        code = CODE_RE.search(query)
        if code:
            return RetrievalResult(products=self._exact(code.group(1)), strategy=self.name, route="identifier")
        plan = self.planner(query)
        filters = [Filter(column=self.backend.catalog.brand, op="not_in" if len(plan.excluded_brands) > 1 else "ne",
                          value=plan.excluded_brands if len(plan.excluded_brands) > 1 else plan.excluded_brands[0])] if plan.excluded_brands else []
        text = " ".join([plan.semantic_query, *plan.attributes]).strip() or query
        products = self.backend.search(SearchRequest(text=text, k=self.candidates, filters=filters)).products
        if not products and filters:
            filters, products = [], self.backend.search(SearchRequest(text=text, k=self.candidates)).products
        outcome = rerank_safely(self.reranker, rerank_request(query, plan), products)
        return RetrievalResult(
            products=outcome.products[: self.k], strategy=self.name, route="exclusion" if filters else "general", filters=filters,
            scores=outcome.scores, rerank_error=outcome.error, diagnostics={"plan": plan.model_dump(), "search_text": text},
        )
