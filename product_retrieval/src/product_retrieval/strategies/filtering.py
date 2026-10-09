"""LLM-planned metadata filters (dynamic filtering), optionally guarded against the catalog vocabulary."""

from __future__ import annotations

from typing import Any, Callable

import mlflow
from databricks_langchain import VectorSearchRetrieverTool
from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities import SpanType

from product_retrieval._llm import chat
from product_retrieval._retry import with_backoff
from product_retrieval.backends import SearchBackend, SearchRequest
from product_retrieval.config import CatalogSchema, RetrievalConfig
from product_retrieval.documents import Filter, RetrievalResult
from product_retrieval.understanding import CatalogVocabulary

_KEY_OPS = {"NOT": "ne", "<": "lt", "<=": "lte", ">": "gt", ">=": "gte", "LIKE": "like"}


def parse_filter_items(items: list[dict[str, Any]]) -> list[Filter]:
    """VectorSearchRetrieverTool FilterItems ({"key": "brand_name NOT", "value": ...}) → neutral Filters."""
    filters: list[Filter] = []
    for item in items:
        parts = str(item.get("key", "")).split()
        if not parts or "OR" in parts or parts[1:] == ["NOT", "LIKE"]:
            continue  # OR / NOT LIKE expressions are not used by these strategies
        op = _KEY_OPS.get(" ".join(parts[1:]), "eq") if len(parts) > 1 else "eq"
        value = item.get("value")
        if isinstance(value, list) and op in ("eq", "ne"):
            op = "in" if op == "eq" else "not_in"
        filters.append(Filter(column=parts[0], op=op, value=value))
    return filters


def guard_filters(filters: list[Filter], vocab: CatalogVocabulary, brand_column: str, category_column: str) -> list[Filter]:
    """Keep only brand/category filters whose values exist in the catalog, in catalog spelling (filters are exact-match)."""
    clean: list[Filter] = []
    for f in filters:
        values = f.value if isinstance(f.value, list) else [f.value]
        if f.column == brand_column:
            values = [b for v in values if (b := vocab.match_brand(str(v)))]
        elif f.column == category_column:
            values = [str(v).upper() for v in values if str(v).upper() in vocab.categories]
        else:
            values = []
        if values:
            op = f.op if len(values) == 1 or f.op in ("in", "not_in") else ("in" if f.op == "eq" else "not_in")
            clean.append(Filter(column=f.column, op=op, value=values if len(values) > 1 else values[0]))
    return clean


def planner_prompt(vocab: CatalogVocabulary, catalog: CatalogSchema) -> str:
    """Filter-planning instructions. Wording matters: earlier evaluation showed that inviting category filters makes the LLM
    apply them twice as often, which removes the right product on descriptive/attribute queries."""
    never = ", ".join(c for c in [catalog.id, catalog.upc, catalog.name, catalog.description] if c)
    return f"""You turn hardware-store shopper requests into exactly one call to the product search tool.
- `query`: the product the shopper wants (type + attributes). Keep attribute words; drop brand-exclusion phrases.
- `filters`: only for constraints the shopper states explicitly:
  * required brand  -> {{"key": "{catalog.brand}", "value": "<BRAND IN UPPERCASE>"}}
  * excluded brand  -> {{"key": "{catalog.brand} NOT", "value": "<BRAND IN UPPERCASE>"}}
  * {catalog.category} -> {{"key": "{catalog.category}", "value": "<EXACT VALUE>"}} only if you are certain which value below applies.
- Never filter on {never}. If there is no explicit constraint, pass no filters.

{catalog.category} values: {"; ".join(vocab.categories)}"""


class LLMFilterRetriever:
    """An LLM fills the typed `filters` argument of `VectorSearchRetrieverTool(dynamic_filter=True)`; the filters run on AI Search.

    guarded=True maps values onto real catalog brands/categories and falls back to an unfiltered search when filters empty the
    result (unguarded LLM filters returned nothing on ~4-5% of eval queries).
    """

    def __init__(
        self,
        config: RetrievalConfig,
        backend: SearchBackend,
        vocabulary: CatalogVocabulary,
        guarded: bool = True,
        candidates: int | None = None,
        name: str | None = None,
        column_selector: Callable[[str], list[str]] | None = None,
    ) -> None:
        """`column_selector` optionally enables the server-side reranker on request-targeted columns."""
        self.backend, self.vocab, self.guarded, self.column_selector = backend, vocabulary, guarded, column_selector
        self.k, self.candidates = config.k, candidates or config.k
        catalog = config.catalog
        self.brand_col, self.category_col = catalog.brand, catalog.category
        self.planner_tool = VectorSearchRetrieverTool(
            index_name=config.index_name, num_results=self.candidates, query_type="HYBRID", columns=catalog.columns(),
            dynamic_filter=True, tool_name="product_search",
            tool_description="Search the product catalog. Returns products ranked by relevance to the shopper request.",
        )
        self.llm = chat(config.llm_endpoint).bind_tools([self.planner_tool], tool_choice="required")
        self.system = planner_prompt(vocabulary, catalog)
        self.name = name or ("llm_filter_guarded" if guarded else "llm_filter")

    @mlflow.trace(span_type=SpanType.CHAIN)
    def plan(self, query: str) -> tuple[str, list[Filter]]:
        msg = with_backoff(lambda: self.llm.invoke([SystemMessage(self.system), HumanMessage(query)]))
        if not msg.tool_calls:
            return query, []
        args = msg.tool_calls[0]["args"]
        return args.get("query") or query, parse_filter_items([dict(f) for f in args.get("filters") or []])

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        text, filters = self.plan(query)
        if self.guarded:
            filters = guard_filters(filters, self.vocab, self.brand_col, self.category_col)
        columns = (self.column_selector(query) if self.column_selector else None) or None
        products = self.backend.search(SearchRequest(text=text, k=self.candidates, filters=filters, rerank_columns=columns)).products
        if not products and filters and self.guarded:
            filters, products = [], self.backend.search(SearchRequest(text=text, k=self.candidates, rerank_columns=columns)).products
        return RetrievalResult(products=products[: self.k], strategy=self.name, filters=filters,
                               diagnostics={"search_text": text, "rerank_columns": columns or []})
