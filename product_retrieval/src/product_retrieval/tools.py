"""LangChain tool factories: the public API of the package.

Every factory returns a `StructuredTool` with the same contract:
  input  {"query": "<the shopper's request, verbatim>"}
  output JSON list of {"page_content": ..., "metadata": {id, name, brand, category, upc, route, score, ...}}

Factories accept a `RetrievalConfig` (or a plain dict, for config-driven frameworks such as dao-ai `type: factory`).
"""

from __future__ import annotations

from typing import Any, Literal

from databricks_langchain import VectorSearchRetrieverTool
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from product_retrieval.backends import QueryType, SearchBackend, build_backend
from product_retrieval.column_selection import RerankColumnSelector
from product_retrieval.config import RetrievalConfig
from product_retrieval.rerankers import AIDecideListwiseReranker, AIDecideReranker, Reranker, ServingEndpointReranker
from product_retrieval.strategies.ai_router import AIDecideRouter
from product_retrieval.strategies.base import Retriever
from product_retrieval.strategies.facets import FacetGuidedRetriever
from product_retrieval.strategies.filtering import LLMFilterRetriever
from product_retrieval.strategies.fusion import FusionRetriever
from product_retrieval.strategies.plain import PlainRetriever
from product_retrieval.strategies.reranked import RerankedRetriever
from product_retrieval.strategies.router import QueryRouter
from product_retrieval.understanding import CatalogVocabulary

DEFAULT_DESCRIPTION: str = (
    "Search the product catalog. Pass the shopper's full request verbatim, including brands to exclude, specs, SKUs or UPCs; "
    "returns the best-matching products ranked by fit."
)


class ProductSearchInput(BaseModel):
    query: str = Field(description="The shopper's request in their own words. Do not strip constraints; the tool parses them.")


def as_tool(retriever: Retriever, name: str = "product_search", description: str = DEFAULT_DESCRIPTION) -> StructuredTool:
    """Wrap any retriever as a LangChain tool."""

    def _run(query: str) -> str:
        return retriever(query).to_tool_output()

    return StructuredTool.from_function(func=_run, name=name, description=description, args_schema=ProductSearchInput)


# ---------------------------------------------------------------------------------------------------------------------
# Shared resolution helpers
# ---------------------------------------------------------------------------------------------------------------------


def _config(config: RetrievalConfig | dict[str, Any]) -> RetrievalConfig:
    return config if isinstance(config, RetrievalConfig) else RetrievalConfig.model_validate(config)


def _vocabulary(
    config: RetrievalConfig, vocabulary: CatalogVocabulary | None, vocabulary_table: str | None, warehouse_id: str | None
) -> CatalogVocabulary:
    if vocabulary is not None:
        return vocabulary
    if not (vocabulary_table and warehouse_id):
        raise ValueError("Pass `vocabulary`, or `vocabulary_table` + `warehouse_id` to load it")
    return CatalogVocabulary.from_warehouse(vocabulary_table, warehouse_id, config.catalog.brand, config.catalog.category)


def make_reranker(
    kind: Literal["none", "ai_decide_noul", "ai_decide_score", "ai_decide_listwise", "cross_encoder"] = "ai_decide_noul",
    endpoint_name: str | None = None,
    details_chars: int = 300,
) -> Reranker | None:
    if kind == "none":
        return None
    if kind == "cross_encoder":
        if not endpoint_name:
            raise ValueError("cross_encoder reranker needs `endpoint_name` (a Model Serving endpoint)")
        return ServingEndpointReranker(endpoint_name)
    if kind == "ai_decide_listwise":
        return AIDecideListwiseReranker(base=AIDecideReranker("noul", details_chars=details_chars), details_chars=details_chars)
    return AIDecideReranker("score" if kind == "ai_decide_score" else "noul", details_chars=details_chars)


# ---------------------------------------------------------------------------------------------------------------------
# Factories (one per strategy)
# ---------------------------------------------------------------------------------------------------------------------


def create_search_tool(
    config: RetrievalConfig | dict[str, Any],
    query_type: QueryType = "HYBRID",
    rerank_columns: list[str] | None = None,
    candidates: int | None = None,
    query_columns: list[str] | None = None,
    name: str = "product_search",
    backend: SearchBackend | None = None,
) -> StructuredTool:
    """Plain AI Search (ANN / FULL_TEXT / HYBRID), optionally with the Databricks server-side reranker on `rerank_columns`."""
    cfg = _config(config)
    retriever = PlainRetriever(backend or build_backend(cfg), query_type, cfg.k, rerank_columns, candidates, query_columns)
    return as_tool(retriever, name)


def create_targeted_rerank_tool(
    config: RetrievalConfig | dict[str, Any], candidates: int = 50, name: str = "product_search", backend: SearchBackend | None = None
) -> StructuredTool:
    """HYBRID + Databricks server-side reranker on columns chosen per request (one ai_decide choice call)."""
    cfg = _config(config)
    be = backend or build_backend(cfg)
    retriever = PlainRetriever(be, "HYBRID", cfg.k, candidates=candidates, column_selector=RerankColumnSelector(be.catalog))
    return as_tool(retriever, name)


def create_dynamic_filter_tool(config: RetrievalConfig | dict[str, Any], name: str = "product_search") -> BaseTool:
    """`VectorSearchRetrieverTool(dynamic_filter=True)`: the *calling agent's* LLM writes the metadata filters."""
    cfg = _config(config)
    return VectorSearchRetrieverTool(
        index_name=cfg.index_name, num_results=cfg.k, query_type="HYBRID", columns=cfg.catalog.columns(),
        dynamic_filter=True, tool_name=name, tool_description=DEFAULT_DESCRIPTION,
    )


def create_guarded_filter_tool(
    config: RetrievalConfig | dict[str, Any],
    vocabulary: CatalogVocabulary | None = None,
    vocabulary_table: str | None = None,
    warehouse_id: str | None = None,
    guarded: bool = True,
    reranker: str = "none",
    targeted_server_rerank: bool = False,
    candidates: int = 25,
    name: str = "product_search",
    backend: SearchBackend | None = None,
) -> StructuredTool:
    """An internal LLM plans filters, values are validated against the catalog, empty results fall back.

    Optional ranking: `reranker` (post-retrieval, e.g. "ai_decide_noul") and/or `targeted_server_rerank` (Databricks reranker on
    request-targeted columns)."""
    cfg = _config(config)
    vocab = _vocabulary(cfg, vocabulary, vocabulary_table, warehouse_id)
    be = backend or build_backend(cfg)
    rr = make_reranker(reranker)  # type: ignore[arg-type]
    selector = RerankColumnSelector(be.catalog) if targeted_server_rerank else None
    deep = rr is not None or targeted_server_rerank
    retriever: Retriever = LLMFilterRetriever(cfg, be, vocab, guarded, candidates if deep else cfg.k, column_selector=selector)
    if rr:
        retriever = RerankedRetriever(retriever, rr, cfg.k)
    return as_tool(retriever, name)


def create_instructed_tool(
    config: RetrievalConfig | dict[str, Any],
    reranker: str = "ai_decide_noul",
    candidates: int = 25,
    endpoint_name: str | None = None,
    name: str = "product_search",
    backend: SearchBackend | None = None,
) -> StructuredTool:
    """HYBRID candidates → instruction-aware rerank (ai_decide pointwise/listwise, or a served cross-encoder)."""
    cfg = _config(config)
    pool = PlainRetriever(backend or build_backend(cfg), "HYBRID", k=candidates)
    rr = make_reranker(reranker, endpoint_name)  # type: ignore[arg-type]
    return as_tool(RerankedRetriever(pool, rr, cfg.k) if rr else pool, name)


def create_facet_guided_tool(
    config: RetrievalConfig | dict[str, Any], name: str = "product_search", backend: SearchBackend | None = None
) -> StructuredTool:
    cfg = _config(config)
    return as_tool(FacetGuidedRetriever(backend or build_backend(cfg), cfg.k), name)


def create_router_tool(
    config: RetrievalConfig | dict[str, Any],
    vocabulary: CatalogVocabulary | None = None,
    vocabulary_table: str | None = None,
    warehouse_id: str | None = None,
    reranker: str = "ai_decide_noul",
    candidates: int = 12,
    endpoint_name: str | None = None,
    name: str = "product_search",
    backend: SearchBackend | None = None,
    classifier: Literal["rules", "ai_decide"] = "rules",
    exclusions: bool = True,
    clean_queries: bool = True,
) -> StructuredTool:
    """The recommended tool. reranker="none" → fast router (~0.3 s); "ai_decide_noul" → quality router (~1.3 s).

    classifier="rules" routes with regex + catalog vocabulary (needs a vocabulary); "ai_decide" routes with one ai_decide call
    over the query's brand facets (no vocabulary needed; handles paraphrased exclusions). `exclusions` / `clean_queries` switch
    off those rules-router stages (the 09 ablations).
    """
    cfg = _config(config)
    rr = make_reranker(reranker, endpoint_name)  # type: ignore[arg-type]
    if classifier == "ai_decide":
        return as_tool(AIDecideRouter(backend or build_backend(cfg), rr, candidates, cfg.k), name)
    vocab = _vocabulary(cfg, vocabulary, vocabulary_table, warehouse_id)
    router = QueryRouter(backend or build_backend(cfg), vocab, rr, candidates, cfg.k, exclusions=exclusions, clean_queries=clean_queries)
    return as_tool(router, name)


def create_fusion_tool(
    configs: list[tuple[RetrievalConfig | dict[str, Any], QueryType, float]],
    k: int = 10,
    rrf_k: int = 60,
    name: str = "product_search",
) -> StructuredTool:
    """Weighted RRF over several (index, query_type, weight) searches, e.g. ANN + FULL_TEXT or products + pseudo-query index."""
    parts = []
    for c, query_type, weight in configs:
        cfg = _config(c)
        parts.append((PlainRetriever(build_backend(cfg), query_type, k=max(k, 25)), weight))
    return as_tool(FusionRetriever(parts, k, rrf_k), name)


def create_dao_ai_instructed_tool(
    config: RetrievalConfig | dict[str, Any],
    vector_search_endpoint: str,
    source_table: str,
    vocabulary: CatalogVocabulary | None = None,
    vocabulary_table: str | None = None,
    warehouse_id: str | None = None,
    rerank: Literal["databricks", "flashrank", "llm"] = "databricks",
    name: str = "product_search",
    **dao_ai_options: Any,
) -> StructuredTool:
    """dao-ai's instructed retriever behind the same contract (requires `pip install product-retrieval[dao-ai]`)."""
    from product_retrieval.strategies.dao_ai import DaoAIInstructedRetriever

    cfg = _config(config)
    vocab = _vocabulary(cfg, vocabulary, vocabulary_table, warehouse_id)
    return as_tool(DaoAIInstructedRetriever(cfg, vocab, vector_search_endpoint, source_table, rerank=rerank, **dao_ai_options), name)


def create_plan_router_tool(
    config: RetrievalConfig | dict[str, Any],
    vocabulary: CatalogVocabulary | None = None,
    vocabulary_table: str | None = None,
    warehouse_id: str | None = None,
    reranker: str = "ai_decide_noul",
    candidates: int = 12,
    endpoint_name: str | None = None,
    name: str = "product_search",
    backend: SearchBackend | None = None,
) -> StructuredTool:
    """LLM structured query plan (cached): excluded brands → filters; required brands/specs → reranker requirements."""
    from product_retrieval.strategies.planning import PlanRouter, QueryPlanner

    cfg = _config(config)
    vocab = _vocabulary(cfg, vocabulary, vocabulary_table, warehouse_id)
    rr = make_reranker(reranker, endpoint_name)  # type: ignore[arg-type]
    return as_tool(PlanRouter(backend or build_backend(cfg), QueryPlanner(cfg.llm_endpoint, vocab), rr, candidates, cfg.k), name)


def create_ltr_tool(
    config: RetrievalConfig | dict[str, Any],
    model_path: str,
    vocabulary: CatalogVocabulary | None = None,
    vocabulary_table: str | None = None,
    warehouse_id: str | None = None,
    signal_endpoints: dict[str, str] | None = None,
    use_ai_decide: bool = True,
    candidates: int = 20,
    name: str = "product_search",
    backend: SearchBackend | None = None,
) -> StructuredTool:
    """Learned ranker (LightGBM LambdaRank, `ltr` extra) over router candidates; `signal_endpoints` maps feature name →
    cross-encoder serving endpoint. Feature names/signals must match those the model was trained with."""
    from product_retrieval.ltr import FeatureCollector, LambdaRankModel, LTRRetriever

    cfg = _config(config)
    vocab = _vocabulary(cfg, vocabulary, vocabulary_table, warehouse_id)
    signals: dict[str, Reranker] = {n: ServingEndpointReranker(e) for n, e in (signal_endpoints or {}).items()}
    if use_ai_decide:
        signals["noul"] = AIDecideReranker("noul")
    collector = FeatureCollector(backend or build_backend(cfg), vocab, signals, candidates)
    return as_tool(LTRRetriever(collector, LambdaRankModel.load(model_path), cfg.k), name)
