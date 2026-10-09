"""Product retrieval on Databricks AI Search, packaged as LangChain tool factories.

Quick start:

    from product_retrieval import RetrievalConfig, CatalogVocabulary, create_router_tool

    config = RetrievalConfig(index_name="catalog.schema.products_enriched_index")
    vocabulary = CatalogVocabulary.from_spark(spark, "catalog.schema.products_enriched")
    tool = create_router_tool(config, vocabulary)          # quality router; reranker="none" for the fast router
    tool.invoke({"query": "brushless impact driver, anything but DeWalt"})
"""

from product_retrieval.column_selection import RerankColumnSelector
from product_retrieval.backends import AISearchBackend, SearchBackend, SearchRequest, SearchResponse, build_backend
from product_retrieval.config import CatalogSchema, EndpointType, QueryEmbedding, RetrievalConfig
from product_retrieval.documents import Filter, Product, RetrievalResult
from product_retrieval.rerankers import (
    AIDecideListwiseReranker,
    AIDecideReranker,
    Reranker,
    ServingEndpointReranker,
    rerank_safely,
)
from product_retrieval.strategies.ai_router import AIDecideRouter
from product_retrieval.strategies.facets import FacetGuidedRetriever
from product_retrieval.strategies.filtering import LLMFilterRetriever
from product_retrieval.strategies.fusion import FusionRetriever
from product_retrieval.strategies.plain import PlainRetriever
from product_retrieval.strategies.planning import PlanRouter, QueryPlan, QueryPlanner
from product_retrieval.strategies.reranked import RerankedRetriever
from product_retrieval.strategies.router import QueryRouter
from product_retrieval.tools import (
    as_tool,
    create_dao_ai_instructed_tool,
    create_dynamic_filter_tool,
    create_facet_guided_tool,
    create_fusion_tool,
    create_guarded_filter_tool,
    create_instructed_tool,
    create_ltr_tool,
    create_plan_router_tool,
    create_router_tool,
    create_search_tool,
    create_targeted_rerank_tool,
    make_reranker,
)
from product_retrieval.understanding import CatalogVocabulary, analyze

__version__ = "0.1.0"
__all__ = [name for name in dir() if not name.startswith("_")]
