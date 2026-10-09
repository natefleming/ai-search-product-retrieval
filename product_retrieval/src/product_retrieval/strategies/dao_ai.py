"""Adapter around dao-ai's instructed retriever (`dao_ai.tools.create_ai_search_tool`); requires the `dao-ai` extra."""

from __future__ import annotations

import json
from typing import Literal

import mlflow
from mlflow.entities import SpanType

from product_retrieval._retry import with_backoff
from product_retrieval.backends import row_to_product
from product_retrieval.config import RetrievalConfig
from product_retrieval.documents import RetrievalResult
from product_retrieval.rerankers import DEFAULT_POLICY
from product_retrieval.understanding import CatalogVocabulary


class DaoAIInstructedRetriever:
    """Decomposition (LLM → filtered subqueries) → parallel search → RRF → rerank, configured through `dao_ai.config`.

    Use a model whose output works with LangChain `with_structured_output` for decomposition (e.g. Claude Haiku);
    dao-ai silently falls back to the unfiltered query when decomposition errors, so configure a fallback model.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        vocabulary: CatalogVocabulary,
        vector_search_endpoint: str,
        source_table: str,
        primary_key: str = "product_id",
        embedding_source_column: str = "description",
        decomposition_llm: str = "databricks-claude-haiku-4-5",
        fallback_llms: tuple[str, ...] = ("databricks-claude-sonnet-4-6",),
        rerank: Literal["databricks", "flashrank", "llm"] = "databricks",
        candidates: int = 50,
        embedding_endpoint: str = "databricks-gte-large-en",
        name: str | None = None,
    ) -> None:
        from dao_ai.config import (
            AiSearchRetrieverModel, ColumnInfo, DecompositionModel, IndexModel, InstructedRetrieverModel,
            InstructionAwareRerankModel, LLMModel, RerankParametersModel, SchemaModel, SearchParametersModel,
            TableModel, VectorSearchEndpoint, VectorStoreModel,
        )
        from dao_ai.tools import create_ai_search_tool

        self.config, self.catalog, self.k = config, config.catalog, config.k
        catalog_name, schema_name, index = config.index_name.split(".")
        llm = LLMModel(name=decomposition_llm, temperature=0.0, max_tokens=1024, fallbacks=list(fallback_llms))
        top_brands = list(vocabulary.brands.values())[:40]
        table_catalog, table_schema, table = source_table.split(".")
        store = VectorStoreModel(
            endpoint=VectorSearchEndpoint(name=vector_search_endpoint),
            index=IndexModel(schema=SchemaModel(catalog_name=catalog_name, schema_name=schema_name), name=index),
            source_table=TableModel(schema=SchemaModel(catalog_name=table_catalog, schema_name=table_schema), name=table),
            embedding_model=LLMModel(name=embedding_endpoint), embedding_source_column=embedding_source_column,
            primary_key=primary_key, columns=self.catalog.columns(),
        )
        instructed = InstructedRetrieverModel(
            columns=[
                ColumnInfo(name=self.catalog.brand, type="string", operators=["", "NOT"],
                           description=f"Brand in catalog spelling (UPPERCASE), exact match. Examples: {', '.join(top_brands)}"),
                ColumnInfo(name=self.catalog.category, type="string", operators=["", "NOT"],
                           description=f"Category, exact match. Examples: {'; '.join(vocabulary.categories[:60])}"),
                ColumnInfo(name=self.catalog.name, type="string", operators=["LIKE"], description="Product title; LIKE matches whole tokens"),
            ],
            constraints=[
                "Use the brand filter only when the shopper names a required brand; use 'NOT' when they exclude one.",
                "Use the category filter only when you are sure of the exact category name.",
                "Never filter on SKU or UPC codes; keep them in the query text.",
            ],
            decomposition=DecompositionModel(model=llm, max_subqueries=3, rrf_k=60, normalize_filter_case="uppercase"),
            rerank=InstructionAwareRerankModel(model=llm, top_n=self.k, instructions=DEFAULT_POLICY) if rerank == "llm" else None,
        )
        rerank_model = (
            RerankParametersModel(model="ms-marco-MiniLM-L-12-v2", top_n=self.k)
            if rerank == "flashrank"
            else RerankParametersModel(columns=[self.catalog.name, self.catalog.category, self.catalog.description], top_n=self.k)
        )
        retriever = AiSearchRetrieverModel(
            vector_store=store, columns=self.catalog.columns(),
            search_parameters=SearchParametersModel(num_results=candidates, query_type="HYBRID"),
            rerank=rerank_model, instructed=instructed,
        )
        self.tool = create_ai_search_tool(retriever=retriever, name="dao_ai_product_search")
        self.name = name or f"dao_ai_instructed_{rerank}"

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        docs = json.loads(with_backoff(lambda: self.tool.invoke({"query": query})))
        products, seen = [], set()
        for d in docs:
            p = row_to_product({**d["metadata"], self.catalog.description: d.get("page_content", "")}, self.catalog)
            if p.id not in seen:
                seen.add(p.id)
                products.append(p)
        return RetrievalResult(products=products[: self.k], strategy=self.name)
