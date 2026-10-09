"""Search backends. Strategies depend only on the `SearchBackend` protocol, so they are testable without Databricks."""

from __future__ import annotations

from typing import Any, Literal, Protocol

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from product_retrieval._retry import with_backoff
from product_retrieval.config import CatalogSchema, EndpointType, RetrievalConfig
from product_retrieval.documents import Filter, Product

QueryType = Literal["ANN", "HYBRID", "FULL_TEXT"]


class SearchRequest(BaseModel):
    text: str
    k: int = 10
    query_type: QueryType = "HYBRID"
    filters: list[Filter] = Field(default_factory=list)
    rerank_columns: list[str] | None = Field(None, description="Enable the Databricks server-side reranker on these columns")
    query_columns: list[str] | None = Field(None, description="Restrict keyword matching to these text columns (FULL_TEXT/HYBRID)")
    facets: list[str] | None = Field(None, description='Facet specs, e.g. ["brand_name TOP 8"] (FULL_TEXT/HYBRID)')


class SearchResponse(BaseModel):
    products: list[Product]
    facets: dict[str, dict[str, int]] = Field(default_factory=dict)


class SearchBackend(Protocol):
    catalog: CatalogSchema

    def search(self, request: SearchRequest) -> SearchResponse: ...


# ---------------------------------------------------------------------------------------------------------------------
# Filter translation
# ---------------------------------------------------------------------------------------------------------------------

_DICT_SUFFIX = {"eq": "", "in": "", "ne": " NOT", "not_in": " NOT", "lt": " <", "lte": " <=", "gt": " >", "gte": " >=", "like": " LIKE"}
_SQL_OP = {"eq": "=", "ne": "!=", "lt": "<", "lte": "<=", "gt": ">", "gte": ">="}


def to_dict_filters(filters: list[Filter]) -> dict[str, Any]:
    """STANDARD endpoint syntax, e.g. {"brand_name NOT": "DEWALT", "sku": "00176279"}."""
    return {f"{f.column}{_DICT_SUFFIX[f.op]}": f.value for f in filters}


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def to_sql_filters(filters: list[Filter]) -> str:
    """STORAGE_OPTIMIZED endpoint syntax, e.g. "brand_name != 'DEWALT' AND sku = '00176279'"."""
    clauses: list[str] = []
    for f in filters:
        if f.op in ("in", "not_in"):
            values = f.value if isinstance(f.value, (list, tuple)) else [f.value]
            clauses.append(f"{f.column} {'NOT IN' if f.op == 'not_in' else 'IN'} ({', '.join(_sql_literal(v) for v in values)})")
        elif f.op == "like":
            clauses.append(f"{f.column} LIKE {_sql_literal(f'%{f.value}%')}")
        elif isinstance(f.value, (list, tuple)):  # eq/ne with several values
            values = ", ".join(_sql_literal(v) for v in f.value)
            clauses.append(f"{f.column} {'NOT IN' if f.op == 'ne' else 'IN'} ({values})")
        else:
            clauses.append(f"{f.column} {_SQL_OP[f.op]} {_sql_literal(f.value)}")
    return " AND ".join(clauses)


# ---------------------------------------------------------------------------------------------------------------------
# Databricks AI Search
# ---------------------------------------------------------------------------------------------------------------------


def row_to_product(row: dict[str, Any], catalog: CatalogSchema) -> Product:
    def text(col: str | None) -> str | None:
        value = row.get(col) if col else None
        return None if value is None else str(value)

    known = {catalog.id, catalog.upc, catalog.brand, catalog.name, catalog.category, catalog.description, "score"}
    return Product(
        id=text(catalog.id) or "",
        name=text(catalog.name) or "",
        brand=text(catalog.brand),
        category=text(catalog.category),
        description=text(catalog.description) or "",
        upc=text(catalog.upc),
        score=row.get("score"),
        extra={k: v for k, v in row.items() if k not in known},
    )


class QueryEmbedder:
    """Embeds queries for self-managed embedding indexes through a Databricks embedding endpoint."""

    def __init__(self, endpoint: str, instruction: str | None, workspace_client: WorkspaceClient) -> None:
        self.endpoint, self.instruction, self.w = endpoint, instruction, workspace_client

    def __call__(self, text: str) -> list[float]:
        body: dict[str, Any] = {"input": [text]}
        if self.instruction:
            body["instruction"] = self.instruction  # Foundation Model API field for instruction-aware embedding models
        resp = with_backoff(lambda: self.w.api_client.do("POST", f"/serving-endpoints/{self.endpoint}/invocations", body=body))
        return resp["data"][0]["embedding"]


class AISearchBackend:
    """Databricks AI Search (Vector Search) index, STANDARD or STORAGE_OPTIMIZED, managed or self-managed embeddings."""

    def __init__(self, config: RetrievalConfig, workspace_client: WorkspaceClient | None = None) -> None:
        from databricks.ai_search.client import VectorSearchClient

        self.config, self.catalog = config, config.catalog
        self.w = workspace_client or WorkspaceClient()
        self.index = VectorSearchClient(disable_notice=True).get_index(index_name=config.index_name)
        self.embed = (
            QueryEmbedder(config.query_embedding.endpoint, config.query_embedding.instruction, self.w)
            if config.query_embedding
            else None
        )

    def _filters(self, filters: list[Filter]) -> dict[str, Any] | str | None:
        if not filters:
            return None
        return to_sql_filters(filters) if self.config.endpoint_type == EndpointType.STORAGE_OPTIMIZED else to_dict_filters(filters)

    @mlflow.trace(name="ai_search", span_type=SpanType.RETRIEVER)
    def search(self, request: SearchRequest) -> SearchResponse:
        from databricks.ai_search.reranker import DatabricksReranker

        kwargs: dict[str, Any] = {
            "columns": self.catalog.columns(),
            "num_results": request.k,
            "query_type": request.query_type,
            "filters": self._filters(request.filters),
        }
        if self.embed and request.query_type != "FULL_TEXT":
            kwargs["query_vector"] = self.embed(request.text)
            if request.query_type == "HYBRID":
                kwargs["query_text"] = request.text
        else:
            kwargs["query_text"] = request.text
        if request.rerank_columns:
            kwargs["reranker"] = DatabricksReranker(columns_to_rerank=request.rerank_columns)
        if request.query_columns:
            kwargs["query_columns"] = request.query_columns
        if request.facets:
            kwargs["facets"] = request.facets

        resp = with_backoff(lambda: self.index.similarity_search(**kwargs))
        names = [c["name"] for c in resp.get("manifest", {}).get("columns", [])]
        products = [row_to_product(dict(zip(names, row)), self.catalog) for row in resp.get("result", {}).get("data_array") or []]
        facets: dict[str, dict[str, int]] = {}
        for column, value, count in resp.get("facet_result", {}).get("facet_array") or []:
            if value not in (None, ""):
                facets.setdefault(column, {})[str(value)] = int(count)
        return SearchResponse(products=products, facets=facets)


def build_backend(config: RetrievalConfig, workspace_client: WorkspaceClient | None = None) -> AISearchBackend:
    return AISearchBackend(config, workspace_client)
