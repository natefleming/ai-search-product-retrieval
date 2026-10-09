"""Configuration objects. Every factory takes a `RetrievalConfig`, so a deployment is described in one place."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EndpointType(str, Enum):
    """AI Search endpoint flavour; they differ in filter syntax (dict vs SQL string) and scale (~320M vs ~1B vectors)."""

    STANDARD = "STANDARD"
    STORAGE_OPTIMIZED = "STORAGE_OPTIMIZED"


class CatalogSchema(BaseModel):
    """Maps logical product fields to the index's column names, so the package works with any catalog layout."""

    id: str = "sku"
    upc: str | None = "upc"
    brand: str = "brand_name"
    name: str = "product_name"
    category: str = "merchandise_class"
    description: str = "description"
    extra: list[str] = Field(default_factory=list, description="Additional columns to return with every product")

    def columns(self) -> list[str]:
        cols = [self.id, self.upc, self.brand, self.name, self.category, self.description, *self.extra]
        return list(dict.fromkeys(c for c in cols if c))


class QueryEmbedding(BaseModel):
    """For self-managed embedding indexes: how to embed the query (e.g. Qwen3 with a query-side instruction)."""

    endpoint: str = "databricks-qwen3-embedding-0-6b"
    instruction: str | None = "Given a shopper's product search request, retrieve the matching catalog products"


class RetrievalConfig(BaseModel):
    index_name: str
    endpoint_type: EndpointType = EndpointType.STANDARD
    catalog: CatalogSchema = Field(default_factory=CatalogSchema)
    k: int = Field(10, description="Results returned to the caller")
    query_embedding: QueryEmbedding | None = Field(
        None, description="Set for self-managed embedding indexes; leave None for Databricks-managed embeddings"
    )
    llm_endpoint: str = Field("databricks-gpt-oss-120b", description="Foundation model used by LLM-based strategies")
