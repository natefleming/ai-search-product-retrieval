"""The product and result types shared by every strategy, reranker and tool."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

FilterOp = Literal["eq", "ne", "in", "not_in", "lt", "lte", "gt", "gte", "like"]


class Filter(BaseModel):
    """Backend-neutral filter; translated to dict (STANDARD) or SQL string (STORAGE_OPTIMIZED) syntax by the backend."""

    column: str
    op: FilterOp = "eq"
    value: Any


class Product(BaseModel):
    id: str
    name: str = ""
    brand: str | None = None
    category: str | None = None
    description: str = ""
    upc: str | None = None
    score: float | None = Field(None, description="Retrieval score from AI Search")
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    products: list[Product]
    strategy: str
    route: str | None = None
    filters: list[Filter] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict, description="Reranker score per product id")
    rerank_error: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return [p.id for p in self.products]

    def to_tool_output(self, description_chars: int = 600) -> str:
        """JSON documents in the same shape as VectorSearchRetrieverTool (page_content + metadata)."""
        docs = [
            {
                "page_content": f"{p.name}\n{p.description[:description_chars]}",
                "metadata": {
                    "id": p.id,
                    "name": p.name,
                    "brand": p.brand,
                    "category": p.category,
                    "upc": p.upc,
                    **p.extra,
                    "route": self.route,
                    "score": self.scores.get(p.id, p.score),
                },
            }
            for p in self.products
        ]
        return json.dumps(docs, default=str)
