"""Request-targeted `columns_to_rerank` for the Databricks server-side reranker.

The reranker reads columns in order and only the first ~2,000 characters, so the column list decides what the request is compared
against. One ai_decide `choice` call maps each request to a column profile.
"""

from __future__ import annotations

import mlflow
from mlflow.entities import SpanType

from product_retrieval.config import CatalogSchema
from product_retrieval.rerankers import AIDecideClient


def default_profiles(catalog: CatalogSchema) -> dict[str, tuple[str, list[str]]]:
    """profile → (when to use it, columns_to_rerank). An empty list means: keep retrieval order (no reranker)."""
    return {
        "title": ("The shopper names a specific product, model or product line.", [catalog.name, catalog.brand]),
        "attributes": ("The shopper specifies attributes: size, voltage, wattage, color, count, material, kit vs tool-only.",
                       [catalog.description, catalog.name]),
        "category": ("The shopper describes a need, use case or general product type without naming a product.",
                     [catalog.category, catalog.name, catalog.description]),
        "brand": ("The shopper requires a specific brand (not an exclusion).", [catalog.brand, catalog.name, catalog.description]),
        "none": ("The shopper looks up an exact code (SKU, UPC or model number); keep the keyword ranking.", []),
    }


class RerankColumnSelector:
    def __init__(
        self,
        catalog: CatalogSchema,
        profiles: dict[str, tuple[str, list[str]]] | None = None,
        default: str = "attributes",
        client: AIDecideClient | None = None,
    ) -> None:
        self.profiles, self.default = profiles or default_profiles(catalog), default
        self.client = client or AIDecideClient()
        self.question = {
            "profile": {
                "type": "choice",
                "instructions": "Which product fields should a reranker compare against this shopper request? If the shopper EXCLUDES "
                                "a brand, never choose 'brand'; choose by what they do want.",
                "criteria": {name: desc for name, (desc, _) in self.profiles.items()},
            }
        }

    @mlflow.trace(name="choose_rerank_columns", span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> list[str]:
        try:
            choice = self.client.decide({"shopper_request": query}, self.question)["profile"].choice
        except Exception:  # selection failure → default profile, never a failed search
            choice = None
        return self.profiles.get(choice or self.default, self.profiles[self.default])[1]
