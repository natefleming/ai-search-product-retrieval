"""Shared fakes: an in-memory backend and reranker so strategies are tested without Databricks."""

from __future__ import annotations

import mlflow
import pytest

mlflow.tracing.disable()  # unit tests never export traces

from product_retrieval.backends import SearchRequest, SearchResponse
from product_retrieval.config import CatalogSchema
from product_retrieval.documents import Product
from product_retrieval.understanding import CatalogVocabulary

CATALOG = [
    Product(id="00000001", name="DeWalt 20V Brushless Impact Driver Kit", brand="DEWALT", category="HEAVY-DUTY POWER TOOLS", upc="0000000000001"),
    Product(id="00000002", name="Milwaukee M18 Brushless Impact Driver Kit", brand="MILWAUKEE", category="HEAVY-DUTY POWER TOOLS", upc="0000000000002"),
    Product(id="00000003", name="Craftsman V20 Impact Driver Kit", brand="CRAFTSMAN", category="CONSUMER POWER TOOLS", upc="0000000000003"),
    Product(id="00000004", name="Black+Decker 20V Drill", brand="BLACK+DECKER", category="CONSUMER POWER TOOLS", upc="0000000000004"),
]


class FakeBackend:
    """Ranks by naive token overlap and applies eq/ne filters, recording every request."""

    def __init__(self, products: list[Product] | None = None) -> None:
        self.products, self.catalog, self.requests = products or CATALOG, CatalogSchema(), []

    def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        field = {"sku": "id", "upc": "upc", "brand_name": "brand", "merchandise_class": "category"}
        hits = []
        for p in self.products:
            ok = True
            for f in request.filters:
                value = getattr(p, field.get(f.column, f.column))
                ok &= (value == f.value) if f.op == "eq" else (value != f.value) if f.op == "ne" else True
            if ok:
                overlap = len(set(request.text.lower().split()) & set(p.name.lower().split()))
                hits.append((overlap, p))
        hits.sort(key=lambda t: -t[0])
        return SearchResponse(products=[p for _, p in hits][: request.k])


class ReverseReranker:
    name = "reverse"

    def rerank(self, query, products):
        return [(p, float(i)) for i, p in enumerate(products)][::-1]


class FailingReranker:
    name = "failing"

    def rerank(self, query, products):
        raise RuntimeError("ai_decide unavailable")


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def vocabulary() -> CatalogVocabulary:
    return CatalogVocabulary.from_values([p.brand for p in CATALOG] + ["MRS. MEYER'S CLEAN DAY", "3M"], [p.category for p in CATALOG])
