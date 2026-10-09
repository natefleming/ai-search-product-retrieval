"""Facet-guided retrieval: AI Search facet counts give the real filter vocabulary; ai_decide `choice` picks from it."""

from __future__ import annotations

import mlflow
from mlflow.entities import SpanType

from product_retrieval.backends import SearchBackend, SearchRequest
from product_retrieval.documents import Filter, RetrievalResult
from product_retrieval.rerankers import AIDecideClient, decide_per_question_on_error

NONE = "none"


def facet_options(values: list[str], none_description: str) -> dict[str, str]:
    """Opaque labels (opt_1..n) with the facet value as description; raw values as labels intermittently fail in ai_decide."""
    return {**{f"opt_{i}": v for i, v in enumerate(values, start=1)}, NONE: none_description}


class FacetGuidedRetriever:
    """Facet counts → three ai_decide choice questions (category, required brand, excluded brand) → filters when confident.

    In the evaluation this over-constrained recall (category filters); facets are better used for shopper refinement UX.
    """

    def __init__(
        self, backend: SearchBackend, k: int = 10, min_confidence: float = 0.7, top_values: int = 8,
        client: AIDecideClient | None = None, name: str = "facet_guided",
    ) -> None:
        self.backend, self.k, self.min_confidence, self.top_values = backend, k, min_confidence, top_values
        self.client, self.name = client or AIDecideClient(), name

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        catalog = self.backend.catalog
        facet_spec = [f"{catalog.category} TOP {self.top_values}", f"{catalog.brand} TOP {self.top_values}"]
        facets = self.backend.search(SearchRequest(text=query, k=self.k, facets=facet_spec)).facets
        labels = {
            "category": facet_options(list(facets.get(catalog.category, {})), "No single category clearly matches."),
            "required_brand": facet_options(list(facets.get(catalog.brand, {})), "The shopper did not require a brand."),
            "excluded_brand": facet_options(list(facets.get(catalog.brand, {})), "No brand is excluded."),
        }
        instructions = {
            "category": "Which category is the shopper explicitly asking for? Choose 'none' unless one clearly matches the product type.",
            "required_brand": "Which brand does the shopper explicitly require? Choose 'none' if they did not name a required brand.",
            "excluded_brand": "Which brand does the shopper explicitly say they do NOT want? Choose 'none' if no brand is excluded.",
        }
        questions = {q: {"type": "choice", "instructions": instructions[q], "criteria": labels[q]} for q in labels}
        answers = decide_per_question_on_error(self.client, {"shopper_request": query, "facet_counts": facets}, questions)
        target = {"category": (catalog.category, "eq"), "required_brand": (catalog.brand, "eq"), "excluded_brand": (catalog.brand, "ne")}
        filters = [
            Filter(column=target[q][0], op=target[q][1], value=labels[q][a.choice])
            for q, a in answers.items()
            if a and a.choice and a.choice != NONE and a.choice in labels[q] and (a.confidence or 0) >= self.min_confidence
        ]
        products = self.backend.search(SearchRequest(text=query, k=self.k, filters=filters)).products
        if not products and filters:
            filters, products = [], self.backend.search(SearchRequest(text=query, k=self.k)).products
        return RetrievalResult(products=products, strategy=self.name, filters=filters, diagnostics={"facets": facets})
