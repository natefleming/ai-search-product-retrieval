"""Router whose query classification is done by ai_decide instead of rules.

ai_decide returns decisions (choice / noul / score), not free text, so open-vocabulary values come from the data: the excluded
brand is chosen from the query's own brand facets, and a SKU/UPC code is read from the text only once ai_decide says the request
is an identifier lookup.
"""

from __future__ import annotations

import re

import mlflow
from mlflow.entities import SpanType

from product_retrieval.backends import SearchBackend, SearchRequest
from product_retrieval.documents import Filter, Product, RetrievalResult
from product_retrieval.rerankers import AIDecideClient, Reranker, decide_per_question_on_error, rerank_safely
from product_retrieval.strategies.facets import NONE, facet_options
from product_retrieval.understanding import CODE_RE, clean_query

ROUTES: dict[str, str] = {
    "identifier": "The shopper looks up an exact SKU, UPC, item or model number.",
    "exclusion": "The shopper explicitly rules out a brand, in any wording ('not X', 'no X', 'done with X', 'skip X', ...).",
    "general": "Anything else: a product type, a need, a required brand, or specs.",
}


def without_brand(query: str, brand: str) -> str:
    """Drop the excluded brand's words so keyword matching doesn't pull that brand back in."""
    words = [w for w in re.split(r"[^\w']+", brand) if len(w) >= 2]
    text = re.sub(r"\b(" + "|".join(map(re.escape, words)) + r")\b", " ", query, flags=re.I) if words else query
    return clean_query(text)


class AIDecideRouter:
    """HYBRID search with brand facets → one ai_decide call (route + excluded brand) → per-route retrieval → optional rerank.

    Same routes as `QueryRouter`, but ai_decide classifies instead of regex + vocabulary: it understands paraphrased exclusions,
    costs one ai_decide call, and can only exclude a brand that appears in the query's top brand facets.
    """

    def __init__(
        self,
        backend: SearchBackend,
        reranker: Reranker | None = None,
        candidates: int = 12,
        k: int = 10,
        facet_brands: int = 10,
        min_confidence: float = 0.6,
        client: AIDecideClient | None = None,
        name: str | None = None,
    ) -> None:
        self.backend, self.reranker, self.candidates, self.k = backend, reranker, candidates, k
        self.facet_brands, self.min_confidence = facet_brands, min_confidence
        self.client = client or AIDecideClient()
        self.name = name or ("ai_decide_router_quality" if reranker else "ai_decide_router_fast")

    @mlflow.trace(name="ai_decide_route", span_type=SpanType.CHAIN)
    def classify(self, query: str, brands: list[str]) -> tuple[str, str | None]:
        """(route, excluded brand). A confident excluded brand makes the route an exclusion whatever the route answer was."""
        labels = facet_options(brands, "No brand is ruled out.")
        questions = {
            "route": {"type": "choice", "instructions": "What kind of product search request is this?", "criteria": ROUTES},
            "excluded_brand": {
                "type": "choice", "criteria": labels,
                "instructions": "Which brand does the shopper explicitly say they do NOT want? Choose 'none' if no brand is ruled out.",
            },
        }
        answers = decide_per_question_on_error(self.client, {"shopper_request": query, "brands_in_results": brands}, questions)
        route_answer, brand_answer = answers.get("route"), answers.get("excluded_brand")
        route = route_answer.choice if route_answer and route_answer.choice in ROUTES else "general"
        excluded = None
        if brand_answer and brand_answer.choice in labels and brand_answer.choice != NONE and (brand_answer.confidence or 0) >= self.min_confidence:
            excluded = labels[brand_answer.choice]
        if route == "exclusion" and not excluded:
            route = "general"  # nothing to filter on
        elif excluded and route != "identifier":
            route = "exclusion"
        return route, excluded

    def _exact(self, code: str) -> list[Product]:
        catalog = self.backend.catalog
        column = catalog.id if len(code) == 8 or not catalog.upc else catalog.upc
        products = self.backend.search(SearchRequest(text=code, k=self.k, filters=[Filter(column=column, value=code)])).products
        return products or self.backend.search(SearchRequest(text=code, k=self.k, query_type="FULL_TEXT")).products

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        catalog = self.backend.catalog
        first = self.backend.search(SearchRequest(text=clean_query(query), k=self.candidates, facets=[f"{catalog.brand} TOP {self.facet_brands}"]))
        route, excluded = self.classify(query, list(first.facets.get(catalog.brand, {})))

        code = CODE_RE.search(query) if route == "identifier" else None
        if code:
            return RetrievalResult(products=self._exact(code.group(1)), strategy=self.name, route="identifier")

        filters: list[Filter] = []
        products = first.products
        if route == "exclusion" and excluded:
            filters = [Filter(column=catalog.brand, op="ne", value=excluded)]
            products = self.backend.search(SearchRequest(text=without_brand(query, excluded), k=self.candidates, filters=filters)).products
        outcome = rerank_safely(self.reranker, query, products)  # reranker sees the original request, constraints included
        return RetrievalResult(
            products=outcome.products[: self.k], strategy=self.name, route="exclusion" if filters else "general",
            filters=filters, scores=outcome.scores, rerank_error=outcome.error, diagnostics={"ai_route": route},
        )
