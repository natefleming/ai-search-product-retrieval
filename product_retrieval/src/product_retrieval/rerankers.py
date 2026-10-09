"""Post-retrieval rerankers. Server-side Databricks reranking is a search option (`SearchRequest.rerank_columns`), not here."""

from __future__ import annotations

from typing import Any, Literal, Protocol

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import BadRequest
from mlflow.entities import SpanType
from pydantic import BaseModel

from product_retrieval._retry import with_backoff
from product_retrieval.documents import Product

DEFAULT_POLICY: str = (
    "You rank catalog products for a hardware-store shopper. Treat every requirement the shopper states as mandatory: "
    "required brand, excluded brands, product type, voltage or battery platform, size, quantity or pack count, color, "
    "wattage, and kit-vs-tool-only. Accessories, parts or refills for the requested product are NOT the requested product."
)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, products: list[Product]) -> list[tuple[Product, float]]:
        """Return products best-first with a score; must keep every input product."""
        ...


class RerankOutcome(BaseModel):
    products: list[Product]
    scores: dict[str, float]
    error: str | None = None


def rerank_safely(reranker: Reranker | None, query: str, products: list[Product]) -> RerankOutcome:
    """Apply a reranker; on any failure keep the retrieval order so the search itself never fails."""
    if reranker is None or not products:
        return RerankOutcome(products=products, scores={})
    try:
        ranked = reranker.rerank(query, products)
        return RerankOutcome(products=[p for p, _ in ranked], scores={p.id: round(s, 4) for p, s in ranked})
    except Exception as e:
        return RerankOutcome(products=products, scores={}, error=f"{reranker.name}: {type(e).__name__}: {str(e)[:200]}")


def _order(products: list[Product], scores: list[float]) -> list[tuple[Product, float]]:
    """Sort best-first; ties keep the incoming (retrieval) order."""
    return [(p, s) for p, s, _ in sorted(zip(products, scores, range(len(products))), key=lambda t: (-t[1], t[2]))]


# ---------------------------------------------------------------------------------------------------------------------
# ai_decide (Databricks AI Functions REST API)
# ---------------------------------------------------------------------------------------------------------------------


class DecideAnswer(BaseModel):
    type: str
    probability: float | None = None
    choice: str | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None

    @property
    def expected_level(self) -> float:
        return sum(int(level) * p for level, p in (self.probabilities or {}).items())


def candidates_state(query: str, products: list[Product], details_chars: int) -> dict[str, Any]:
    return {
        "shopper_request": query,
        "candidates": {
            f"c{i}": {"name": p.name, "brand": p.brand, "category": p.category, "details": p.description[:details_chars]}
            for i, p in enumerate(products)
        },
    }


class AIDecideClient:
    def __init__(self, workspace_client: WorkspaceClient | None = None) -> None:
        self.w = workspace_client or WorkspaceClient()

    @mlflow.trace(name="ai_decide", span_type=SpanType.TOOL)
    def decide(self, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, DecideAnswer]:
        resp = with_backoff(lambda: self.w.ai_functions.ai_decide(state=state, questions=questions))
        return {qid: DecideAnswer.model_validate(a) for qid, a in resp.response["answers"].items()}


def decide_per_question_on_error(client: AIDecideClient, state: dict[str, Any], questions: dict[str, Any]) -> dict[str, DecideAnswer | None]:
    """Ask all questions in one call; if ai_decide (beta) rejects it, ask one question per call so one bad question drops only itself."""
    try:
        return dict(client.decide(state, questions))
    except BadRequest:
        answers: dict[str, DecideAnswer | None] = {}
        for qid, q in questions.items():
            try:
                answers[qid] = client.decide(state, {qid: q})[qid]
            except BadRequest:
                answers[qid] = None
        return answers


SCORE_CRITERIA: list[str] = [
    "Unrelated product",
    "Related category but a different product type (or an accessory/part for it)",
    "Right product type but violates at least one stated requirement (brand, exclusion, size, voltage, count, kit/tool-only, color)",
    "Right product type and satisfies every stated requirement",
]


class AIDecideReranker:
    """Pointwise instruction-aware rerank: one batched ai_decide call, one question per candidate.

    `noul` (default) returns a continuous probability and separates candidates better than the 4-level `score` question.
    """

    def __init__(
        self,
        question: Literal["noul", "score"] = "noul",
        policy: str = DEFAULT_POLICY,
        details_chars: int = 300,
        client: AIDecideClient | None = None,
    ) -> None:
        self.question, self.policy, self.details_chars = question, policy, details_chars
        self.client = client or AIDecideClient()
        self.name = f"ai_decide_{question}"

    def _questions(self, n: int) -> dict[str, Any]:
        if self.question == "score":
            return {
                f"c{i}": {"type": "score", "instructions": f"{self.policy} How well does candidate c{i} satisfy the shopper_request?",
                          "criteria": SCORE_CRITERIA}
                for i in range(n)
            }
        return {
            f"c{i}": {
                "type": "noul",
                "instructions": f"{self.policy} Is candidate c{i} the right product type AND does it satisfy every stated requirement?",
                "criteria": {"true": "Right product type and every stated requirement is met.",
                             "false": "Wrong product type or a requirement is violated."},
            }
            for i in range(n)
        }

    @mlflow.trace(name="ai_decide_rerank", span_type=SpanType.RERANKER)
    def rerank(self, query: str, products: list[Product]) -> list[tuple[Product, float]]:
        answers = self.client.decide(candidates_state(query, products, self.details_chars), self._questions(len(products)))
        scores = [a.expected_level if a.type == "score" else (a.probability or 0.0) for a in (answers[f"c{i}"] for i in range(len(products)))]
        return _order(products, scores)


class AIDecideListwiseReranker:
    """Listwise step: after an optional pointwise reranker, one ai_decide `choice` picks the best of the top N for rank 1."""

    def __init__(
        self,
        base: Reranker | None = None,
        top_n: int = 5,
        min_confidence: float = 0.5,
        policy: str = DEFAULT_POLICY,
        details_chars: int = 300,
        client: AIDecideClient | None = None,
    ) -> None:
        self.base, self.top_n, self.min_confidence, self.policy, self.details_chars = base, top_n, min_confidence, policy, details_chars
        self.client = client or AIDecideClient()
        self.name = f"{base.name}+listwise" if base else "ai_decide_listwise"

    @mlflow.trace(name="ai_decide_listwise", span_type=SpanType.RERANKER)
    def rerank(self, query: str, products: list[Product]) -> list[tuple[Product, float]]:
        ranked = self.base.rerank(query, products) if self.base else [(p, 0.0) for p in products]
        head = [p for p, _ in ranked[: self.top_n]]
        question = {
            "best": {
                "type": "choice",
                "instructions": f"{self.policy} Which candidate best satisfies every requirement in the shopper_request?",
                "criteria": {f"c{i}": f"candidate c{i}" for i in range(len(head))},
            }
        }
        answer = self.client.decide(candidates_state(query, head, self.details_chars), question)["best"]
        if answer.choice and (answer.confidence or 0) >= self.min_confidence:
            best = head[int(answer.choice[1:])]
            ranked = [(best, max(s for _, s in ranked) + 1.0)] + [(p, s) for p, s in ranked if p.id != best.id]
        return ranked


# ---------------------------------------------------------------------------------------------------------------------
# Cross-encoder served on Databricks Model Serving
# ---------------------------------------------------------------------------------------------------------------------


class ServingEndpointReranker:
    """Cross-encoder (e.g. bge-reranker-v2-m3, Qwen3-Reranker) served on Databricks Model Serving.

    Endpoint contract (MLflow pyfunc): `dataframe_records=[{"query": str, "document": str}, ...]` → `predictions=[score, ...]`.
    """

    def __init__(self, endpoint_name: str, document_chars: int = 600, workspace_client: WorkspaceClient | None = None) -> None:
        self.endpoint_name, self.document_chars = endpoint_name, document_chars
        self.w = workspace_client or WorkspaceClient()
        self.name = f"cross_encoder:{endpoint_name}"

    @staticmethod
    def document_text(p: Product, chars: int) -> str:
        return f"{p.name} | Brand: {p.brand} | Category: {p.category} | {p.description}"[:chars]

    @mlflow.trace(name="cross_encoder_rerank", span_type=SpanType.RERANKER)
    def rerank(self, query: str, products: list[Product]) -> list[tuple[Product, float]]:
        records = [{"query": query, "document": self.document_text(p, self.document_chars)} for p in products]
        resp = with_backoff(
            lambda: self.w.api_client.do("POST", f"/serving-endpoints/{self.endpoint_name}/invocations", body={"dataframe_records": records})
        )
        scores = [float(s) for s in resp["predictions"]]
        return _order(products, scores)
