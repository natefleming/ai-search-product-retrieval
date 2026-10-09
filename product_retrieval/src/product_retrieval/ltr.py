"""Learning-to-rank over retrieval and reranker signals (requires the `ltr` extra: lightgbm, pandas).

Pattern: production search stacks distil many signals (lexical, dense, cross-encoder/LLM relevance, constraint matches) into a
fast learned ranker (e.g. LambdaRank). Candidates come from the router's pool; features are computed in parallel.
"""

from __future__ import annotations

import contextvars
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import mlflow
from mlflow.entities import SpanType

from product_retrieval.backends import SearchBackend, SearchRequest
from product_retrieval.documents import Product, RetrievalResult
from product_retrieval.rerankers import Reranker
from product_retrieval.strategies.router import QueryRouter
from product_retrieval.understanding import CatalogVocabulary, analyze, brand_key

_TOKEN = re.compile(r"[a-z0-9]+(?:[./][0-9]+)?")
_NUMBER = re.compile(r"\d+(?:[./]\d+)?")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def lexical_features(query: str, product: Product, excluded_brand: str | None) -> dict[str, float]:
    q_tokens, title_tokens = _tokens(query), _tokens(product.name)
    q_numbers, p_numbers = set(_NUMBER.findall(query)), set(_NUMBER.findall(f"{product.name} {product.description}"))
    brand = brand_key(product.brand or "")
    return {
        "title_overlap": len(q_tokens & title_tokens) / max(len(q_tokens), 1),
        "number_overlap": len(q_numbers & p_numbers) / max(len(q_numbers), 1) if q_numbers else -1.0,
        "brand_in_query": float(bool(brand) and brand in brand_key(query)),
        "is_excluded_brand": float(excluded_brand is not None and product.brand == excluded_brand),
    }


class FeatureCollector:
    """Candidate pool + per-candidate features for one query."""

    def __init__(
        self,
        backend: SearchBackend,
        vocabulary: CatalogVocabulary,
        signals: dict[str, Reranker] | None = None,
        candidates: int = 20,
        rank_depth: int = 50,
    ) -> None:
        self.backend, self.vocab, self.signals = backend, vocabulary, signals or {}
        self.pool = QueryRouter(backend, vocabulary, candidates=candidates, k=candidates)
        self.rank_depth = rank_depth

    def _ranks(self, query: str, query_type: str) -> dict[str, int]:
        products = self.backend.search(SearchRequest(text=query, k=self.rank_depth, query_type=query_type)).products  # type: ignore[arg-type]
        return {p.id: i for i, p in enumerate(products, start=1)}

    def _signal(self, reranker: Reranker, query: str, products: list[Product]) -> dict[str, float]:
        try:
            return {p.id: s for p, s in reranker.rerank(query, products)}
        except Exception:  # a failed signal becomes a missing feature, not a failed query
            return {}

    @mlflow.trace(name="ltr_features", span_type=SpanType.CHAIN)
    def collect(self, query: str) -> tuple[RetrievalResult, list[dict[str, float]]]:
        pool = self.pool(query)
        if pool.route == "identifier" or not pool.products:
            return pool, []
        analysis = analyze(query, self.vocab)
        jobs: dict[str, Any] = {
            "ann": (self._ranks, query, "ANN"),
            "full_text": (self._ranks, query, "FULL_TEXT"),
            **{name: (self._signal, r, query, pool.products) for name, r in self.signals.items()},
        }
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = {k: ex.submit(contextvars.copy_context().run, fn, *args) for k, (fn, *args) in jobs.items()}
            results = {k: f.result() for k, f in futures.items()}
        rows = []
        for rank, p in enumerate(pool.products, start=1):
            row = {
                "pool_rank": float(rank),
                "ann_rank": float(results["ann"].get(p.id, self.rank_depth + 1)),
                "full_text_rank": float(results["full_text"].get(p.id, self.rank_depth + 1)),
                "route_exclusion": float(pool.route == "exclusion"),
                **lexical_features(query, p, analysis.excluded_brand),
            }
            for name in self.signals:
                row[name] = float(results[name].get(p.id, -1.0))
            rows.append(row)
        return pool, rows


class LambdaRankModel:
    """Thin wrapper around a LightGBM ranker with a fixed feature order."""

    def __init__(self, booster: Any, feature_names: list[str]) -> None:
        self.booster, self.feature_names = booster, feature_names

    @classmethod
    def train(cls, rows: list[dict[str, float]], labels: list[int], groups: list[int], params: dict[str, Any] | None = None,
              num_boost_round: int = 200) -> LambdaRankModel:
        import lightgbm as lgb
        import pandas as pd

        X = pd.DataFrame(rows)
        dataset = lgb.Dataset(X, label=labels, group=groups)
        defaults = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 5, 10], "learning_rate": 0.05,
                    "num_leaves": 15, "min_data_in_leaf": 10, "verbose": -1}
        return cls(lgb.train({**defaults, **(params or {})}, dataset, num_boost_round=num_boost_round), list(X.columns))

    def save(self, path: str) -> None:
        """Writes `<path>.txt` (LightGBM model) and `<path>.features.json`. Staged through a local temp dir because
        LightGBM seeks while writing, which Unity Catalog volume paths don't support."""
        import json
        import os
        import shutil
        import tempfile

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "model.txt")
            self.booster.save_model(local)
            shutil.copyfile(local, f"{path}.txt")
        with open(f"{path}.features.json", "w") as f:
            json.dump(self.feature_names, f)

    @classmethod
    def load(cls, path: str) -> LambdaRankModel:
        import json
        import os
        import shutil
        import tempfile

        import lightgbm as lgb

        with open(f"{path}.features.json") as f:
            features = json.load(f)
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "model.txt")
            shutil.copyfile(f"{path}.txt", local)
            return cls(lgb.Booster(model_file=local), features)

    def score(self, rows: list[dict[str, float]]) -> list[float]:
        import pandas as pd

        return list(self.booster.predict(pd.DataFrame(rows)[self.feature_names]))


class LTRRetriever:
    """Router pool → features → learned ranking (identifier queries keep the exact lookup)."""

    def __init__(self, collector: FeatureCollector, model: LambdaRankModel, k: int = 10, name: str = "ltr") -> None:
        self.collector, self.model, self.k, self.name = collector, model, k, name

    @mlflow.trace(span_type=SpanType.CHAIN)
    def __call__(self, query: str) -> RetrievalResult:
        pool, rows = self.collector.collect(query)
        if not rows:
            return pool.model_copy(update={"strategy": self.name, "products": pool.products[: self.k]})
        scores = self.model.score(rows)
        order = sorted(range(len(rows)), key=lambda i: (-scores[i], i))
        products = [pool.products[i] for i in order]
        return pool.model_copy(update={
            "strategy": self.name, "products": products[: self.k], "scores": {pool.products[i].id: round(scores[i], 4) for i in order},
        })
