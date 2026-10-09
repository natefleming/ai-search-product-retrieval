"""MLflow evaluation harness: score any retriever against an MLflow evaluation dataset (requires the `evaluation` extra).

Every strategy is evaluated on the same dataset rows, so runs line up in the MLflow Evaluations → Compare view.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Callable

import mlflow
from mlflow.entities import SpanType, Trace
from mlflow.genai.scorers import scorer
from pydantic import BaseModel, Field

from product_retrieval._retry import with_backoff
from product_retrieval.documents import RetrievalResult

K = 10
LLM_SPAN_TYPES = {SpanType.CHAT_MODEL, SpanType.LLM}

# ---------------------------------------------------------------------------------------------------------------------
# Retrieval metrics (plain functions, reused for per-type breakdowns and bootstrap CIs)
# ---------------------------------------------------------------------------------------------------------------------


def _satisfies(product: dict[str, Any], constraints: dict[str, Any]) -> bool:
    if constraints.get("brand_name") and product.get("brand") != constraints["brand_name"]:
        return False
    if constraints.get("merchandise_class") and product.get("category") != constraints["merchandise_class"]:
        return False
    if constraints.get("exclude_brand") and product.get("brand") == constraints["exclude_brand"]:
        return False
    return True


def retrieval_metrics(outputs: dict[str, Any], expectations: dict[str, Any], k: int = K) -> dict[str, float | None]:
    ids: list[str] = outputs["skus"][:k]
    products: list[dict[str, Any]] = outputs.get("products", [])[:k]
    expected = set(expectations["expected_skus"])
    constraints = expectations.get("constraints") or {}
    rank = next((i for i, s in enumerate(ids, start=1) if s in expected), None)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(expected), k) + 1))
    dcg = sum(1 / math.log2(r + 1) for r, s in enumerate(ids, start=1) if s in expected)
    return {
        "hit_at_1": float(rank == 1),
        "hit_at_5": float(rank is not None and rank <= 5),
        "hit_at_10": float(rank is not None),
        "mrr_at_10": 1 / rank if rank else 0.0,
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "constraint_precision_at_10": (sum(_satisfies(p, constraints) for p in products) / len(products) if products else 0.0)
        if any(constraints.values()) else None,
        "excluded_brand_leak_at_10": float(any(p.get("brand") == constraints["exclude_brand"] for p in products))
        if constraints.get("exclude_brand") else None,
        "zero_results": float(not ids),
    }


RETRIEVAL_METRICS: list[str] = list(retrieval_metrics({"skus": []}, {"expected_skus": []}))


def _metric_scorer(metric: str):
    @scorer(name=metric)
    def _score(outputs: dict[str, Any], expectations: dict[str, Any]) -> float | None:
        return retrieval_metrics(outputs, expectations)[metric]

    return _score


# ---------------------------------------------------------------------------------------------------------------------
# Latency / cost scorers (read the trace)
# ---------------------------------------------------------------------------------------------------------------------


def p99(values: list[float]) -> float:
    import numpy as np

    return float(np.percentile(values, 99))


LATENCY_AGGREGATIONS = ["mean", "median", "p90", p99]


def _outermost(trace: Trace, predicate: Callable[[Any], bool]) -> list[Any]:
    """Matching spans with no matching ancestor (avoids double-counting nested autolog spans)."""
    by_id = {s.span_id: s for s in trace.data.spans}

    def nested(span: Any) -> bool:
        parent = by_id.get(span.parent_id)
        while parent is not None:
            if predicate(parent):
                return True
            parent = by_id.get(parent.parent_id)
        return False

    return [s for s in trace.data.spans if predicate(s) and not nested(s)]


def _span_ms(trace: Trace, predicate: Callable[[Any], bool]) -> float:
    return sum((s.end_time_ns - s.start_time_ns) / 1e6 for s in _outermost(trace, predicate))


@scorer(aggregations=LATENCY_AGGREGATIONS)
def latency_ms(trace: Trace) -> float:
    return float(trace.info.execution_duration)


@scorer(aggregations=LATENCY_AGGREGATIONS)
def retrieval_ms(trace: Trace) -> float:
    return _span_ms(trace, lambda s: s.span_type == SpanType.RETRIEVER)


@scorer(aggregations=LATENCY_AGGREGATIONS)
def rerank_ms(trace: Trace) -> float:
    return _span_ms(trace, lambda s: s.span_type == SpanType.RERANKER)


@scorer(aggregations=LATENCY_AGGREGATIONS)
def llm_ms(trace: Trace) -> float:
    return _span_ms(trace, lambda s: s.span_type in LLM_SPAN_TYPES)


@scorer
def llm_calls(trace: Trace) -> int:
    return len(_outermost(trace, lambda s: s.span_type in LLM_SPAN_TYPES))


@scorer
def ai_decide_calls(trace: Trace) -> int:
    return sum(s.name == "ai_decide" for s in trace.data.spans)


@scorer
def total_tokens(trace: Trace) -> int:
    return int((trace.info.token_usage or {}).get("total_tokens", 0))


TRACE_SCORERS = [latency_ms, retrieval_ms, rerank_ms, llm_ms, llm_calls, ai_decide_calls, total_tokens]


def judge_scorers() -> list:
    from mlflow.genai.scorers import Correctness, Guidelines, RetrievalGroundedness, RetrievalRelevance, RetrievalSufficiency

    return [
        Correctness(), RetrievalRelevance(), RetrievalSufficiency(), RetrievalGroundedness(),
        Guidelines(name="answer_policy", guidelines=[
            "The response only recommends products that appear in the retrieved context.",
            "The response includes the SKU of every product it recommends.",
            "If the request excludes a brand, the response does not recommend that brand.",
        ]),
    ]


# ---------------------------------------------------------------------------------------------------------------------
# End-to-end answer (judged runs)
# ---------------------------------------------------------------------------------------------------------------------

ANSWER_PROMPT = (
    "You are a hardware-store product expert. Answer the shopper using ONLY the products below. "
    "Recommend the best match first, include each recommended product's SKU, and keep it under 120 words.\n\nProducts:\n{context}"
)


@mlflow.trace(name="final_top_k", span_type=SpanType.RETRIEVER)
def final_documents(result: RetrievalResult, top_n: int = 5) -> list[dict[str, Any]]:
    """Re-emit the final ranking as a RETRIEVER span so MLflow retrieval judges see what the answer used."""
    return [
        {"page_content": f"SKU {p.id} | {p.name} | Brand: {p.brand} | Category: {p.category}\n{p.description[:800]}",
         "metadata": {"doc_uri": p.id, "chunk_id": p.id}}
        for p in result.products[:top_n]
    ]


def answer(query: str, result: RetrievalResult, llm_endpoint: str) -> str:
    from product_retrieval._llm import chat, message_text

    context = "\n\n".join(d["page_content"] for d in final_documents(result))
    return message_text(with_backoff(lambda: chat(llm_endpoint).invoke([("system", ANSWER_PROMPT.format(context=context)), ("user", query)])))


# ---------------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------------


def _outputs(result: RetrievalResult) -> dict[str, Any]:
    return {
        "skus": result.ids,
        "products": [{"id": p.id, "brand": p.brand, "category": p.category} for p in result.products],
        "route": result.route,
        "filters": [f.model_dump() for f in result.filters],
        "scores": result.scores,
        "rerank_error": result.rerank_error,
    }


class Evaluator:
    """Runs strategies against an MLflow evaluation dataset and logs metrics, per-type metrics and a per-query table."""

    def __init__(
        self,
        dataset: str,
        query_types: dict[str, str],
        judge_dataset: str | None = None,
        tags: dict[str, str] | None = None,
        answer_llm: str = "databricks-gpt-oss-120b",
    ) -> None:
        """`query_types` maps each dataset query to its type (for per-type metrics)."""
        self.dataset, self.judge_dataset, self.query_types = dataset, judge_dataset, query_types
        self.tags, self.answer_llm = tags or {}, answer_llm

    def run(
        self,
        name: str,
        retriever: Callable[[str], RetrievalResult],
        params: dict[str, Any] | None = None,
        judged: bool = False,
        dataset: str | None = None,
    ) -> str:
        """Evaluate one strategy; `judged=True` also generates answers and runs LLM judges on the judge dataset."""
        import pandas as pd

        data = mlflow.genai.datasets.get_dataset(name=dataset or (self.judge_dataset if judged else self.dataset))
        rows: dict[str, dict[str, Any]] = {}

        @mlflow.trace(name=name, span_type=SpanType.CHAIN)
        def predict_fn(query: str) -> dict[str, Any]:
            start = time.perf_counter()
            try:
                result = with_backoff(lambda: retriever(query))
                error = None
            except Exception as e:  # a failed query is scored as an empty result, never silently dropped
                result, error = RetrievalResult(products=[], strategy=name), f"{type(e).__name__}: {str(e)[:200]}"
            out = {**_outputs(result), "error": error}
            if judged:
                out["answer"] = answer(query, result, self.answer_llm)
            rows[query] = {**out, "latency_ms": (time.perf_counter() - start) * 1000}
            return out

        run_name = f"{name}_judged" if judged else name
        scorers = [_metric_scorer(m) for m in RETRIEVAL_METRICS] + TRACE_SCORERS + (judge_scorers() if judged else [])
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({**self.tags, "strategy": name, "judged": str(judged), "dataset": data.name})
            mlflow.log_params(params or {})
            records = data.to_df()
            mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=scorers)
            expectations = {r["inputs"]["query"]: r["expectations"] for _, r in records.iterrows()}
            table = pd.DataFrame([
                {"query": q, "query_type": self.query_types.get(q), **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in out.items()},
                 **retrieval_metrics(out, expectations[q])}
                for q, out in rows.items()
            ])
            mlflow.log_metrics({"prediction_errors": int(table.error.notna().sum()), "queries_scored": len(table), **per_type_metrics(table)})
            mlflow.log_table(table, artifact_file="per_query.json")
        return run.info.run_id


def per_type_metrics(table: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for qtype, grp in table.groupby("query_type"):
        for m in RETRIEVAL_METRICS + ["latency_ms"]:
            vals = grp[m].dropna()
            if len(vals):
                metrics[f"{m}.{qtype}"] = float(vals.mean())
    return metrics


def paired_bootstrap(a: Any, b: Any, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% CI of (b - a) over the same queries."""
    import numpy as np

    d = (b - a).dropna().to_numpy()
    boots = np.random.default_rng(seed).choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------------------------------------------------
# Loading results back (comparison notebooks, reports)
# ---------------------------------------------------------------------------------------------------------------------


def load_runs(experiment_id: str, tags: dict[str, str] | None = None, judged: bool = False) -> Any:
    """Latest FINISHED run per strategy with metrics as columns, indexed by strategy."""
    clauses = [f"tags.{k} = '{v}'" for k, v in (tags or {}).items()] + [f"tags.judged = '{judged}'", "attributes.status = 'FINISHED'"]
    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=" and ".join(clauses), order_by=["start_time DESC"])
    return runs.drop_duplicates("tags.strategy").set_index("tags.strategy").sort_index()


def load_per_query(runs: Any) -> Any:
    """Per-query table of every run (one row per strategy × query), with JSON columns decoded."""
    import pandas as pd

    frames = []
    for strategy, run_id in runs["run_id"].items():
        df = mlflow.load_table("per_query.json", run_ids=[run_id])
        for col in ("skus", "products", "filters", "scores"):
            if col in df:
                df[col] = df[col].map(lambda v: json.loads(v) if isinstance(v, str) else v)
        frames.append(df.assign(strategy=strategy))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------
# Graded relevance judging (ESCI: exact / substitute / complement / irrelevant)
# ---------------------------------------------------------------------------------------------------------------------

ESCI_GAIN = {"exact": 3, "substitute": 2, "complement": 1, "irrelevant": 0}


ESCI_PROMPT = """You judge product search results for a hardware store, using Amazon's ESCI definitions:
- exact: the product is what the shopper asked for and satisfies every stated requirement (brand, exclusions, specs, pack size).
- substitute: a different product that could reasonably replace it, but misses at least one stated requirement.
- complement: not the requested product but used with it (accessory, part, refill).
- irrelevant: none of the above.
Reply with exactly one word: exact, substitute, complement or irrelevant."""


class GradedJudge:
    """LLM judge for ESCI labels; use a strong model (e.g. databricks-claude-sonnet-5-5) and spot-check a sample by hand.

    Plain-text single-word replies: newer Claude endpoints reject sampling parameters and forced tool choice."""

    def __init__(self, llm_endpoint: str = "databricks-claude-sonnet-5-5") -> None:
        from databricks_langchain import ChatDatabricks

        self.llm = ChatDatabricks(endpoint=llm_endpoint, max_tokens=20)

    def judge(self, query: str, name: str, brand: str | None, category: str | None, description: str) -> str:
        from product_retrieval._llm import message_text

        product = f"{name} | Brand: {brand} | Category: {category}\n{description[:800]}"
        msg = with_backoff(lambda: self.llm.invoke([("system", ESCI_PROMPT), ("user", f"Shopper request: {query}\n\nProduct: {product}")]))
        words = re.findall(r"[a-z]+", message_text(msg).lower())
        return next((w for w in words if w in ESCI_GAIN), "irrelevant")


def graded_ndcg(labels: list[str], k: int = 5) -> float:
    """nDCG@k with ESCI gains, normalized by the ideal ordering of the same labels."""
    gains = [ESCI_GAIN[l] for l in labels[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gains, reverse=True)))
    return dcg / ideal if ideal else 0.0
