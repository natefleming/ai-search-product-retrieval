# Databricks notebook source
# MAGIC %md
# MAGIC # 90 · Compare strategies
# MAGIC Pulls every strategy run from the MLflow experiment, builds a leaderboard, breaks quality down by query type, and tests
# MAGIC whether each improvement over the HYBRID baseline is statistically real (paired bootstrap, 95% CI, same 600 queries).
# MAGIC
# MAGIC Row-level side-by-side comparisons are available in the MLflow UI: **Experiment → Evaluations → select two runs → Compare**.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow matplotlib

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import matplotlib.pyplot as plt

runs = strategy_runs()
per_query = load_per_query(runs)
STRATEGIES: list[str] = list(runs.index)
print(STRATEGIES)


def metric(name: str, agg: str = "mean") -> pd.Series:
    for col in (f"metrics.{name}/{agg}", f"metrics.{name}"):
        if col in runs:
            return runs[col]
    return pd.Series(index=runs.index, dtype=float)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Leaderboard

# COMMAND ----------

QUALITY: list[str] = ["hit_at_1", "hit_at_10", "mrr_at_10", "ndcg_at_10", "constraint_precision_at_10", "excluded_brand_leak_at_10", "zero_results"]
leaderboard = pd.DataFrame({m: metric(m) for m in QUALITY})
leaderboard["latency_ms_p50"] = metric("latency_ms", "median")
leaderboard["latency_ms_p90"] = metric("latency_ms", "p90")
leaderboard["latency_ms_p99"] = metric("latency_ms", "p99")
leaderboard["prediction_errors"] = runs.get("metrics.prediction_errors")
leaderboard["queries_scored"] = runs.get("metrics.queries_scored")
leaderboard["llm_calls"] = metric("llm_calls")
leaderboard["ai_decide_calls"] = metric("ai_decide_calls")
leaderboard = leaderboard.sort_values("mrr_at_10", ascending=False)
display(leaderboard.round(3).reset_index(names="strategy"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Is it real? Paired bootstrap vs the HYBRID baseline

# COMMAND ----------

pivot = {m: per_query.pivot_table(index="query", columns="strategy", values=m) for m in ["mrr_at_10", "hit_at_1", "constraint_precision_at_10"]}
significance = []
for s in STRATEGIES:
    if s == BASELINE_RUN:
        continue
    row: dict[str, Any] = {"strategy": s}
    for m, p in pivot.items():
        mean, lo, hi = paired_bootstrap(p[BASELINE_RUN], p[s])
        row |= {f"Δ{m}": round(mean, 3), f"{m} 95% CI": f"[{lo:+.3f}, {hi:+.3f}]", f"{m} verdict": "better" if lo > 0 else ("worse" if hi < 0 else "no clear difference")}
    significance.append(row)
significance_df = pd.DataFrame(significance).sort_values("Δmrr_at_10", ascending=False)
display(significance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality by query type
# MAGIC Different methods win on different query shapes, which is the core argument for a layered retrieval design.

# COMMAND ----------

by_type = per_query.pivot_table(index="strategy", columns="query_type", values="mrr_at_10").loc[leaderboard.index, QUERY_TYPES]
fig, ax = plt.subplots(figsize=(14, 5))
by_type.T.plot.bar(ax=ax, width=0.85)
ax.set_ylabel("MRR@10")
ax.set_title("MRR@10 by query type")
ax.legend(fontsize=7, ncol=3)
plt.tight_layout()
display(fig)
display(by_type.round(3).reset_index())

# COMMAND ----------

cp = per_query.pivot_table(index="strategy", columns="query_type", values="constraint_precision_at_10")
leak = per_query.pivot_table(index="strategy", columns="query_type", values="excluded_brand_leak_at_10")
display(pd.concat({"constraint_precision@10": cp, "excluded_brand_leak@10": leak}, axis=1).round(3).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality vs latency (Pareto)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(leaderboard.latency_ms_p50, leaderboard.mrr_at_10)
for s, r in leaderboard.iterrows():
    ax.annotate(s, (r.latency_ms_p50, r.mrr_at_10), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.set_xscale("log")
ax.set_xlabel("p50 latency (ms, log)")
ax.set_ylabel("MRR@10")
ax.set_title("Quality vs latency")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommended default per query type
# MAGIC The highest-MRR strategy per type, plus the fastest strategy within 0.02 MRR of it.

# COMMAND ----------

lat = leaderboard.latency_ms_p50
recs = []
for qtype in QUERY_TYPES:
    col = by_type[qtype]
    best = col.idxmax()
    near = col[col >= col.max() - 0.02].index
    recs.append({"query_type": qtype, "best": best, "best_mrr": round(col.max(), 3), "fastest_near_best": lat[near].idxmin(),
                 "baseline_mrr": round(col[BASELINE_RUN], 3)})
display(pd.DataFrame(recs))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Judged subset: answer correctness & retrieval judges

# COMMAND ----------

judged = strategy_runs(judged=True)
JUDGES = ["correctness", "retrieval_relevance", "retrieval_sufficiency", "retrieval_groundedness", "answer_policy", "mrr_at_10", "latency_ms"]
judged_table = pd.DataFrame(
    {j: next((judged[c] for c in (f"metrics.{j}/mean", f"metrics.{j}") if c in judged), pd.Series(index=judged.index, dtype=float)) for j in JUDGES}
)
display(judged_table.round(3).reset_index(names="strategy"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Holdout: do the finalists hold up on queries from unseen products?

# COMMAND ----------

holdout = strategy_runs(split="holdout")
if len(holdout):
    cols = {"mrr_at_10": "MRR@10", "hit_at_1": "hit@1", "excluded_brand_leak_at_10": "leak"}
    dev_vs_holdout = pd.concat(
        {"dev": pd.DataFrame({v: metric(k) for k, v in cols.items()}),
         "holdout": pd.DataFrame({v: holdout.get(f"metrics.{k}/mean") for k, v in cols.items()})}, axis=1
    ).dropna(subset=[("holdout", "MRR@10")]).sort_values(("holdout", "MRR@10"), ascending=False)
    display(dev_vs_holdout.round(3).reset_index(names="strategy"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Label noise: judged exact@1 (notebook 11)

# COMMAND ----------

measurement = mlflow.search_runs([experiment.experiment_id], "tags.demo = 'retrieval_measurement'", order_by=["start_time DESC"], max_results=1)
if len(measurement):
    m = measurement.iloc[0]
    judged_tbl = pd.DataFrame({
        "seed hit@1": metric("hit_at_1"),
        "judged exact@1": {c.split(".")[1]: m[c] for c in measurement.columns if c.endswith(".judged_exact_at_1")},
        "graded nDCG@5 (judged subset)": {c.split(".")[1]: m[c] for c in measurement.columns if c.endswith(".graded_ndcg_at_5")},
    }).dropna(subset=["judged exact@1"])
    display(judged_tbl.round(3).reset_index(names="strategy"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scale: 38K vs ~5M rows on a storage-optimized endpoint (notebook 18)

# COMMAND ----------

pairs = {"18_scale_hybrid": "10_enriched_hybrid", "18_scale_router_fast": "10_enriched_router_fast", "18_scale_router_quality": "10_enriched_router_quality"}
scale = pd.DataFrame([
    {"strategy": small, "MRR@10 38K": metric("mrr_at_10").get(small), "MRR@10 5M": metric("mrr_at_10").get(big),
     "p50 38K": metric("latency_ms", "median").get(small), "p50 5M": metric("latency_ms", "median").get(big),
     "p99 38K": metric("latency_ms", "p99").get(small), "p99 5M": metric("latency_ms", "p99").get(big)}
    for big, small in pairs.items() if big in runs.index
])
if len(scale):
    display(scale.round(3))
