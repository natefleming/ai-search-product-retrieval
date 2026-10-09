# Databricks notebook source
# MAGIC %md
# MAGIC # 11 · Measurement: how much of the remaining error is real?
# MAGIC The evaluation's ground truth is the **seed product** a query was generated from. For constrained queries other products can be
# MAGIC equally correct (another non-DeWalt impact driver), and pack-size variants can be near-identical, so some "misses" may be label
# MAGIC noise. Following Amazon's ESCI scheme, an LLM judge (`product_retrieval.evaluation.GradedJudge`, Claude Sonnet 5.5) labels results as
# MAGIC **exact / substitute / complement / irrelevant**:
# MAGIC
# MAGIC 1. **judged exact@1** over all 600 queries for the leading strategies: the rank-1 product is the seed *or* the judge says it
# MAGIC    exactly satisfies the request.
# MAGIC 2. **graded nDCG@5** (ESCI gains 3/2/1/0) on the 102-query judged subset.
# MAGIC
# MAGIC A sample of judgements is shown for manual spot-checking; LLM judges are biased, so treat these as estimates.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.evaluation import GradedJudge, graded_ndcg

STRATEGIES = ["03_hybrid", "09_router_fast", "09_router_quality", "10_enriched_router_quality"]
runs = strategy_runs()
per_query = load_per_query(runs.loc[[s for s in STRATEGIES if s in runs.index]])
cat, src = product_catalog(), eval_source()
judge = GradedJudge("databricks-claude-sonnet-5-5")


def label(query: str, sku: str) -> str:
    p = cat.get(sku)
    return judge.judge(query, p.product_name, p.brand_name, p.merchandise_class, p.description) if p else "irrelevant"


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · judged exact@1 (all queries)

# COMMAND ----------

pairs = {(r.query, r.skus[0]) for r in per_query.itertuples() if r.skus and r.skus[0] not in set(src.loc[r.query, "expected_skus"])}
mlflow.tracing.disable()
with ThreadPoolExecutor(max_workers=8) as pool:
    labels = dict(zip(pairs, pool.map(lambda qs: label(*qs), pairs)))
mlflow.tracing.enable()


def top1(r) -> str:
    if not r.skus:
        return "none"
    return "seed" if r.skus[0] in set(src.loc[r.query, "expected_skus"]) else labels[(r.query, r.skus[0])]


per_query["top1"] = [top1(r) for r in per_query.itertuples()]
summary = per_query.groupby("strategy").top1.value_counts(normalize=True).unstack(fill_value=0).round(3)
summary["judged_exact_at_1"] = summary.get("seed", 0) + summary.get("exact", 0)
display(summary.reset_index())
by_type = per_query.assign(ok=per_query.top1.isin(["seed", "exact"])).pivot_table(index="strategy", columns="query_type", values="ok").round(3)
display(by_type.reset_index())

# COMMAND ----------

display(per_query[per_query.top1 != "seed"].sample(min(15, (per_query.top1 != "seed").sum()), random_state=1)
        .assign(top1_product=lambda d: d.skus.map(lambda s: cat[s[0]].product_name if s and s[0] in cat else ""))
        [["strategy", "query_type", "query", "top1_product", "top1"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · graded nDCG@5 (judged subset)

# COMMAND ----------

judge_queries = set(spark.table(JUDGE_DATASET).toPandas()["inputs"].map(lambda i: (json.loads(i) if isinstance(i, str) else i)["query"]))
subset = per_query[per_query["query"].isin(judge_queries)]
top5 = {(r.query, s) for r in subset.itertuples() for s in r.skus[:5]}
with ThreadPoolExecutor(max_workers=8) as pool:
    graded = dict(zip(top5, pool.map(lambda qs: label(*qs), top5)))
subset = subset.assign(graded_ndcg_at_5=[graded_ndcg([graded[(r.query, s)] for s in r.skus[:5]]) for r in subset.itertuples()])
ndcg = subset.groupby("strategy").graded_ndcg_at_5.mean().round(3)
display(ndcg.reset_index())

# COMMAND ----------

with mlflow.start_run(run_name="11_measurement"):
    mlflow.set_tags({"demo": "retrieval_measurement", "judge": "databricks-claude-sonnet-5-5"})
    for strategy, row in summary.iterrows():
        mlflow.log_metric(f"{strategy}.judged_exact_at_1", float(row["judged_exact_at_1"]))
        mlflow.log_metric(f"{strategy}.graded_ndcg_at_5", float(ndcg.get(strategy, float("nan"))))
    mlflow.log_table(summary.reset_index(), artifact_file="judged_top1.json")
    mlflow.log_table(per_query[["strategy", "query", "query_type", "top1"]], artifact_file="judged_top1_per_query.json")
dbutils.notebook.exit(json.dumps({"judged_exact_at_1": summary["judged_exact_at_1"].to_dict(), "graded_ndcg_at_5": ndcg.to_dict()}))
