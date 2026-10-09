# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Built-in reranker (`columns_to_rerank`)
# MAGIC AI Search can rerank candidates server-side with a cross-encoder (`SearchRequest.rerank_columns` →
# MAGIC `DatabricksReranker(columns_to_rerank=[...])`). It reads columns **in order** and uses only the first ~2,000 characters, so the
# MAGIC column list decides what the request is compared against.
# MAGIC
# MAGIC | run | columns_to_rerank | candidates |
# MAGIC |---|---|---|
# MAGIC | `04_rerank_name_desc` | product_name, description | 50 → top 10 |
# MAGIC | `04_rerank_all` | product_name, brand_name, merchandise_class, description | 50 → top 10 |
# MAGIC | `04_rerank_targeted` | **chosen per request** by an `ai_decide` choice (`RerankColumnSelector`) | 50 → top 10 |
# MAGIC
# MAGIC Tool factories: `pr.create_search_tool(config, rerank_columns=[...], candidates=50)` and `pr.create_targeted_rerank_tool(config)`.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.column_selection import RerankColumnSelector, default_profiles

DEPTH: int = 50
backend = pr.build_backend(config_original)
fixed = {
    "04_rerank_name_desc": ["product_name", "description"],
    "04_rerank_all": ["product_name", "brand_name", "merchandise_class", "description"],
}
selector = RerankColumnSelector(backend.catalog)  # ai_decide choice over the profiles below
targeted = pr.PlainRetriever(backend, "HYBRID", k=K, candidates=DEPTH, column_selector=selector, name="04_rerank_targeted")
display(pd.DataFrame([(name, when, ", ".join(cols) or "(no rerank)") for name, (when, cols) in default_profiles(backend.catalog).items()],
                     columns=["profile", "when", "columns_to_rerank"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before / after on one query

# COMMAND ----------

example = eval_source().query("query_type == 'exclusion'").iloc[0]
print(example.name)
for label, retriever in [("HYBRID", pr.PlainRetriever(backend, "HYBRID", k=K)),
                         ("rerank on all columns", pr.PlainRetriever(backend, rerank_columns=fixed["04_rerank_all"], candidates=DEPTH)),
                         ("rerank on targeted columns", targeted)]:
    print(f"\n{label}")
    for i, p in enumerate(retriever(example.name).products[:5], start=1):
        print(f"{i}. {'✓' if p.id in example.expected_skus else ' '} {p.brand:<14} {p.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate

# COMMAND ----------

for name, cols in fixed.items():
    run_strategy(name, pr.PlainRetriever(backend, "HYBRID", k=K, rerank_columns=cols, candidates=DEPTH, name=name),
                 {"query_type": "HYBRID", "num_results": DEPTH, "reranker": "DatabricksReranker", "columns_to_rerank": ",".join(cols)})

TARGETED_PARAMS = {"query_type": "HYBRID", "num_results": DEPTH, "reranker": "DatabricksReranker", "columns_to_rerank": "per-request (ai_decide profile)"}
run_strategy("04_rerank_targeted", targeted, TARGETED_PARAMS)
run_strategy("04_rerank_targeted", targeted, TARGETED_PARAMS, judged=True)
