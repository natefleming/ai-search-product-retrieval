# Databricks notebook source
# MAGIC %md
# MAGIC # 19 · Holdout check for the finalists
# MAGIC The routing rules, prompts and LTR features were developed against the 600-query dev set. This notebook re-runs the
# MAGIC baseline and the leading strategies on the **holdout** set (300 queries generated from products never used as dev seeds),
# MAGIC so the final numbers aren't tuned-on-test. Runs carry the `split=holdout` tag.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.rerankers import AIDecideListwiseReranker, AIDecideReranker

original, enriched = pr.build_backend(config_original), pr.build_backend(config_enriched)
vocab = vocabulary()
noul = AIDecideReranker("noul", details_chars=300)
finalists = {
    "03_hybrid": pr.PlainRetriever(original, "HYBRID", k=K, name="03_hybrid"),
    "03_full_text": pr.PlainRetriever(original, "FULL_TEXT", k=K, name="03_full_text"),
    "06_filter_ai_decide": pr.RerankedRetriever(pr.LLMFilterRetriever(config_original, original, vocab, candidates=25),
                                                AIDecideReranker("score", details_chars=500), K, name="06_filter_ai_decide"),
    "09_router_fast": pr.QueryRouter(original, vocab, name="09_router_fast"),
    "09_router_quality": pr.QueryRouter(original, vocab, reranker=noul, candidates=12, name="09_router_quality"),
    "10_enriched_router_quality": pr.QueryRouter(enriched, vocab, reranker=noul, candidates=12, name="10_enriched_router_quality"),
    "17_router_listwise": pr.QueryRouter(enriched, vocab, reranker=AIDecideListwiseReranker(base=noul), candidates=12, name="17_router_listwise"),
}
CROSS_ENCODERS = {"bge": "retrieval-bge-reranker-v2-m3", "qwen3": "retrieval-qwen3-reranker-0-6b"}
dbutils.widgets.text("cross_encoders", "", "Comma-separated cross-encoders (bge, qwen3) to add as router finalists")
for short in [e.strip() for e in dbutils.widgets.get("cross_encoders").split(",") if e.strip()]:
    finalists[f"13_router_{short}"] = pr.QueryRouter(original, vocab, reranker=pr.ServingEndpointReranker(CROSS_ENCODERS[short]),
                                                     candidates=25, name=f"13_router_{short}")

for name, retriever in finalists.items():
    run_strategy(name, retriever, {"split": "holdout"}, holdout=True)
