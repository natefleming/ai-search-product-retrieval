# Databricks notebook source
# MAGIC %md
# MAGIC # 16 · LLM structured query plans (hard vs soft constraints)
# MAGIC Industry pattern (Instacart's LLM query-understanding engine; Amazon hint-augmented reranking): parse the request into a
# MAGIC typed **plan**, validate it against the catalog, cache it for head queries, and keep inferred preferences soft.
# MAGIC
# MAGIC `pr.PlanRouter` / `pr.create_plan_router_tool(config, vocab)`:
# MAGIC * SKU/UPC → exact lookup (no LLM)
# MAGIC * otherwise `QueryPlan` (gpt-oss-120b tool call, LRU-cached): **excluded brands → hard filter** (validated against the catalog);
# MAGIC   **required brands and specs → explicit requirements for the reranker**, not filters (hard brand/category filters removed the
# MAGIC   right product too often in notebook 05)
# MAGIC * HYBRID on the plan's semantic query + specs → `ai_decide` noul rerank
# MAGIC
# MAGIC Compared with `10_enriched_router_quality` (regex router, same index and reranker): does LLM query understanding beat the
# MAGIC deterministic router, and at what latency?

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.rerankers import AIDecideReranker

backend = pr.build_backend(config_enriched)
planner = pr.QueryPlanner(LLM_ENDPOINT, vocabulary())
src = eval_source()
display(pd.DataFrame([{"query": q, **planner(q).model_dump()} for q in src.groupby("query_type").head(2).index if not re.search(r"\d{8}", q)]))

# COMMAND ----------

plan_router = pr.PlanRouter(backend, planner, reranker=AIDecideReranker("noul", details_chars=300), candidates=12, name="16_plan_router_quality")
plan_router_fast = pr.PlanRouter(backend, planner, reranker=None, candidates=12, name="16_plan_router_fast")
run_strategy("16_plan_router_quality", plan_router, {"planner_llm": LLM_ENDPOINT, "rerank": "ai_decide noul", "candidates": 12, "index": ENRICHED_INDEX})
run_strategy("16_plan_router_fast", plan_router_fast, {"planner_llm": LLM_ENDPOINT, "rerank": "none", "index": ENRICHED_INDEX})
