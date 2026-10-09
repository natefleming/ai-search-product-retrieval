# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Instructed retrieval with `ai_decide`
# MAGIC [`ai_decide`](https://www.databricks.com/blog/introducing-aidecide-make-fast-decisions-your-governed-data) is a decision model
# MAGIC (not a text generator): given a `state` and typed `questions` it returns calibrated probabilities. That makes it a good
# MAGIC **instruction-aware reranker**: retrieve broadly, then ask for every candidate whether it satisfies the shopper's stated
# MAGIC requirements under a business policy written in plain English. All candidates go in **one** call (one question each).
# MAGIC
# MAGIC REST via the SDK: `w.ai_functions.ai_decide(state=..., questions=...)` → `POST /api/2.0/ai-functions/ai-decide`.
# MAGIC
# MAGIC | run | candidates | reranker | tool factory |
# MAGIC |---|---|---|---|
# MAGIC | `06_ai_decide_score` | HYBRID top 25 | `score` (0–3) | `pr.create_instructed_tool(config, reranker="ai_decide_score")` |
# MAGIC | `06_ai_decide_noul` | HYBRID top 25 | `noul` probability | `pr.create_instructed_tool(config)` |
# MAGIC | `06_filter_ai_decide` | guarded LLM filters top 25 | `score` | `pr.create_guarded_filter_tool(config, vocab, reranker="ai_decide_score")` |

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.rerankers import AIDecideReranker
from product_retrieval.strategies.filtering import LLMFilterRetriever

CANDIDATES: int = 25
backend = pr.build_backend(config_original)
pool = pr.PlainRetriever(backend, "HYBRID", k=CANDIDATES)
score, noul = AIDecideReranker("score", details_chars=500), AIDecideReranker("noul", details_chars=500)

# COMMAND ----------

# MAGIC %md
# MAGIC ## One call scores every candidate

# COMMAND ----------

example = eval_source().query("query_type == 'exclusion'").iloc[0]
print(example.name)
for p, s in score.rerank(example.name, pool(example.name).products)[:10]:
    print(f"{s:.2f}  {p.brand:<14} {p.name[:70]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate

# COMMAND ----------

params = {"query_type": "HYBRID", "candidates": CANDIDATES, "reranker": "ai_decide", "details_chars": 500}
run_strategy("06_ai_decide_score", pr.RerankedRetriever(pool, score, K, name="06_ai_decide_score"), {**params, "question": "score"})
run_strategy("06_ai_decide_noul", pr.RerankedRetriever(pool, noul, K, name="06_ai_decide_noul"), {**params, "question": "noul"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guarded LLM filters → `ai_decide`
# MAGIC Hard filters (notebook 05) narrow the pool; `ai_decide` then enforces the softer requirements (size, voltage, kit).

# COMMAND ----------

filtered_pool = LLMFilterRetriever(config_original, backend, vocabulary(), guarded=True, candidates=CANDIDATES)
filter_ai_decide = pr.RerankedRetriever(filtered_pool, score, K, name="06_filter_ai_decide")
fparams = {**params, "dynamic_filter": True, "guarded": True, "question": "score"}
run_strategy("06_filter_ai_decide", filter_ai_decide, fparams)
run_strategy("06_filter_ai_decide", filter_ai_decide, fparams, judged=True)
