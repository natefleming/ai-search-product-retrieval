# Databricks notebook source
# MAGIC %md
# MAGIC # 13 · Open cross-encoder rerankers (served on Model Serving)
# MAGIC Compares the GPU-served cross-encoders deployed in notebook 12 with `ai_decide` as the reranking stage, on the original index
# MAGIC so results line up with notebooks 06 and 09.
# MAGIC
# MAGIC | run | candidates | reranker | tool factory |
# MAGIC |---|---|---|---|
# MAGIC | `13_hybrid50_bge` / `13_hybrid50_qwen3` | HYBRID top 50 | cross-encoder | `pr.create_instructed_tool(config, reranker="cross_encoder", endpoint_name=..., candidates=50)` |
# MAGIC | `13_router_bge` / `13_router_qwen3` | router pool of 25 | cross-encoder | `pr.create_router_tool(config, vocab, reranker="cross_encoder", endpoint_name=..., candidates=25)` |

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

ENDPOINTS = {"bge": "retrieval-bge-reranker-v2-m3", "qwen3": "retrieval-qwen3-reranker-0-6b"}
backend = pr.build_backend(config_original)
vocab = vocabulary()
rerankers = {k: pr.ServingEndpointReranker(e) for k, e in ENDPOINTS.items()}
for r in rerankers.values():  # wake scale-to-zero endpoints before timing anything
    r.rerank("warm up", pr.PlainRetriever(backend, "HYBRID", k=5)("drill").products)

example = eval_source().query("query_type == 'exclusion'").iloc[0]
print(example.name)
for k, r in rerankers.items():
    ranked = r.rerank(example.name, pr.PlainRetriever(backend, "HYBRID", k=50)(example.name).products)
    print(k, [(p.brand, round(s, 3)) for p, s in ranked[:5]])

# COMMAND ----------

pool50 = pr.PlainRetriever(backend, "HYBRID", k=50)
strategies = {}
for k, r in rerankers.items():
    strategies[f"13_hybrid50_{k}"] = pr.RerankedRetriever(pool50, r, K, name=f"13_hybrid50_{k}")
    strategies[f"13_router_{k}"] = pr.QueryRouter(backend, vocab, reranker=r, candidates=25, name=f"13_router_{k}")
for name, retriever in strategies.items():
    run_strategy(name, retriever, {"reranker": "cross_encoder", "endpoint": ENDPOINTS[name.rsplit("_", 1)[1]],
                                   "candidates": 50 if "hybrid50" in name else 25})
