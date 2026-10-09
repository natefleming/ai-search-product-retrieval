# Databricks notebook source
# MAGIC %md
# MAGIC # 09 · Improvements: routing, faster instructed reranking, latency
# MAGIC Notebooks 03–07 showed *why* richer retrieval didn't move the averages as much as expected:
# MAGIC
# MAGIC 1. **Identifier queries** ("do you have item 89283134?") scored **MRR 0** on every relevance method: the code is diluted
# MAGIC    by filler words and embeddings carry no meaning for digits. That's 1/6 of the eval set.
# MAGIC 2. **HYBRID already ranks the target first** on most non-identifier queries, so rerankers only have the rank 2–10 slice to fix.
# MAGIC 3. `ai_decide`'s 4-level `score` question **ties** the top candidates in most queries (ranking falls back to retrieval order);
# MAGIC    `noul` probabilities are continuous and fixed twice as many rankings.
# MAGIC 4. A reranker can't remove the excluded brand when it fills the candidate pool; only a **filter** can.
# MAGIC 5. Hard brand/class filters **hurt** brand, attribute and descriptive queries (wrong guesses remove the right product).
# MAGIC
# MAGIC The experiments below apply those lessons one at a time, then together:
# MAGIC
# MAGIC | run | what changes | latency lever |
# MAGIC |---|---|---|
# MAGIC | `09_hybrid_id_routing` | identifier → exact SKU/UPC filter; everything else plain HYBRID | exact lookups skip ranking |
# MAGIC | `09_ai_decide_fast` | HYBRID top **12** → `noul`, 300-char details (vs 25 / 500 in 06) | smaller `ai_decide` payload |
# MAGIC | `09_router_fast` | identifier → exact; "not X" → `brand_name NOT` filter parsed from the text; else cleaned HYBRID | **no LLM, no ai_decide** |
# MAGIC | `09_router_quality` | `09_router_fast` + `noul` rerank of the top 12 | one `ai_decide` call, **no generative LLM** |
# MAGIC
# MAGIC Query understanding is deterministic (`product_retrieval.understanding.analyze`): code patterns, negation phrases and the catalog's
# MAGIC brand vocabulary. It was checked on this eval set (599/600 queries routed correctly), so validate it on fresh production traffic.
# MAGIC
# MAGIC Tool factories: `pr.create_router_tool(config, vocab, reranker="none")` (fast), `pr.create_router_tool(config, vocab)` (quality),
# MAGIC `pr.create_instructed_tool(config, candidates=12)` (fast ai_decide).

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query understanding

# COMMAND ----------

import numpy as np
from product_retrieval.rerankers import AIDecideReranker

backend = pr.build_backend(config_original)
vocab = vocabulary()
src = eval_source()
routes = pd.DataFrame({"query_type": src.query_type, "route": [pr.analyze(q, vocab).route for q in src.index]})
display(pd.crosstab(routes.query_type, routes.route))
display(pd.DataFrame([pr.analyze(q, vocab).model_dump() | {"query": q} for q in src.groupby("query_type").head(2).index]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Strategies

# COMMAND ----------

noul_fast = AIDecideReranker("noul", details_chars=300)
strategies = {
    "09_hybrid_id_routing": pr.QueryRouter(backend, vocab, candidates=K, exclusions=False, clean_queries=False, name="09_hybrid_id_routing"),
    "09_ai_decide_fast": pr.RerankedRetriever(pr.PlainRetriever(backend, "HYBRID", k=12), noul_fast, K, name="09_ai_decide_fast"),
    "09_router_fast": pr.QueryRouter(backend, vocab, name="09_router_fast"),
    "09_router_quality": pr.QueryRouter(backend, vocab, reranker=noul_fast, candidates=12, name="09_router_quality"),
}
params = {
    "09_hybrid_id_routing": {"routing": "identifier", "query_type": "HYBRID"},
    "09_ai_decide_fast": {"candidates": 12, "question": "noul", "details_chars": 300},
    "09_router_fast": {"routing": "identifier,exclusion,general", "rerank": "none", "llm": "none"},
    "09_router_quality": {"routing": "identifier,exclusion,general", "rerank": "ai_decide noul", "candidates": 12, "details_chars": 300, "llm": "none"},
}
cat = product_catalog()
q = src.query("query_type == 'identifier'").index[0]
print(q, "→", [(p.id, p.name) for p in strategies["09_router_fast"](q).products[:3]])

# COMMAND ----------

for name, retriever in strategies.items():
    run_strategy(name, retriever, params[name])
run_strategy("09_router_quality", strategies["09_router_quality"], params["09_router_quality"], judged=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tail-latency diagnostic: is HYBRID's p99 the endpoint or the load?
# MAGIC The same 120 queries, sequentially and with 8 concurrent callers (the evaluation's concurrency). Logged as a diagnostic run.

# COMMAND ----------

sample = list(src.sample(120, random_state=3).index)
hybrid = pr.PlainRetriever(backend, "HYBRID", k=K)


def timed(query: str) -> float:
    t = time.perf_counter()
    hybrid(query)
    return (time.perf_counter() - t) * 1000


mlflow.tracing.disable()
sequential = [timed(q) for q in sample]
with ThreadPoolExecutor(max_workers=8) as pool:
    concurrent = list(pool.map(timed, sample))
mlflow.tracing.enable()

diag = pd.DataFrame({
    mode: {"p50": np.percentile(v, 50), "p90": np.percentile(v, 90), "p99": np.percentile(v, 99)}
    for mode, v in [("sequential", sequential), ("8 concurrent", concurrent)]
}).round(0)
display(diag)
with mlflow.start_run(run_name="09_latency_diagnostic"):
    mlflow.set_tags({"demo": "retrieval_diagnostic"})
    mlflow.log_metrics({f"hybrid_{m.replace(' ', '_')}_{p}_ms": float(diag.loc[p, m]) for m in diag.columns for p in diag.index})
