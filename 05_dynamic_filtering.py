# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Dynamic filtering (LLM-generated filters)
# MAGIC `VectorSearchRetrieverTool(dynamic_filter=True)` exposes a typed `filters` argument to the LLM, so *"brushless impact driver,
# MAGIC anything but DeWalt"* becomes `query="brushless impact driver"` + `filters=[{"key": "brand_name NOT", "value": "DEWALT"}]`, a
# MAGIC hard constraint applied inside the index. Filters are exact-match, so wrong values (casing, a guessed category) silently
# MAGIC return nothing or the wrong slice.
# MAGIC
# MAGIC | run | what it does | tool factory |
# MAGIC |---|---|---|
# MAGIC | — | the calling agent's LLM writes the filters | `pr.create_dynamic_filter_tool(config)` |
# MAGIC | `05_dynamic_filter` | internal planner LLM, raw filters | `pr.create_guarded_filter_tool(config, vocab, guarded=False)` |
# MAGIC | `05_dynamic_filter_guarded` | values validated against the catalog; empty result → unfiltered | `pr.create_guarded_filter_tool(config, vocab)` |
# MAGIC | `05_dynamic_filter_guarded_rerank` | guarded + Databricks reranker on request-targeted columns | `…(targeted_server_rerank=True)` |

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.column_selection import RerankColumnSelector
from product_retrieval.strategies.filtering import LLMFilterRetriever, guard_filters

backend = pr.build_backend(config_original)
vocab = vocabulary()
raw = LLMFilterRetriever(config_original, backend, vocab, guarded=False, name="05_dynamic_filter")
guarded = LLMFilterRetriever(config_original, backend, vocab, guarded=True, name="05_dynamic_filter_guarded")
guarded_rerank = LLMFilterRetriever(config_original, backend, vocab, guarded=True, candidates=50,
                                    column_selector=RerankColumnSelector(backend.catalog), name="05_dynamic_filter_guarded_rerank")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The calling agent writes the filters (`create_dynamic_filter_tool`)

# COMMAND ----------

dyn_tool = pr.create_dynamic_filter_tool(config_original)
msg = chat(LLM_ENDPOINT).bind_tools([dyn_tool]).invoke("Find a brushless impact driver kit, but not DeWalt. Brand values are stored in UPPERCASE.")
print(msg.tool_calls[0]["args"] if msg.tool_calls else msg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the internal planner decides (raw vs guarded)

# COMMAND ----------

src = eval_source()
for qtype in ["exclusion", "brand_constrained", "attribute_constrained", "descriptive"]:
    q = src.query(f"query_type == '{qtype}'").index[0]
    text, filters = raw.plan(q)
    clean = guard_filters(filters, vocab, "brand_name", "merchandise_class")
    print(f"[{qtype}] {q}\n   -> query={text!r} raw={[(f.column, f.op, f.value) for f in filters]} "
          f"guarded={[(f.column, f.op, f.value) for f in clean]}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate

# COMMAND ----------

base = {"query_type": "HYBRID", "dynamic_filter": True, "llm": LLM_ENDPOINT}
run_strategy("05_dynamic_filter", raw, {**base, "guarded": False})
run_strategy("05_dynamic_filter_guarded", guarded, {**base, "guarded": True})
rerank_params = {**base, "guarded": True, "reranker": "DatabricksReranker", "columns_to_rerank": "per-request (ai_decide profile)", "num_results": 50}
run_strategy("05_dynamic_filter_guarded_rerank", guarded_rerank, rerank_params)
run_strategy("05_dynamic_filter_guarded_rerank", guarded_rerank, rerank_params, judged=True)
