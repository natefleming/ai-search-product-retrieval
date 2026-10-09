# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · dao-ai instructed retrieval
# MAGIC [dao-ai](https://pypi.org/project/dao-ai/) ships a configurable **instructed retriever** behind `dao_ai.tools.create_ai_search_tool`,
# MAGIC driven by the `dao_ai.config` object model: LLM **decomposition** into filtered subqueries → parallel search → **RRF** →
# MAGIC **rerank** (Databricks server-side, FlashRank, or an instruction-aware LLM). The package wraps it behind the same tool
# MAGIC contract: `pr.create_dao_ai_instructed_tool(config, endpoint, source_table, vocabulary, rerank=...)`.
# MAGIC
# MAGIC | run | rerank |
# MAGIC |---|---|
# MAGIC | `07_dao_ai_instructed` | Databricks server-side on name, category, description |
# MAGIC | `07_dao_ai_instructed_flashrank` | FlashRank `ms-marco-MiniLM-L-12-v2` (CPU) |
# MAGIC | `07_dao_ai_instructed_llm_rerank` | server-side + instruction-aware LLM rerank |
# MAGIC
# MAGIC **Operational notes:** decomposition uses `with_structured_output`, which can't parse gpt-oss-120b's reasoning blocks, so it
# MAGIC uses Claude Haiku 4.5 (dao-ai's own example model) with a Sonnet fallback. dao-ai falls back *silently* to the unfiltered
# MAGIC query when decomposition errors; under 8 concurrent workers the pay-per-token endpoint returned 429 for half the queries, so
# MAGIC these runs use 4 workers and log `decomposition_errors` from the traces.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow "dao-ai[rerank]"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.strategies.dao_ai import DaoAIInstructedRetriever

os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = "4"
LIB_VERSIONS["dao-ai"] = md.version("dao-ai")
vocab = vocabulary()
retrievers = {
    f"07_dao_ai_instructed{suffix}": DaoAIInstructedRetriever(
        config_original, vocab, vector_search_endpoint=VS_ENDPOINT, source_table=PRODUCTS_TABLE, rerank=rerank,
        name=f"07_dao_ai_instructed{suffix}",
    )
    for rerank, suffix in [("databricks", ""), ("flashrank", "_flashrank"), ("llm", "_llm_rerank")]
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## One call
# MAGIC Open the trace: `decompose_query` → subqueries with filters → RRF → rerank.

# COMMAND ----------

example = eval_source().query("query_type == 'exclusion'").iloc[0]
print(example.name)
for p in retrievers["07_dao_ai_instructed"](example.name).products[:5]:
    print(f"  {'✓' if p.id in example.expected_skus else ' '} {p.brand:<14} {p.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate

# COMMAND ----------


def log_decomposition_errors(run_id: str) -> int:
    """Count decompose_query spans that errored (dao-ai falls back to the unfiltered query on these)."""
    traces = mlflow.search_traces(run_id=run_id, max_results=1000, return_type="list")
    errors = sum(1 for t in traces for sp in t.data.spans if sp.name == "decompose_query" and str(sp.status.status_code).endswith("ERROR"))
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("decomposition_errors", errors)
    return errors


for name, retriever in retrievers.items():
    run_id = run_strategy(name, retriever, {"framework": "dao-ai", "query_type": "HYBRID", "num_results": 50,
                                            "decomposition_llm": "databricks-claude-haiku-4-5", "max_subqueries": 3})
    print(name, "decomposition errors:", log_decomposition_errors(run_id))

run_strategy("07_dao_ai_instructed", retrievers["07_dao_ai_instructed"], {"framework": "dao-ai"}, judged=True)
