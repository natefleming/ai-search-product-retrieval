# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Baseline: plain RAG retrieval
# MAGIC The starting point most teams ship: send the shopper's text straight to the index and take the top 10.
# MAGIC Three AI Search query types, built with `product_retrieval.create_search_tool` / `PlainRetriever`:
# MAGIC
# MAGIC * **ANN**: dense embeddings only (semantic)
# MAGIC * **FULL_TEXT**: keyword / BM25-style only
# MAGIC * **HYBRID**: both, fused with RRF. This becomes the **baseline** every later strategy is compared to.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.evaluation import answer

backend = pr.build_backend(config_original)
retrievers = {qt: pr.PlainRetriever(backend, qt, k=K) for qt in ["ANN", "FULL_TEXT", "HYBRID"]}


def show(result: pr.RetrievalResult, expected: list[str], n: int = 5) -> pd.DataFrame:
    return pd.DataFrame([{"rank": i, "hit": "✓" if p.id in expected else "", "sku": p.id, "brand": p.brand, "product": p.name}
                         for i, p in enumerate(result.products[:n], start=1)])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Same query, three query types
# MAGIC An exclusion query shows the core problem: retrieval matches the words, including the brand the shopper rejected.

# COMMAND ----------

example = eval_source().query("query_type == 'exclusion'").iloc[0]
print(example.name, "| constraints:", example.constraints)
for qt, retriever in retrievers.items():
    print(f"\n{qt}")
    display(show(retriever(example.name), list(example.expected_skus)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## End-to-end RAG answer
# MAGIC Retrieval feeds generation; if the excluded brand is retrieved, it tends to be recommended.

# COMMAND ----------

print(answer(example.name, retrievers["HYBRID"](example.name), LLM_ENDPOINT))

# COMMAND ----------

# MAGIC %md
# MAGIC ## As a LangChain tool
# MAGIC The same strategy, packaged for an agent.

# COMMAND ----------

tool = pr.create_search_tool(config_original, query_type="HYBRID")
print(tool.name, "→", json.loads(tool.invoke({"query": example.name}))[0]["metadata"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate on the full dataset
# MAGIC Same MLflow dataset and scorers for every run, so runs compare side by side in the experiment's **Evaluations** tab.

# COMMAND ----------

for qt, retriever in retrievers.items():
    run_strategy(f"03_{qt.lower()}", retriever, {"query_type": qt, "num_results": K, "reranker": "none"})

run_strategy(BASELINE_RUN, retrievers["HYBRID"], {"query_type": "HYBRID", "num_results": K, "reranker": "none"}, judged=True)
