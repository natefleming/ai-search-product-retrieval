# Databricks notebook source
# MAGIC %md
# MAGIC # Integration test: every tool factory, live on AI Search
# MAGIC Installs the published wheel, builds each LangChain tool from its factory, and checks the output contract plus a few
# MAGIC behaviours (exact SKU lookup, exclusion filter, reranker fallback).

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow "dao-ai[rerank]"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import json
import time

import product_retrieval as pr

dbutils.widgets.text("index_name", "retail_consumer_goods.product_search.products_enriched_index")
dbutils.widgets.text("vocabulary_table", "retail_consumer_goods.product_search.products_enriched")
dbutils.widgets.text("vector_search_endpoint", "dao_ai_workshop_vs")
dbutils.widgets.text("warehouse_id", "", "SQL warehouse ID (for the warehouse-loaded vocabulary check)")
w = {k: dbutils.widgets.get(k) for k in ["index_name", "vocabulary_table", "vector_search_endpoint", "warehouse_id"]}

config = pr.RetrievalConfig(index_name=w["index_name"])
vocabulary = pr.CatalogVocabulary.from_spark(spark, w["vocabulary_table"])
print(pr.__version__, len(vocabulary.brands), "brands", len(vocabulary.categories), "categories")

# COMMAND ----------

QUERIES = {
    "identifier": "Do you have item 00176279?",
    "exclusion": "brushless impact driver kit, anything but DeWalt",
    "general": "60W equivalent soft white LED bulbs, 4 pack",
}

tools = {
    "search_hybrid": pr.create_search_tool(config),
    "search_full_text": pr.create_search_tool(config, query_type="FULL_TEXT"),
    "search_rerank": pr.create_search_tool(config, rerank_columns=["product_name", "description"], candidates=50),
    "guarded_filter": pr.create_guarded_filter_tool(config, vocabulary),
    "guarded_filter_ai_decide": pr.create_guarded_filter_tool(config, vocabulary, reranker="ai_decide_noul"),
    "instructed_noul": pr.create_instructed_tool(config),
    "instructed_listwise": pr.create_instructed_tool(config, reranker="ai_decide_listwise", candidates=12),
    "facet_guided": pr.create_facet_guided_tool(config),
    "router_fast": pr.create_router_tool(config, vocabulary, reranker="none"),
    "router_quality": pr.create_router_tool(config, vocabulary),
    "router_ai_decide": pr.create_router_tool(config, classifier="ai_decide", reranker="none"),
    "fusion_ann_full_text": pr.create_fusion_tool([(config, "ANN", 1.0), (config, "FULL_TEXT", 1.0)]),
    "dao_ai_instructed": pr.create_dao_ai_instructed_tool(
        config, w["vector_search_endpoint"], w["vocabulary_table"], vocabulary, embedding_source_column="search_text"
    ),
}
if w["warehouse_id"]:  # vocabulary loaded through a SQL warehouse (runtimes without Spark)
    tools["router_quality_warehouse_vocab"] = pr.create_router_tool(config, vocabulary_table=w["vocabulary_table"], warehouse_id=w["warehouse_id"])

report: dict[str, dict] = {}
for name, tool in tools.items():
    report[name] = {}
    for qtype, q in QUERIES.items():
        t = time.perf_counter()
        try:
            docs = json.loads(tool.invoke({"query": q}))
            meta = docs[0]["metadata"] if docs else {}
            report[name][qtype] = {"n": len(docs), "top": meta.get("id"), "top_brand": meta.get("brand"),
                                   "dewalt_in_top10": any(d["metadata"].get("brand") == "DEWALT" for d in docs),
                                   "ms": round((time.perf_counter() - t) * 1000)}
        except Exception as e:
            report[name][qtype] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
print(json.dumps(report, indent=1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dynamic-filter tool (VectorSearchRetrieverTool): the calling agent writes filters

# COMMAND ----------

from databricks_langchain import ChatDatabricks

dyn = pr.create_dynamic_filter_tool(config)
msg = ChatDatabricks(endpoint="databricks-gpt-oss-120b", temperature=0, extra_params={"reasoning_effort": "low"}).bind_tools([dyn]).invoke(
    "Find a brushless impact driver kit, but not DeWalt. Brands are stored in UPPERCASE."
)
dyn_call = msg.tool_calls[0]["args"] if msg.tool_calls else None
dyn_out = str(dyn.invoke(dyn_call))[:300] if dyn_call else None
print(dyn_call, dyn_out)

# COMMAND ----------

checks = {
    "all_tools_returned_results": all(v.get("n", 0) > 0 for r in report.values() for v in r.values()),
    "router_identifier_exact": report["router_quality"]["identifier"].get("top") == "00176279",
    "router_exclusion_no_dewalt": not report["router_quality"]["exclusion"].get("dewalt_in_top10", True),
    "warehouse_vocab_same_result": report["router_quality_warehouse_vocab"]["exclusion"].get("top") is not None,
    "dynamic_filter_tool_called": dyn_call is not None,
}
print(checks)
dbutils.notebook.exit(json.dumps({"checks": checks, "report": report, "dynamic_filter_call": dyn_call}, default=str))
