# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Enriched index: embed what shoppers search for
# MAGIC The original index embeds `description`: ~740 characters that start with marketing copy and bury the structured attributes
# MAGIC (`Category: …; Volts: 20V MAX; Kit or Tool Only: …`) at the end. Embedding models and the reranker (first ~2,000 characters)
# MAGIC see mostly prose.
# MAGIC
# MAGIC This notebook builds a second table + index whose embedding column is a compact **`search_text`**:
# MAGIC
# MAGIC `product name | Brand: … | Class: … | <structured attributes> | SKU … | UPC …`
# MAGIC
# MAGIC and re-runs the key strategies against it (the original index is untouched, so quality compares directly). The index lives
# MAGIC on `dao_ai_workshop_vs` (see the note in the next cell), so its **latency** isn't directly comparable to notebooks 03–09.
# MAGIC
# MAGIC | run | strategy on the enriched index |
# MAGIC |---|---|
# MAGIC | `10_enriched_hybrid` | plain HYBRID (vs `03_hybrid`) |
# MAGIC | `10_enriched_full_text` | plain FULL_TEXT (vs `03_full_text`) |
# MAGIC | `10_enriched_rerank` | HYBRID + `DatabricksReranker(columns_to_rerank=["search_text"])` (vs `04_*`) |
# MAGIC | `10_enriched_router_quality` | the full router + `ai_decide` (vs `09_router_quality`) |

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F

# ENRICHED_TABLE / ENRICHED_INDEX / ENRICHED_ENDPOINT come from 00_config. New Delta Sync indexes on `dbdemos_vs_endpoint` stalled
# in PROVISIONING_INITIAL_SNAPSHOT (twice, >1h, 0 rows) while its existing indexes kept syncing; an identical index on
# `dao_ai_workshop_vs` synced in ~15 min. Quality compares directly; latency is measured on a different (also STANDARD) endpoint.
from databricks.ai_search.client import VectorSearchClient

dbutils.widgets.dropdown("rebuild", "false", ["false", "true"])
vsc = VectorSearchClient(disable_notice=True)
index_ready = ENRICHED_INDEX in {i["name"] for i in vsc.list_indexes(ENRICHED_ENDPOINT).get("vector_indexes", [])} and \
    vsc.get_index(endpoint_name=ENRICHED_ENDPOINT, index_name=ENRICHED_INDEX).describe()["status"].get("ready", False)
REBUILD: bool = dbutils.widgets.get("rebuild") == "true" or not index_ready
print("rebuild table + index:", REBUILD)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build `search_text`
# MAGIC The attribute block is everything from `Category:` onward in the description; the marketing prose before it is dropped.

# COMMAND ----------

attributes = F.when(F.instr("description", "Category:") > 0, F.expr("substring(description, instr(description, 'Category:'))")).otherwise(F.col("description"))
enriched = spark.table(PRODUCTS_TABLE).withColumn(
    "search_text",
    F.concat_ws(
        " | ",
        F.col("product_name"),
        F.concat(F.lit("Brand: "), F.coalesce("brand_name", F.lit(""))),
        F.concat(F.lit("Class: "), F.coalesce("merchandise_class", F.lit(""))),
        F.substring(attributes, 1, 1200),
        F.concat(F.lit("SKU "), F.col("sku"), F.lit(" UPC "), F.col("upc")),
    ),
)

if REBUILD:
    spark.sql(f"DROP TABLE IF EXISTS {ENRICHED_TABLE}")
    enriched.write.option("delta.enableChangeDataFeed", "true").saveAsTable(ENRICHED_TABLE)
    spark.sql(f"ALTER TABLE {ENRICHED_TABLE} ALTER COLUMN product_id SET NOT NULL")
lengths = spark.table(ENRICHED_TABLE).select(F.avg(F.length("description")).alias("description_chars"), F.avg(F.length("search_text")).alias("search_text_chars"))
display(lengths)
display(spark.table(ENRICHED_TABLE).select("sku", "search_text").limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the index

# COMMAND ----------

if REBUILD:
    if ENRICHED_INDEX in {i["name"] for i in vsc.list_indexes(ENRICHED_ENDPOINT).get("vector_indexes", [])}:
        vsc.delete_index(endpoint_name=ENRICHED_ENDPOINT, index_name=ENRICHED_INDEX)
        while ENRICHED_INDEX in {i["name"] for i in vsc.list_indexes(ENRICHED_ENDPOINT).get("vector_indexes", [])}:
            time.sleep(10)
    vsc.create_delta_sync_index(
        endpoint_name=ENRICHED_ENDPOINT,
        index_name=ENRICHED_INDEX,
        source_table_name=ENRICHED_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="product_id",
        embedding_source_column="search_text",
        embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    )
    n_rows = spark.table(ENRICHED_TABLE).count()
    deadline = time.time() + 5400  # initial sync on a shared endpoint can take a while
    while True:
        status = vsc.get_index(endpoint_name=ENRICHED_ENDPOINT, index_name=ENRICHED_INDEX).describe()["status"]
        if status.get("ready") and status.get("indexed_row_count", 0) >= n_rows:
            break
        if time.time() > deadline:
            raise TimeoutError(status)
        time.sleep(30)
    print(status)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate on the enriched index
# MAGIC Tool factories: `pr.create_search_tool(config_enriched, ...)`, `pr.create_router_tool(config_enriched, vocab)`.

# COMMAND ----------

from product_retrieval.rerankers import AIDecideReranker

backend = pr.build_backend(config_enriched)
strategies = {
    "10_enriched_hybrid": pr.PlainRetriever(backend, "HYBRID", k=K, name="10_enriched_hybrid"),
    "10_enriched_full_text": pr.PlainRetriever(backend, "FULL_TEXT", k=K, name="10_enriched_full_text"),
    "10_enriched_rerank": pr.PlainRetriever(backend, "HYBRID", k=K, rerank_columns=["search_text"], candidates=50, name="10_enriched_rerank"),
    "10_enriched_router_fast": pr.QueryRouter(backend, vocabulary(), name="10_enriched_router_fast"),
    "10_enriched_router_quality": pr.QueryRouter(backend, vocabulary(), reranker=AIDecideReranker("noul", details_chars=300),
                                                 candidates=12, name="10_enriched_router_quality"),
}
for name, retriever in strategies.items():
    run_strategy(name, retriever, {"index": ENRICHED_INDEX, "endpoint": ENRICHED_ENDPOINT, "embedding_column": "search_text"})
