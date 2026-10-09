# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest the product catalog & rebuild the AI Search index
# MAGIC Drops and recreates `retail_consumer_goods.product_search.products` from the raw parquet and rebuilds a Delta Sync
# MAGIC AI Search index (managed `databricks-gte-large-en` embeddings on `description`) that supports ANN, FULL_TEXT and HYBRID queries.
# MAGIC
# MAGIC **Prereq:** `products.snappy.parquet` uploaded to `/Volumes/retail_consumer_goods/product_search/raw/`.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the catalog
# MAGIC Column comments matter: `VectorSearchRetrieverTool(dynamic_filter=True)` surfaces them to the LLM when it writes filters.

# COMMAND ----------

from pyspark.sql import functions as F

COLUMN_COMMENTS: dict[str, str] = {
    "product_id": "Unique product identifier.",
    "sku": "8-digit SKU, e.g. '00176279'.",
    "upc": "UPC barcode.",
    "brand_name": "Brand in UPPERCASE, e.g. 'MILWAUKEE', 'DEWALT', 'BLACK+DECKER'. Exact match.",
    "product_name": "Full product title including brand, size and model details.",
    "merchandise_class": "Merchandise class in UPPERCASE, e.g. 'HEAVY-DUTY POWER TOOLS', 'LED LIGHT BULBS'. Exact match.",
    "class_cd": "Merchandise class code.",
    "description": "Marketing copy followed by structured attributes ('Category: ...; Brand Name: ...; Volts: ...').",
}

raw = spark.read.parquet(f"{RAW_VOLUME_PATH}/products.snappy.parquet")
products = raw.select(
    F.col("product_id").cast("bigint"),
    *[F.col(c).cast("string") for c in ["sku", "upc", "brand_name", "product_name", "merchandise_class", "class_cd", "description"]],
).dropDuplicates(["product_id"])

spark.sql(f"DROP TABLE IF EXISTS {PRODUCTS_TABLE}")
(
    products.write.option("delta.enableChangeDataFeed", "true")
    .saveAsTable(PRODUCTS_TABLE, comment="Hardware-store product catalog for the AI Search retrieval demo")
)
for col, comment in COLUMN_COMMENTS.items():
    spark.sql(f"ALTER TABLE {PRODUCTS_TABLE} ALTER COLUMN {col} COMMENT \"{comment}\"")
spark.sql(f"ALTER TABLE {PRODUCTS_TABLE} ALTER COLUMN product_id SET NOT NULL")

print(f"{PRODUCTS_TABLE}: {spark.table(PRODUCTS_TABLE).count():,} rows")
display(spark.table(PRODUCTS_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rebuild the AI Search index
# MAGIC Triggered Delta Sync index with Databricks-managed embeddings; all columns are synced so they can be returned, filtered and faceted.

# COMMAND ----------

from databricks.ai_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

if INDEX_NAME in {i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])}:
    vsc.delete_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME)
    while INDEX_NAME in {i["name"] for i in vsc.list_indexes(VS_ENDPOINT).get("vector_indexes", [])}:
        time.sleep(10)

index = vsc.create_delta_sync_index(
    endpoint_name=VS_ENDPOINT,
    index_name=INDEX_NAME,
    source_table_name=PRODUCTS_TABLE,
    pipeline_type="TRIGGERED",
    primary_key="product_id",
    embedding_source_column="description",
    embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
)

# COMMAND ----------

deadline = time.time() + 3600
while True:
    status = vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=INDEX_NAME).describe()["status"]
    if status.get("ready") and status.get("indexed_row_count", 0) >= products.count():
        break
    if time.time() > deadline:
        raise TimeoutError(status)
    print(status.get("detailed_state"), status.get("indexed_row_count"))
    time.sleep(30)
print(status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test through the package

# COMMAND ----------

tool = pr.create_search_tool(config_original)
for d in json.loads(tool.invoke({"query": "20V cordless drill kit with battery"}))[:5]:
    print(d["metadata"]["id"], d["metadata"]["brand"], "|", d["metadata"]["name"])
