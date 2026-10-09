# Databricks notebook source
# MAGIC %md
# MAGIC # 18 · Scale test: ~5M products on a storage-optimized endpoint
# MAGIC The retailer's production index will hold ~100M embeddings. Results on a 38K catalog can overstate production quality (ANN recall,
# MAGIC filter selectivity and keyword statistics all change with corpus size), and production-scale indexes belong on
# MAGIC **storage-optimized** endpoints (~1B vectors per endpoint, SQL-string filters, ~400 ms reference ANN latency at 100M).
# MAGIC
# MAGIC This notebook adds **~5M realistic distractors** to the real catalog: variants of real products with the brand swapped for
# MAGIC another brand in the same category and sizes/counts perturbed, so they compete with the real items for the same queries.
# MAGIC Distractor SKUs/UPCs are non-numeric, so they can never collide with identifier queries.
# MAGIC
# MAGIC * `stage=build`: create the table, the storage-optimized endpoint and the index (hours: embedding ~5M rows)
# MAGIC * `stage=evaluate`: run the leading strategies against it; the only config change is `endpoint_type=STORAGE_OPTIMIZED`

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from databricks.ai_search.client import VectorSearchClient
from pyspark.sql import functions as F

dbutils.widgets.dropdown("stage", "build", ["build", "evaluate"])
dbutils.widgets.text("variants_per_product", "130")
STAGE = dbutils.widgets.get("stage")
VARIANTS = int(dbutils.widgets.get("variants_per_product"))
SCALE_TABLE = f"{CATALOG}.{SCHEMA}.products_scale"
SCALE_INDEX = f"{CATALOG}.{SCHEMA}.products_scale_index"
SCALE_ENDPOINT = "retrieval_scale_storage_optimized"
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build: distractors + storage-optimized index

# COMMAND ----------

if STAGE == "build":
    real = spark.table(ENRICHED_TABLE)
    # categories with no branded products can't supply a replacement brand; their products get no distractors
    brands_by_class = (real.groupBy("merchandise_class").agg(F.collect_set("brand_name").alias("class_brands"))
                       .where(F.size("class_brands") > 0))
    variants = (
        real.join(brands_by_class, "merchandise_class")
        .withColumn("v", F.explode(F.sequence(F.lit(1), F.lit(VARIANTS))))
        .withColumn("new_brand", F.element_at("class_brands", (F.abs(F.hash("sku", "v")) % F.size("class_brands")) + 1))
        .withColumn("factor", F.element_at(F.array(*[F.lit(x) for x in ["0.5", "0.75", "1.5", "2", "3", "4"]]), (F.abs(F.hash("v", "upc")) % 6) + 1))
    )

    def perturb(col: str) -> F.Column:
        """Swap the brand and rewrite the first number (size / count / voltage) so the variant is a near-miss, not a duplicate."""
        swapped = F.expr(f"regexp_replace({col}, concat('(?i)', regexp_replace(coalesce(brand_name, '~~'), '([^A-Za-z0-9 ])', '\\\\\\\\$1')), new_brand)")
        number = F.regexp_extract(swapped, r"(\d+(?:\.\d+)?)", 1)
        scaled = F.format_number(number.cast("double") * F.col("factor").cast("double"), 0)
        return F.when(number != "", F.regexp_replace(swapped, r"\d+(?:\.\d+)?", scaled)).otherwise(swapped)

    distractors = variants.select(
        (F.lit(10**12) + F.col("product_id") * 1000 + F.col("v")).cast("bigint").alias("product_id"),
        F.concat(F.lit("S"), F.col("sku"), F.lpad(F.col("v").cast("string"), 3, "0")).alias("sku"),
        F.concat(F.lit("X"), F.col("upc"), F.lpad(F.col("v").cast("string"), 3, "0")).alias("upc"),
        F.col("new_brand").alias("brand_name"),
        perturb("product_name").alias("product_name"),
        "merchandise_class", "class_cd",
        perturb("description").alias("description"),
        perturb("search_text").alias("search_text"),
    )
    scale = real.select(*distractors.columns).unionByName(distractors)
    spark.sql(f"DROP TABLE IF EXISTS {SCALE_TABLE}")
    scale.write.option("delta.enableChangeDataFeed", "true").saveAsTable(SCALE_TABLE)
    spark.sql(f"ALTER TABLE {SCALE_TABLE} ALTER COLUMN product_id SET NOT NULL")
    n_rows = spark.table(SCALE_TABLE).count()
    print(f"{SCALE_TABLE}: {n_rows:,} rows")
    display(spark.table(SCALE_TABLE).where("sku like 'S%'").select("sku", "brand_name", "product_name").limit(5))

# COMMAND ----------

if STAGE == "build":
    if SCALE_ENDPOINT not in {e["name"] for e in vsc.list_endpoints().get("endpoints", [])}:
        vsc.create_endpoint_and_wait(name=SCALE_ENDPOINT, endpoint_type="STORAGE_OPTIMIZED")
    if SCALE_INDEX in {i["name"] for i in vsc.list_indexes(SCALE_ENDPOINT).get("vector_indexes", [])}:
        vsc.delete_index(endpoint_name=SCALE_ENDPOINT, index_name=SCALE_INDEX)
        while SCALE_INDEX in {i["name"] for i in vsc.list_indexes(SCALE_ENDPOINT).get("vector_indexes", [])}:
            time.sleep(30)
    vsc.create_delta_sync_index(
        endpoint_name=SCALE_ENDPOINT, index_name=SCALE_INDEX, source_table_name=SCALE_TABLE, pipeline_type="TRIGGERED",
        primary_key="product_id", embedding_source_column="search_text", embedding_model_endpoint_name=EMBEDDING_ENDPOINT,
    )
    started = time.time()
    while True:
        status = vsc.get_index(endpoint_name=SCALE_ENDPOINT, index_name=SCALE_INDEX).describe()["status"]
        if status.get("ready") and status.get("indexed_row_count", 0) >= n_rows:
            break
        if time.time() - started > 3.8 * 3600:
            raise TimeoutError(status)
        time.sleep(120)
    build_minutes = (time.time() - started) / 60
    print(status, f"{build_minutes:.0f} min")
    with mlflow.start_run(run_name="18_scale_index_build"):
        mlflow.set_tags({"demo": "retrieval_diagnostic"})
        mlflow.log_metrics({"rows": n_rows, "build_minutes": build_minutes})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate: same strategies, 5M-row storage-optimized index

# COMMAND ----------

if STAGE == "evaluate":
    from product_retrieval.rerankers import AIDecideReranker

    config_scale = pr.RetrievalConfig(index_name=SCALE_INDEX, endpoint_type=pr.EndpointType.STORAGE_OPTIMIZED, llm_endpoint=LLM_ENDPOINT)
    backend = pr.build_backend(config_scale)
    rows = spark.table(SCALE_TABLE).count()
    strategies = {
        "18_scale_hybrid": pr.PlainRetriever(backend, "HYBRID", k=K, name="18_scale_hybrid"),
        "18_scale_router_fast": pr.QueryRouter(backend, vocabulary(), name="18_scale_router_fast"),
        "18_scale_router_quality": pr.QueryRouter(backend, vocabulary(), reranker=AIDecideReranker("noul", details_chars=300),
                                                  candidates=12, name="18_scale_router_quality"),
        # deeper candidate pools: at scale the target competes with many near-identical variants
        "18_scale_router_quality_50": pr.QueryRouter(backend, vocabulary(), reranker=AIDecideReranker("noul", details_chars=300),
                                                     candidates=50, name="18_scale_router_quality_50"),
        "18_scale_router_bge_50": pr.QueryRouter(backend, vocabulary(), reranker=pr.ServingEndpointReranker("retrieval-bge-reranker-v2-m3"),
                                                 candidates=50, name="18_scale_router_bge_50"),
    }
    dbutils.widgets.text("only", "", "Comma-separated strategy names to run (default: all)")
    only = [n.strip() for n in dbutils.widgets.get("only").split(",") if n.strip()]
    strategies = {n: r for n, r in strategies.items() if not only or n in only}
    for name, retriever in strategies.items():
        run_strategy(name, retriever, {"index": SCALE_INDEX, "endpoint_type": "STORAGE_OPTIMIZED", "rows": rows})
