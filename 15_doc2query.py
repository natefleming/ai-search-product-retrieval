# Databricks notebook source
# MAGIC %md
# MAGIC # 15 · Doc2Query document expansion
# MAGIC Product-side expansion: an LLM writes the searches a shopper might type for each product, offline, and they are indexed with
# MAGIC the product. Industry evidence: AliExpress SAM-D2Q (+3.4% GMV in production, 2026); Doc2Query++ (2025) shows appending
# MAGIC generated queries helps keyword retrieval but can blur dense embeddings, and recommends a **separate pseudo-query index fused
# MAGIC with RRF**. Both designs are tested:
# MAGIC
# MAGIC | run | design | tool factory |
# MAGIC |---|---|---|
# MAGIC | `15_d2q_hybrid` / `15_d2q_router_quality` | (a) pseudo-queries appended to `search_text` in a new index | `create_search_tool` / `create_router_tool` on the new index |
# MAGIC | `15_fusion_enriched_pseudoq` | (b) separate pseudo-query index, RRF-fused with the enriched index | `pr.create_fusion_tool([...])` |
# MAGIC
# MAGIC **Caveat:** the evaluation queries were also LLM-generated from product text, so Doc2Query is likely to look better here than
# MAGIC on real shopper traffic; treat gains as optimistic until confirmed on real query logs.
# MAGIC At ~100M products, generation cost is the main consideration; tokens and throughput are logged.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval._retry import with_backoff
from product_retrieval.rerankers import AIDecideReranker
from databricks.ai_search.client import VectorSearchClient
from pyspark.sql import functions as F

D2Q_TABLE = f"{CATALOG}.{SCHEMA}.products_doc2query"
D2Q_INDEX = f"{CATALOG}.{SCHEMA}.products_d2q_index"
PSEUDO_TABLE = f"{CATALOG}.{SCHEMA}.products_pseudoq"
PSEUDO_INDEX = f"{CATALOG}.{SCHEMA}.products_pseudoq_index"
GEN_ENDPOINT = LLM_ENDPOINT
dbutils.widgets.dropdown("regenerate", "false", ["false", "true"])
vsc = VectorSearchClient(disable_notice=True)

D2Q_PROMPT = (
    "Write 5 different searches a hardware-store shopper might type to find this exact product: vary the wording (product type, "
    "use case, key specs, brand, colloquial names). One per line, no numbering, no quotes.\n\n"
    "Product: {name}\nBrand: {brand}\nCategory: {category}\nDetails: {details}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate pseudo-queries (offline, once per product)

# COMMAND ----------

if dbutils.widgets.get("regenerate") == "true" or not spark.catalog.tableExists(D2Q_TABLE):
    products = spark.table(ENRICHED_TABLE).select("product_id", "product_name", "brand_name", "merchandise_class", "description").toPandas()
    gen = chat(GEN_ENDPOINT, temperature=0.7, max_tokens=400)
    usage = {"tokens": 0}

    def generate(row: Any) -> str:
        prompt = D2Q_PROMPT.format(name=row.product_name, brand=row.brand_name, category=row.merchandise_class, details=row.description[:600])
        try:
            msg = with_backoff(lambda: gen.invoke(prompt))
            usage["tokens"] += int((msg.response_metadata or {}).get("usage", {}).get("total_tokens", 0))
            lines = [l.strip(" -•\"'") for l in message_text(msg).splitlines() if l.strip()]
            return "; ".join(lines[:5])
        except Exception:
            return ""

    mlflow.langchain.autolog(disable=True)
    started = time.time()
    with ThreadPoolExecutor(max_workers=24) as pool:
        products["pseudo_queries"] = list(pool.map(generate, products.itertuples()))
    minutes = (time.time() - started) / 60
    mlflow.langchain.autolog()
    spark.createDataFrame(products[["product_id", "pseudo_queries"]]).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(D2Q_TABLE)
    empty = int((products.pseudo_queries == "").sum())
    print(f"{len(products):,} products in {minutes:.0f} min, {usage['tokens']:,} tokens, {empty} empty")
    with mlflow.start_run(run_name="15_doc2query_generation"):
        mlflow.set_tags({"demo": "retrieval_diagnostic"})
        mlflow.log_metrics({"products": len(products), "minutes": minutes, "total_tokens": usage["tokens"], "empty": empty,
                            "tokens_per_product": usage["tokens"] / len(products)})
display(spark.table(D2Q_TABLE).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build both indexes

# COMMAND ----------


def build_index(table: str, index: str, source_column: str) -> None:
    if index in {i["name"] for i in vsc.list_indexes(ENRICHED_ENDPOINT).get("vector_indexes", [])}:
        status = vsc.get_index(endpoint_name=ENRICHED_ENDPOINT, index_name=index).describe()["status"]
        if status.get("ready") and dbutils.widgets.get("regenerate") != "true":
            return
        vsc.delete_index(endpoint_name=ENRICHED_ENDPOINT, index_name=index)
        while index in {i["name"] for i in vsc.list_indexes(ENRICHED_ENDPOINT).get("vector_indexes", [])}:
            time.sleep(10)
    vsc.create_delta_sync_index(endpoint_name=ENRICHED_ENDPOINT, index_name=index, source_table_name=table, pipeline_type="TRIGGERED",
                                primary_key="product_id", embedding_source_column=source_column, embedding_model_endpoint_name=EMBEDDING_ENDPOINT)
    n_rows, deadline = spark.table(table).count(), time.time() + 5400
    while True:
        status = vsc.get_index(endpoint_name=ENRICHED_ENDPOINT, index_name=index).describe()["status"]
        if status.get("ready") and status.get("indexed_row_count", 0) >= n_rows:
            return
        if time.time() > deadline:
            raise TimeoutError(status)
        time.sleep(30)


d2q = spark.table(D2Q_TABLE)
for table, df in [
    (f"{CATALOG}.{SCHEMA}.products_d2q", spark.table(ENRICHED_TABLE).join(d2q, "product_id")
        .withColumn("search_text", F.concat_ws(" | Searches: ", "search_text", "pseudo_queries"))),
    (PSEUDO_TABLE, spark.table(ENRICHED_TABLE).join(d2q, "product_id").where("pseudo_queries != ''")
        .withColumn("search_text", F.col("pseudo_queries"))),
]:
    if dbutils.widgets.get("regenerate") == "true" or not spark.catalog.tableExists(table):
        spark.sql(f"DROP TABLE IF EXISTS {table}")
        df.write.option("delta.enableChangeDataFeed", "true").saveAsTable(table)
        spark.sql(f"ALTER TABLE {table} ALTER COLUMN product_id SET NOT NULL")
build_index(f"{CATALOG}.{SCHEMA}.products_d2q", D2Q_INDEX, "search_text")
build_index(PSEUDO_TABLE, PSEUDO_INDEX, "search_text")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate

# COMMAND ----------

dbutils.widgets.dropdown("evaluate", "true", ["true", "false"])
if dbutils.widgets.get("evaluate") == "false":  # build-only run (evaluations run sequentially in the main chain)
    dbutils.notebook.exit("built")


config_d2q = pr.RetrievalConfig(index_name=D2Q_INDEX, llm_endpoint=LLM_ENDPOINT)
config_pseudo = pr.RetrievalConfig(index_name=PSEUDO_INDEX, llm_endpoint=LLM_ENDPOINT)
d2q_backend = pr.build_backend(config_d2q)
enriched_hybrid = pr.PlainRetriever(pr.build_backend(config_enriched), "HYBRID", k=25, name="enriched")
pseudo_hybrid = pr.PlainRetriever(pr.build_backend(config_pseudo), "HYBRID", k=25, name="pseudoq")
strategies = {
    "15_d2q_hybrid": pr.PlainRetriever(d2q_backend, "HYBRID", k=K, name="15_d2q_hybrid"),
    "15_d2q_router_quality": pr.QueryRouter(d2q_backend, vocabulary(), reranker=AIDecideReranker("noul", details_chars=300),
                                            candidates=12, name="15_d2q_router_quality"),
    "15_fusion_enriched_pseudoq": pr.FusionRetriever([(enriched_hybrid, 1.0), (pseudo_hybrid, 1.0)], k=K, name="15_fusion_enriched_pseudoq"),
}
for name, retriever in strategies.items():
    run_strategy(name, retriever, {"doc2query_llm": GEN_ENDPOINT, "queries_per_product": 5,
                                   "design": "dual-index RRF" if "fusion" in name else "appended field"})
