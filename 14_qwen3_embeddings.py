# Databricks notebook source
# MAGIC %md
# MAGIC # 14 · Instruction-aware embeddings: Qwen3-Embedding (self-managed vectors in AI Search)
# MAGIC Qwen3-Embedding models are instruction-aware: queries are embedded with a task instruction, documents without one.
# MAGIC Databricks hosts `databricks-qwen3-embedding-0-6b` (1024-dim) on the Foundation Model API with an `instruction` field.
# MAGIC Because the query and document sides are embedded differently, the index uses **self-managed** embeddings: we compute
# MAGIC the document vectors and store them in AI Search; the package embeds queries (`RetrievalConfig.query_embedding`).
# MAGIC
# MAGIC Same `search_text` as the gte-large enriched index (notebook 10), so the comparison isolates the embedding model.
# MAGIC At a ~100M-row production scale, switching embedding models means re-embedding the catalog; the throughput measured here is logged.
# MAGIC
# MAGIC | run | query embedding |
# MAGIC |---|---|
# MAGIC | `14_qwen3_ann` / `14_qwen3_hybrid` | with product-search instruction |
# MAGIC | `14_qwen3_hybrid_noinstr` | no instruction (ablation) |
# MAGIC | `14_qwen3_router_quality` | instruction; the quality router on this index |

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
from pyspark.sql.types import ArrayType, FloatType

QWEN_ENDPOINT = "databricks-qwen3-embedding-0-6b"
QWEN_TABLE = f"{CATALOG}.{SCHEMA}.products_qwen3"
QWEN_INDEX = f"{CATALOG}.{SCHEMA}.products_qwen3_index"
DIM, BATCH = 1024, 32
dbutils.widgets.dropdown("rebuild", "false", ["false", "true"])
vsc = VectorSearchClient(disable_notice=True)
exists = QWEN_INDEX in {i["name"] for i in vsc.list_indexes(ENRICHED_ENDPOINT).get("vector_indexes", [])}
REBUILD = dbutils.widgets.get("rebuild") == "true" or not exists

# COMMAND ----------

# MAGIC %md
# MAGIC ## Embed the catalog (document side: no instruction)

# COMMAND ----------


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = with_backoff(lambda: w.api_client.do("POST", f"/serving-endpoints/{QWEN_ENDPOINT}/invocations", body={"input": texts}))
    return [d["embedding"] for d in sorted(resp["data"], key=lambda d: d["index"])]


if REBUILD:
    rows = spark.table(ENRICHED_TABLE).select("product_id", "search_text").toPandas()
    batches = [rows.search_text.iloc[i:i + BATCH].tolist() for i in range(0, len(rows), BATCH)]
    started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        vectors = [v for batch in pool.map(embed_batch, batches) for v in batch]
    embed_minutes = (time.time() - started) / 60
    rows["embedding"] = vectors
    emb = spark.createDataFrame(rows[["product_id", "embedding"]]).withColumn("embedding", F.col("embedding").cast(ArrayType(FloatType())))
    spark.sql(f"DROP TABLE IF EXISTS {QWEN_TABLE}")
    spark.table(ENRICHED_TABLE).join(emb, "product_id").write.option("delta.enableChangeDataFeed", "true").saveAsTable(QWEN_TABLE)
    spark.sql(f"ALTER TABLE {QWEN_TABLE} ALTER COLUMN product_id SET NOT NULL")
    print(f"embedded {len(rows):,} products in {embed_minutes:.1f} min ({len(rows) / embed_minutes / 60:.0f} rows/s)")
    with mlflow.start_run(run_name="14_qwen3_embedding_throughput"):
        mlflow.set_tags({"demo": "retrieval_diagnostic"})
        mlflow.log_metrics({"rows": len(rows), "minutes": embed_minutes, "rows_per_second": len(rows) / embed_minutes / 60})

# COMMAND ----------

if REBUILD:
    if exists:
        vsc.delete_index(endpoint_name=ENRICHED_ENDPOINT, index_name=QWEN_INDEX)
        while QWEN_INDEX in {i["name"] for i in vsc.list_indexes(ENRICHED_ENDPOINT).get("vector_indexes", [])}:
            time.sleep(10)
    vsc.create_delta_sync_index(endpoint_name=ENRICHED_ENDPOINT, index_name=QWEN_INDEX, source_table_name=QWEN_TABLE,
                                pipeline_type="TRIGGERED", primary_key="product_id", embedding_dimension=DIM,
                                embedding_vector_column="embedding")
    n_rows = spark.table(QWEN_TABLE).count()
    deadline = time.time() + 5400
    while True:
        status = vsc.get_index(endpoint_name=ENRICHED_ENDPOINT, index_name=QWEN_INDEX).describe()["status"]
        if status.get("ready") and status.get("indexed_row_count", 0) >= n_rows:
            break
        if time.time() > deadline:
            raise TimeoutError(status)
        time.sleep(30)
    print(status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate
# MAGIC Tool factories: any factory with `RetrievalConfig(index_name=QWEN_INDEX, query_embedding=pr.QueryEmbedding(...))`.

# COMMAND ----------

dbutils.widgets.dropdown("evaluate", "true", ["true", "false"])
if dbutils.widgets.get("evaluate") == "false":  # build-only run (evaluations run sequentially in the main chain)
    dbutils.notebook.exit("built")


with_instruction = pr.RetrievalConfig(index_name=QWEN_INDEX, llm_endpoint=LLM_ENDPOINT, query_embedding=pr.QueryEmbedding(endpoint=QWEN_ENDPOINT))
no_instruction = pr.RetrievalConfig(index_name=QWEN_INDEX, llm_endpoint=LLM_ENDPOINT,
                                    query_embedding=pr.QueryEmbedding(endpoint=QWEN_ENDPOINT, instruction=None))
backend, backend_plain = pr.build_backend(with_instruction), pr.build_backend(no_instruction)
strategies = {
    "14_qwen3_ann": pr.PlainRetriever(backend, "ANN", k=K, name="14_qwen3_ann"),
    "14_qwen3_hybrid": pr.PlainRetriever(backend, "HYBRID", k=K, name="14_qwen3_hybrid"),
    "14_qwen3_hybrid_noinstr": pr.PlainRetriever(backend_plain, "HYBRID", k=K, name="14_qwen3_hybrid_noinstr"),
    "14_qwen3_router_quality": pr.QueryRouter(backend, vocabulary(), reranker=AIDecideReranker("noul", details_chars=300),
                                              candidates=12, name="14_qwen3_router_quality"),
}
for name, retriever in strategies.items():
    run_strategy(name, retriever, {"index": QWEN_INDEX, "embedding": QWEN_ENDPOINT, "instruction": "noinstr" not in name})
