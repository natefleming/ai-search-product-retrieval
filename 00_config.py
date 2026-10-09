# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Shared config
# MAGIC Loaded by every notebook via `%run ./00_config`, after the notebook installs the `product-retrieval` wheel. All retrieval logic
# MAGIC lives in the package (`product_retrieval/`); this notebook only holds names, the evaluation harness wiring and a catalog lookup
# MAGIC used by the data-preparation notebooks.

# COMMAND ----------

import functools
import importlib.metadata as md
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import product_retrieval as pr
import mlflow
import pandas as pd
from product_retrieval._llm import chat, message_text
from product_retrieval.evaluation import Evaluator, load_per_query, load_runs, paired_bootstrap
from databricks.sdk import WorkspaceClient
from pydantic import BaseModel

# COMMAND ----------

CATALOG: str = "retail_consumer_goods"
SCHEMA: str = "product_search"
PRODUCTS_TABLE: str = f"{CATALOG}.{SCHEMA}.products"
ENRICHED_TABLE: str = f"{CATALOG}.{SCHEMA}.products_enriched"
INDEX_NAME: str = f"{CATALOG}.{SCHEMA}.products_index"
ENRICHED_INDEX: str = f"{CATALOG}.{SCHEMA}.products_enriched_index"
VS_ENDPOINT: str = "dbdemos_vs_endpoint"  # original index
ENRICHED_ENDPOINT: str = "dao_ai_workshop_vs"  # new indexes (dbdemos_vs_endpoint stopped provisioning new indexes)
EMBEDDING_ENDPOINT: str = "databricks-gte-large-en"
RAW_VOLUME_PATH: str = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
REPORT_VOLUME_PATH: str = f"{RAW_VOLUME_PATH}/reports"
WAREHOUSE_ID: str = os.environ.get("RETRIEVAL_WAREHOUSE_ID", "")  # only for CatalogVocabulary.from_warehouse outside Spark

LLM_ENDPOINT: str = "databricks-gpt-oss-120b"

EVAL_SOURCE_TABLE: str = f"{CATALOG}.{SCHEMA}.retrieval_eval_source"
EVAL_DATASET: str = f"{CATALOG}.{SCHEMA}.retrieval_eval"
JUDGE_DATASET: str = f"{CATALOG}.{SCHEMA}.retrieval_eval_judge"
HOLDOUT_SOURCE_TABLE: str = f"{CATALOG}.{SCHEMA}.retrieval_eval_holdout_source"
HOLDOUT_DATASET: str = f"{CATALOG}.{SCHEMA}.retrieval_eval_holdout"

K: int = 10
QUERY_TYPES: list[str] = ["known_item", "descriptive", "brand_constrained", "attribute_constrained", "exclusion", "identifier"]
BASELINE_RUN: str = "03_hybrid"

w = WorkspaceClient()
USER: str = w.current_user.me().user_name
EXPERIMENT_PATH: str = f"/Users/{USER}/ai-search-product-retrieval/retrieval_experiments"
experiment = mlflow.set_experiment(EXPERIMENT_PATH)
mlflow.langchain.autolog()
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "8")

LIB_VERSIONS: dict[str, str] = {
    p: md.version(p) for p in ["product-retrieval", "databricks-langchain", "databricks-ai-search", "databricks-sdk", "mlflow"]
}
print(LIB_VERSIONS)

# Retrieval configs for the two indexes (managed gte-large embeddings)
config_original = pr.RetrievalConfig(index_name=INDEX_NAME, llm_endpoint=LLM_ENDPOINT)
config_enriched = pr.RetrievalConfig(index_name=ENRICHED_INDEX, llm_endpoint=LLM_ENDPOINT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Catalog lookup & vocabulary

# COMMAND ----------


class CatalogProduct(BaseModel):
    product_id: int
    sku: str
    upc: str
    brand_name: str | None
    product_name: str
    merchandise_class: str | None
    description: str


@functools.cache
def product_catalog() -> dict[str, CatalogProduct]:
    rows = spark.table(PRODUCTS_TABLE).select(*CatalogProduct.model_fields).collect()
    return {r.sku: CatalogProduct(**r.asDict()) for r in rows}


@functools.cache
def vocabulary() -> pr.CatalogVocabulary:
    return pr.CatalogVocabulary.from_spark(spark, PRODUCTS_TABLE)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC Every strategy is evaluated on the same MLflow dataset rows (dev = 600 queries; holdout for finalists).

# COMMAND ----------


@functools.cache
def _query_types(table: str) -> dict[str, str]:
    return {r.query: r.query_type for r in spark.table(table).select("query", "query_type").collect()}


def eval_source(table: str = EVAL_SOURCE_TABLE) -> pd.DataFrame:
    return spark.table(table).toPandas().set_index("query")


def evaluator(holdout: bool = False) -> Evaluator:
    return Evaluator(
        dataset=HOLDOUT_DATASET if holdout else EVAL_DATASET,
        judge_dataset=JUDGE_DATASET,
        query_types=_query_types(HOLDOUT_SOURCE_TABLE if holdout else EVAL_SOURCE_TABLE),
        tags={"demo": "product_retrieval", "split": "holdout" if holdout else "dev", **LIB_VERSIONS},
        answer_llm=LLM_ENDPOINT,
    )


def run_strategy(name: str, retriever: Callable[[str], pr.RetrievalResult], params: dict[str, Any] | None = None,
                 judged: bool = False, holdout: bool = False) -> str:
    run_id = evaluator(holdout).run(name, retriever, params, judged=judged)
    print(f"{name}{'_judged' if judged else ''}{' [holdout]' if holdout else ''}: {run_id}")
    return run_id


def strategy_runs(judged: bool = False, split: str = "dev") -> pd.DataFrame:
    """Latest finished run per strategy for this demo (dev or holdout split)."""
    return load_runs(experiment.experiment_id, {"demo": "product_retrieval", "split": split}, judged)


def satisfies(sku: str, constraints: dict[str, Any]) -> bool:
    """Does a catalog product meet a query's brand / category / exclusion constraints (for report annotations)."""
    p = product_catalog().get(sku)
    if p is None:
        return False
    return not ((constraints.get("brand_name") and p.brand_name != constraints["brand_name"])
                or (constraints.get("merchandise_class") and p.merchandise_class != constraints["merchandise_class"])
                or (constraints.get("exclude_brand") and p.brand_name == constraints["exclude_brand"]))
