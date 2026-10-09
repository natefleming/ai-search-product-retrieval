# Databricks notebook source
# MAGIC %md
# MAGIC # 12 · Deploy open cross-encoder rerankers on Model Serving (GPU)
# MAGIC Two open-weight rerankers recommended by 2026 product-relevance benchmarks (ShopRank-Bench: Qwen3-Reranker-0.6B 73.5%,
# MAGIC bge-reranker-v2-m3 73.3% pairwise accuracy; both Apache-2.0), served on small GPU endpoints with scale-to-zero:
# MAGIC
# MAGIC | endpoint | model | family |
# MAGIC |---|---|---|
# MAGIC | `retrieval-bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` (~0.57B) | sentence-transformers CrossEncoder |
# MAGIC | `retrieval-qwen3-reranker-0-6b` | `Qwen/Qwen3-Reranker-0.6B` | instruction-aware yes/no-logit scoring |
# MAGIC
# MAGIC The model code is `product_retrieval/serving/cross_encoder_model.py` (MLflow models-from-code, no dependency on the package);
# MAGIC the client is `product_retrieval.ServingEndpointReranker(endpoint_name)`. CPU in-process serving was ruled out: a 33M-parameter
# MAGIC FlashRank model already took ~10 s per 50 documents on serverless CPU.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow huggingface_hub "transformers>=4.51" "sentence-transformers>=3.4" torch

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import product_retrieval.serving as serving_pkg
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType
from huggingface_hub import snapshot_download

mlflow.set_registry_uri("databricks-uc")
MODEL_CODE = os.path.join(os.path.dirname(serving_pkg.__file__), "cross_encoder_model.py")
PIP = [f"torch=={md.version('torch').split('+')[0]}", f"transformers=={md.version('transformers')}",
       f"sentence-transformers=={md.version('sentence-transformers')}", "accelerate", f"mlflow=={md.version('mlflow')}", "pandas"]
MODELS = {
    "retrieval-bge-reranker-v2-m3": {"repo": "BAAI/bge-reranker-v2-m3", "uc": f"{CATALOG}.{SCHEMA}.bge_reranker_v2_m3",
                               "config": {"family": "bge", "max_length": 512, "batch_size": 64}},
    "retrieval-qwen3-reranker-0-6b": {"repo": "Qwen/Qwen3-Reranker-0.6B", "uc": f"{CATALOG}.{SCHEMA}.qwen3_reranker_0_6b",
                                "config": {"family": "qwen3", "max_length": 512, "batch_size": 32}},
}
example = pd.DataFrame({"query": ["cordless drill not DeWalt"], "document": ["Milwaukee M18 1/2 in. Cordless Drill Kit"]})
print(PIP)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log, register and serve

# COMMAND ----------

versions: dict[str, str] = {}
for endpoint, spec in MODELS.items():
    local = snapshot_download(spec["repo"], local_dir=f"/tmp/hf/{spec['repo'].split('/')[-1]}")  # serverless: /tmp is writable
    with mlflow.start_run(run_name=f"register_{endpoint}") as run:
        mlflow.set_tags({"demo": "retrieval_model_registration"})
        info = mlflow.pyfunc.log_model(
            name="model", python_model=MODEL_CODE, artifacts={"model_dir": local}, model_config=spec["config"],
            pip_requirements=PIP, input_example=example, registered_model_name=spec["uc"],
        )
    versions[endpoint] = str(info.registered_model_version)
    print(endpoint, spec["uc"], "v", versions[endpoint])

# COMMAND ----------

existing = {e.name for e in w.serving_endpoints.list()}
for endpoint, spec in MODELS.items():
    entity = ServedEntityInput(entity_name=spec["uc"], entity_version=versions[endpoint], workload_type=ServingModelWorkloadType.GPU_SMALL,
                               workload_size="Small", scale_to_zero_enabled=True)
    if endpoint in existing:
        w.serving_endpoints.update_config(name=endpoint, served_entities=[entity])
    else:
        w.serving_endpoints.create(name=endpoint, config=EndpointCoreConfigInput(name=endpoint, served_entities=[entity]))
    print("deploying", endpoint)

# COMMAND ----------

deadline = time.time() + 3 * 3600
pending = set(MODELS)
while pending and time.time() < deadline:
    for endpoint in list(pending):
        state = w.serving_endpoints.get(endpoint).state
        if str(state.ready).endswith("READY") and str(state.config_update).endswith("NOT_UPDATING"):
            pending.discard(endpoint)
        elif str(state.config_update).endswith("UPDATE_FAILED"):
            raise RuntimeError(f"{endpoint} failed to deploy: {state}")
    time.sleep(60)
assert not pending, f"not ready: {pending}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Request-level latency for a 50-candidate rerank

# COMMAND ----------

import numpy as np

pool = pr.PlainRetriever(pr.build_backend(config_enriched), "HYBRID", k=50)
queries = list(eval_source().sample(30, random_state=1).index)
latency: dict[str, dict[str, float]] = {}
for endpoint in MODELS:
    reranker = pr.ServingEndpointReranker(endpoint)
    reranker.rerank(queries[0], pool(queries[0]).products)  # warm up (scale-from-zero)
    times = []
    for q in queries:
        products = pool(q).products
        t = time.perf_counter()
        reranker.rerank(q, products)
        times.append((time.perf_counter() - t) * 1000)
    latency[endpoint] = {"p50_ms": float(np.percentile(times, 50)), "p90_ms": float(np.percentile(times, 90)), "p99_ms": float(np.percentile(times, 99))}
display(pd.DataFrame(latency).T.round(0))
with mlflow.start_run(run_name="12_cross_encoder_latency"):
    mlflow.set_tags({"demo": "retrieval_diagnostic"})
    mlflow.log_metrics({f"{e.replace('-', '_')}_{k}": v for e, m in latency.items() for k, v in m.items()})
dbutils.notebook.exit(json.dumps({"versions": versions, "latency": latency}))
