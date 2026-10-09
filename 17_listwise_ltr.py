# Databricks notebook source
# MAGIC %md
# MAGIC # 17 · Listwise reranking and a learned ranker (LambdaRank)
# MAGIC Two remaining ranking patterns, on the enriched index (compare with `10_enriched_router_quality`):
# MAGIC
# MAGIC 1. **Listwise** (RankGPT-style): after the pointwise `noul` scores, one `ai_decide` `choice` over the top 5 picks the single
# MAGIC    best match for rank 1 (`pr.create_router_tool(config, vocab, reranker="ai_decide_listwise")`).
# MAGIC 2. **Learned ranker** (production pattern at Walmart/Etsy: distil many signals into a fast model): LightGBM LambdaRank over
# MAGIC    router-pool candidates with features = pool rank, ANN rank, FULL_TEXT rank, `ai_decide` noul, two served cross-encoders,
# MAGIC    title/number overlap, brand-in-query, excluded-brand match. Labels: the evaluation's expected SKUs.
# MAGIC    **5-fold cross-validation by query on dev** gives an honest dev estimate (`17_ltr_cv`); the model trained on all dev is
# MAGIC    then scored on the **holdout** set (`17_ltr`, holdout split). Tool: `pr.create_ltr_tool(config, model_path, vocab, ...)`.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow lightgbm

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.ltr import FeatureCollector, LambdaRankModel, LTRRetriever
from product_retrieval.rerankers import AIDecideListwiseReranker, AIDecideReranker

backend = pr.build_backend(config_enriched)
vocab = vocabulary()
noul = AIDecideReranker("noul", details_chars=300)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Listwise rank-1 decision

# COMMAND ----------

listwise = pr.QueryRouter(backend, vocab, reranker=AIDecideListwiseReranker(base=noul, top_n=5), candidates=12, name="17_router_listwise")
run_strategy("17_router_listwise", listwise, {"rerank": "ai_decide noul + listwise top5", "candidates": 12, "index": ENRICHED_INDEX})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Learned ranker
# MAGIC Collect features once for dev and holdout queries (each signal is a live call), then train/evaluate.

# COMMAND ----------

SIGNALS = {"noul": noul, "ce_bge": pr.ServingEndpointReranker("retrieval-bge-reranker-v2-m3"),
           "ce_qwen3": pr.ServingEndpointReranker("retrieval-qwen3-reranker-0-6b")}
collector = FeatureCollector(backend, vocab, SIGNALS, candidates=20)


def collect(table: str) -> dict[str, tuple[pr.RetrievalResult, list[dict[str, float]]]]:
    queries = list(spark.table(table).select("query").toPandas()["query"])
    mlflow.tracing.disable()
    with ThreadPoolExecutor(max_workers=8) as pool:
        out = dict(zip(queries, pool.map(collector.collect, queries)))
    mlflow.tracing.enable()
    return out


dev_features, holdout_features = collect(EVAL_SOURCE_TABLE), collect(HOLDOUT_SOURCE_TABLE)
expected = {**{q: set(r.expected_skus) for q, r in eval_source().iterrows()},
            **{q: set(r.expected_skus) for q, r in eval_source(HOLDOUT_SOURCE_TABLE).iterrows()}}


def training_rows(features: dict, queries: list[str]) -> tuple[list[dict], list[int], list[int]]:
    rows, labels, groups = [], [], []
    for q in queries:
        pool, feats = features[q]
        if not feats:
            continue
        rows += feats
        labels += [int(p.id in expected[q]) for p in pool.products]
        groups.append(len(feats))
    return rows, labels, groups


trainable = [q for q, (_, f) in dev_features.items() if f]
print(f"dev queries with LTR features: {len(trainable)} (identifier queries keep exact lookup)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5-fold cross-validation on dev (out-of-fold rankings)

# COMMAND ----------

import random

shuffled = trainable[:]
random.Random(7).shuffle(shuffled)
folds = [shuffled[i::5] for i in range(5)]  # plain str keys (numpy arrays would turn them into np.str_)
oof: dict[str, pr.RetrievalResult] = {}
for i, test_queries in enumerate(folds):
    train_queries = [q for q in trainable if q not in set(test_queries)]
    model = LambdaRankModel.train(*training_rows(dev_features, train_queries))
    for q in test_queries:
        pool, feats = dev_features[q]
        scores = model.score(feats)
        order = sorted(range(len(feats)), key=lambda j: (-scores[j], j))
        oof[q] = pool.model_copy(update={"products": [pool.products[j] for j in order][:K], "strategy": "17_ltr_cv"})


def precomputed(features: dict, ranked: dict[str, pr.RetrievalResult]):
    """Evaluate fixed rankings (identifier queries fall back to the router pool, i.e. the exact lookup)."""
    return lambda q: ranked.get(q) or features[q][0].model_copy(update={"products": features[q][0].products[:K]})


feature_names = list(dev_features[trainable[0]][1][0])
run_strategy("17_ltr_cv", precomputed(dev_features, oof), {"model": "LightGBM LambdaRank", "folds": 5, "features": ",".join(feature_names)})

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final model: train on all dev, score on holdout

# COMMAND ----------

final = LambdaRankModel.train(*training_rows(dev_features, trainable))
model_path = f"{RAW_VOLUME_PATH}/models/ltr_lambdarank"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
final.save(model_path)
importance = dict(zip(final.feature_names, final.booster.feature_importance("gain").round(1)))
print(model_path, importance)

ltr = LTRRetriever(collector, final, K, name="17_ltr")
run_strategy("17_ltr", ltr, {"model_path": model_path, "trained_on": "dev", "importance": json.dumps(importance)}, holdout=True)
