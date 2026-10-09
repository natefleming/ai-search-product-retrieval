# Databricks notebook source
# MAGIC %md
# MAGIC # 20 · ai_decide as the router
# MAGIC The routers so far classify requests with rules: a SKU/UPC regex and a negation regex matched against the catalog brand
# MAGIC vocabulary. `pr.AIDecideRouter` replaces the rules with **one ai_decide call** holding two `choice` questions:
# MAGIC
# MAGIC | Question | Options |
# MAGIC |---|---|
# MAGIC | `route` | identifier · exclusion · general |
# MAGIC | `excluded_brand` | the query's top-10 **brand facets** from AI Search (opaque labels) + `none` |
# MAGIC
# MAGIC ai_decide only returns decisions, so the open-vocabulary parts come from data: the excluded brand must be one of the facet
# MAGIC values, and the SKU/UPC is read from the text once ai_decide says "identifier". Retrieval per route and reranking are the
# MAGIC same as in `QueryRouter`, so each pair below differs only in the classifier.
# MAGIC
# MAGIC Three test sets:
# MAGIC * **dev** (600) and **holdout** (300): the standard sets. Their exclusion queries were written with "not X" / "anything but X",
# MAGIC   which the rules were tuned for.
# MAGIC * **paraphrase**: the dev exclusion queries rewritten so the rules miss them ("I'm done with DeWalt", "DeWalt's out"). Same
# MAGIC   expected products and constraints. This is where an LLM classifier should earn its cost.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval._retry import with_backoff
from product_retrieval.understanding import CODE_RE, brand_key

dbutils.widgets.dropdown("regenerate", "false", ["true", "false"], "Regenerate the paraphrase set")
dbutils.widgets.text("splits", "paraphrase,dev,holdout", "Comma-separated splits to evaluate")
SPLITS = [s.strip() for s in dbutils.widgets.get("splits").split(",") if s.strip()]

PARAPHRASE_SOURCE_TABLE: str = f"{CATALOG}.{SCHEMA}.retrieval_eval_paraphrase_source"
PARAPHRASE_DATASET: str = f"{CATALOG}.{SCHEMA}.retrieval_eval_paraphrase"
BGE_ENDPOINT = "retrieval-bge-reranker-v2-m3"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paraphrase set
# MAGIC gpt-oss rewrites each dev exclusion query. A rewrite is kept only if the brand is still named, there is no code, and the
# MAGIC rules **don't** detect the exclusion (`analyze(...).route != "exclusion"`), so every kept query is a known miss for the rules.

# COMMAND ----------

PARAPHRASE_PROMPT = """Rewrite this hardware-store search request so it keeps the same meaning: the shopper wants the same product \
and still does NOT want the brand {brand}. Express the brand exclusion in a casual, indirect way and name the brand. Do NOT use \
the words "not", "no", "anything but", "except", "excluding", "other than", "without", or a minus sign. Examples of the style: \
"I'm done with {brand}", "{brand} keeps breaking on me, need something else", "skip {brand}", "rather avoid {brand}".
Request: {query}
Reply with only the rewritten request."""

vocab = vocabulary()
if dbutils.widgets.get("regenerate") == "true" or not spark.catalog.tableExists(PARAPHRASE_SOURCE_TABLE):
    source = eval_source().reset_index()
    exclusions = source[source.query_type == "exclusion"]
    gen = chat(LLM_ENDPOINT, temperature=0.8, max_tokens=300)

    def rewrite(row: Any) -> str | None:
        brand = json.loads(row.constraints)["exclude_brand"]
        for _ in range(3):  # a few attempts until the rewrite is a valid rules miss
            try:
                text = message_text(with_backoff(lambda: gen.invoke(PARAPHRASE_PROMPT.format(brand=brand, query=row.query)))).strip(" \"'\n")
            except Exception:
                continue
            named = brand_key(brand) in brand_key(text)
            if text and named and not CODE_RE.search(text) and pr.analyze(text, vocab).route != "exclusion":
                return text
        return None

    mlflow.langchain.autolog(disable=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        rewrites = list(pool.map(rewrite, exclusions.itertuples()))
    paraphrased = exclusions.assign(original_query=exclusions["query"].values, query=rewrites).dropna(subset=["query"])
    paraphrased = paraphrased.drop_duplicates("query").assign(
        expected_skus=lambda d: d.expected_skus.map(list), expected_facts=lambda d: d.expected_facts.map(list))
    print(f"kept {len(paraphrased)} of {len(exclusions)} exclusion queries")
    spark.createDataFrame(paraphrased).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PARAPHRASE_SOURCE_TABLE)

    spark.sql(f"DROP TABLE IF EXISTS {PARAPHRASE_DATASET}")
    ds = mlflow.genai.datasets.create_dataset(name=PARAPHRASE_DATASET, experiment_id=experiment.experiment_id)
    ds.merge_records([
        {"inputs": {"query": r.query},
         "expectations": {"expected_skus": list(r.expected_skus), "constraints": json.loads(r.constraints),
                          "expected_facts": list(r.expected_facts), "query_type": r.query_type}}
        for r in paraphrased.itertuples()
    ])

paraphrase_source = spark.table(PARAPHRASE_SOURCE_TABLE).toPandas()
display(paraphrase_source[["original_query", "query"]].head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Strategies
# MAGIC Pairs with the same retrieval and reranker; only the classifier changes.
# MAGIC
# MAGIC | Rules | ai_decide | Reranker |
# MAGIC |---|---|---|
# MAGIC | `09_router_fast` | `20_ai_router_fast` | none |
# MAGIC | `09_router_quality` | `20_ai_router_quality` | ai_decide noul, 12 candidates |
# MAGIC | `13_router_bge` | `20_ai_router_bge` | bge-reranker-v2-m3, 25 candidates |

# COMMAND ----------

backend = pr.build_backend(config_original)
noul = pr.AIDecideReranker("noul", details_chars=300)
bge = pr.ServingEndpointReranker(BGE_ENDPOINT)
bge.rerank("warm up", pr.PlainRetriever(backend, "HYBRID", k=5)("drill").products)  # wake the scale-to-zero endpoint

ai_routers = {
    "20_ai_router_fast": pr.AIDecideRouter(backend, name="20_ai_router_fast"),
    "20_ai_router_quality": pr.AIDecideRouter(backend, reranker=noul, candidates=12, name="20_ai_router_quality"),
    "20_ai_router_bge": pr.AIDecideRouter(backend, reranker=bge, candidates=25, name="20_ai_router_bge"),
}
rule_routers = {  # already evaluated on dev/holdout by 09, 13 and 19; re-run here only on the new paraphrase set
    "09_router_fast": pr.QueryRouter(backend, vocab, name="09_router_fast"),
    "09_router_quality": pr.QueryRouter(backend, vocab, reranker=noul, candidates=12, name="09_router_quality"),
    "13_router_bge": pr.QueryRouter(backend, vocab, reranker=bge, candidates=25, name="13_router_bge"),
}
PAIRS = list(zip(rule_routers, ai_routers))

for example in ["cordless drill, I'm done with DeWalt", "brushless impact driver, anything but DeWalt", "item 10003485"]:
    r = ai_routers["20_ai_router_fast"](example)
    print(f"{example!r}: route={r.route} filters={[f.model_dump() for f in r.filters]} top={[p.brand for p in r.products[:3]]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## As a LangChain tool
# MAGIC The handover form: the same router from the tool factory, with `classifier="ai_decide"`. No catalog vocabulary is needed,
# MAGIC because the excluded-brand options come from each query's facets. The evaluation below uses the retrievers directly
# MAGIC (the evaluator takes `query -> RetrievalResult` callables), but this tool wraps the identical `AIDecideRouter`.

# COMMAND ----------

ai_router_tool = pr.create_router_tool(config_original, classifier="ai_decide", reranker="cross_encoder", candidates=25,
                                       endpoint_name=BGE_ENDPOINT)
print(ai_router_tool.name, "-", ai_router_tool.description[:120])
docs = json.loads(ai_router_tool.invoke({"query": "cordless drill, I'm done with DeWalt"}))
display(pd.DataFrame([d["metadata"] for d in docs])[["id", "name", "brand", "route"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate
# MAGIC Runs are tagged `split=dev|holdout|paraphrase` like the rest of the experiment, so they show up in MLflow comparisons.

# COMMAND ----------

paraphrase_eval = Evaluator(
    dataset=PARAPHRASE_DATASET, query_types=dict(zip(paraphrase_source["query"], paraphrase_source.query_type)),
    tags={"demo": "product_retrieval", "split": "paraphrase", **LIB_VERSIONS}, answer_llm=LLM_ENDPOINT,
)
for split in SPLITS:
    if split == "paraphrase":
        for name, retriever in {**rule_routers, **ai_routers}.items():
            print(name, "[paraphrase]:", paraphrase_eval.run(name, retriever, {"split": "paraphrase"}))
    else:
        for name, retriever in ai_routers.items():
            run_strategy(name, retriever, {"classifier": "ai_decide", "split": split}, holdout=split == "holdout")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC Quality, latency and **routing accuracy**: expected route = `identifier` / `exclusion` from the query type, else
# MAGIC `general`; an exclusion counts as correct only if the filtered brand is the one the shopper excluded.

# COMMAND ----------

EXPECTED_ROUTE = {"identifier": "identifier", "exclusion": "exclusion"}
constraints_by_query = {
    **{q: json.loads(c) for q, c in eval_source().constraints.items()},
    **{q: json.loads(c) for q, c in eval_source(HOLDOUT_SOURCE_TABLE).constraints.items()},
    **{r.query: json.loads(r.constraints) for r in paraphrase_source.itertuples()},
}


def routed_correctly(row: pd.Series) -> bool:
    expected = EXPECTED_ROUTE.get(row.query_type, "general")
    if row.route != expected:
        return False
    if expected == "exclusion":
        return any(f["op"] == "ne" and f["value"] == constraints_by_query[row.query]["exclude_brand"] for f in row.filters)
    return True


summaries, comparisons = [], []
for split in ["dev", "holdout", "paraphrase"]:
    runs = strategy_runs(split=split)
    names = [n for pair in PAIRS for n in pair if n in runs.index]
    if not names:
        continue
    per_query = load_per_query(runs.loc[names])
    per_query["routed_correctly"] = per_query.apply(routed_correctly, axis=1).astype(float)
    for name, grp in per_query.groupby("strategy"):
        summaries.append({
            "split": split, "strategy": name, "classifier": "ai_decide" if name.startswith("20_") else "rules", "queries": len(grp),
            "mrr_at_10": grp.mrr_at_10.mean(), "hit_at_1": grp.hit_at_1.mean(),
            "excluded_brand_leak_at_10": grp.excluded_brand_leak_at_10.mean(), "routing_accuracy": grp.routed_correctly.mean(),
            "exclusion_routing_accuracy": grp[grp.query_type == "exclusion"].routed_correctly.mean(),
            "latency_ms_p50": grp.latency_ms.median(), "latency_ms_p90": grp.latency_ms.quantile(0.9),
        })
    mrr = per_query.pivot_table(index="query", columns="strategy", values="mrr_at_10")
    for rules, ai in PAIRS:
        if rules in mrr and ai in mrr:
            mean, lo, hi = paired_bootstrap(mrr[rules], mrr[ai])
            comparisons.append({"split": split, "rules": rules, "ai_decide": ai, "Δmrr_at_10": round(mean, 3), "95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
                                "verdict": "ai_decide better" if lo > 0 else ("rules better" if hi < 0 else "no clear difference")})

summary_df = pd.DataFrame(summaries).round(3)
comparison_df = pd.DataFrame(comparisons)
display(summary_df)
display(comparison_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where the two classifiers disagree (paraphrase set)

# COMMAND ----------

if "paraphrase" in set(summary_df.split):
    pq = load_per_query(strategy_runs(split="paraphrase").loc[["09_router_fast", "20_ai_router_fast"]])
    wide = pq.pivot_table(index="query", columns="strategy", values="mrr_at_10")
    wide["filters_ai"] = pq[pq.strategy == "20_ai_router_fast"].set_index("query").filters.map(lambda fs: [f["value"] for f in fs])
    display(wide.sort_values("20_ai_router_fast", ascending=False).head(20))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"summary": summary_df.to_dict("records"), "comparisons": comparison_df.to_dict("records"),
                                  "paraphrase_queries": len(paraphrase_source)}, default=str))
