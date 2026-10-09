# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · AI Search facets
# MAGIC **Facets** (Full-Text Search beta) return value counts for matching results alongside the hits, e.g.
# MAGIC `facets=["merchandise_class TOP 8", "brand_name TOP 8"]`; they work with `FULL_TEXT` and `HYBRID` queries
# MAGIC (`SearchRequest.facets` in the package).
# MAGIC
# MAGIC Two uses:
# MAGIC 1. **Shopper refinement UX**: "Brand: Milwaukee (29) · DeWalt (18) ..." chips next to results.
# MAGIC 2. **Facet-guided retrieval** (`08_facet_guided`, `pr.create_facet_guided_tool(config)`): facet values are the real filter
# MAGIC    vocabulary for this query, so three `ai_decide` choice questions pick category / required brand / excluded brand from them.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from product_retrieval.backends import SearchRequest

backend = pr.build_backend(config_original)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Refinement UX

# COMMAND ----------

query = "cordless impact driver"
resp = backend.search(SearchRequest(text=query, k=K, facets=["merchandise_class TOP 8", "brand_name TOP 8"]))
display(pd.DataFrame([(c, v, n) for c, vals in resp.facets.items() for v, n in vals.items()], columns=["facet", "value", "count"]))

top_brand = next(iter(resp.facets["brand_name"]))
print(f"Shopper clicks Brand = {top_brand}")
for p in backend.search(SearchRequest(text=query, k=5, filters=[pr.Filter(column="brand_name", value=top_brand)])).products:
    print(f"  {p.brand:<12} {p.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Facet-guided retrieval with `ai_decide`

# COMMAND ----------

facet_guided = pr.FacetGuidedRetriever(backend, k=K, name="08_facet_guided")
src = eval_source()
for qtype in ["exclusion", "brand_constrained", "attribute_constrained", "known_item"]:
    q = src.query(f"query_type == '{qtype}'").index[1]
    print(f"[{qtype}] {q}\n   -> {[(f.column, f.op, f.value) for f in facet_guided(q).filters]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate

# COMMAND ----------

run_strategy("08_facet_guided", facet_guided, {"query_type": "HYBRID", "facets": "merchandise_class,brand_name TOP 8",
                                               "filter_selector": "ai_decide", "min_confidence": 0.7})
