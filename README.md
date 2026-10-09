# Product retrieval on Databricks AI Search

Interactive notebooks that measure how different retrieval methods change product-search quality on a 38K-SKU
hardware catalog, with every run traced and scored in MLflow.

**Data:** bring your own product catalog as `products.snappy.parquet` (columns `product_id`, `sku`, `upc`, `brand_name`,
`product_name`, `merchandise_class`, `class_cd`, `description`) in `/Volumes/retail_consumer_goods/product_search/raw/`; no data is
included. `report/` holds the results from the evaluation run (HTML deck + PDF). Catalog, schema and column names are configurable (`00_config`, `CatalogSchema`).

| Notebook | What it does |
|---|---|
| `00_config` | Shared names, `VectorSearchRetrieverTool` helpers, `ai_decide` client, MLflow scorers, strategy runner |
| `01_ingest_and_index` | Drop/recreate `retail_consumer_goods.product_search.products` and the AI Search index |
| `02_build_eval_dataset` | 600 labelled queries (6 types), stored as MLflow evaluation datasets |
| `03_baseline_rag` | Plain RAG: ANN vs FULL_TEXT vs HYBRID |
| `04_reranker` | `DatabricksReranker(columns_to_rerank=...)`: fixed column lists vs columns targeted to each request (chosen by `ai_decide`) |
| `05_dynamic_filtering` | `VectorSearchRetrieverTool(dynamic_filter=True)` + gpt-oss-120b, raw vs guarded |
| `06_instructed_retrieval` | `ai_decide` (REST) instruction-aware reranking |
| `07_dao_ai_instructed` | dao-ai's instructed retriever (`dao_ai.tools.create_ai_search_tool` + `dao_ai.config` models): decomposition → RRF → rerank |
| `08_facets` | AI Search facets: refinement UX + facet-guided filters chosen by `ai_decide` |
| `09_improvements` | Fixes from the analysis: SKU/UPC routing, LLM-free exclusion filters, faster `noul` rerank, tail-latency diagnostic |
| `10_enriched_index` | Second index embedding a compact `search_text` (title, brand, class, attributes, codes) instead of marketing copy |
| `90_compare` | Leaderboard, per-type breakdown, paired-bootstrap significance, latency (p50/p90/p99) trade-off |
| `91_report` | Self-contained HTML presentation with side-by-side examples (`report/product_retrieval_report.html`; `make_pdf.sh` adds the PDF) |

## Running

* Compute: **serverless, environment version 5** (pinned in the published notebooks' metadata, required for `%uv`).
* Run 01 → 91 in order, **one at a time**: 03–10 evaluate against the same UC-backed MLflow dataset, and concurrent evaluations conflict on it.
* MLflow experiment: `/Users/<you>/ai-search-product-retrieval/retrieval_experiments`. Open **Evaluations**, select two runs, and click
  **Compare** for row-level side-by-side results (all runs share the same MLflow dataset).

## Editing & publishing

The `.py` files are Databricks source notebooks (the source of truth). `python publish.py [stem ...]` converts them to
`.ipynb` pinned to serverless env 5 and imports them to `/Users/<you>/ai-search-product-retrieval` in the workspace of CLI profile `$RETRIEVAL_PROFILE` (default `DEFAULT`).
`run_notebook.sh <stem>` runs one on serverless; `make_pdf.sh` prints the HTML report to PDF.

## Handover package: `product_retrieval/`

Every strategy above is available as a factory function returning a LangChain tool (`product_retrieval.create_*_tool`), built from
small injectable parts (config, AI Search backend, catalog vocabulary, rerankers). See `product_retrieval/README.md`.

* `product_retrieval/notebooks/build_wheel`: runs the unit tests and publishes the wheel to `/Volumes/.../raw/wheels/`
* `product_retrieval/notebooks/integration_test`: builds and calls every tool factory live
* `docs/implementing_enriched_index_and_quality_router.md`: step-by-step guide to the enriched index and the quality router

## Notebooks 11–19 (state-of-the-art experiments)

| Notebook | What it does |
|---|---|
| `11_measurement` | ESCI-style LLM judge: how much of the remaining error is label noise (judged exact@1, graded nDCG@5) |
| `12_deploy_cross_encoders` | Serves bge-reranker-v2-m3 and Qwen3-Reranker-0.6B on GPU Model Serving endpoints |
| `13_cross_encoder_rerank` | Served cross-encoders as the reranking stage (vs ai_decide) |
| `14_qwen3_embeddings` | Instruction-aware Qwen3 embeddings as self-managed vectors in AI Search |
| `15_doc2query` | Doc2Query document expansion: appended field vs dual pseudo-query index with RRF |
| `16_query_plan` | LLM structured query plans: hard constraints → filters, soft → reranker requirements |
| `17_listwise_ltr` | Listwise ai_decide rank-1 decision; LightGBM LambdaRank learned ranker (CV on dev, scored on holdout) |
| `18_scale_test` | ~5M-row storage-optimized index (real catalog + realistic distractors) |
| `19_holdout_finalists` | Finalists on the 300-query holdout set (products never used for development) |
