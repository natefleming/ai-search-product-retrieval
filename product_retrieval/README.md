# product-retrieval

Product-search retrieval strategies on **Databricks AI Search** (Vector Search), each available as a **factory function that
returns a LangChain tool**. Drop a tool into any LangChain/LangGraph agent, or reference a factory from a config-driven framework
(e.g. dao-ai `type: factory`).

* Embeddings are stored in Databricks AI Search (managed or self-managed vectors).
* All models are Databricks-hosted (Foundation Model APIs) or served on Databricks Model Serving.
* Works with STANDARD and STORAGE_OPTIMIZED endpoints (the latter for ~100M+ vectors).

## Install

```bash
pip install /Volumes/<catalog>/<schema>/<volume>/wheels/product_retrieval-0.1.0-py3-none-any.whl
# extras: [evaluation] (MLflow harness), [ltr] (LightGBM learned ranker), [dao-ai] (dao-ai adapter)
```

In a Databricks notebook (serverless environment 5):
`%uv pip install -U <wheel> databricks-langchain databricks-ai-search databricks-sdk mlflow` then `%restart_python`.
The package pins `mcp<2`: mcp 2.0 removed `mcp.shared.context.RequestContext`, which `langchain-mcp-adapters` 0.3.x (a
dependency of `databricks-langchain`) still imports.

## Quick start

```python
import product_retrieval as pr

config = pr.RetrievalConfig(index_name="catalog.schema.products_enriched_index")
vocabulary = pr.CatalogVocabulary.from_spark(spark, "catalog.schema.products_enriched")   # or .from_warehouse(table, warehouse_id)

tool = pr.create_router_tool(config, vocabulary)               # the recommended tool (quality router)
tool.invoke({"query": "brushless impact driver kit, anything but DeWalt"})
```

Every tool has the same contract:

* **input** `{"query": "<the shopper's request, verbatim>"}`. Don't let the agent strip constraints; the tool parses them.
* **output** JSON list of `{"page_content": ..., "metadata": {id, name, brand, category, upc, route, score, ...}}` (the same shape
  as `VectorSearchRetrieverTool`).

## Building blocks

| Module | What it provides |
|---|---|
| `config` | `RetrievalConfig` (index, endpoint type, `k`, LLM endpoint, optional query embedding), `CatalogSchema` (maps product fields to your column names), `QueryEmbedding` |
| `backends` | `AISearchBackend`: the only code that calls AI Search. Translates neutral `Filter`s to dict syntax (STANDARD) or SQL strings (STORAGE_OPTIMIZED), embeds queries for self-managed indexes, retries throttling |
| `understanding` | `CatalogVocabulary` (exact brand/category values) and `analyze()`: deterministic SKU/UPC and "not <brand>" detection, query cleaning |
| `rerankers` | `AIDecideReranker` (pointwise noul/score), `AIDecideListwiseReranker`, `ServingEndpointReranker` (cross-encoder on Model Serving), `rerank_safely` (a reranker failure never fails the search) |
| `column_selection` | `RerankColumnSelector`: picks `columns_to_rerank` per request for the Databricks server-side reranker |
| `strategies` | `PlainRetriever`, `LLMFilterRetriever`, `RerankedRetriever`, `QueryRouter`, `AIDecideRouter`, `FacetGuidedRetriever`, `FusionRetriever`, `PlanRouter`, `DaoAIInstructedRetriever` |
| `ltr` | `FeatureCollector`, `LambdaRankModel`, `LTRRetriever` |
| `tools` | `as_tool(retriever)` plus one `create_*_tool` factory per strategy |
| `evaluation` | MLflow evaluation harness (`Evaluator`), retrieval/latency scorers, ESCI `GradedJudge`, result loaders |

Strategies are plain callables `query -> RetrievalResult` composed from injected parts (backend, vocabulary, reranker), so you
can build variants without new code and test them with fakes (see `tests/`).

## Tool factories

| Factory | Strategy | Notes |
|---|---|---|
| `create_search_tool(config, query_type="HYBRID", rerank_columns=None, candidates=None)` | Plain AI Search (ANN / FULL_TEXT / HYBRID), optional server-side reranker | Baseline |
| `create_targeted_rerank_tool(config)` | HYBRID + server reranker on columns chosen per request | One ai_decide choice call |
| `create_dynamic_filter_tool(config)` | `VectorSearchRetrieverTool(dynamic_filter=True)` | The *calling agent* writes filters |
| `create_guarded_filter_tool(config, vocabulary, reranker="none", targeted_server_rerank=False)` | Internal LLM plans filters, validated against the catalog, empty-result fallback | `reranker="ai_decide_score"` = filter → ai_decide |
| `create_instructed_tool(config, reranker="ai_decide_noul", candidates=25, endpoint_name=None)` | HYBRID pool → instruction-aware rerank | `reranker="cross_encoder"` + `endpoint_name` for a served cross-encoder |
| `create_facet_guided_tool(config)` | Facet counts → ai_decide picks filters | Better used for refinement UX |
| `create_router_tool(config, vocabulary, reranker="ai_decide_noul", candidates=12)` | **Recommended.** SKU/UPC → exact; "not X" → filter; else cleaned HYBRID; then rerank | `reranker="none"` = fast router |
| `create_router_tool(config, classifier="ai_decide", reranker=...)` | Same routes, classified by one ai_decide call (route + excluded brand chosen from the query's brand facets) | No vocabulary needed; handles paraphrased exclusions; +1 ai_decide call |
| `create_plan_router_tool(config, vocabulary)` | LLM structured query plan (cached) → hard/soft constraints | |
| `create_fusion_tool([(config, query_type, weight), ...])` | Weighted RRF across searches/indexes | e.g. product + pseudo-query index |
| `create_ltr_tool(config, model_path, vocabulary, signal_endpoints=...)` | Learned LambdaRank ranker over router candidates | `ltr` extra |
| `create_dao_ai_instructed_tool(config, endpoint, source_table, vocabulary, rerank=...)` | dao-ai instructed retriever behind the same contract | `dao-ai` extra |

### Every evaluated strategy as a tool

Each run name in the MLflow experiment and the report maps to one factory call (`cfg` = the index config for that run, e.g. the
enriched, Qwen3 self-managed, Doc2Query or storage-optimized index; `bge` = `"retrieval-bge-reranker-v2-m3"`):

| Run | Factory call |
|---|---|
| `03_ann` / `03_full_text` / `03_hybrid` | `create_search_tool(cfg, query_type="ANN" \| "FULL_TEXT" \| "HYBRID")` |
| `04_rerank_*` (fixed columns) | `create_search_tool(cfg, rerank_columns=[...], candidates=50)` |
| `04_rerank_targeted` | `create_targeted_rerank_tool(cfg)` |
| `05_dynamic_filter` / `_guarded` / `_guarded_rerank` | `create_guarded_filter_tool(cfg, vocab, guarded=False \| True, targeted_server_rerank=...)`; agent-written filters: `create_dynamic_filter_tool(cfg)` |
| `06_ai_decide_score` / `_noul` | `create_instructed_tool(cfg, reranker="ai_decide_score" \| "ai_decide_noul")` |
| `06_filter_ai_decide` | `create_guarded_filter_tool(cfg, vocab, reranker="ai_decide_score")` |
| `07_dao_ai_*` | `create_dao_ai_instructed_tool(cfg, endpoint, source_table, vocab, rerank=...)` |
| `08_facet_guided` | `create_facet_guided_tool(cfg)` |
| `09_hybrid_id_routing` | `create_router_tool(cfg, vocab, reranker="none", exclusions=False, clean_queries=False)` |
| `09_ai_decide_fast` | `create_instructed_tool(cfg, candidates=12)` |
| `09/10/14/15/18_*router_fast` | `create_router_tool(cfg, vocab, reranker="none")` |
| `09/10/14/15/18_*router_quality` | `create_router_tool(cfg, vocab)` (`candidates=50` for `_quality_50`) |
| `13_hybrid50_bge` / `_qwen3` | `create_instructed_tool(cfg, reranker="cross_encoder", candidates=50, endpoint_name=...)` |
| `13_router_bge` / `_qwen3`, `18_scale_router_bge_50` | `create_router_tool(cfg, vocab, reranker="cross_encoder", candidates=25 \| 50, endpoint_name=...)` |
| `15_fusion_enriched_pseudoq` | `create_fusion_tool([(enriched_cfg, "HYBRID", 1.0), (pseudo_query_cfg, "HYBRID", 1.0)])` |
| `16_plan_router_fast` / `_quality` | `create_plan_router_tool(cfg, vocab, reranker="none" \| "ai_decide_noul")` |
| `17_router_listwise` | `create_router_tool(cfg, vocab, reranker="ai_decide_listwise")` |
| `17_ltr` | `create_ltr_tool(cfg, model_path, vocab, signal_endpoints={"bge": bge})` |
| `20_ai_router_fast` / `_quality` / `_bge` | `create_router_tool(cfg, classifier="ai_decide", reranker="none" \| "ai_decide_noul" \| "cross_encoder", ...)` |

Factories that need the vocabulary accept `vocabulary=` **or** `vocabulary_table=` + `warehouse_id=` (loads it through a SQL warehouse,
for runtimes without Spark). Every factory accepts the config as a `RetrievalConfig` or a plain dict.

### Config-driven use (dao-ai)

```yaml
tools:
  product_search:
    name: product_search
    function:
      type: factory
      name: product_retrieval.create_router_tool
      args:
        config: {index_name: catalog.schema.products_enriched_index}
        vocabulary_table: catalog.schema.products_enriched
        warehouse_id: <warehouse-id>
        reranker: ai_decide_noul
```

## Scale and endpoint types

```python
config = pr.RetrievalConfig(index_name="catalog.schema.products_index", endpoint_type=pr.EndpointType.STORAGE_OPTIMIZED)
```

Only the backend changes: filters become SQL strings. Use storage-optimized endpoints for ~100M+ vectors (≈1B per endpoint).

## Self-managed (instruction-aware) embeddings

```python
config = pr.RetrievalConfig(
    index_name="catalog.schema.products_qwen3_index",
    query_embedding=pr.QueryEmbedding(endpoint="databricks-qwen3-embedding-0-6b"),  # query-side instruction applied
)
```

Document vectors must be computed with the same model **without** the instruction (see notebook `14_qwen3_embeddings`).

## Cross-encoder rerankers on Model Serving

`product_retrieval/serving/cross_encoder_model.py` is an MLflow models-from-code pyfunc for `BAAI/bge-reranker-v2-m3` (`family: bge`) and
`Qwen/Qwen3-Reranker-0.6B` (`family: qwen3`), with no dependency on this package. Log it with the Hugging Face snapshot as the
`model_dir` artifact and serve it on a GPU endpoint (see notebook `12_deploy_cross_encoders`). Use it with
`pr.ServingEndpointReranker(endpoint_name)` or any factory's `reranker="cross_encoder", endpoint_name=...`.

## Evaluation

```python
from product_retrieval.evaluation import Evaluator

evaluator = Evaluator(dataset="catalog.schema.retrieval_eval", query_types={...}, tags={"demo": "product_retrieval"})
evaluator.run("my_strategy", retriever)   # MLflow run: hit@k, MRR, nDCG, constraint precision, leak, latency p50/p90/p99
```

Every strategy is evaluated against the same MLflow evaluation dataset, so runs line up in **Evaluations → Compare**.

## Development

```bash
pytest -q          # unit tests with fake backends (no Databricks needed)
```

`notebooks/build_wheel` runs the tests and publishes the wheel to a UC volume; `notebooks/integration_test` exercises every factory
live. Version in `pyproject.toml` / `product_retrieval.__version__`.
