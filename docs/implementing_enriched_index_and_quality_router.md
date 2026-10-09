# Implementing the enriched index and the quality router

A step-by-step guide to the changes that took product retrieval on a hardware-store catalog from **MRR@10 0.645 → 0.92**
(hit@1 57% → 88%) on a 300-query **holdout** set built from products never used during development. The router is the core;
the reranking stage is pluggable (`ai_decide`, or a GPU cross-encoder: Step B6).

| Strategy (holdout, 300 queries) | MRR@10 | hit@1 | p50 (dev) |
|---|---|---|---|
| HYBRID baseline (`03_hybrid`) | 0.645 | 56.7% | ~0.3 s |
| Fast router, no reranker (`09_router_fast`) | 0.845 | 77.7% | ~0.26 s |
| Quality router, `ai_decide` (`09_router_quality`) | 0.900 | 85.0% | ~1.2 s |
| Quality router + enriched index (`10_enriched_router_quality`) | 0.909 | 85.7% | ~1.2 s |
| **Router + bge-reranker-v2-m3 cross-encoder** (`13_router_bge`) | **0.919** | **88.0%** | **~0.6 s** |
| Learned ranker over all signals (`17_ltr`, for reference) | 0.934 | 89.0% | ~3.6 s |

**Recommended:** the router with the bge cross-encoder (best quality per millisecond); use the fast router where no GPU endpoint is
available, and the `ai_decide` quality router where a serving endpoint isn't wanted. On a ~5M-row index with near-duplicate
distractors, use a deeper pool (`candidates=50`): router + bge reached MRR 0.783 vs 0.667 for the 12-candidate `ai_decide` router.

Files referenced here (repo root locally; workspace `/Users/<you>/ai-search-product-retrieval`):

| File | What it is |
|---|---|
| `product_retrieval/` | The handover package: `QueryRouter`, `CatalogVocabulary`, `create_router_tool` and every other strategy as a tool factory |
| `product_retrieval/notebooks/integration_test` | Builds and calls every tool factory live against the workspace |
| `10_enriched_index.py` | Notebook that builds the enriched table and index |
| `00_config.py` / `09_improvements.py` | The experiment versions of the same logic, with MLflow evaluation |

---

## How the pieces fit

```
shopper request
      │
      ▼
classify()  ── regex + catalog brand vocabulary, no LLM, microseconds
      │
      ├── identifier ("do you have item 00176279")  → equality filter on sku / upc ─────────────────────► top 10
      │
      ├── exclusion ("impact driver, anything but DeWalt")
      │        → strip the exclusion phrase, filter  brand_name NOT 'DEWALT'  → HYBRID top 12 ─┐
      │                                                                                        ├► ai_decide (1 call, noul
      └── general (everything else)                                                            │   per candidate) → top 10
               → strip filler words ("do you carry", "?")                → HYBRID top 12 ──────┘
```

Every search goes to the **enriched index**, whose embedding column is a compact `search_text` field rather than marketing copy.
Use the **fast router** (no `ai_decide` step) when latency matters most; it is HYBRID-speed at MRR 0.861.

---

## Prerequisites

* Databricks workspace with AI Search (Vector Search) and a **STANDARD** endpoint, the `databricks-gte-large-en` embedding
  endpoint, and `ai_decide` (Beta) enabled (`POST /api/2.0/ai-functions/ai-decide`).
* Unity Catalog: `USE CATALOG`/`USE SCHEMA`, `CREATE TABLE` on the target schema, `SELECT` on the product table.
* Python packages (tested versions): `databricks-langchain` 0.20.0, `databricks-ai-search` 0.78, `databricks-sdk` 0.147.0,
  `mlflow` 3.16–3.17. In notebooks use serverless **environment version 5** and
  `%uv pip install -U databricks-langchain databricks-ai-search databricks-sdk mlflow` then `%restart_python`.
* A product table with at least: `product_id` (primary key, NOT NULL), `sku`, `upc`, `brand_name`, `product_name`,
  `merchandise_class`, `description`. In this catalog `brand_name` and `merchandise_class` are UPPERCASE and `description`
  is marketing copy followed by `Category: …; Brand Name: …; Volts: …; …` attributes.

---

## Part A: The enriched index

**Why:** the original index embeds `description` (~740 characters) whose first half is marketing prose; the structured attributes
and the codes shoppers type (SKU, UPC) are buried or absent. A compact field puts what shoppers search for first.
On its own this lifted HYBRID 0.661 → 0.689 and FULL_TEXT 0.677 → 0.726; under the quality router it adds ~+0.01.

### Step A1: Build `search_text`

Format: `product name | Brand: … | Class: … | <attribute block> | SKU … UPC …`. The attribute block is everything from
`Category:` onward in the description (the prose before it is dropped), capped at 1,200 characters.

```python
from pyspark.sql import functions as F

SOURCE = "retail_consumer_goods.product_search.products"
ENRICHED_TABLE = "retail_consumer_goods.product_search.products_enriched"

attributes = (
    F.when(F.instr("description", "Category:") > 0, F.expr("substring(description, instr(description, 'Category:'))"))
    .otherwise(F.col("description"))
)
enriched = spark.table(SOURCE).withColumn(
    "search_text",
    F.concat_ws(
        " | ",
        F.col("product_name"),
        F.concat(F.lit("Brand: "), F.coalesce("brand_name", F.lit(""))),
        F.concat(F.lit("Class: "), F.coalesce("merchandise_class", F.lit(""))),
        F.substring(attributes, 1, 1200),
        F.concat(F.lit("SKU "), F.col("sku"), F.lit(" UPC "), F.col("upc")),
    ),
)
enriched.write.option("delta.enableChangeDataFeed", "true").saveAsTable(ENRICHED_TABLE)  # CDF is required for Delta Sync
spark.sql(f"ALTER TABLE {ENRICHED_TABLE} ALTER COLUMN product_id SET NOT NULL")
```

Example output row:

```
Wrap-It MagSnap 12.25 in. L X 3.25 in. W Black Magnetic Tool Holder 1 pk | Brand: WRAP-IT | Class: KITCHEN/HOUSEHOLD STORAGE |
Category: KITCHEN/HOUSEHOLD STORAGE; Brand Name: Wrap-It; Color: Black; ... | SKU 00176279 UPC 0...
```

Keep all original columns in the table: they are returned with results, used as filters (`brand_name`, `sku`, `upc`), and fed to
`ai_decide` (`description`).

### Step A2: Keep it fresh in production

The demo builds the table once. In production, produce `search_text` wherever the product table is maintained so it never drifts:

* add it as a column in the existing ETL (Lakeflow Declarative Pipeline / job) that writes the product table, **or**
* make `products_enriched` a downstream table refreshed on the same schedule.

Either way keep **Change Data Feed** on, and trigger an index sync after each refresh (Step A3), or create the index with
`pipeline_type="CONTINUOUS"` if catalog changes must be searchable within minutes.

### Step A3: Create the Delta Sync index

```python
from databricks.ai_search.client import VectorSearchClient

ENDPOINT = "dao_ai_workshop_vs"  # any ONLINE STANDARD endpoint (see troubleshooting below)
ENRICHED_INDEX = "retail_consumer_goods.product_search.products_enriched_index"

vsc = VectorSearchClient(disable_notice=True)
vsc.create_delta_sync_index(
    endpoint_name=ENDPOINT,
    index_name=ENRICHED_INDEX,
    source_table_name=ENRICHED_TABLE,
    pipeline_type="TRIGGERED",
    primary_key="product_id",
    embedding_source_column="search_text",          # the enriched field, not description
    embedding_model_endpoint_name="databricks-gte-large-en",
)
```

All table columns are synced, so they can be returned and filtered. Initial sync of 38K rows took ~15 minutes.
Wait for `ready` and a full `indexed_row_count`:

```python
import time

n_rows = spark.table(ENRICHED_TABLE).count()
while True:
    status = vsc.get_index(endpoint_name=ENDPOINT, index_name=ENRICHED_INDEX).describe()["status"]
    if status.get("ready") and status.get("indexed_row_count", 0) >= n_rows:
        break
    time.sleep(30)
```

Subsequent refreshes: `WorkspaceClient().vector_search_indexes.sync_index(index_name=ENRICHED_INDEX)` (CLI:
`databricks vector-search-indexes sync-index <index>`), e.g. as the last task of the product-ETL job.

**Troubleshooting.** In our test workspace, new indexes on `dbdemos_vs_endpoint` twice sat in `PROVISIONING_INITIAL_SNAPSHOT` with 0 rows for
over an hour, while that endpoint's existing indexes kept syncing normally and an identical index on `dao_ai_workshop_vs` synced
in ~15 minutes. If initial sync shows no progress after ~30 minutes, check the sync pipeline's latest update
(`databricks pipelines get <pipeline_id>`); if it is stuck in `INITIALIZING`, create the index on another endpoint and raise it
with support.

### Step A4: Validate the index

```python
from databricks_langchain import DatabricksVectorSearch

vs = DatabricksVectorSearch(index_name=ENRICHED_INDEX, columns=["sku", "brand_name", "product_name"])
for d in vs.similarity_search("20V cordless drill kit with battery", k=5, query_type="HYBRID"):
    print(d.metadata["sku"], d.metadata["brand_name"], d.metadata["product_name"])
print(vs.similarity_search("00176279", k=1, query_type="FULL_TEXT")[0].metadata["sku"])  # codes are now searchable text
```

---

## Part B: The quality router

**Why each design choice** (all measured on the 600-query set):

| Choice | Evidence |
|---|---|
| Route SKU/UPC lookups to an exact filter | every ranking method (HYBRID, FULL_TEXT, rerankers, ai_decide) scored **MRR 0.00** on the 100 identifier queries |
| Parse "not X" with code, not an LLM | exclusion queries leaked the excluded brand into the top 10 **79%** of the time; a filter removes it; an LLM planner adds ~1 s and can invent brand names |
| No hard filters on other queries | LLM-chosen brand/class filters *lowered* MRR on brand, attribute and descriptive queries (a wrong class removes the right product) |
| `noul` rerank, not 4-level `score` | `score` left ≥3 candidates tied at the top in 61% of queries; `noul` in 22%, and it fixed 41% vs 20% of mis-ranked queries |
| 12 candidates, 300-char details, one batched call | HYBRID already has the target in its top 10 for 96% of non-identifier queries; the smaller payload cut `ai_decide` from ~2.0 s to ~1.3 s; the Beta endpoint is throttled per call |

The implementation is `product_retrieval.QueryRouter` (`product_retrieval/src/product_retrieval/strategies/router.py`, query understanding in
`understanding.py`, rerankers in `rerankers.py`); the steps below walk through it.

### Step B1: Load the brand vocabulary

Exclusion parsing maps what the shopper typed onto the **exact** catalog brand (filters are exact-match). Load it once at startup
and refresh when the catalog changes.

```python
import product_retrieval as pr

# In a notebook / job with Spark:
vocabulary = pr.CatalogVocabulary.from_spark(spark, ENRICHED_TABLE)
# Without Spark (model serving, Apps, agents): through a SQL warehouse
vocabulary = pr.CatalogVocabulary.from_warehouse(ENRICHED_TABLE, warehouse_id="<warehouse-id>")
```

Keys are normalized (`brand_key("Black & Decker") == "BLACKANDDECKER"`), values are catalog spellings (`"BLACK+DECKER"`).

### Step B2: Query understanding (`classify`)

Three regular expressions plus the vocabulary; no model call:

```python
CODE_RE = re.compile(r"(?<!\d)(\d{12,13}|\d{8})(?!\d)")          # 8-digit SKU or 12/13-digit UPC
EXCLUSION_RE = re.compile(
    r"(?:\b(?:not|anything but|except|excluding|other than|without|no)\s+|\s-)([\w&+.'’‐-― -]{2,40})", re.I)
FILLER_RE = re.compile(r"\b(do you (have|carry|sell|stock)( any)?|i('m| am) looking for|looking for|i need|...)\b|site:\S+|[?!]", re.I)

def classify(query, brands):
    if CODE_RE.search(query):
        return "identifier"
    if excluded_brand(query, brands):   # longest catalog brand within 4 words after a negation
        return "exclusion"
    return "general"
```

`excluded_brand` tries the longest phrase first (`"open road brands"` before `"open"`) and accepts a unique prefix match for
partial brand names (`"Mrs. Meyer's"` → `"MRS. MEYER'S CLEAN DAY"`). On the eval set this routed **599 of 600** queries correctly
(100/100 identifiers, 99/100 exclusions, 0 false exclusions on the other 400).

> The rules were tuned on this eval set. They are generic (code patterns, negation words, the catalog's own vocabulary), but
> **check the route distribution on real production query logs** before relying on the numbers (Step D3).

### Step B3: Retrieval per route

`DatabricksVectorSearch` (the class `VectorSearchRetrieverTool` wraps) returns structured `Document`s; filters use the same
dictionary syntax as the tool.

```python
vs = DatabricksVectorSearch(index_name=ENRICHED_INDEX, columns=RETURN_COLUMNS)

# identifier: exact filter, FULL_TEXT on the bare code as fallback
code = CODE_RE.search(query).group(1)
docs = vs.similarity_search(code, k=10, filter={"sku" if len(code) == 8 else "upc": code}, query_type="HYBRID") \
       or vs.similarity_search(code, k=10, query_type="FULL_TEXT")

# exclusion: drop the "not X" phrase from the text, enforce it as a filter
docs = vs.similarity_search(without_exclusion(query), k=12, filter={"brand_name NOT": excluded_brand(query, brands)}, query_type="HYBRID")

# general: strip filler words, no filters
docs = vs.similarity_search(clean_query(query), k=12, query_type="HYBRID")
```

### Step B4: Instruction-aware rerank with `ai_decide`

One REST call scores all candidates: the request and the 12 candidates go into one `state`, with one `noul` question per
candidate. The questions carry the business policy.

```python
STORE_POLICY = ("You rank catalog products for a hardware-store shopper. Treat every requirement the shopper states as mandatory: "
              "required brand, excluded brands, product type, voltage or battery platform, size, quantity or pack count, color, "
              "wattage, and kit-vs-tool-only. Accessories, parts or refills for the requested product are NOT the requested product.")

state = {"shopper_request": query,
         "candidates": {f"c{i}": {"name": ..., "brand": ..., "category": ..., "details": description[:300]} for i, d in enumerate(docs)}}
questions = {f"c{i}": {"type": "noul",
                       "instructions": f"{STORE_POLICY} Is candidate c{i} the right product type AND does it satisfy every stated requirement?",
                       "criteria": {"true": "Right product type and every stated requirement is met.",
                                    "false": "Wrong product type or a requirement is violated."}}
             for i in range(len(docs))}

resp = WorkspaceClient().ai_functions.ai_decide(state=state, questions=questions)   # POST /api/2.0/ai-functions/ai-decide
p = {qid: a["probability"] for qid, a in resp.response["answers"].items()}
ranked = sorted(range(len(docs)), key=lambda i: (-p[f"c{i}"], i))                  # ties keep retrieval order
```

Two production rules built into the package (`product_retrieval/_retry.py`, `rerankers.rerank_safely`):

* **Retry throttling** (`429`, `RESOURCE_EXHAUSTED`, `REQUEST_LIMIT_EXCEEDED`) with exponential backoff (`with_backoff`).
* **Never fail the search because of the reranker**: if `ai_decide` still errors (it is Beta and occasionally rejects requests),
  return the retrieval order and record `rerank_error`. The retrieval order alone is the fast router (MRR 0.861).

### Step B5: Put it together

```python
config = pr.RetrievalConfig(index_name=ENRICHED_INDEX)
backend = pr.build_backend(config)
router = pr.QueryRouter(backend, vocabulary, reranker=pr.AIDecideReranker("noul", details_chars=300), candidates=12)  # quality
fast = pr.QueryRouter(backend, vocabulary)                                                                       # fast

r = router("brushless impact driver kit, anything but DeWalt")
r.route, r.filters, r.skus[:5], r.scores, r.rerank_error
# ('exclusion', {'brand_name NOT': 'DEWALT'}, [...Craftsman / Milwaukee / Bosch...], {...}, None)
```

`QueryRouter`, the AI Search backend and the rerankers are MLflow-traced (CHAIN / RETRIEVER / RERANKER spans), so every request shows
its route, filters, candidates and `ai_decide` latency in the trace.

---

### Step B6 (recommended): Rerank with a GPU cross-encoder instead of `ai_decide`

The reranker is pluggable. Serving the open cross-encoder **bge-reranker-v2-m3** (Apache-2.0, ~0.57B) on a small GPU Model
Serving endpoint gave the best quality per millisecond: holdout MRR@10 0.919 vs 0.900 for `ai_decide`, at about half the median
latency (~0.6 s vs ~1.2 s with 25 candidates).

1. **Deploy** (notebook `12_deploy_cross_encoders`): download the Hugging Face snapshot, log
   `product_retrieval/serving/cross_encoder_model.py` (MLflow models-from-code; `model_config={"family": "bge", "max_length": 512}`,
   artifact `model_dir`), register it in Unity Catalog, and serve it with `workload_type=GPU_SMALL` (scale-to-zero for dev,
   provisioned for production). Request-level latency for 50 candidates on GPU_SMALL: p50 ~0.67 s, p99 ~0.81 s.
2. **Use it** in the router:

```python
tool = pr.create_router_tool(config, vocabulary, reranker="cross_encoder",
                             endpoint_name="retrieval-bge-reranker-v2-m3", candidates=25)
# or: pr.QueryRouter(backend, vocabulary, reranker=pr.ServingEndpointReranker("retrieval-bge-reranker-v2-m3"), candidates=25)
```

The cross-encoder scores `name | brand | category | description` (first 600 characters) against the original request.
Like `ai_decide`, it only reorders candidates, so the router's SKU routing and exclusion filters remain essential (on plain
HYBRID top 50 it reached only 0.750). Qwen3-Reranker-0.6B was slower (~5 s) and no better on the same GPU size.

### Step B7 (optional): Classify with `ai_decide` instead of rules

Step B2 classifies with regexes and the brand vocabulary. `pr.AIDecideRouter` replaces that with **one `ai_decide` call** with
two `choice` questions: the route (identifier / exclusion / general) and the excluded brand, whose options are the query's top-10
brand facets from AI Search plus "none". Retrieval and reranking per route are unchanged, and no vocabulary is needed.

```python
tool = pr.create_router_tool(config, classifier="ai_decide", reranker="cross_encoder",
                             endpoint_name="retrieval-bge-reranker-v2-m3", candidates=25)
```

Notebook `20_ai_decide_router` compares each rules router with its `ai_decide` twin (same retrieval and reranker):

| Test set | Rules + bge MRR@10 | ai_decide + bge MRR@10 | Excluded-brand leak (rules → ai_decide) | p50 (rules → ai_decide) |
|---|---|---|---|---|
| dev (600) | 0.930 | 0.923 | 1% → 10% | 596 → 802 ms |
| holdout (300) | 0.919 | 0.920 | 0% → 10% | 674 → 842 ms |
| paraphrased exclusions (100, e.g. "I'm done with DeWalt") | 0.771 | **0.855** | 83% → 14% | 688 → 1,054 ms |

`ai_decide` routes paraphrased exclusions the rules can't parse (83% routed correctly vs 0%). But on plain "not X" queries
it misses about 16% of exclusions: the excluded brand isn't always in the top-10 facets, or the answer falls below the 0.6
confidence cut-off. It also adds ~0.2 s. **Recommendation:** keep the rules as the primary classifier. Call `ai_decide` only
when the rules find no exclusion but the query names a catalog brand, so the two complement each other.

## Part C: Reuse it as a LangChain tool

`pr.create_router_tool` (or `pr.as_tool(any_retriever)`) returns a `StructuredTool` with the same contract as `VectorSearchRetrieverTool`
(input `query`, output a JSON list of `{page_content, metadata}`), adding `route` and `fit_probability` to each document's metadata.

```python
tool = pr.create_router_tool(config, vocabulary)                  # quality router; reranker="none" → fast router
tool = pr.as_tool(router, name="product_search")                  # or wrap any retriever you composed yourself
tool.invoke({"query": "60W equivalent soft white LED bulbs, 4 pack"})
```

The input schema tells the agent to pass the request **verbatim**, so the router (not the agent) parses constraints. Validated
with `databricks-gpt-oss-120b` tool calling in our test workspace:

| User says | Agent's tool call | Top result |
|---|---|---|
| "Do you carry item 00176279?" | `{"query": "00176279"}` | Wrap-It MagSnap Magnetic Tool Holder (exact SKU) |
| "I need a cordless drill kit but not DeWalt" | `{"query": "cordless drill kit not DeWalt"}` | Craftsman / Black+Decker drill kits (no DeWalt) |
| "60W equivalent soft white LED bulbs, 4 pack" | `{"query": "60W equivalent soft white LED bulbs 4 pack"}` | Feit A19 60W-equivalent soft white |

**LangGraph / LangChain agents:** pass the tool like any other (`create_react_agent(llm, [tool])`, `llm.bind_tools([tool])`).

**Config-driven frameworks (e.g. dao-ai):** use the factory, which takes only primitives and loads the brand vocabulary through a
SQL warehouse (validated: identical vocabulary to the Spark load, correct UPC lookup):

```yaml
tools:
  product_search_tool:
    name: product_search_tool
    function:
      type: factory
      name: product_retrieval.create_router_tool
      args:
        config: {index_name: retail_consumer_goods.product_search.products_enriched_index}
        vocabulary_table: retail_consumer_goods.product_search.products_enriched
        warehouse_id: <warehouse-id>
        reranker: ai_decide_noul
```

Install the `product-retrieval` wheel with the agent, and grant the agent's identity `SELECT` on the index and
brand table, `CAN USE` on the warehouse, and access to `ai_decide`.

---

## Part D: Validate before rollout

### Step D1: Re-run the evaluation

Use the same MLflow evaluation dataset and scorers as the experiments, so the new run lines up row for row with the baseline in
the MLflow **Evaluations → Compare** view:

```python
from product_retrieval.evaluation import Evaluator
Evaluator(dataset=EVAL_DATASET, query_types=query_types).run("router_quality_v1", router)
```

and run `product_retrieval/notebooks/integration_test` to check every tool factory end to end.
Gate the rollout on: MRR@10 and hit@1 not below the current production strategy, `rerank_errors` ≈ 0, excluded-brand leak ≈ 0,
p90 latency within budget.

### Step D2: Latency budget

| Stage | Typical |
|---|---|
| `classify` | < 1 ms |
| HYBRID retrieval (12 candidates) | ~250–300 ms median; p99 ~2 s (endpoint-side tail, unchanged by client concurrency) |
| `ai_decide` (12 candidates × 300 chars, one call) | ~1.0–1.3 s, more under throttling |
| Total quality router | p50 ~1.3–1.5 s, p90 ~2.2–2.9 s |

If the budget is tighter: serve the fast router for interactive search and call the quality router only for agent / "best match"
answers; or lower `candidates` (the target is in HYBRID's top 10 for 96% of non-identifier queries).

### Step D3: Monitor in production

From the MLflow traces (or by logging `RouterResult`):

* **route distribution**: a drop in `identifier`/`exclusion` share suggests new phrasings the regexes miss;
* **`rerank_error` rate** and `ai_decide` latency;
* **zero-result rate** per route (a stale brand vocabulary makes exclusion filters miss);
* **sampled quality**: periodically add real queries with labels to the MLflow evaluation dataset and re-run Step D1.

---

## Checklist

- [ ] `products_enriched` built with `search_text`, CDF on, `product_id NOT NULL` (A1), refreshed by the product ETL (A2)
- [ ] Delta Sync index on `search_text` ONLINE with full row count (A3, A4)
- [ ] Brand vocabulary loaded from the same table, refreshed with the catalog (B1)
- [ ] Router routes checked on a sample of real queries (B2, D3)
- [ ] `ai_decide` access, throttling retry and rerank fallback verified (B4)
- [ ] Tool registered with the agent; identity has index, table, warehouse and `ai_decide` access (C)
- [ ] Evaluation run against the shared MLflow dataset meets the rollout gate (D1)
- [ ] Dashboards/alerts on route mix, rerank errors, zero results and latency (D3)
