# Databricks notebook source
# MAGIC %md
# MAGIC # 91 · HTML results presentation
# MAGIC Builds a self-contained HTML deck (no external assets) from the MLflow runs: leaderboard, significance, per-query-type
# MAGIC quality, latency trade-offs, answer-quality judges and **side-by-side examples** of baseline vs improved retrieval.
# MAGIC It is logged to MLflow (run `91_report`) and written to the `raw/reports` volume.

# COMMAND ----------

# MAGIC %uv pip install -U /Volumes/retail_consumer_goods/product_search/raw/wheels/product_retrieval-0.1.0-py3-none-any.whl databricks-langchain databricks-ai-search databricks-sdk mlflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import datetime as dt
from typing import Iterable
import html
import math

runs = strategy_runs()
judged = strategy_runs(judged=True)
per_query = load_per_query(runs)
cat = product_catalog()
src = eval_source()
HOST = w.config.host.rstrip("/")
EXP_URL = f"{HOST}/ml/experiments/{experiment.experiment_id}"


def metric(df: pd.DataFrame, name: str, agg: str = "mean") -> pd.Series:
    for col in (f"metrics.{name}/{agg}", f"metrics.{name}"):
        if col in df:
            return df[col]
    return pd.Series(index=df.index, dtype=float)


lb = pd.DataFrame({m: metric(runs, m) for m in ["hit_at_1", "hit_at_10", "mrr_at_10", "ndcg_at_10", "constraint_precision_at_10", "excluded_brand_leak_at_10", "zero_results"]})
lb["p50_ms"] = metric(runs, "latency_ms", "median")
lb["p90_ms"] = metric(runs, "latency_ms", "p90")
lb["p99_ms"] = metric(runs, "latency_ms", "p99")
lb = lb.sort_values("mrr_at_10", ascending=False)
by_type = per_query.pivot_table(index="strategy", columns="query_type", values="mrr_at_10")
mrr_pivot = per_query.pivot_table(index="query", columns="strategy", values="mrr_at_10")
best = lb.index[0]
base = lb.loc[BASELINE_RUN]
esc = html.escape

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chart & table builders (inline SVG)

# COMMAND ----------


def hbar(values: pd.Series, fmt: str = "{:.3f}", highlight: str | None = BASELINE_RUN, width: int = 560, vmax: float | None = None) -> str:
    """Single-series horizontal bars with direct labels; the baseline bar is gray."""
    row_h, label_w, pad = 26, 230, 60
    vmax = vmax or float(values.max()) or 1.0
    bars = []
    for i, (name, v) in enumerate(values.items()):
        y, w_ = i * row_h, max(2.0, (width - label_w - pad) * float(v) / vmax)
        fill = "var(--neutral)" if name == highlight else "var(--series-1)"
        bars.append(
            f'<g><title>{esc(name)}: {fmt.format(v)}</title>'
            f'<text x="{label_w - 8}" y="{y + 17}" text-anchor="end" class="lbl">{esc(name)}</text>'
            f'<rect x="{label_w}" y="{y + 5}" width="{w_:.1f}" height="16" rx="4" fill="{fill}"/>'
            f'<text x="{label_w + w_ + 6:.1f}" y="{y + 17}" class="val">{fmt.format(v)}</text></g>'
        )
    return f'<svg viewBox="0 0 {width} {len(values) * row_h + 4}" class="chart" role="img">{"".join(bars)}</svg>'


def scatter(x: pd.Series, y: pd.Series, width: int = 760, height: int = 380) -> str:
    """Quality vs log-latency with direct labels."""
    l, r, t, b = 60, 170, 20, 46
    keep = x[x > 0].index  # precomputed (offline) runs have no latency and can't sit on a log axis
    x, y = x[keep], y[keep]
    lx = x.map(math.log10)
    x0, x1, y0, y1 = lx.min() - 0.1, lx.max() + 0.1, max(0.0, y.min() - 0.05), min(1.0, y.max() + 0.05)
    px = lambda v: l + (math.log10(v) - x0) / (x1 - x0) * (width - l - r)
    py = lambda v: height - b - (v - y0) / (y1 - y0) * (height - t - b)
    ticks = [10**e for e in range(math.floor(x0), math.ceil(x1) + 1) if x0 <= e <= x1] or [10 ** round(x0)]
    grid = "".join(f'<line x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{t}" y2="{height - b}" class="grid"/><text x="{px(v):.1f}" y="{height - b + 16}" text-anchor="middle" class="tick">{v:,.0f} ms</text>' for v in ticks)
    pts = "".join(
        f'<g><title>{esc(s)}: MRR {y[s]:.3f}, p50 {x[s]:,.0f} ms</title>'
        f'<circle cx="{px(x[s]):.1f}" cy="{py(y[s]):.1f}" r="5" fill="{"var(--neutral)" if s == BASELINE_RUN else "var(--series-1)"}" stroke="var(--surface)" stroke-width="2"/>'
        f'<text x="{px(x[s]) + 8:.1f}" y="{py(y[s]) + 4:.1f}" class="lbl">{esc(s)}</text></g>'
        for s in x.index
    )
    axes = f'<text x="{(width - r + l) / 2}" y="{height - 6}" text-anchor="middle" class="tick">p50 latency per query (log scale)</text><text x="14" y="{t + 10}" class="tick">MRR@10</text>'
    return f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">{grid}{axes}{pts}</svg>'


def table(df: pd.DataFrame, pct_cols: Iterable[str] = (), lower_better: Iterable[str] = ()) -> str:
    """HTML table with a subtle per-column heat tint (blue = better)."""
    pct_cols, lower_better = set(pct_cols), set(lower_better)
    head = "".join(f"<th>{esc(str(c))}</th>" for c in [df.index.name or "", *df.columns])
    rows = []
    for idx, r in df.iterrows():
        cells = [f'<td class="name">{esc(str(idx))}</td>']
        for c in df.columns:
            v = r[c]
            if isinstance(v, (int, float)) and not pd.isna(v) and df[c].dtype != object:
                col = df[c].dropna()
                span = (col.max() - col.min()) or 1
                good = (col.max() - v) / span if c in lower_better else (v - col.min()) / span
                txt = f"{v:.1%}" if c in pct_cols else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3f}")
                cells.append(f'<td style="background: color-mix(in oklab, var(--series-1) {good * 28:.0f}%, transparent)">{txt}</td>')
            else:
                cells.append(f"<td>{esc('' if pd.isna(v) else str(v))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# COMMAND ----------

# MAGIC %md
# MAGIC ## Side-by-side examples
# MAGIC For each query type: the query where the type's best strategy gained the most MRR over the baseline. Plus the two worst
# MAGIC regressions of the overall best strategy, to keep the story honest.

# COMMAND ----------


def result_list(row: pd.Series, constraints: dict[str, Any], expected: list[str]) -> str:
    items = []
    for s in row["skus"][:5]:
        p = cat[s]
        mark = '<span class="ok">✓ expected</span>' if s in expected else ('<span class="bad">✗ violates request</span>' if any(constraints.values()) and not satisfies(s, constraints) else "")
        score = f' <span class="muted">rerank score {row["scores"][s]:.2f}</span>' if s in row["scores"] else ""
        items.append(f"<li><b>{esc(p.brand_name or '')}</b> {esc(p.product_name)} <span class='muted'>SKU {s}</span> {mark}{score}</li>")
    filters = f'<div class="muted">filters: <code>{esc(json.dumps(row["filters"]))}</code></div>' if row["filters"] else ""
    return f"<ol>{''.join(items) or '<li>(no results)</li>'}</ol>{filters}"


def example_card(query: str, left: str, right: str, title: str) -> str:
    meta = src.loc[query]
    constraints, expected = json.loads(meta["constraints"]), list(meta["expected_skus"])
    rows = per_query[(per_query["query"] == query)].set_index("strategy")
    l, r = rows.loc[left], rows.loc[right]
    want = cat[meta["seed_sku"]]
    return f"""<div class="card"><div class="kicker">{esc(title)} · {esc(meta['query_type'])}</div>
<h3>“{esc(query)}”</h3><div class="muted">target: {esc(want.product_name)} (SKU {want.sku}){' · constraints ' + esc(json.dumps({k: v for k, v in constraints.items() if v})) if any(constraints.values()) else ''}</div>
<div class="sbs"><div><h4>{esc(left)} <span class="muted">MRR {l['mrr_at_10']:.2f}</span></h4>{result_list(l, constraints, expected)}</div>
<div><h4>{esc(right)} <span class="muted">MRR {r['mrr_at_10']:.2f}</span></h4>{result_list(r, constraints, expected)}</div></div></div>"""


examples: list[str] = []
for qtype in QUERY_TYPES:
    winner = by_type[qtype].drop(BASELINE_RUN).idxmax()
    qs = src[src.query_type == qtype].index
    gain = (mrr_pivot.loc[qs, winner] - mrr_pivot.loc[qs, BASELINE_RUN]).sort_values(ascending=False)
    if len(gain) and gain.iloc[0] > 0:
        examples.append(example_card(gain.index[0], BASELINE_RUN, winner, "Improvement"))
regress = (mrr_pivot[best] - mrr_pivot[BASELINE_RUN]).sort_values().head(2)
for q, d in regress.items():
    if d < 0:
        examples.append(example_card(q, BASELINE_RUN, best, "Regression (honest view)"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble the deck

# COMMAND ----------

sig_rows = []
for s in lb.index.drop(BASELINE_RUN):
    m, lo, hi = paired_bootstrap(mrr_pivot[BASELINE_RUN], mrr_pivot[s])
    sig_rows.append({"strategy": s, "ΔMRR@10": m, "95% CI": f"[{lo:+.3f}, {hi:+.3f}]", "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "no clear difference")})
sig = pd.DataFrame(sig_rows).set_index("strategy").sort_values("ΔMRR@10", ascending=False)

judge_cols = ["correctness", "retrieval_relevance", "retrieval_sufficiency", "retrieval_groundedness", "answer_policy", "mrr_at_10"]
jt = pd.DataFrame({c: metric(judged, c) for c in judge_cols}).dropna(how="all")
jt.index.name = "strategy (judged subset)"

types_html = "".join(
    f'<div class="mini"><h4>{esc(t)}</h4>{hbar(by_type[t].sort_values(ascending=False), vmax=1.0, width=520)}</div>' for t in QUERY_TYPES
)
types_table = table(by_type.loc[lb.mrr_at_10.sort_values(ascending=False).index.intersection(by_type.index), list(QUERY_TYPES)]
                    .rename_axis("strategy (MRR@10 by query type)"))  # print/PDF stand-in for the per-type bar charts
lb_display = lb.rename(columns={"hit_at_1": "hit@1", "hit_at_10": "hit@10", "mrr_at_10": "MRR@10", "ndcg_at_10": "nDCG@10",
                                "constraint_precision_at_10": "constraint prec@10", "excluded_brand_leak_at_10": "excluded-brand leak", "zero_results": "zero results"})
lb_display.index.name = "strategy"
gain_best = lb.loc[best, "mrr_at_10"] - base["mrr_at_10"]
n_queries = len(src)

def diagnosis_slide() -> str:
    """Why reranking alone moved the averages less than expected, computed from the baseline and ai_decide runs."""
    base_q = per_query[per_query.strategy == BASELINE_RUN].set_index("query")
    ident = base_q.query_type == "identifier"
    non_id = base_q[~ident]
    headroom = int(((non_id.mrr_at_10 > 0) & (non_id.mrr_at_10 < 1)).sum())
    already = int((non_id.hit_at_1 == 1).sum())
    ident_mrr = per_query[per_query.query_type == "identifier"].pivot_table(index="strategy", values="mrr_at_10").mrr_at_10
    reorder_only = [s for s in ident_mrr.index if s.startswith(("03_", "04_", "06_ai_decide"))]
    items = [
        f"<b>Identifier lookups are invisible to relevance ranking.</b> {int(ident.sum())} of {len(base_q)} queries are SKU/UPC lookups; "
        f"every strategy that only ranks text scored at most MRR {ident_mrr.reindex(reorder_only).max():.2f} on them, which caps the overall average.",
        f"<b>HYBRID leaves little headroom on the rest.</b> Of {len(non_id)} non-identifier queries the target is already at rank 1 for "
        f"{already}; only {headroom} have it at rank 2–10, the only slice a reranker can improve.",
        "<b>Rerankers can only reorder.</b> When the excluded brand fills the candidate pool, reranking cannot remove it; only a filter can.",
        "<b>Coarse decisions tie.</b> A 4-level <code>score</code> question leaves several candidates tied at the top, so order falls back "
        "to retrieval; continuous <code>noul</code> probabilities separate them.",
        "<b>A cross-encoder is not instruction-following.</b> It matches the words in the request, including the brand being excluded, and "
        "treats pack-size variants as equivalent.",
    ]
    return f"<section><h2>Why reranking alone moved the average less than expected</h2><ol class='big'>{''.join(f'<li>{i}</li>' for i in items)}</ol></section>"


def recommendations_slide() -> str:
    """Prioritized findings computed from the runs (holdout numbers where available) so the slide stays true on re-runs."""
    holdout = strategy_runs(split="holdout")

    def v(strategy: str, col: str) -> float:
        return float(lb.loc[strategy, col]) if strategy in lb.index else float("nan")

    def h(strategy: str, metric_name: str = "mrr_at_10") -> float:
        col = f"metrics.{metric_name}/mean"
        return float(holdout.loc[strategy, col]) if strategy in holdout.index and col in holdout else float("nan")

    def hp50(strategy: str) -> float:
        col = "metrics.latency_ms/median"
        return float(holdout.loc[strategy, col]) if strategy in holdout.index and col in holdout else float("nan")

    measurement = mlflow.search_runs([experiment.experiment_id], "tags.demo = 'retrieval_measurement'", order_by=["start_time DESC"], max_results=1)
    judged_best = float(measurement.iloc[0].get("metrics.10_enriched_router_quality.judged_exact_at_1", float("nan"))) if len(measurement) else float("nan")
    noise_gap = judged_best - v("10_enriched_router_quality", "hit_at_1")

    para = strategy_runs(split="paraphrase")

    def pm(strategy: str) -> float:
        return float(para.loc[strategy, "metrics.mrr_at_10"]) if strategy in para.index and "metrics.mrr_at_10" in para else float("nan")

    items = [
        f"<b>Route by query shape first; it's the biggest, cheapest win.</b> SKU/UPC → exact filter, \"not X\" → <code>brand_name NOT</code> "
        f"filter parsed from the text (no LLM), else cleaned HYBRID. Fast router: MRR {h('09_router_fast'):.3f} on holdout at "
        f"~{v('09_router_fast', 'p50_ms'):.0f} ms p50 vs HYBRID {h(BASELINE_RUN):.3f}.",
        f"<b>Rerank the router's candidates with a GPU cross-encoder.</b> Router + bge-reranker-v2-m3 (Model Serving, 25 candidates): "
        f"MRR {v('13_router_bge', 'mrr_at_10'):.3f} dev / {h('13_router_bge'):.3f} holdout at ~{v('13_router_bge', 'p50_ms'):.0f} ms p50, "
        f"versus the ai_decide quality router {v('09_router_quality', 'mrr_at_10'):.3f} / {h('09_router_quality'):.3f} at "
        f"~{v('09_router_quality', 'p50_ms') / 1000:.1f}s. Qwen3-Reranker-0.6B was slower and no better on a small GPU.",
        f"<b>For the last points, learn the ranking.</b> A LightGBM LambdaRank model over router candidates (pool/ANN/FULL_TEXT ranks, "
        f"ai_decide, two cross-encoders, lexical features) reached MRR {v('17_ltr_cv', 'mrr_at_10'):.3f} (5-fold CV on dev) and "
        f"{h('17_ltr'):.3f} on holdout, the best result, but at ~{hp50('17_ltr') / 1000:.1f}s p50 because every signal is computed per "
        f"request; distil to the few strongest signals for production.",
        f"<b>Embed and enrich what shoppers search for.</b> A compact title | brand | category | attributes | codes field helped every "
        f"strategy (router {v('09_router_quality', 'mrr_at_10'):.3f} → {v('10_enriched_router_quality', 'mrr_at_10'):.3f}); Doc2Query "
        f"pseudo-queries added a little more ({v('15_d2q_router_quality', 'mrr_at_10'):.3f}) but are likely optimistic on this LLM-generated "
        f"eval set. Qwen3 instruction-aware embeddings matched gte-large under the router ({v('14_qwen3_router_quality', 'mrr_at_10'):.3f}).",
        f"<b>Keep LLMs out of the hot path unless they earn it.</b> An LLM structured query plan scored {v('16_plan_router_quality', 'mrr_at_10'):.3f} "
        f"but leaked the excluded brand on {v('16_plan_router_quality', 'excluded_brand_leak_at_10'):.0%} of exclusion queries vs "
        f"{v('10_enriched_router_quality', 'excluded_brand_leak_at_10'):.0%} for the rules; LLM-planned filters varied run to run. Use them for "
        f"the long tail the rules don't cover, with catalog validation and caching.",
        f"<b>Keep the rules as the router; add ai_decide only as a fallback.</b> Classifying the route with one ai_decide call "
        f"(<code>create_router_tool(..., classifier=\"ai_decide\")</code>) tied the rules on holdout ({h('20_ai_router_bge'):.3f} vs "
        f"{h('13_router_bge'):.3f} with bge) but missed some plain \"not X\" exclusions (brand leak {v('20_ai_router_bge', 'excluded_brand_leak_at_10'):.0%} "
        f"vs {v('13_router_bge', 'excluded_brand_leak_at_10'):.0%}) and added ~{v('20_ai_router_bge', 'p50_ms') - v('13_router_bge', 'p50_ms'):.0f} ms p50. "
        f"On paraphrased exclusions the rules can't parse (\"I'm done with DeWalt\") it lifted MRR {pm('13_router_bge'):.3f} → {pm('20_ai_router_bge'):.3f}. "
        f"Run it only when the rules find no exclusion but the query names a catalog brand.",
        f"<b>Plan for scale: depth matters.</b> On a ~5M-row storage-optimized index with near-duplicate distractors, the 12-candidate "
        f"quality router fell to MRR {v('18_scale_router_quality', 'mrr_at_10'):.3f} (hit@10 {v('18_scale_router_quality', 'hit_at_10'):.0%}); "
        f"50 candidates: {v('18_scale_router_quality_50', 'mrr_at_10'):.3f} with ai_decide, {v('18_scale_router_bge_50', 'mrr_at_10'):.3f} with "
        f"the bge cross-encoder. At 100M, budget deeper candidate pools and a fast cross-encoder.",
        f"<b>Measure honestly.</b> An ESCI-style LLM judge found ~{noise_gap * 100:.0f} points of apparent rank-1 misses were valid "
        f"alternatives (judged exact@1 {judged_best:.0%} for the enriched quality router), so true rank-1 accuracy is higher than hit@1 shows; "
        f"confirm on real production query logs with graded labels before rollout. The MLflow dataset, scorers and traces make every change a "
        f"measured experiment.",
    ]
    return f"<section><h2>Recommendations</h2><ol class='big'>{''.join(f'<li>{i}</li>' for i in items)}</ol></section>"


def research_slide() -> str:
    rows = [
        ("Instruction-aware embeddings", "Qwen3-Embedding (MTEB top tier)", "databricks-qwen3-embedding-0-6b, self-managed vectors", "14"),
        ("Open cross-encoder rerankers", "ShopRank-Bench 2026; Qwen3-Reranker, bge-reranker-v2-m3", "Model Serving GPU endpoints", "12-13"),
        ("Doc2Query expansion", "AliExpress SAM-D2Q (+3.4% GMV); Doc2Query++ dual-index fusion", "offline LLM generation + RRF", "15"),
        ("Structured query plans", "Instacart Intent Engine; Amazon hint-augmented reranking", "gpt-oss-120b plan, cached, validated", "16"),
        ("Listwise reranking", "RankGPT (EMNLP 2023)", "ai_decide choice over top 5", "17"),
        ("Learned ranking / distillation", "Walmart LLM-labelled dense retrieval (+5.1% nDCG); Etsy <10 ms student", "LightGBM LambdaRank", "17"),
        ("Graded relevance evaluation", "Amazon ESCI; bias-corrected LLM judging", "Claude Sonnet ESCI judge + holdout set", "11, 19"),
        ("Billion-scale vector search", "Databricks storage-optimized endpoints (~1B vectors)", "5M-row storage-optimized test", "18"),
    ]
    body = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>" for a, b, c, d in rows)
    return ("<section><h2>State-of-the-art patterns tested</h2><p class='muted'>All models are Databricks-hosted or served on Databricks; "
            "all vectors live in Databricks AI Search.</p><table class='plain'><tr><th>pattern</th><th>industry evidence</th>"
            f"<th>implementation</th><th>notebook</th></tr>{body}</table></section>")


def holdout_slide() -> str:
    holdout = strategy_runs(split="holdout")
    if not len(holdout):
        return ""
    df = pd.DataFrame({"dev MRR@10": lb["mrr_at_10"], "holdout MRR@10": metric(holdout, "mrr_at_10"),
                       "dev hit@1": lb["hit_at_1"], "holdout hit@1": metric(holdout, "hit_at_1")}).dropna(subset=["holdout MRR@10"])
    df = df.sort_values("holdout MRR@10", ascending=False)
    df.index.name = "strategy"
    return ("<section><h2>Holdout check</h2><p class='muted'>300 queries generated from products never used to build or tune the strategies. "
            f"Rules, prompts and the learned ranker were developed on dev; these are the unbiased numbers.</p>{table(df)}</section>")


AI_ROUTER_PAIRS = [("09_router_fast", "20_ai_router_fast"), ("09_router_quality", "20_ai_router_quality"), ("13_router_bge", "20_ai_router_bge")]


def ai_router_slide() -> str:
    """Rules vs ai_decide classification for the router, on dev, holdout and the paraphrased-exclusion set (notebook 20)."""
    para = strategy_runs(split="paraphrase")
    if not len(para):
        return ""
    holdout = strategy_runs(split="holdout")
    rows = []
    for rules, ai in AI_ROUTER_PAIRS:
        for name, classifier in [(rules, "rules"), (ai, "ai_decide")]:
            pick = lambda df, m, agg="mean": float(metric(df, m, agg).get(name, float("nan")))
            rows.append({"strategy": name, "classifier": classifier, "dev MRR@10": pick(runs, "mrr_at_10"),
                         "holdout MRR@10": pick(holdout, "mrr_at_10"), "paraphrase MRR@10": pick(para, "mrr_at_10"),
                         "paraphrase brand leak": pick(para, "excluded_brand_leak_at_10"), "dev p50_ms": pick(runs, "latency_ms", "median")})
    df = pd.DataFrame(rows).set_index("strategy")
    return ("<section><h2>ai_decide as the router</h2><p class='muted'>Same retrieval and reranker in each pair; only the classifier "
            "differs. Rules = SKU/UPC regex + negation regex over the brand vocabulary. ai_decide = one call with two choice questions "
            "(route; excluded brand chosen from the query's top-10 brand facets). The paraphrase set rewrites the dev exclusion queries "
            "so the rules miss them (\"I'm done with DeWalt\").</p>"
            f"{table(df, pct_cols=['paraphrase brand leak'], lower_better=['paraphrase brand leak', 'dev p50_ms'])}</section>")


def noise_slide() -> str:
    m = mlflow.search_runs([experiment.experiment_id], "tags.demo = 'retrieval_measurement'", order_by=["start_time DESC"], max_results=1)
    if not len(m):
        return ""
    r = m.iloc[0]
    df = pd.DataFrame({
        "seed hit@1": lb["hit_at_1"],
        "judged exact@1": {c.split(".")[1]: r[c] for c in m.columns if c.endswith(".judged_exact_at_1")},
        "graded nDCG@5": {c.split(".")[1]: r[c] for c in m.columns if c.endswith(".graded_ndcg_at_5")},
    }).dropna(subset=["judged exact@1"])
    df.index.name = "strategy"
    return ("<section><h2>How much of the remaining error is real?</h2><p class='muted'>Ground truth is the seed product a query was "
            "generated from. An ESCI-style LLM judge (Claude Sonnet 5.5) re-labels rank-1 results that aren't the seed: judged exact@1 "
            f"counts the seed or a product judged an exact match. LLM judges are biased; treat as an estimate.</p>{table(df)}</section>")


def scale_slide() -> str:
    pairs = {"18_scale_hybrid": "10_enriched_hybrid", "18_scale_router_fast": "10_enriched_router_fast",
             "18_scale_router_quality": "10_enriched_router_quality", "18_scale_router_quality_50": "10_enriched_router_quality",
             "18_scale_router_bge_50": "13_router_bge"}
    rows = [{"strategy at ~5M": big, "38K counterpart": small, "MRR@10 38K": lb["mrr_at_10"].get(small), "MRR@10 ~5M": lb["mrr_at_10"].get(big),
             "hit@10 ~5M": lb["hit_at_10"].get(big), "p50 ms ~5M": lb["p50_ms"].get(big), "p99 ms ~5M": lb["p99_ms"].get(big)}
            for big, small in pairs.items() if big in lb.index]
    if not rows:
        return ""
    df = pd.DataFrame(rows).set_index("strategy at ~5M")
    return ("<section><h2>Scale: 38K → ~5M products (storage-optimized endpoint)</h2><p class='muted'>Realistic distractors (brand swapped "
            "within category, sizes/counts perturbed) added to the real catalog; same strategies, only the endpoint type changes. The retailer's "
            f"production index (~100M) is ~20× larger again. Deeper candidate pools (50) and a fast cross-encoder recover much of the loss.</p>{table(df)}</section>")


slides = [
    f"""<section class="title"><div class="kicker">Hardware retail × Databricks · {dt.date.today():%B %d, %Y}</div>
<h1>Improving product retrieval on Databricks AI Search</h1>
<p class="lead">{n_queries} labelled shopper queries · {len(cat):,} SKUs · {len(lb)} retrieval strategies, all traced and scored in MLflow.</p>
<div class="stats"><div><b>{base['mrr_at_10']:.3f}</b><span>baseline MRR@10 (HYBRID)</span></div>
<div><b>{lb.loc[best, 'mrr_at_10']:.3f}</b><span>best MRR@10 ({esc(best)})</span></div>
<div><b>{gain_best:+.3f}</b><span>absolute gain</span></div></div></section>""",
    """<section><h2>The ask</h2><ul class="big">
<li>Improve retrieval accuracy on <b>structured retail product data</b> where shoppers mix intent, brand, specs and exclusions.</li>
<li>Show which Databricks AI Search capabilities move the needle, with <b>measured</b> evidence rather than anecdotes.</li>
<li>Give a reproducible, governed evaluation harness the retailer can run on its own catalog.</li></ul></section>""",
    f"""<section><h2>How we measured</h2><div class="two"><div><h4>Data &amp; index</h4><ul>
<li><code>{PRODUCTS_TABLE}</code>: {len(cat):,} products, 500+ merchandise classes</li>
<li>Delta Sync AI Search index, managed <code>{EMBEDDING_ENDPOINT}</code> embeddings, ANN / FULL_TEXT / HYBRID</li>
<li>MLflow evaluation dataset <code>{EVAL_DATASET}</code>: {n_queries} queries, 6 types, ground truth from the catalog</li></ul></div>
<div><h4>Metrics (MLflow scorers)</h4><ul><li><b>hit@k, MRR@10, nDCG@10</b>: is the right product found, and how high</li>
<li><b>constraint precision@10</b>: share of results honoring brand / class / exclusion</li><li><b>excluded-brand leak</b>: “not DeWalt” but DeWalt shown</li>
<li><b>latency</b> (p50 / p90 / p99 from traces), LLM &amp; ai_decide call counts</li><li><b>Correctness</b> + retrieval judges on a 100-query subset</li></ul></div></div></section>""",
    """<section><h2>Methods compared</h2><table class="plain"><tr><th>family</th><th>what it does</th><th>Databricks capability</th></tr>
<tr><td>Plain RAG</td><td>query → top-10</td><td><code>VectorSearchRetrieverTool</code> ANN / FULL_TEXT / HYBRID</td></tr>
<tr><td>Reranker</td><td>cross-encoder reorders candidates; columns fixed or targeted to the request</td><td><code>DatabricksReranker(columns_to_rerank=…)</code> + <code>ai_decide</code> column profile</td></tr>
<tr><td>Dynamic filtering</td><td>LLM turns brand/exclusion/class intent into index filters</td><td><code>VectorSearchRetrieverTool(dynamic_filter=True)</code> + gpt-oss-120b</td></tr>
<tr><td>Instructed retrieval</td><td>every candidate judged against the request + store policy</td><td><code>ai_decide</code> REST (score / noul questions)</td></tr>
<tr><td>dao-ai instructed</td><td>LLM decomposition into filtered subqueries → RRF → rerank</td><td><code>dao_ai.tools.create_ai_search_tool</code> + <code>dao_ai.config</code></td></tr>
<tr><td>Router (09)</td><td>SKU/UPC → exact lookup; "not X" → brand NOT filter; else cleaned HYBRID; optional ai_decide</td><td><code>VectorSearchRetrieverTool</code> filters + <code>ai_decide</code>, no generative LLM</td></tr>
<tr><td>Enriched index (10)</td><td>embed a compact title | brand | class | attributes | codes field</td><td>second Delta Sync index on <code>search_text</code></td></tr>
<tr><td>Cross-encoder rerank (13)</td><td>open rerankers score router candidates on GPU</td><td>bge-reranker-v2-m3 / Qwen3-Reranker on Model Serving</td></tr>
<tr><td>Doc2Query (15)</td><td>LLM-written shopper searches indexed with each product</td><td>offline generation + new index / RRF fusion</td></tr>
<tr><td>Query plan (16)</td><td>LLM parses hard vs soft constraints, cached</td><td>gpt-oss-120b tool call, catalog-validated</td></tr>
<tr><td>Learned ranker (17)</td><td>LambdaRank over retrieval, ai_decide, cross-encoder and lexical signals</td><td>LightGBM, 5-fold CV + holdout</td></tr>
<tr><td>Facet-guided</td><td>facet counts give the real filter vocabulary; ai_decide picks</td><td>AI Search <code>facets</code> + <code>ai_decide</code> choice</td></tr></table></section>""",
    research_slide(),
    diagnosis_slide(),
    f"""<section><h2>Leaderboard <span class="muted">(all {n_queries} queries)</span></h2>{table(lb_display, pct_cols=['hit@1', 'hit@10', 'constraint prec@10', 'excluded-brand leak', 'zero results'], lower_better=['excluded-brand leak', 'zero results', 'p50_ms', 'p90_ms', 'p99_ms'])}</section>""",
    f"""<section class="screen-only"><h2>MRR@10 by strategy</h2>{hbar(lb['mrr_at_10'], vmax=1.0)}<p class="muted">Gray = HYBRID baseline. Hover for values.</p></section>""",
    f"""<section><h2>Are the gains real?</h2><p class="muted">Paired bootstrap over the same queries (2,000 resamples). better / worse = the 95% CI of the MRR@10 change excludes 0.</p>{table(sig)}</section>""",
    f"""<section><h2>Different query shapes, different winners</h2><div class="grid3 screen-only">{types_html}</div><div class="print-only">{types_table}</div></section>""",
    f"""<section><h2>Quality vs latency</h2>{scatter(lb['p50_ms'], lb['mrr_at_10'])}<p class="muted">Pick the cheapest strategy that is good enough per query shape; route only hard queries to the heavier pipeline.</p></section>""",
    f"""<section><h2>Answer quality (end-to-end RAG)</h2><p class="muted">gpt-oss-120b answers from each strategy's top-5; MLflow judges on the 100-query judged dataset.</p>{table(jt) if len(jt) else '<p>No judged runs found.</p>'}</section>""",
    holdout_slide(),
    ai_router_slide(),
    noise_slide(),
    scale_slide(),
    f"""<section><h2>Side by side: baseline vs improved</h2>{''.join(examples)}</section>""",
    recommendations_slide(),
    f"""<section><h2>Reproduce &amp; explore in MLflow</h2><ul class="big"><li>Experiment: <a href="{EXP_URL}">{esc(EXPERIMENT_PATH)}</a>. Open <b>Evaluations</b>, select two runs, then <b>Compare</b> for row-level side by side.</li>
<li>Every run evaluates the same MLflow dataset <code>{EVAL_DATASET}</code>, so rows align across runs.</li>
<li>Notebooks: <code>/Users/{esc(USER)}/ai-search-product-retrieval</code> (00–91). Library versions: {esc(json.dumps(LIB_VERSIONS))}</li></ul>
<table class="plain"><tr><th>run</th><th>run id</th></tr>{''.join(f'<tr><td>{esc(s)}</td><td><a href="{EXP_URL}/runs/{rid}">{rid}</a></td></tr>' for s, rid in runs['run_id'].items())}</table></section>""",
]

CSS = """
:root{--surface:#fcfcfb;--card:#ffffff;--text:#0b0b0b;--muted:#52514e;--grid:#e4e3df;--series-1:#2a78d6;--neutral:#a3a29c;--ok:#008300;--bad:#c62f2e;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--card:#232322;--text:#fff;--muted:#c3c2b7;--grid:#383835;--series-1:#3987e5;--neutral:#77766f;--ok:#3fae3f;--bad:#e66767;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{scroll-snap-type:y proximity}section{min-height:100vh;padding:56px 7vw;scroll-snap-align:start;border-bottom:1px solid var(--grid)}
h1{font-size:44px;line-height:1.1;margin:.2em 0}h2{font-size:30px;margin:0 0 20px}h3{margin:.3em 0}h4{margin:.4em 0}
.kicker{text-transform:uppercase;letter-spacing:.08em;font-size:12px;color:var(--muted)}.lead{font-size:20px;color:var(--muted)}
.title{display:flex;flex-direction:column;justify-content:center}.stats{display:flex;gap:48px;margin-top:32px}.stats b{display:block;font-size:44px}.stats span{color:var(--muted)}
.muted{color:var(--muted);font-size:.9em}.big li{font-size:20px;margin:.5em 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:40px}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td.name,.plain td,.plain th{text-align:left}
.chart{width:100%;max-width:820px;display:block}.chart .lbl{font-size:12px;fill:var(--text)}.chart .val,.chart .tick{font-size:11px;fill:var(--muted)}.chart .grid{stroke:var(--grid)}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}.mini{background:var(--card);padding:12px 16px;border-radius:10px}
.card{background:var(--card);border-radius:12px;padding:20px 24px;margin:0 0 24px;box-shadow:0 1px 3px rgba(0,0,0,.08)}.sbs{display:grid;grid-template-columns:1fr 1fr;gap:28px}
.sbs ol{padding-left:20px;margin:6px 0}.sbs li{margin:4px 0;font-size:14px}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
code{font-size:.88em;background:color-mix(in oklab,var(--grid) 60%,transparent);padding:1px 5px;border-radius:4px}a{color:var(--series-1)}
.print-only{display:none}
@page{size:A4 landscape;margin:12mm}
@media print{:root{--surface:#fff;--card:#fff;--text:#0b0b0b;--muted:#52514e;--grid:#e4e3df;--series-1:#2a78d6;--neutral:#a3a29c;--ok:#008300;--bad:#c62f2e;color-scheme:light}
body{font-size:13px;-webkit-print-color-adjust:exact;print-color-adjust:exact}main{scroll-snap-type:none}
section{min-height:auto;padding:0;border:0;break-before:page}section:first-child{break-before:auto}.title{min-height:170mm}
h1{font-size:36px}h2{font-size:24px;margin-bottom:12px;break-after:avoid}.big li{font-size:16px}
table{font-size:11px}th,td{padding:3px 8px}tr,.card,.mini,.chart,svg{break-inside:avoid}.card{box-shadow:none;border:1px solid var(--grid);padding:12px 16px;margin-bottom:12px}
.grid3{grid-template-columns:1fr 1fr}.sbs li{font-size:11px}a{color:inherit;text-decoration:none}}
@media print{svg.chart{width:auto;max-width:100%;max-height:172mm;margin:0 auto}.mini svg.chart{max-height:165mm}.grid3{grid-template-columns:1fr 1fr;gap:16px}.mini{padding:6px 8px}}
@media print{.screen-only{display:none!important}.print-only{display:block}}
"""
report_html = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Hardware retail · AI Search retrieval results</title><style>{CSS}</style></head><body><main>{''.join(slides)}</main></body></html>"

# COMMAND ----------

dbutils.fs.mkdirs(REPORT_VOLUME_PATH)
report_path = f"{REPORT_VOLUME_PATH}/product_retrieval_report.html"
with open(report_path, "w") as f:
    f.write(report_html)
with mlflow.start_run(run_name="91_report"):
    mlflow.set_tags({"demo": "product_retrieval_report"})
    mlflow.log_artifact(report_path)
print(report_path, f"{len(report_html):,} bytes")

# COMMAND ----------

displayHTML(report_html)
