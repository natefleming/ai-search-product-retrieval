import json

from conftest import FailingReranker, FakeBackend, ReverseReranker

from product_retrieval.documents import Product
from product_retrieval.rerankers import DecideAnswer, rerank_safely
from product_retrieval.strategies.ai_router import AIDecideRouter
from product_retrieval.strategies.filtering import guard_filters, parse_filter_items
from product_retrieval.strategies.fusion import rrf
from product_retrieval.strategies.plain import PlainRetriever
from product_retrieval.strategies.reranked import RerankedRetriever
from product_retrieval.strategies.router import QueryRouter
from product_retrieval.tools import as_tool, create_router_tool


def test_router_identifier_uses_exact_filter(backend, vocabulary):
    result = QueryRouter(backend, vocabulary)("do you have item 00000002?")
    assert result.route == "identifier" and result.ids[0] == "00000002"
    assert backend.requests[0].filters[0].column == "sku"


def test_router_upc_uses_upc_column(backend, vocabulary):
    result = QueryRouter(backend, vocabulary)("upc 0000000000003")
    assert result.ids[0] == "00000003" and backend.requests[0].filters[0].column == "upc"


def test_router_exclusion_filters_brand(backend, vocabulary):
    result = QueryRouter(backend, vocabulary)("brushless impact driver kit, anything but DeWalt")
    assert result.route == "exclusion" and "DEWALT" not in {p.brand for p in result.products}
    assert result.filters[0].op == "ne" and result.filters[0].value == "DEWALT"


def test_router_reranker_failure_keeps_retrieval_order(backend, vocabulary):
    fast = QueryRouter(backend, vocabulary)("impact driver kit")
    degraded = QueryRouter(backend, vocabulary, reranker=FailingReranker())("impact driver kit")
    assert degraded.ids == fast.ids and degraded.rerank_error.startswith("failing")


def test_reranked_retriever_applies_reranker(backend):
    base = PlainRetriever(backend, k=4)
    assert RerankedRetriever(base, ReverseReranker(), k=4)("impact driver").ids == base("impact driver").ids[::-1]


def test_rerank_safely_noop_without_reranker():
    p = [Product(id="a")]
    assert rerank_safely(None, "q", p).products == p


def test_parse_and_guard_llm_filters(vocabulary):
    parsed = parse_filter_items([{"key": "brand_name NOT", "value": "dewalt"}, {"key": "merchandise_class", "value": "made up"},
                                 {"key": "sku", "value": "1"}, {"key": "brand_name", "value": ["milwaukee", "nope"]}])
    assert [(f.column, f.op) for f in parsed] == [("brand_name", "ne"), ("merchandise_class", "eq"), ("sku", "eq"), ("brand_name", "in")]
    guarded = guard_filters(parsed, vocabulary, "brand_name", "merchandise_class")
    assert [(f.column, f.op, f.value) for f in guarded] == [("brand_name", "ne", "DEWALT"), ("brand_name", "in", "MILWAUKEE")]


def test_rrf_weights_and_dedupes():
    a, b, c = Product(id="a"), Product(id="b"), Product(id="c")
    fused = rrf([([a, b], 1.0), ([b, c], 1.0)])
    assert [p.id for p, _ in fused][0] == "b" and len(fused) == 3


def test_tool_contract(backend, vocabulary):
    tool = as_tool(QueryRouter(backend, vocabulary), name="product_search")
    docs = json.loads(tool.invoke({"query": "impact driver not dewalt"}))
    assert tool.name == "product_search" and set(tool.args) == {"query"}
    assert {"page_content", "metadata"} <= set(docs[0]) and docs[0]["metadata"]["route"] == "exclusion"


def test_router_ablation_without_exclusions(backend, vocabulary):
    result = QueryRouter(backend, vocabulary, exclusions=False, clean_queries=False)("impact driver kit not dewalt")
    assert result.filters == [] and result.route == "general"
    assert backend.requests[-1].text == "impact driver kit not dewalt"


def test_plan_router_uses_validated_exclusions(backend, vocabulary):
    from product_retrieval.strategies.planning import PlanRouter, QueryPlan, rerank_request

    class FakePlanner:
        def __call__(self, query):
            return QueryPlan(semantic_query="impact driver kit", product_type="impact driver", excluded_brands=["DEWALT"],
                             attributes=["brushless"])

    result = PlanRouter(backend, FakePlanner(), reranker=ReverseReranker())("impact driver, not dewalt, brushless")
    assert result.filters[0].value == "DEWALT" and "DEWALT" not in {p.brand for p in result.products}
    assert "required specs: brushless" in rerank_request("q", FakePlanner()("q"))


def test_ltr_lexical_features():
    from product_retrieval.ltr import lexical_features

    f = lexical_features("milwaukee 18v impact driver not dewalt", Product(id="1", name="Milwaukee M18 18V Impact Driver", brand="MILWAUKEE"), "DEWALT")
    assert f["brand_in_query"] == 1.0 and f["is_excluded_brand"] == 0.0 and f["title_overlap"] > 0.5 and f["number_overlap"] == 1.0


class FakeDecideClient:
    """Answers ai_decide with fixed choices; the excluded brand is given by value and mapped to its opaque label."""

    def __init__(self, route: str, excluded: str | None = None, confidence: float = 0.9) -> None:
        self.route, self.excluded, self.confidence, self.calls = route, excluded, confidence, []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        labels = questions["excluded_brand"]["criteria"]
        brand = next((label for label, value in labels.items() if value == self.excluded), "none")
        return {
            "route": DecideAnswer(type="choice", choice=self.route, confidence=self.confidence),
            "excluded_brand": DecideAnswer(type="choice", choice=brand, confidence=self.confidence),
        }


def _faceted(backend):
    """FakeBackend plus brand facet counts, as AI Search returns them for a facet request."""
    search = backend.search

    def with_facets(request):
        response = search(request)
        if request.facets:
            response.facets = {"brand_name": {p.brand: 1 for p in backend.products}}
        return response

    backend.search = with_facets
    return backend


def test_ai_router_paraphrased_exclusion_filters_brand(backend):
    client = FakeDecideClient("exclusion", excluded="DEWALT")
    result = AIDecideRouter(_faceted(backend), client=client)("impact driver kit, I'm done with DeWalt")
    assert result.route == "exclusion" and result.filters[0].op == "ne" and result.filters[0].value == "DEWALT"
    assert "DEWALT" not in {p.brand for p in result.products}
    assert "dewalt" not in backend.requests[-1].text.lower()  # brand words removed from the filtered search
    assert set(client.calls[0][1]["excluded_brand"]["criteria"].values()) >= {"DEWALT", "MILWAUKEE"}


def test_ai_router_identifier_uses_exact_filter(backend):
    result = AIDecideRouter(_faceted(backend), client=FakeDecideClient("identifier"))("item 00000002")
    assert result.route == "identifier" and result.ids == ["00000002"]


def test_ai_router_low_confidence_or_missing_brand_is_general(backend):
    low = AIDecideRouter(_faceted(backend), client=FakeDecideClient("exclusion", "DEWALT", confidence=0.3))("impact driver not dewalt")
    missing = AIDecideRouter(_faceted(FakeBackend()), client=FakeDecideClient("exclusion", "RYOBI"))("impact driver not ryobi")
    assert low.route == missing.route == "general" and not low.filters and not missing.filters


def test_ai_router_tool_needs_no_vocabulary(backend, monkeypatch):
    monkeypatch.setattr("product_retrieval.strategies.ai_router.AIDecideClient", lambda: FakeDecideClient("exclusion", "MILWAUKEE"))
    tool = create_router_tool({"index_name": "c.s.i"}, reranker="none", backend=_faceted(backend), classifier="ai_decide")
    out = json.loads(tool.invoke({"query": "impact driver kit, skip the Milwaukee stuff"}))
    assert out and {d["metadata"]["route"] for d in out} == {"exclusion"} and "MILWAUKEE" not in {d["metadata"]["brand"] for d in out}


def test_router_tool_ablation_options(backend, vocabulary):
    tool = create_router_tool({"index_name": "c.s.i"}, vocabulary, reranker="none", backend=backend, exclusions=False, clean_queries=False)
    docs = json.loads(tool.invoke({"query": "do you have impact driver kit, anything but DeWalt"}))
    assert {d["metadata"]["route"] for d in docs} == {"general"}
    assert backend.requests[-1].text == "do you have impact driver kit, anything but DeWalt" and not backend.requests[-1].filters
