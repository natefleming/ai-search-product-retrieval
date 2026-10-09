import pytest

from product_retrieval.understanding import analyze, brand_key, clean_query


@pytest.mark.parametrize(
    "query,route,code",
    [
        ("Do you have item 00176279?", "identifier", "00176279"),
        ("do you carry upc 0243925469525", "identifier", "0243925469525"),
        ("brushless impact driver, anything but DeWalt", "exclusion", None),
        ("20V cordless drill kit", "general", None),
        ("5 in 320 grit sanding discs 15 pack not 3M", "exclusion", None),
    ],
)
def test_routes(vocabulary, query, route, code):
    analysis = analyze(query, vocabulary)
    assert analysis.route == route
    assert analysis.code == code


def test_excluded_brand_uses_catalog_spelling(vocabulary):
    assert analyze("cordless drill, not Black & Decker", vocabulary).excluded_brand is None  # '&' → AND ≠ BLACK+DECKER
    assert analyze("cordless drill, not black+decker", vocabulary).excluded_brand == "BLACK+DECKER"
    assert analyze("impact driver -dewalt", vocabulary).excluded_brand == "DEWALT"


def test_unique_prefix_brand_match(vocabulary):
    assert analyze("stain remover powder not Mrs. Meyer's", vocabulary).excluded_brand == "MRS. MEYER'S CLEAN DAY"


def test_negation_without_brand_is_general(vocabulary):
    assert analyze("drill bits that do not rust", vocabulary).route == "general"


def test_exclusion_phrase_removed_from_search_text(vocabulary):
    assert analyze("brushless impact driver, anything but DeWalt", vocabulary).search_text == "brushless impact driver"


def test_clean_query_strips_filler():
    assert clean_query("Do you have any LED bulbs in stock?") == "LED bulbs"
    assert brand_key("Black & Decker") == "BLACKANDDECKER"
