from product_retrieval.backends import row_to_product, to_dict_filters, to_sql_filters
from product_retrieval.config import CatalogSchema
from product_retrieval.documents import Filter


def test_dict_filters_standard_endpoint():
    filters = [Filter(column="brand_name", op="ne", value="DEWALT"), Filter(column="sku", value="00176279"),
               Filter(column="price", op="lt", value=100), Filter(column="brand_name", op="in", value=["A", "B"])]
    assert to_dict_filters(filters) == {"brand_name NOT": "DEWALT", "sku": "00176279", "price <": 100, "brand_name": ["A", "B"]}


def test_sql_filters_storage_optimized_endpoint():
    filters = [Filter(column="brand_name", op="ne", value="MRS. MEYER'S"), Filter(column="price", op="gte", value=10),
               Filter(column="brand_name", op="not_in", value=["A", "B"]), Filter(column="product_name", op="like", value="drill")]
    assert to_sql_filters(filters) == (
        "brand_name != 'MRS. MEYER''S' AND price >= 10 AND brand_name NOT IN ('A', 'B') AND product_name LIKE '%drill%'"
    )


def test_row_to_product_maps_schema_and_keeps_extras():
    row = {"sku": "1", "product_name": "Drill", "brand_name": "ACME", "merchandise_class": "TOOLS", "description": "d",
           "upc": "9", "price": 5.0, "score": 0.8}
    p = row_to_product(row, CatalogSchema())
    assert (p.id, p.name, p.brand, p.category, p.upc, p.score, p.extra) == ("1", "Drill", "ACME", "TOOLS", "9", 0.8, {"price": 5.0})
