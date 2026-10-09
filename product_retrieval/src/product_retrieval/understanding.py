"""Deterministic query understanding: catalog vocabulary, SKU/UPC detection, brand exclusions, query cleaning."""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal

from databricks.sdk import WorkspaceClient
from pydantic import BaseModel, Field

Route = Literal["identifier", "exclusion", "general"]

CODE_RE = re.compile(r"(?<!\d)(\d{12,13}|\d{8})(?!\d)")  # 8-digit SKU or 12/13-digit UPC
FILLER_RE = re.compile(
    r"\b(do you (have|carry|sell|stock)( any)?|i('m| am) looking for|looking for|i need|can i (get|find)|where can i find|"
    r"please|in stock|item( number| no\.?)?|sku|upc)\b|site:\S+|[?!]",
    re.I,
)
EXCLUSION_RE = re.compile(
    r"(?:\b(?:not|anything but|except|excluding|other than|without|no)\s+|\s-)([\w&+.'’‐-― -]{2,40})", re.I
)


def brand_key(value: Any) -> str:
    """Normalize a brand for matching: 'Black & Decker' → 'BLACKANDDECKER', 'BLACK+DECKER' → 'BLACKDECKER'."""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper().replace("&", "AND"))


class CatalogVocabulary(BaseModel):
    """Exact brand and category values present in the index; filters are exact-match, so values must come from here."""

    brands: dict[str, str] = Field(default_factory=dict, description="normalized key → brand as stored")
    categories: list[str] = Field(default_factory=list)

    @classmethod
    def from_values(cls, brands: Iterable[str | None], categories: Iterable[str | None] = ()) -> CatalogVocabulary:
        vocab: dict[str, str] = {}
        for name in brands:
            if name:
                vocab.setdefault(brand_key(name), name)
        return cls(brands=vocab, categories=sorted({c for c in categories if c}))

    @classmethod
    def from_spark(cls, spark: Any, table: str, brand_column: str = "brand_name", category_column: str = "merchandise_class") -> CatalogVocabulary:
        rows = spark.table(table).select(brand_column, category_column).distinct().collect()
        return cls.from_values((r[0] for r in rows), (r[1] for r in rows))

    @classmethod
    def from_warehouse(
        cls,
        table: str,
        warehouse_id: str,
        brand_column: str = "brand_name",
        category_column: str = "merchandise_class",
        workspace_client: WorkspaceClient | None = None,
    ) -> CatalogVocabulary:
        """For runtimes without Spark (Model Serving, Databricks Apps)."""
        w = workspace_client or WorkspaceClient()
        resp = w.statement_execution.execute_statement(
            statement=f"SELECT DISTINCT {brand_column}, {category_column} FROM {table}", warehouse_id=warehouse_id, wait_timeout="50s"
        )
        rows = list(resp.result.data_array or [])
        chunk = resp.result.next_chunk_index
        while chunk is not None:
            part = w.statement_execution.get_statement_result_chunk_n(resp.statement_id, chunk)
            rows += part.data_array or []
            chunk = part.next_chunk_index
        return cls.from_values((r[0] for r in rows), (r[1] for r in rows))

    def match_brand(self, text: str) -> str | None:
        """Catalog brand for a phrase: exact normalized match, else a unique prefix match ('Mrs. Meyer's' → 'MRS. MEYER'S CLEAN DAY')."""
        key = brand_key(text)
        if len(key) >= 2 and key in self.brands:
            return self.brands[key]
        if len(key) >= 6:
            prefixed = [b for k, b in self.brands.items() if k.startswith(key)]
            if len(prefixed) == 1:
                return prefixed[0]
        return None


class QueryAnalysis(BaseModel):
    route: Route
    search_text: str
    code: str | None = None
    excluded_brand: str | None = None


def clean_query(query: str) -> str:
    cleaned = re.sub(r"\s+", " ", FILLER_RE.sub(" ", query)).strip(" ,.-")
    return cleaned or query


def find_excluded_brand(query: str, vocab: CatalogVocabulary) -> str | None:
    """Longest catalog brand within four words after a negation ('anything but DeWalt' → 'DEWALT')."""
    for match in EXCLUSION_RE.finditer(query):
        words = match.group(1).split()
        for n in range(min(len(words), 4), 0, -1):
            brand = vocab.match_brand(" ".join(words[:n]))
            if brand:
                return brand
    return None


def analyze(query: str, vocab: CatalogVocabulary) -> QueryAnalysis:
    """Route a request: identifier (SKU/UPC) → exclusion ("not <brand>") → general."""
    code = CODE_RE.search(query)
    if code:
        return QueryAnalysis(route="identifier", search_text=code.group(1), code=code.group(1))
    brand = find_excluded_brand(query, vocab)
    if brand:
        return QueryAnalysis(route="exclusion", search_text=clean_query(EXCLUSION_RE.sub(" ", query)), excluded_brand=brand)
    return QueryAnalysis(route="general", search_text=clean_query(query))
