from __future__ import annotations

from typing import Protocol

from product_retrieval.documents import RetrievalResult


class Retriever(Protocol):
    """Every strategy: a named callable from the shopper's request to ranked products."""

    name: str

    def __call__(self, query: str) -> RetrievalResult: ...
