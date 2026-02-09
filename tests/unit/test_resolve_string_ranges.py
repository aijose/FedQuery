"""Unit tests for ChromaStore._resolve_string_ranges."""

from datetime import date
from unittest.mock import MagicMock

import chromadb

from src.vectorstore.chroma_store import ChromaStore


def _make_store_with_dates(dates: list[str]) -> ChromaStore:
    """Create an in-memory ChromaStore with chunks at the given dates."""
    store = ChromaStore(path=":memory:")
    # ChromaDB in-memory shares state; wipe collection for clean fixture
    store._client.delete_collection("fomc_chunks")
    store._collection = store._client.get_or_create_collection(
        name="fomc_chunks", metadata={"hnsw:space": "cosine"},
    )
    if not dates:
        return store

    dim = 3
    ids = [f"chunk-{i}" for i in range(len(dates))]
    embeddings = [[0.1 * (i + 1)] * dim for i in range(len(dates))]
    documents = [f"Text for chunk {i}" for i in range(len(dates))]
    metadatas = [{"document_date": d, "document_id": f"doc-{i}"} for i, d in enumerate(dates)]

    store._collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return store


class TestResolveStringRanges:
    """_resolve_string_ranges converts $gte/$lte on strings to $in."""

    def test_no_and_clause_passthrough(self):
        store = _make_store_with_dates(["2024-12-18"])
        where = {"document_date": "2024-12-18"}
        assert store._resolve_string_ranges(where) == where

    def test_single_month_range(self):
        store = _make_store_with_dates([
            "2024-11-07", "2024-12-18", "2025-01-29",
        ])
        where = {"$and": [
            {"document_date": {"$gte": "2024-12-01"}},
            {"document_date": {"$lte": "2024-12-31"}},
        ]}
        resolved = store._resolve_string_ranges(where)

        # Should become a simple $in with the one matching date
        assert resolved == {"document_date": {"$in": ["2024-12-18"]}}

    def test_year_range(self):
        store = _make_store_with_dates([
            "2023-12-13", "2024-01-31", "2024-06-12", "2024-12-18", "2025-01-29",
        ])
        where = {"$and": [
            {"document_date": {"$gte": "2024-01-01"}},
            {"document_date": {"$lte": "2024-12-31"}},
        ]}
        resolved = store._resolve_string_ranges(where)

        assert resolved == {"document_date": {"$in": ["2024-01-31", "2024-06-12", "2024-12-18"]}}

    def test_no_matching_dates(self):
        store = _make_store_with_dates(["2024-01-31", "2024-06-12"])
        where = {"$and": [
            {"document_date": {"$gte": "2099-01-01"}},
            {"document_date": {"$lte": "2099-12-31"}},
        ]}
        resolved = store._resolve_string_ranges(where)

        # No matching dates → returns impossible $in to guarantee empty results
        assert resolved == {"document_date": {"$in": ["__no_match__"]}}

    def test_numeric_gte_lte_not_resolved(self):
        """Numeric $gte/$lte should pass through unchanged (ChromaDB handles them)."""
        store = _make_store_with_dates(["2024-12-18"])
        where = {"$and": [
            {"token_count": {"$gte": 100}},
            {"token_count": {"$lte": 500}},
        ]}
        resolved = store._resolve_string_ranges(where)

        # Numeric ranges should not be touched
        assert resolved == where


class TestQueryWithDateFilter:
    """ChromaStore.query() should work end-to-end with date range where clauses."""

    def test_filtered_query_returns_matching_dates(self):
        store = _make_store_with_dates([
            "2024-01-31", "2024-06-12", "2024-12-18", "2025-01-29",
        ])
        # Use a query embedding close to chunk-2 (index 2 → 2024-12-18)
        query_emb = [0.3] * 3

        where = {"$and": [
            {"document_date": {"$gte": "2024-12-01"}},
            {"document_date": {"$lte": "2024-12-31"}},
        ]}
        results = store.query(query_embedding=query_emb, top_k=10, where=where)

        dates = [r["metadata"]["document_date"] for r in results]
        assert all("2024-12-01" <= d <= "2024-12-31" for d in dates)
        assert len(results) == 1  # Only one chunk matches December 2024

    def test_unfiltered_query_returns_all(self):
        store = _make_store_with_dates([
            "2024-01-31", "2024-06-12", "2024-12-18", "2025-01-29",
        ])
        query_emb = [0.1] * 3
        results = store.query(query_embedding=query_emb, top_k=10)

        assert len(results) == 4
