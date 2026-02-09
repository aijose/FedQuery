"""ChromaDB vector store integration for FOMC document chunks."""

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.embedding.provider import EmbeddingProvider
from src.models.chunk import DocumentChunk
from src.models.document import FOMCDocument

logger = logging.getLogger(__name__)

COLLECTION_NAME = "fomc_chunks"


class ChromaStore:
    """ChromaDB-backed vector store for FOMC document chunks.

    Manages a single collection ('fomc_chunks') with cosine distance.
    Stores embeddings plus 7 metadata fields per data-model.md.
    """

    def __init__(self, path: str = "./data/chroma"):
        if path == ":memory:":
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(
                path=path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk], document: FOMCDocument) -> int:
        """Add document chunks to the collection.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.chunk_text)
            metadatas.append({
                "document_id": document.id,
                "document_title": document.title,
                "document_date": document.date.isoformat(),
                "document_type": document.document_type.value,
                "chunk_index": chunk.chunk_index,
                "section_header": chunk.section_header or "",
                "token_count": chunk.token_count,
                "source_url": document.source_url,
            })

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def has_document(self, source_url: str) -> bool:
        """Check if a document with the given source_url already exists."""
        results = self._collection.get(
            where={"source_url": source_url},
            limit=1,
        )
        return len(results["ids"]) > 0

    def query(
        self,
        query_embedding: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Query the collection for similar chunks.

        Provide either query_embedding (for pre-embedded queries) or
        query_text (uses ChromaDB's built-in embedding -- not recommended,
        prefer using the EmbeddingProvider externally).

        Args:
            where: Optional ChromaDB where filter (e.g. date range).

        Returns a list of dicts with keys: id, text, metadata, distance.
        """
        kwargs = {"n_results": top_k}

        if query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding]
        elif query_text is not None:
            kwargs["query_texts"] = [query_text]
        else:
            raise ValueError("Must provide either query_embedding or query_text")

        if where is not None:
            kwargs["where"] = self._resolve_string_ranges(where)

        try:
            results = self._collection.query(**kwargs)
        except Exception:
            # n_results may exceed matching docs; retry without cap
            if "where" in kwargs:
                kwargs.pop("n_results", None)
                kwargs["n_results"] = self._collection.count()
                results = self._collection.query(**kwargs)
            else:
                raise

        output = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                output.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })
        return output

    def get_document_chunks(self, document_id: str) -> list[dict]:
        """Retrieve all chunks for a specific document."""
        results = self._collection.get(
            where={"document_id": document_id},
        )
        output = []
        for i in range(len(results["ids"])):
            output.append({
                "id": results["ids"][i],
                "text": results["documents"][i] if results["documents"] else "",
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })
        return output

    def _resolve_string_ranges(self, where: dict) -> dict:
        """Convert $gte/$lte on string fields to $in with exact values.

        ChromaDB only supports $gte/$lte on numeric types. For string
        metadata like document_date, we resolve the range to an $in
        filter by scanning distinct values from the collection.
        """
        if "$and" not in where:
            return where

        field_ranges: dict[str, dict] = {}
        other_clauses = []

        for clause in where["$and"]:
            is_string_range = False
            for field, condition in clause.items():
                if isinstance(condition, dict):
                    op = next(iter(condition))
                    val = condition[op]
                    if op in ("$gte", "$lte") and isinstance(val, str):
                        field_ranges.setdefault(field, {})[op] = val
                        is_string_range = True
            if not is_string_range:
                other_clauses.append(clause)

        if not field_ranges:
            return where

        all_meta = self._collection.get(include=["metadatas"])["metadatas"]

        for field, ops in field_ranges.items():
            gte = ops.get("$gte", "")
            lte = ops.get("$lte", "\uffff")
            matching = sorted({
                m.get(field, "") for m in all_meta
                if gte <= m.get(field, "") <= lte
            })
            if matching:
                other_clauses.append({field: {"$in": matching}})
            else:
                # No values in range — use impossible match to guarantee empty results
                other_clauses.append({field: {"$in": ["__no_match__"]}})

        if not other_clauses:
            return where
        if len(other_clauses) == 1:
            return other_clauses[0]
        return {"$and": other_clauses}

    @property
    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()
