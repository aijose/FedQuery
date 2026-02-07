"""FAISS index builders for HNSW vs IVF benchmark."""

import numpy as np
import faiss


def build_hnsw_index(vectors: np.ndarray, dim: int, M: int = 32, ef_search: int = 64) -> faiss.Index:
    """Build an HNSW flat index (what ChromaDB uses internally)."""
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efSearch = ef_search
    index.add(vectors)
    return index


def build_ivf_index(vectors: np.ndarray, dim: int, nlist: int = 100, nprobe: int = 10) -> faiss.Index:
    """Build an IVF flat index."""
    # Ensure nlist is reasonable relative to vector count
    # FAISS requires at least 39 * nlist training vectors
    max_nlist = max(1, len(vectors) // 39)
    actual_nlist = min(nlist, max_nlist)
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist)
    index.nprobe = nprobe
    index.train(vectors)
    index.add(vectors)
    return index


def build_flat_index(vectors: np.ndarray, dim: int) -> faiss.Index:
    """Build a brute-force flat L2 index (baseline)."""
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index
