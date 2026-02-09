# FedQuery HNSW vs IVF Benchmark Analysis

**Date**: 2026-02-09
**Corpus**: 1,545 chunks from 81 FOMC documents (2021-2025)
**Embedding model**: BAAI/bge-small-en-v1.5 (384 dimensions)
**Query count**: 200 random vectors per benchmark run
**Index implementations**: FAISS HNSW and FAISS IVF (parameter sweeps for both)
**Ground truth**: Brute-force flat L2 index (exact nearest neighbors)
**Run command**: `fedquery benchmark --sweep -v`

---

## 1. What This Benchmark Measures

This benchmark measures **index-level approximation fidelity**: does the approximate index (HNSW or IVF) return the same top-k neighbors as an exact brute-force search?

It uses **random query vectors**, not real user questions. The recall numbers tell you how well the ANN index approximates exact search — not whether the system retrieves the right FOMC documents for a question. Retrieval quality is measured separately by `fedquery evaluate` using the golden QA dataset.

---

## 2. Brute-Force Baseline (1,545 vectors)

| Metric | Value |
|--------|-------|
| Recall@5 | 1.0000 (by definition) |
| Avg Latency | 0.034ms |
| P99 Latency | 0.054ms |
| Index Size | 2,318 KB |

At 1,545 vectors, a full linear scan completes in 0.034ms. This sets the bar: any ANN index must be faster than this to justify its complexity.

---

## 3. HNSW Parameter Sweep

HNSW builds a multi-layer navigable graph. `M` controls edges per node (graph density). `efSearch` controls the beam width at query time (more nodes explored = better recall = higher latency).

| Config | Recall@5 | Recall@10 | Avg ms | P50 ms | P99 ms | Size KB |
|--------|----------|-----------|--------|--------|--------|---------|
| M=16 ef=16 | 0.4950 | 0.4830 | 0.007 | 0.007 | 0.011 | 2,535 |
| M=16 ef=32 | 0.6160 | 0.6100 | 0.012 | 0.012 | 0.019 | 2,535 |
| M=16 ef=64 | 0.7670 | 0.7745 | 0.021 | 0.021 | 0.029 | 2,535 |
| M=16 ef=128 | 0.8900 | 0.8985 | 0.042 | 0.042 | 0.055 | 2,535 |
| M=32 ef=16 | 0.6380 | 0.6290 | 0.011 | 0.011 | 0.017 | 2,727 |
| M=32 ef=32 | 0.7970 | 0.7905 | 0.018 | 0.018 | 0.027 | 2,727 |
| M=32 ef=64 | 0.9120 | 0.9090 | 0.032 | 0.031 | 0.046 | 2,727 |
| M=32 ef=128 | 0.9840 | 0.9790 | 0.056 | 0.056 | 0.069 | 2,727 |
| M=48 ef=16 | 0.7100 | 0.7030 | 0.013 | 0.013 | 0.020 | 2,920 |
| M=48 ef=32 | 0.8310 | 0.8260 | 0.020 | 0.020 | 0.031 | 2,920 |
| M=48 ef=64 | 0.9450 | 0.9485 | 0.035 | 0.035 | 0.050 | 2,920 |
| M=48 ef=128 | 0.9880 | 0.9875 | 0.059 | 0.058 | 0.088 | 2,920 |

**Observations:**
- Higher M = denser graph = better recall + more memory. M=16 to M=48 adds 15% to index size (2,535 to 2,920 KB).
- efSearch is the latency/recall knob. Doubling ef roughly doubles latency and adds ~10-15% recall.
- Best recall: M=48 ef=128 at 98.8%, but costs 0.059ms — 1.7x slower than brute-force (0.034ms).
- ChromaDB defaults (M=16, efSearch=100) would land around 85-89% recall on this benchmark.

---

## 4. IVF Parameter Sweep

IVF partitions vectors into `nlist` Voronoi cells via k-means. At query time, `nprobe` cells are searched. More cells = finer partitions. More nprobe = higher recall = closer to brute-force cost.

| Config | Recall@5 | Recall@10 | Avg ms | P50 ms | P99 ms | Size KB |
|--------|----------|-----------|--------|--------|--------|---------|
| nlist=10 nprobe=1 | 0.4680 | 0.4465 | 0.005 | 0.005 | 0.011 | 2,345 |
| nlist=10 nprobe=3 | 0.8090 | 0.7925 | 0.011 | 0.010 | 0.018 | 2,345 |
| nlist=10 nprobe=5 | 0.9170 | 0.9085 | 0.018 | 0.018 | 0.024 | 2,345 |
| nlist=10 nprobe=10 | 0.9990 | 1.0000 | 0.033 | 0.034 | 0.037 | 2,345 |
| nlist=25 nprobe=1 | 0.3980 | 0.3805 | 0.003 | 0.003 | 0.005 | 2,367 |
| nlist=25 nprobe=3 | 0.6670 | 0.6525 | 0.007 | 0.006 | 0.011 | 2,367 |
| nlist=25 nprobe=5 | 0.7870 | 0.7795 | 0.009 | 0.009 | 0.013 | 2,367 |
| nlist=25 nprobe=10 | 0.9240 | 0.9220 | 0.015 | 0.015 | 0.019 | 2,367 |
| nlist=39 nprobe=1 | 0.3230 | 0.3090 | 0.003 | 0.003 | 0.005 | 2,388 |
| nlist=39 nprobe=3 | 0.5820 | 0.5595 | 0.005 | 0.005 | 0.009 | 2,388 |
| nlist=39 nprobe=5 | 0.7150 | 0.6930 | 0.007 | 0.007 | 0.010 | 2,388 |
| nlist=39 nprobe=10 | 0.8810 | 0.8645 | 0.011 | 0.011 | 0.017 | 2,388 |

**Observations:**
- nlist=10, nprobe=10 searches all 10 cells — effectively brute-force. 99.9% recall at 0.033ms (vs 0.034ms brute-force). The 0.1% recall loss is from cell boundary effects.
- At low nprobe (1), recall drops to 32-47%. The query lands in one cell and misses neighbors in adjacent cells.
- IVF memory is slightly lower than HNSW (2,345 vs 2,727 KB) — inverted lists are just assignments, not graph structures.
- More cells (higher nlist) = finer partitions = lower recall at the same nprobe, because relevant vectors are spread across more cells.

---

## 5. Recall-Matched Fair Comparison

For each recall target, the cheapest (lowest latency) config from each index that meets it:

| Target | Best HNSW (recall, latency) | Best IVF (recall, latency) | Verdict |
|--------|----------------------------|---------------------------|---------|
| >=60% | M=32 ef=16 (63.8%, 0.011ms) | nl=25 np=3 (66.7%, 0.007ms) | IVF faster |
| >=80% | M=48 ef=32 (83.1%, 0.020ms) | nl=10 np=3 (80.9%, 0.011ms) | IVF faster |
| >=90% | M=32 ef=64 (91.2%, 0.032ms) | nl=25 np=10 (92.4%, 0.015ms) | IVF 2x faster |
| >=93% | M=48 ef=64 (94.5%, 0.035ms) | nl=10 np=10 (99.9%, 0.033ms) | Comparable |
| >=98% | M=32 ef=128 (98.4%, 0.056ms) | nl=10 np=10 (99.9%, 0.033ms) | IVF faster |

IVF is faster at every recall tier. However, its highest-recall config (nlist=10, nprobe=10) is brute-force in disguise — it searches all cells at 0.033ms, which is the same as the 0.034ms brute-force baseline.

---

## 6. Key Finding

**At 1,545 vectors, neither ANN index is faster than brute-force for high recall.**

- IVF at 99.9% recall: 0.033ms (= brute-force with cell overhead)
- HNSW at 98.4% recall: 0.056ms (1.7x slower than brute-force due to graph traversal)
- Brute-force: 0.034ms with perfect recall

ANN indexes add overhead (graph traversal for HNSW, cell assignment for IVF) that exceeds the benefit of skipping vectors at this corpus size. They only provide meaningful speedups at >10K-100K vectors where linear scan becomes expensive.

---

## 7. Why HNSW Remains the Right Default

The choice is operational, not performance-driven:

1. **ChromaDB only supports HNSW.** Using IVF would require a parallel FAISS index with storage duplication and sync complexity. ChromaDB's built-in HNSW is zero-configuration.
2. **HNSW scales without retuning.** IVF requires periodic k-means retraining when `nlist` needs adjustment for a growing corpus. HNSW's graph adapts automatically.
3. **The ANN step is invisible in practice.** 0.034ms vector search vs 2-5 second LLM API call. The index algorithm contributes <0.01% of end-to-end latency.
4. **Projected growth is modest.** ~16 documents/year at ~19 chunks each = ~300 chunks/year. Even 20 years would yield ~6,000 chunks — well within brute-force territory.

---

## 8. ChromaDB HNSW Configuration

Current ChromaDB settings in `src/vectorstore/chroma_store.py`:

```python
metadata={"hnsw:space": "cosine"}  # All other params use ChromaDB defaults
```

ChromaDB defaults: M=16, efSearch=100, efConstruction=100. Based on the sweep, M=16/ef=100 yields ~85-89% recall on random vectors. At current corpus size this is adequate — all configs achieve equivalent results on the real FOMC document space.

---

## 9. Relation to Retrieval Quality

This benchmark measures **index approximation fidelity** using random vectors. It answers: "does HNSW/IVF return the same neighbors as exact search?"

**Retrieval quality** — "does the system find the right FOMC documents for a user's question?" — is measured by `fedquery evaluate` using the golden QA dataset (24 questions). Current metrics after the bge-small-en-v1.5 switch: MRR=0.283, up from 0.134 with all-MiniLM-L6-v2.

At the current corpus size, the index is not the bottleneck. Retrieval quality improvements come from the embedding model, chunking strategy, cross-encoder reranking, and two-pass date-filtered retrieval (see `optimization_efforts.md`).
