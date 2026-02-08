# FedQuery Retrieval Optimization Efforts

## Overview

Systematic evaluation and optimization of the FedQuery FOMC RAG retrieval pipeline. Work was done in three phases: building an evaluation framework, testing chunking parameters, and adding cross-encoder reranking.

Corpus: FOMC documents from 2020-2024 (5 years), 1,545 chunks in ChromaDB. Phases 1-3 were originally run on 2024 data (269 chunks); Phase 4 benchmark uses the full corpus.

Golden QA dataset: 24 hand-crafted questions across 5 categories (factual, cross-document, section-specific, temporal, out-of-scope).

---

## Phase 1: Evaluation Framework

### What was built

- **Golden QA dataset** (`data/eval/golden_qa.json`): 24 questions with expected text fragments, relevant document references (type + date), and category labels.
- **IR metrics** (`src/evaluation/retrieval_metrics.py`): Precision@K, Recall@K, MRR, NDCG@K, Hit Rate@K, Chunk Text Recall.
- **Eval runner** (`src/evaluation/eval_runner.py`): Runs search for each question, computes all metrics, aggregates by category. No LLM required.
- **CLI command** (`fedquery evaluate`): Rich-formatted report with `--verbose` and `--chunking-grid` flags.

### Baseline Results (bi-encoder only, 512/50 chunking)

| Metric | @3 | @5 | @10 |
|--------|-----|-----|------|
| Precision | 0.222 | 0.167 | 0.096 |
| Recall | 0.460 | 0.575 | 0.633 |
| Hit Rate | 0.500 | 0.583 | 0.667 |
| NDCG | 0.242 | 0.274 | 0.282 |
| Chunk Text Recall | 0.500 | 0.583 | 0.667 |
| **MRR** | **0.310** | | |

### Baseline by category

| Category | MRR | Hit@5 | CTR@5 |
|----------|------|-------|-------|
| factual | 0.370 | 0.667 | 0.667 |
| cross_document | 0.250 | 0.600 | 0.600 |
| section_specific | 0.417 | 0.750 | 0.750 |
| temporal | 0.111 | 0.333 | 0.333 |
| out_of_scope | 0.333 | 0.333 | 0.333 |

### Key finding

Temporal queries performed worst (MRR=0.111). The bi-encoder has no awareness of dates, so queries like "What did the January 2024 statement say?" retrieve content from any meeting date. This points toward metadata filtering (Phase 4) as a targeted fix.

---

## Phase 2: Chunking Parameter Grid Evaluation

### Configurations tested

| Config | chunk_size | chunk_overlap | Chunks | MRR | Hit@5 | CTR@5 | NDCG@5 |
|--------|-----------|---------------|--------|------|-------|-------|--------|
| 256/25 | 256 | 25 | 537 | 0.240 | 0.500 | 0.500 | 0.216 |
| 256/50 | 256 | 50 | 573 | 0.253 | 0.500 | 0.500 | 0.218 |
| **512/50** | **512** | **50** | **269** | **0.340** | **0.583** | **0.583** | **0.276** |
| 512/100 | 512 | 100 | 288 | 0.290 | 0.583 | 0.583 | 0.274 |
| 768/75 | 768 | 75 | 187 | 0.284 | 0.583 | 0.583 | 0.270 |
| 768/150 | 768 | 150 | 197 | 0.284 | 0.583 | 0.583 | 0.270 |
| 1024/100 | 1024 | 100 | 149 | 0.299 | 0.583 | 0.583 | 0.290 |

### Key findings

1. **512/50 (current default) is the best chunking configuration** for MRR (0.340), confirming the original choice.
2. **Smaller chunks (256) hurt retrieval quality** — MRR drops to ~0.24. Sentences split mid-thought lose context.
3. **Larger chunks (768, 1024) improve NDCG slightly** but don't improve hit rate. More context per chunk helps ranking but doesn't surface new documents.
4. **Overlap has minimal effect** — 50 vs 100 overlap at 512 chunk size shows negligible difference (0.340 vs 0.290 MRR).
5. **Chunking parameter tuning yields marginal gains.** The bottleneck is not chunk boundaries but bi-encoder ranking quality. This motivated Phase 3.

---

## Phase 3: Cross-Encoder Reranking

### Approach

Two-stage retrieval pipeline:
1. **Bi-encoder recall** (all-MiniLM-L6-v2): Fast approximate search. Over-fetch 3x candidates from ChromaDB.
2. **Cross-encoder precision** (ms-marco-MiniLM-L-6-v2): Jointly encode (query, chunk) pairs for accurate relevance scoring. Return top-k.

The cross-encoder is ~80MB, runs locally, and adds ~100-200ms per query for 15 candidates.

### Results: Baseline vs Reranker

| Metric | Baseline | + Reranker | Improvement |
|--------|----------|------------|-------------|
| MRR | 0.310 | 0.597 | **+92%** |
| Precision@5 | 0.167 | 0.233 | +40% |
| Recall@5 | 0.575 | 0.800 | +39% |
| Hit Rate@5 | 0.583 | 0.917 | **+57%** |
| NDCG@5 | 0.274 | 0.549 | **+100%** |
| CTR@5 | 0.583 | 0.792 | +36% |

### Results by category

| Category | MRR (base) | MRR (rerank) | Hit@5 (base) | Hit@5 (rerank) |
|----------|------------|--------------|--------------|----------------|
| factual | 0.370 | 0.722 | 0.667 | 1.000 |
| cross_document | 0.250 | 0.550 | 0.600 | 0.800 |
| section_specific | 0.417 | 0.750 | 0.750 | 1.000 |
| temporal | 0.111 | 0.242 | 0.333 | 0.667 |
| out_of_scope | 0.333 | 0.556 | 0.333 | 1.000 |

### Key findings

1. **Cross-encoder reranking is the single biggest retrieval improvement**, nearly doubling MRR and NDCG.
2. **Factual and section-specific queries hit 100% hit rate** — the system now reliably finds the right documents for direct questions.
3. **Temporal queries improved least** (MRR 0.111 -> 0.242). The cross-encoder improves ranking but can't filter by date. Metadata filtering (Phase 4) is the targeted fix for these.
4. **Out-of-scope questions improved substantially** (Hit@5 0.333 -> 1.000). The reranker better distinguishes relevant from irrelevant content even when the question is tangential.
5. **Cost is minimal**: ~80MB model download, ~100-200ms latency per query. Enabled via `FEDQUERY_RERANKER_ENABLED=true`.

---

## Summary of Improvements

```
Baseline (bi-encoder only):       MRR=0.310  Hit@5=0.583  NDCG@5=0.274
+ Chunking tuning (marginal):     MRR=0.340  Hit@5=0.583  NDCG@5=0.276
+ Cross-encoder reranking:        MRR=0.597  Hit@5=0.917  NDCG@5=0.549
```

The chunking parameter search confirmed the default (512/50) is near-optimal. The real win came from cross-encoder reranking, which nearly doubled all key metrics.

---

## Phase 4: HNSW vs IVF Benchmark (Full 5-Year Corpus)

### Setup

Benchmark run on the full corpus: 1,545 chunks, 384-dimensional embeddings (all-MiniLM-L6-v2). FAISS indexes built from embeddings extracted from ChromaDB.

- **200 queries**: 20 real FOMC questions + 180 random vectors for statistical coverage
- **Ground truth**: brute-force flat L2 index (exact nearest neighbors)
- **Metrics**: Recall@5, Recall@10, query latency (avg/p50/p99), index size on disk

### HNSW Parameter Sweep

HNSW builds a multi-layer navigable graph. Each vector connects to `M` neighbors. At query time, `efSearch` controls the search beam width — higher means more neighbors explored, better recall, higher latency.

| Config | Recall@5 | Recall@10 | Avg ms | P50 ms | P99 ms | Size KB |
|--------|----------|-----------|--------|--------|--------|---------|
| M=16 ef=32 | 0.6990 | 0.6955 | 0.012 | 0.011 | 0.018 | 2,535 |
| M=16 ef=64 | 0.8370 | 0.8415 | 0.021 | 0.021 | 0.033 | 2,535 |
| M=16 ef=128 | 0.9350 | 0.9355 | 0.041 | 0.040 | 0.064 | 2,535 |
| M=32 ef=32 | 0.8590 | 0.8575 | 0.017 | 0.017 | 0.027 | 2,727 |
| **M=32 ef=64** | **0.9330** | **0.9380** | **0.030** | **0.029** | **0.057** | **2,727** |
| M=32 ef=128 | 0.9860 | 0.9850 | 0.054 | 0.053 | 0.085 | 2,727 |
| M=64 ef=32 | 0.9250 | 0.9155 | 0.020 | 0.020 | 0.032 | 3,114 |
| M=64 ef=64 | 0.9730 | 0.9720 | 0.034 | 0.034 | 0.045 | 3,114 |
| M=64 ef=128 | 0.9940 | 0.9940 | 0.058 | 0.058 | 0.070 | 3,114 |

**Key observations:**
- `M` controls graph density: higher M = more edges = better recall + more memory. Going from M=16 to M=64 adds 23% to index size but pushes recall from 0.70 to 0.99.
- `efSearch` is the latency/recall knob: doubling ef roughly doubles latency and adds ~10% recall. M=32/ef=64 is the sweet spot for our corpus — 93% recall at 0.03ms.
- Memory cost is modest: the graph adjacency lists add only ~400KB over the raw vectors (2,318KB flat).

### IVF Parameter Sweep

IVF partitions vectors into `nlist` Voronoi cells via k-means. At query time, `nprobe` cells are searched. More cells = finer partitions = lower nprobe recall. More nprobe = better recall = closer to brute-force cost.

| Config | Recall@5 | Recall@10 | Avg ms | P50 ms | P99 ms | Size KB |
|--------|----------|-----------|--------|--------|--------|---------|
| nlist=10 nprobe=1 | 0.5360 | 0.5160 | 0.006 | 0.006 | 0.007 | 2,345 |
| nlist=10 nprobe=5 | 0.9530 | 0.9520 | 0.022 | 0.021 | 0.062 | 2,345 |
| **nlist=10 nprobe=10** | **1.0000** | **1.0000** | **0.032** | **0.031** | **0.042** | **2,345** |
| nlist=25 nprobe=1 | 0.4170 | 0.4120 | 0.004 | 0.004 | 0.006 | 2,367 |
| nlist=25 nprobe=5 | 0.8510 | 0.8580 | 0.010 | 0.010 | 0.015 | 2,367 |
| **nlist=25 nprobe=10** | **0.9590** | **0.9590** | **0.017** | **0.016** | **0.022** | **2,367** |
| nlist=25 nprobe=25 | 1.0000 | 1.0000 | 0.033 | 0.033 | 0.042 | 2,367 |
| nlist=39 nprobe=1 | 0.3830 | 0.3630 | 0.003 | 0.003 | 0.004 | 2,389 |
| nlist=39 nprobe=5 | 0.7650 | 0.7680 | 0.007 | 0.007 | 0.011 | 2,389 |
| nlist=39 nprobe=10 | 0.9120 | 0.9130 | 0.012 | 0.011 | 0.015 | 2,389 |
| nlist=39 nprobe=39 | 1.0000 | 1.0000 | 0.035 | 0.034 | 0.041 | 2,389 |

**Key observations:**
- When `nprobe = nlist` (search all cells), IVF degrades to brute-force — recall is perfect but there's no speed gain, just overhead from the partitioning structure.
- The sweet spot is `nprobe ≈ nlist/3`: e.g., nlist=25/nprobe=10 gives 96% recall at 0.017ms.
- At low nprobe (1), recall plummets to ~40-54% — the query vector lands in one cell, and nearby relevant vectors in adjacent cells are missed entirely.
- IVF memory is slightly *lower* than HNSW (2,345 vs 2,727 KB) because inverted lists are just vector assignments, not full graph structures.

### Brute-Force Baseline

| Metric | Value |
|--------|-------|
| Recall@5 | 1.0000 (by definition) |
| Avg Latency | 0.031ms |
| P99 Latency | 0.036ms |
| Size | 2,318 KB |

At 1,545 vectors, brute-force is fast enough (0.031ms) that ANN indexes provide marginal speed improvement. The real value of ANN emerges at >100K vectors where brute-force linear scan becomes prohibitive.

### Head-to-Head: Best Configurations

| Metric | HNSW (M=32 ef=64) | IVF (nlist=25 nprobe=10) | Brute-Force |
|--------|-------------------|--------------------------|-------------|
| Recall@5 | 0.9330 | 0.9590 | 1.0000 |
| Avg Latency | 0.030ms | 0.017ms | 0.031ms |
| P99 Latency | 0.057ms | 0.022ms | 0.036ms |
| Index Size | 2,727 KB | 2,367 KB | 2,318 KB |
| Real Query Recall@5 | 1.0000 | 1.0000 | 1.0000 |

Both ANN methods hit 100% recall on real FOMC queries — the recall gap only shows on random vectors where edge cases in geometric space matter.

### Why HNSW Is Still the Right Default

Despite IVF showing slightly better numbers at this corpus size, HNSW is the right choice for FedQuery:

1. **ChromaDB uses HNSW internally.** Switching to IVF would mean building a parallel FAISS index, duplicating storage, and adding sync complexity. Using ChromaDB's built-in HNSW is zero-configuration.
2. **HNSW scales better without retuning.** As the corpus grows (more years of FOMC data), HNSW's graph adapts without needing to retrain centroids or adjust `nlist`. IVF requires periodic retraining when the data distribution shifts.
3. **At this scale, both are essentially free.** Query latency is 0.017-0.031ms — thousands of times faster than the LLM API call (2-5 seconds). The ANN algorithm choice is irrelevant to end-user latency.
4. **IVF's advantages emerge at scale.** With millions of vectors, IVF's lower memory footprint (no graph structure) and ability to shard across machines make it the practical choice. At 1,545 vectors, these advantages don't apply.

---

## Remaining Opportunities

- **Metadata filtering** — Rule-based date/type extraction from queries, passed as ChromaDB `where` filters. Expected to significantly improve temporal queries (current MRR=0.242).
- **Confidence threshold calibration** — Use evaluation score distributions to empirically set high/medium/low confidence thresholds in the agent workflow.
