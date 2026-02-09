"""Benchmark runner for HNSW vs IVF index comparison."""

import os
import time
import tracemalloc
import logging

# Prevent FAISS OMP crash on macOS ARM when running alongside ChromaDB
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import faiss

from src.models.benchmark import BenchmarkResult
from src.models.enums import IndexType
from src.vectorstore.faiss_benchmark import build_flat_index, build_hnsw_index, build_ivf_index

logger = logging.getLogger(__name__)


def compute_recall_at_k(
    ground_truth: list[list[int]],
    predictions: list[list[int]],
    k: int,
) -> float:
    """Compute recall@k: fraction of true top-k found in predicted top-k."""
    recalls = []
    for gt, pred in zip(ground_truth, predictions):
        gt_set = set(gt[:k])
        pred_set = set(pred[:k])
        if len(gt_set) == 0:
            recalls.append(1.0)
        else:
            recalls.append(len(gt_set & pred_set) / len(gt_set))
    return np.mean(recalls)


def measure_latency(
    index: faiss.Index, queries: np.ndarray, k: int = 5,
) -> tuple[float, float, float]:
    """Measure query latency in milliseconds over all queries.

    Returns (avg_ms, p50_ms, p99_ms).
    """
    times = []
    for q in queries:
        q = q.reshape(1, -1)
        start = time.perf_counter()
        index.search(q, k)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    arr = np.array(times)
    return float(np.mean(arr)), float(np.median(arr)), float(np.percentile(arr, 99))


def measure_memory(build_fn) -> float:
    """Measure peak memory usage in MB during index construction."""
    tracemalloc.start()
    build_fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def _measure_index_size(build_fn, dim: int) -> float:
    """Measure serialized index size in KB."""
    index = build_fn()
    writer = faiss.VectorIOWriter()
    faiss.write_index(index, writer)
    return len(faiss.vector_to_array(writer.data)) / 1024


def run_benchmark(
    vectors: np.ndarray,
    n_queries: int = 100,
    k: int = 5,
) -> list[BenchmarkResult]:
    """Run the full HNSW vs IVF benchmark with default parameters.

    Args:
        vectors: Corpus embeddings as numpy array (N x dim).
        n_queries: Number of queries for latency measurement.
        k: Top-k for recall and search.

    Returns:
        List of BenchmarkResult objects (one per index type).
    """
    dim = vectors.shape[1]
    n_vectors = vectors.shape[0]

    queries = np.random.rand(n_queries, dim).astype(np.float32)

    logger.info("Building brute-force baseline...")
    flat_index = build_flat_index(vectors, dim)
    _, gt_indices_5 = flat_index.search(queries, k)
    _, gt_indices_10 = flat_index.search(queries, 10)
    gt5 = gt_indices_5.tolist()
    gt10 = gt_indices_10.tolist()

    results = []

    # HNSW benchmark
    logger.info("Benchmarking HNSW (M=32, ef=64)...")
    hnsw_index = build_hnsw_index(vectors, dim)
    _, hnsw_idx5 = hnsw_index.search(queries, k)
    _, hnsw_idx10 = hnsw_index.search(queries, 10)
    hnsw_recall5 = compute_recall_at_k(gt5, hnsw_idx5.tolist(), k)
    hnsw_recall10 = compute_recall_at_k(gt10, hnsw_idx10.tolist(), 10)
    avg, p50, p99 = measure_latency(hnsw_index, queries, k)
    hnsw_memory = measure_memory(lambda: build_hnsw_index(vectors, dim))

    results.append(BenchmarkResult(
        index_type=IndexType.HNSW,
        query_latency_ms=round(avg, 3),
        recall_at_k=round(hnsw_recall5, 4),
        recall_at_10=round(hnsw_recall10, 4),
        memory_usage_mb=round(hnsw_memory, 2),
        p50_latency_ms=round(p50, 3),
        p99_latency_ms=round(p99, 3),
        corpus_size=n_vectors,
        params={"M": 32, "ef_search": 64},
    ))

    # IVF benchmark
    logger.info("Benchmarking IVF (nlist=auto, nprobe=10)...")
    try:
        ivf_index = build_ivf_index(vectors, dim)
        _, ivf_idx5 = ivf_index.search(queries, k)
        _, ivf_idx10 = ivf_index.search(queries, 10)
        ivf_recall5 = compute_recall_at_k(gt5, ivf_idx5.tolist(), k)
        ivf_recall10 = compute_recall_at_k(gt10, ivf_idx10.tolist(), 10)
        avg, p50, p99 = measure_latency(ivf_index, queries, k)
        ivf_memory = measure_memory(lambda: build_ivf_index(vectors, dim))

        results.append(BenchmarkResult(
            index_type=IndexType.IVF,
            query_latency_ms=round(avg, 3),
            recall_at_k=round(ivf_recall5, 4),
            recall_at_10=round(ivf_recall10, 4),
            memory_usage_mb=round(ivf_memory, 2),
            p50_latency_ms=round(p50, 3),
            p99_latency_ms=round(p99, 3),
            corpus_size=n_vectors,
            params={"nlist": "auto", "nprobe": 10},
        ))
    except Exception as e:
        logger.warning("IVF benchmark failed (known issue on macOS ARM): %s", e)
        results.append(BenchmarkResult(
            index_type=IndexType.IVF,
            query_latency_ms=0.0,
            recall_at_k=0.0,
            memory_usage_mb=0.0,
            corpus_size=n_vectors,
        ))

    return results


def run_parameter_sweep(
    vectors: np.ndarray,
    n_queries: int = 200,
    k: int = 5,
) -> list[BenchmarkResult]:
    """Run parameter sweeps for both HNSW and IVF, plus brute-force baseline.

    HNSW sweep: M in {16, 32, 48}, efSearch in {16, 32, 64, 128}
    IVF sweep: nlist in {10, 25, 39}, nprobe in {1, 3, 5, 10}

    Returns list of BenchmarkResult with params metadata.
    """
    dim = vectors.shape[1]
    n_vectors = vectors.shape[0]

    queries = np.random.rand(n_queries, dim).astype(np.float32)

    # Brute-force ground truth
    logger.info("Building brute-force baseline for %d vectors...", n_vectors)
    flat_index = build_flat_index(vectors, dim)
    _, gt5 = flat_index.search(queries, k)
    _, gt10 = flat_index.search(queries, 10)
    gt5_list = gt5.tolist()
    gt10_list = gt10.tolist()

    # Measure brute-force latency
    bf_avg, bf_p50, bf_p99 = measure_latency(flat_index, queries, k)
    bf_size = _measure_index_size(lambda: build_flat_index(vectors, dim), dim)

    results = []

    # Brute-force baseline result
    results.append(BenchmarkResult(
        index_type=IndexType.HNSW,  # Use HNSW as placeholder for brute-force
        query_latency_ms=round(bf_avg, 3),
        recall_at_k=1.0,
        recall_at_10=1.0,
        memory_usage_mb=0.0,
        p50_latency_ms=round(bf_p50, 3),
        p99_latency_ms=round(bf_p99, 3),
        index_size_kb=round(bf_size, 1),
        corpus_size=n_vectors,
        params={"type": "brute_force"},
    ))

    # HNSW parameter sweep
    hnsw_m_values = [16, 32, 48]
    hnsw_ef_values = [16, 32, 64, 128]

    for M in hnsw_m_values:
        for ef in hnsw_ef_values:
            logger.info("HNSW: M=%d ef=%d", M, ef)
            try:
                idx = build_hnsw_index(vectors, dim, M=M, ef_search=ef)
                _, pred5 = idx.search(queries, k)
                _, pred10 = idx.search(queries, 10)
                r5 = compute_recall_at_k(gt5_list, pred5.tolist(), k)
                r10 = compute_recall_at_k(gt10_list, pred10.tolist(), 10)
                avg, p50, p99 = measure_latency(idx, queries, k)
                size_kb = _measure_index_size(
                    lambda _M=M, _ef=ef: build_hnsw_index(vectors, dim, M=_M, ef_search=_ef),
                    dim,
                )
                results.append(BenchmarkResult(
                    index_type=IndexType.HNSW,
                    query_latency_ms=round(avg, 3),
                    recall_at_k=round(r5, 4),
                    recall_at_10=round(r10, 4),
                    memory_usage_mb=0.0,
                    p50_latency_ms=round(p50, 3),
                    p99_latency_ms=round(p99, 3),
                    index_size_kb=round(size_kb, 1),
                    corpus_size=n_vectors,
                    params={"M": M, "ef_search": ef},
                ))
            except Exception as e:
                logger.warning("HNSW M=%d ef=%d failed: %s", M, ef, e)

    # IVF parameter sweep
    max_nlist = max(1, n_vectors // 39)
    ivf_nlist_values = [n for n in [10, 25, 39] if n <= max_nlist]
    ivf_nprobe_values = [1, 3, 5, 10]

    for nlist in ivf_nlist_values:
        for nprobe in ivf_nprobe_values:
            if nprobe > nlist:
                continue
            logger.info("IVF: nlist=%d nprobe=%d", nlist, nprobe)
            try:
                idx = build_ivf_index(vectors, dim, nlist=nlist, nprobe=nprobe)
                _, pred5 = idx.search(queries, k)
                _, pred10 = idx.search(queries, 10)
                r5 = compute_recall_at_k(gt5_list, pred5.tolist(), k)
                r10 = compute_recall_at_k(gt10_list, pred10.tolist(), 10)
                avg, p50, p99 = measure_latency(idx, queries, k)
                size_kb = _measure_index_size(
                    lambda _nl=nlist, _np=nprobe: build_ivf_index(vectors, dim, nlist=_nl, nprobe=_np),
                    dim,
                )
                results.append(BenchmarkResult(
                    index_type=IndexType.IVF,
                    query_latency_ms=round(avg, 3),
                    recall_at_k=round(r5, 4),
                    recall_at_10=round(r10, 4),
                    memory_usage_mb=0.0,
                    p50_latency_ms=round(p50, 3),
                    p99_latency_ms=round(p99, 3),
                    index_size_kb=round(size_kb, 1),
                    corpus_size=n_vectors,
                    params={"nlist": nlist, "nprobe": nprobe},
                ))
            except Exception as e:
                logger.warning("IVF nlist=%d nprobe=%d failed: %s", nlist, nprobe, e)

    return results


def _find_cheapest_at_recall(
    results: list[BenchmarkResult], target: float,
) -> tuple[BenchmarkResult | None, BenchmarkResult | None]:
    """Find cheapest HNSW and IVF configs that meet a recall target exactly.

    Returns (best_hnsw, best_ivf) — either may be None if no config meets target.
    """
    best_hnsw = None
    best_ivf = None
    for r in results:
        if r.params.get("type") == "brute_force":
            continue
        if r.recall_at_k < target:
            continue
        if r.index_type == IndexType.HNSW:
            if best_hnsw is None or r.query_latency_ms < best_hnsw.query_latency_ms:
                best_hnsw = r
        elif r.index_type == IndexType.IVF:
            if best_ivf is None or r.query_latency_ms < best_ivf.query_latency_ms:
                best_ivf = r
    return best_hnsw, best_ivf


def format_benchmark_report(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a comparison table."""
    lines = [
        "HNSW vs IVF Benchmark Report",
        "=" * 60,
        "",
        f"{'Metric':<25} {'HNSW':>15} {'IVF':>15}",
        "-" * 60,
    ]

    hnsw = next((r for r in results if r.index_type == IndexType.HNSW), None)
    ivf = next((r for r in results if r.index_type == IndexType.IVF), None)

    if hnsw and ivf:
        lines.append(f"{'Query Latency (ms)':<25} {hnsw.query_latency_ms:>15.3f} {ivf.query_latency_ms:>15.3f}")
        lines.append(f"{'Recall@5':<25} {hnsw.recall_at_k:>15.4f} {ivf.recall_at_k:>15.4f}")
        lines.append(f"{'Recall@10':<25} {hnsw.recall_at_10:>15.4f} {ivf.recall_at_10:>15.4f}")
        lines.append(f"{'P50 Latency (ms)':<25} {hnsw.p50_latency_ms:>15.3f} {ivf.p50_latency_ms:>15.3f}")
        lines.append(f"{'P99 Latency (ms)':<25} {hnsw.p99_latency_ms:>15.3f} {ivf.p99_latency_ms:>15.3f}")
        lines.append(f"{'Memory (MB)':<25} {hnsw.memory_usage_mb:>15.2f} {ivf.memory_usage_mb:>15.2f}")
        lines.append(f"{'Corpus Size':<25} {hnsw.corpus_size:>15} {ivf.corpus_size:>15}")
        h_params = " ".join(f"{k}={v}" for k, v in hnsw.params.items())
        i_params = " ".join(f"{k}={v}" for k, v in ivf.params.items())
        lines.append(f"{'Params':<25} {h_params:>15} {i_params:>15}")
        lines.append("-" * 60)

    return "\n".join(lines)


def format_sweep_report(results: list[BenchmarkResult]) -> str:
    """Format parameter sweep results with fair recall-matched comparison."""
    lines = [
        "HNSW vs IVF Parameter Sweep Report",
        "=" * 90,
    ]

    # Extract brute-force baseline
    bf = next((r for r in results if r.params.get("type") == "brute_force"), None)
    corpus_size = results[0].corpus_size if results else 0

    if bf:
        lines.append("")
        lines.append(f"Corpus: {corpus_size} vectors, {results[0].recall_at_k if results else 0} dims")
        lines.append(f"Brute-force baseline: Avg={bf.query_latency_ms:.3f}ms  "
                      f"P50={bf.p50_latency_ms:.3f}ms  P99={bf.p99_latency_ms:.3f}ms  "
                      f"Size={bf.index_size_kb:.0f}KB")

    # HNSW sweep table
    hnsw_results = [r for r in results if r.index_type == IndexType.HNSW and r.params.get("type") != "brute_force"]
    if hnsw_results:
        lines.append("")
        lines.append("HNSW Parameter Sweep")
        lines.append("-" * 90)
        lines.append(f"{'Config':<16} {'Recall@5':>10} {'Recall@10':>10} {'Avg ms':>8} {'P50 ms':>8} {'P99 ms':>8} {'Size KB':>10}")
        lines.append("-" * 90)
        for r in sorted(hnsw_results, key=lambda x: (x.params.get("M", 0), x.params.get("ef_search", 0))):
            label = f"M={r.params['M']} ef={r.params['ef_search']}"
            lines.append(f"{label:<16} {r.recall_at_k:>10.4f} {r.recall_at_10:>10.4f} "
                         f"{r.query_latency_ms:>8.3f} {r.p50_latency_ms:>8.3f} {r.p99_latency_ms:>8.3f} "
                         f"{r.index_size_kb:>10.0f}")

    # IVF sweep table
    ivf_results = [r for r in results if r.index_type == IndexType.IVF]
    if ivf_results:
        lines.append("")
        lines.append("IVF Parameter Sweep")
        lines.append("-" * 90)
        lines.append(f"{'Config':<20} {'Recall@5':>10} {'Recall@10':>10} {'Avg ms':>8} {'P50 ms':>8} {'P99 ms':>8} {'Size KB':>10}")
        lines.append("-" * 90)
        for r in sorted(ivf_results, key=lambda x: (x.params.get("nlist", 0), x.params.get("nprobe", 0))):
            label = f"nlist={r.params['nlist']} nprobe={r.params['nprobe']}"
            lines.append(f"{label:<20} {r.recall_at_k:>10.4f} {r.recall_at_10:>10.4f} "
                         f"{r.query_latency_ms:>8.3f} {r.p50_latency_ms:>8.3f} {r.p99_latency_ms:>8.3f} "
                         f"{r.index_size_kb:>10.0f}")

    # Recall-matched comparison
    recall_targets = [0.60, 0.80, 0.90, 0.93, 0.98]
    lines.append("")
    lines.append("Recall-Matched Comparison (fairest head-to-head)")
    lines.append("For each recall target, the cheapest (lowest latency) config that meets it.")
    lines.append("-" * 105)
    lines.append(f"{'Target':<9} {'Best HNSW (recall, latency)':<36} {'Best IVF (recall, latency)':<36} {'Verdict':>18}")
    lines.append("-" * 105)

    for target in recall_targets:
        best_h, best_i = _find_cheapest_at_recall(results, target)
        h_str = "--- (none meets target)"
        i_str = "--- (none meets target)"
        verdict = "---"
        if best_h:
            h_label = f"M={best_h.params['M']} ef={best_h.params['ef_search']}"
            h_str = f"{h_label} ({best_h.recall_at_k:.2%}, {best_h.query_latency_ms:.3f}ms)"
        if best_i:
            i_label = f"nl={best_i.params['nlist']} np={best_i.params['nprobe']}"
            i_str = f"{i_label} ({best_i.recall_at_k:.2%}, {best_i.query_latency_ms:.3f}ms)"
        if best_h and best_i:
            if best_h.query_latency_ms < best_i.query_latency_ms:
                verdict = "HNSW faster"
            elif best_i.query_latency_ms < best_h.query_latency_ms:
                verdict = "IVF faster"
            else:
                verdict = "Tie"
        elif best_h:
            verdict = "HNSW only"
        elif best_i:
            verdict = "IVF only"

        lines.append(f">={target:<8.0%} {h_str:<36} {i_str:<36} {verdict:>18}")

    lines.append("-" * 105)

    return "\n".join(lines)
