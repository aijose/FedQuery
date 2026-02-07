"""IR metrics for evaluating retrieval quality.

All functions are pure computation on search outputs — no LLM required.
"""

import math


def precision_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Fraction of top-k retrieved documents that are relevant.

    Returns 0.0 if k <= 0 or no documents retrieved.
    """
    if k <= 0 or not retrieved_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / k


def recall_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Fraction of relevant documents found in top-k results.

    Returns 1.0 if there are no relevant documents (vacuous truth for
    out-of-scope questions).
    """
    if not relevant_doc_ids:
        return 1.0
    if k <= 0 or not retrieved_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    hits = len(top_k & relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def mrr(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant result.

    Returns 0.0 if no relevant document is found.
    """
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (1 if relevant, 0 otherwise).
    Returns 0.0 if k <= 0 or no relevant documents exist.
    """
    if k <= 0 or not relevant_doc_ids:
        return 0.0 if relevant_doc_ids else 1.0

    top_k = retrieved_doc_ids[:k]

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = 1.0 if doc_id in relevant_doc_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant docs first
    ideal_count = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int,
) -> float:
    """Binary: 1.0 if any relevant document appears in top-k, else 0.0.

    Returns 1.0 for out-of-scope questions (no relevant documents).
    """
    if not relevant_doc_ids:
        return 1.0
    if k <= 0 or not retrieved_doc_ids:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    return 1.0 if top_k & relevant_doc_ids else 0.0


def chunk_text_recall(
    retrieved_texts: list[str],
    expected_fragments: list[str],
    k: int,
) -> float:
    """Fraction of expected text fragments found as substrings in top-k chunks.

    Case-insensitive substring matching. Returns 1.0 if no expected fragments
    (vacuous truth for out-of-scope questions).
    """
    if not expected_fragments:
        return 1.0
    if k <= 0 or not retrieved_texts:
        return 0.0

    top_k_texts = retrieved_texts[:k]
    combined = " ".join(t.lower() for t in top_k_texts)

    hits = sum(
        1 for frag in expected_fragments
        if frag.lower() in combined
    )
    return hits / len(expected_fragments)
