"""Tests for retrieval quality metrics."""

import math

import pytest

from src.evaluation.retrieval_metrics import (
    chunk_text_recall,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 1.0

    def test_none_relevant(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 4) == 0.5

    def test_k_smaller_than_results(self):
        retrieved = ["a", "b", "x", "y"]
        relevant = {"a", "b"}
        assert precision_at_k(retrieved, relevant, 2) == 1.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecallAtK:
    def test_all_found(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 1.0

    def test_partial_found(self):
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.5

    def test_none_found(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 3) == 0.0

    def test_no_relevant_docs(self):
        """Out-of-scope: vacuous truth."""
        assert recall_at_k(["x", "y"], set(), 2) == 1.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}, 5) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a", "c"], {"a"}) == 0.5

    def test_third_position(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3)

    def test_not_found(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant(self):
        """MRR uses first relevant result."""
        assert mrr(["x", "a", "b"], {"a", "b"}) == 0.5


class TestNDCGAtK:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, 3) == 1.0

    def test_worst_ranking(self):
        retrieved = ["x", "y", "a"]
        relevant = {"a"}
        # DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(4)) / (1.0 / math.log2(2))
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected)

    def test_no_relevant_empty_set(self):
        """Empty relevant set: vacuous, returns 1.0."""
        assert ndcg_at_k(["a", "b"], set(), 2) == 1.0

    def test_k_zero(self):
        assert ndcg_at_k(["a"], {"a"}, 0) == 0.0

    def test_no_hits(self):
        assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


class TestHitRateAtK:
    def test_hit(self):
        assert hit_rate_at_k(["x", "a", "y"], {"a"}, 3) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["x", "y", "z"], {"a"}, 3) == 0.0

    def test_hit_within_k(self):
        assert hit_rate_at_k(["x", "a", "y", "z"], {"a"}, 2) == 1.0

    def test_hit_outside_k(self):
        assert hit_rate_at_k(["x", "y", "a", "z"], {"a"}, 2) == 0.0

    def test_no_relevant(self):
        """Out-of-scope: vacuous truth."""
        assert hit_rate_at_k(["x", "y"], set(), 2) == 1.0


class TestChunkTextRecall:
    def test_all_found(self):
        texts = ["the cat sat on the mat", "the dog played fetch"]
        fragments = ["cat sat", "dog played"]
        assert chunk_text_recall(texts, fragments, 2) == 1.0

    def test_partial_found(self):
        texts = ["the cat sat on the mat", "hello world"]
        fragments = ["cat sat", "dog played"]
        assert chunk_text_recall(texts, fragments, 2) == 0.5

    def test_none_found(self):
        texts = ["hello world", "foo bar"]
        fragments = ["cat sat", "dog played"]
        assert chunk_text_recall(texts, fragments, 2) == 0.0

    def test_case_insensitive(self):
        texts = ["The Cat SAT on the Mat"]
        fragments = ["cat sat"]
        assert chunk_text_recall(texts, fragments, 1) == 1.0

    def test_no_expected_fragments(self):
        """Out-of-scope: vacuous truth."""
        assert chunk_text_recall(["hello"], [], 1) == 1.0

    def test_k_limits_search(self):
        texts = ["no match here", "the target fragment is here"]
        fragments = ["target fragment"]
        assert chunk_text_recall(texts, fragments, 1) == 0.0
        assert chunk_text_recall(texts, fragments, 2) == 1.0

    def test_empty_texts(self):
        assert chunk_text_recall([], ["hello"], 5) == 0.0
