"""Tests for cmm_embedding.evaluation.benchmark_runner module."""

from __future__ import annotations

import pytest

from cmm_embedding.evaluation.benchmark_runner import (
    BenchmarkRunner,
    EntityResolutionMetrics,
    EvaluationConfig,
    RetrievalMetrics,
    SupplyChainMetrics,
)

# ============================================================================
# RetrievalMetrics
# ============================================================================


class TestRecallAtK:
    """Tests for ``RetrievalMetrics.recall_at_k``."""

    def test_perfect_recall(self) -> None:
        """All relevant items in top-k should yield 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        """Only some relevant items in top-k should give fractional recall."""
        retrieved = ["a", "x", "y"]
        relevant = {"a", "b"}
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)

    def test_no_relevant_items(self) -> None:
        """Empty relevant set should return 0.0."""
        retrieved = ["a", "b"]
        relevant: set[str] = set()
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_k_smaller_than_retrieved(self) -> None:
        """Only the first k retrieved items should be considered."""
        retrieved = ["a", "b", "c", "d"]
        relevant = {"c", "d"}
        # k=2 considers only ["a", "b"] -- no hits
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        """If k > len(retrieved), use all of retrieved."""
        retrieved = ["a"]
        relevant = {"a", "b"}
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, k=10) == pytest.approx(0.5)


class TestPrecisionAtK:
    """Tests for ``RetrievalMetrics.precision_at_k``."""

    def test_all_relevant(self) -> None:
        """All top-k items being relevant should yield 1.0."""
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_half_relevant(self) -> None:
        """Half of top-k being relevant should yield 0.5."""
        retrieved = ["a", "x"]
        relevant = {"a"}
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)

    def test_k_zero(self) -> None:
        """k=0 should return 0.0."""
        assert RetrievalMetrics.precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_no_hits(self) -> None:
        """No relevant items in top-k should yield 0.0."""
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.0)


class TestMRR:
    """Tests for ``RetrievalMetrics.mrr``."""

    def test_first_item_relevant(self) -> None:
        """First relevant item at rank 1 should give MRR = 1.0."""
        assert RetrievalMetrics.mrr(["a", "b"], {"a"}) == pytest.approx(1.0)

    def test_second_item_relevant(self) -> None:
        """First relevant at rank 2 should give MRR = 0.5."""
        assert RetrievalMetrics.mrr(["x", "a"], {"a"}) == pytest.approx(0.5)

    def test_third_item_relevant(self) -> None:
        """First relevant at rank 3 should give MRR = 1/3."""
        assert RetrievalMetrics.mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_found(self) -> None:
        """No relevant items should yield MRR = 0.0."""
        assert RetrievalMetrics.mrr(["x", "y"], {"a"}) == 0.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list should yield 0.0."""
        assert RetrievalMetrics.mrr([], {"a"}) == 0.0


class TestNDCGAtK:
    """Tests for ``RetrievalMetrics.ndcg_at_k``."""

    def test_perfect_ranking(self) -> None:
        """All relevant at the top should yield NDCG = 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_no_relevant(self) -> None:
        """No relevant items should yield NDCG = 0.0."""
        retrieved = ["x", "y"]
        relevant: set[str] = set()
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=2) == 0.0

    def test_single_relevant_at_top(self) -> None:
        """One relevant at rank 1 out of 1 total relevant should give 1.0."""
        retrieved = ["a", "x", "y"]
        relevant = {"a"}
        assert RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_single_relevant_not_at_top(self) -> None:
        """One relevant at rank 2 should give NDCG < 1.0."""
        retrieved = ["x", "a", "y"]
        relevant = {"a"}
        result = RetrievalMetrics.ndcg_at_k(retrieved, relevant, k=3)
        assert 0.0 < result < 1.0

    def test_empty_retrieved_list(self) -> None:
        """Empty retrieved list with non-empty relevant should return 0.0."""
        assert RetrievalMetrics.ndcg_at_k([], {"a"}, k=5) == 0.0


# ============================================================================
# EntityResolutionMetrics
# ============================================================================


class TestEntityResolutionComputeF1:
    """Tests for ``EntityResolutionMetrics.compute_f1``."""

    def test_perfect_match(self) -> None:
        """Identical clusters should yield P=R=F1=1.0."""
        predicted = {"a", "b", "c"}
        truth = {"a", "b", "c"}
        p, r, f1 = EntityResolutionMetrics.compute_f1(predicted, truth)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        """Disjoint clusters should yield P=R=F1=0.0."""
        predicted = {"x", "y"}
        truth = {"a", "b"}
        p, r, f1 = EntityResolutionMetrics.compute_f1(predicted, truth)
        assert p == 0.0
        assert r == 0.0
        assert f1 == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap should give values between 0 and 1."""
        predicted = {"a", "b", "x"}
        truth = {"a", "b", "c"}
        p, r, f1 = EntityResolutionMetrics.compute_f1(predicted, truth)
        # intersection = {a, b} size 2
        # precision = 2/3, recall = 2/3, f1 = 2/3
        assert p == pytest.approx(2 / 3)
        assert r == pytest.approx(2 / 3)
        assert f1 == pytest.approx(2 / 3)

    def test_empty_predicted(self) -> None:
        """Empty predicted cluster should yield all zeros."""
        p, r, f1 = EntityResolutionMetrics.compute_f1(set(), {"a", "b"})
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_empty_ground_truth(self) -> None:
        """Empty ground truth cluster should yield all zeros."""
        p, r, f1 = EntityResolutionMetrics.compute_f1({"a"}, set())
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_precision_higher_than_recall(self) -> None:
        """When predicted is a subset of truth, precision > recall."""
        predicted = {"a"}
        truth = {"a", "b", "c"}
        p, r, _f1 = EntityResolutionMetrics.compute_f1(predicted, truth)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1 / 3)

    def test_recall_higher_than_precision(self) -> None:
        """When truth is a subset of predicted, recall > precision."""
        predicted = {"a", "b", "c"}
        truth = {"a"}
        p, r, _f1 = EntityResolutionMetrics.compute_f1(predicted, truth)
        assert p == pytest.approx(1 / 3)
        assert r == pytest.approx(1.0)


# ============================================================================
# SupplyChainMetrics
# ============================================================================


class TestPathAccuracy:
    """Tests for ``SupplyChainMetrics.path_accuracy``."""

    def test_exact_match(self) -> None:
        """An exact match with ground truth should return 1.0."""
        path = ["A", "B", "C"]
        gt = ["A", "B", "C"]
        assert SupplyChainMetrics.path_accuracy(path, gt) == 1.0

    def test_no_match(self) -> None:
        """A completely wrong path should return 0.0."""
        path = ["X", "Y", "Z"]
        gt = ["A", "B", "C"]
        assert SupplyChainMetrics.path_accuracy(path, gt) == 0.0

    def test_alternative_path_match(self) -> None:
        """Matching an alternative path should also return 1.0."""
        path = ["A", "D", "C"]
        gt = ["A", "B", "C"]
        alternatives = [["A", "D", "C"], ["A", "E", "C"]]
        assert SupplyChainMetrics.path_accuracy(path, gt, alternatives) == 1.0

    def test_partial_match_still_zero(self) -> None:
        """Partial overlap (not exact) should return 0.0."""
        path = ["A", "B"]
        gt = ["A", "B", "C"]
        assert SupplyChainMetrics.path_accuracy(path, gt) == 0.0


class TestPartialPathScore:
    """Tests for ``SupplyChainMetrics.partial_path_score``."""

    def test_identical_paths(self) -> None:
        """Identical paths should score 1.0."""
        path = ["A", "B", "C"]
        assert SupplyChainMetrics.partial_path_score(path, path) == pytest.approx(1.0)

    def test_empty_ground_truth(self) -> None:
        """Empty ground truth should return 0.0."""
        assert SupplyChainMetrics.partial_path_score(["A"], []) == 0.0

    def test_empty_predicted(self) -> None:
        """Empty predicted path against non-empty truth should return 0.0."""
        assert SupplyChainMetrics.partial_path_score([], ["A", "B"]) == 0.0

    def test_partial_subsequence(self) -> None:
        """A predicted path sharing a subsequence should get partial credit."""
        predicted = ["A", "X", "C"]
        gt = ["A", "B", "C"]
        score = SupplyChainMetrics.partial_path_score(predicted, gt)
        # LCS is ["A", "C"] length 2, gt length 3 -> 2/3
        assert score == pytest.approx(2 / 3)

    def test_no_common_elements(self) -> None:
        """Completely disjoint paths should score 0.0."""
        assert SupplyChainMetrics.partial_path_score(["X", "Y"], ["A", "B"]) == 0.0

    def test_subsequence_order_matters(self) -> None:
        """LCS is order-dependent -- reversed elements should yield lower score."""
        predicted = ["C", "B", "A"]
        gt = ["A", "B", "C"]
        score = SupplyChainMetrics.partial_path_score(predicted, gt)
        # LCS could be just ["A"] or ["B"] or ["C"] -- length 1
        assert score == pytest.approx(1 / 3)


# ============================================================================
# BenchmarkRunner._aggregate_metrics
# ============================================================================


class TestAggregateMetrics:
    """Tests for ``BenchmarkRunner._aggregate_metrics``."""

    def _make_runner(self) -> BenchmarkRunner:
        """Create a minimal BenchmarkRunner with a stub model."""

        class StubModel:
            pass

        return BenchmarkRunner(StubModel(), EvaluationConfig())

    def test_empty_list(self) -> None:
        """Empty input should return an empty dict."""
        runner = self._make_runner()
        assert runner._aggregate_metrics([]) == {}

    def test_single_item(self) -> None:
        """A single item should propagate its values as the mean."""
        runner = self._make_runner()
        items = [{"mrr": 0.5, "recall@10": 0.8, "category": "test"}]
        agg = runner._aggregate_metrics(items)
        assert agg["mrr"] == pytest.approx(0.5)
        assert agg["recall@10"] == pytest.approx(0.8)

    def test_averages_correctly(self) -> None:
        """Multiple items should be averaged."""
        runner = self._make_runner()
        items = [
            {"mrr": 1.0, "f1": 0.8, "name": "a"},
            {"mrr": 0.5, "f1": 0.6, "name": "b"},
            {"mrr": 0.0, "f1": 0.4, "name": "c"},
        ]
        agg = runner._aggregate_metrics(items)
        assert agg["mrr"] == pytest.approx(0.5)
        assert agg["f1"] == pytest.approx(0.6)

    def test_non_numeric_keys_excluded(self) -> None:
        """String-valued keys should not appear in the aggregate."""
        runner = self._make_runner()
        items = [{"mrr": 0.5, "category": "test"}]
        agg = runner._aggregate_metrics(items)
        assert "category" not in agg
