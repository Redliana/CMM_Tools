from __future__ import annotations

# embedding/evaluation/benchmark_runner.py
"""
CMM Benchmark Evaluation Runner

This module provides the evaluation runner for assessing embedding models
against the CMM benchmark suite.

Metrics computed:
- Recall@K: Proportion of relevant items in top-K results
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant item
- NDCG@K: Normalized Discounted Cumulative Gain
- Entity Resolution F1: Precision/Recall for entity matching
- Path Accuracy: Correct supply chain paths
- Scale Bridging Score: Cross-scale retrieval quality

Usage:
    from embedding.evaluation import BenchmarkRunner, EvaluationConfig

    runner = BenchmarkRunner(embedding_model, config)
    results = runner.run_full_evaluation(benchmark_suite)
    results.save_report("evaluation_report.json")
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .cmm_benchmark_spec import (
    BenchmarkSuite,
    CrossModalAlignmentItem,
    CrossScaleRetrievalItem,
    EntityResolutionItem,
    SupplyChainTraversalItem,
    TemporalConsistencyItem,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Evaluation Configuration
# =============================================================================


@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""

    # Retrieval settings
    top_k_values: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    similarity_threshold: float = 0.5

    # Entity resolution settings
    entity_similarity_threshold: float = 0.85

    # Supply chain settings
    max_path_length: int = 10
    partial_path_credit: bool = True

    # Scoring weights
    category_weights: dict[str, float] = field(
        default_factory=lambda: {
            "cross_scale_retrieval": 0.25,
            "cross_modal_alignment": 0.25,
            "entity_resolution": 0.20,
            "supply_chain_traversal": 0.20,
            "temporal_consistency": 0.10,
        }
    )

    # Output settings
    output_dir: str = "./evaluation_results"
    save_per_item_results: bool = True
    compute_confidence_intervals: bool = True
    bootstrap_samples: int = 1000


# =============================================================================
# Metric Functions
# =============================================================================


class RetrievalMetrics:
    """Retrieval evaluation metrics."""

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Compute Recall@K.

        Recall@K = |relevant ∩ retrieved_top_k| / |relevant|
        """
        if not relevant_ids:
            return 0.0

        retrieved_set = set(retrieved_ids[:k])
        hits = len(retrieved_set & relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Compute Precision@K.

        Precision@K = |relevant ∩ retrieved_top_k| / K
        """
        if k == 0:
            return 0.0

        retrieved_set = set(retrieved_ids[:k])
        hits = len(retrieved_set & relevant_ids)
        return hits / k

    @staticmethod
    def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """
        Compute Mean Reciprocal Rank.

        MRR = 1 / rank of first relevant item (0 if none found)
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain @ K.

        NDCG = DCG / IDCG where DCG = Σ (2^rel - 1) / log2(i + 1)
        """
        if not relevant_ids:
            return 0.0

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1

        # IDCG (ideal: all relevant docs at top)
        ideal_length = min(len(relevant_ids), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def hit_rate(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        """
        Compute Hit Rate @ K.

        1 if any relevant item in top-K, 0 otherwise.
        """
        retrieved_set = set(retrieved_ids[:k])
        return 1.0 if (retrieved_set & relevant_ids) else 0.0


class EntityResolutionMetrics:
    """Entity resolution evaluation metrics."""

    @staticmethod
    def compute_f1(
        predicted_cluster: set[str], ground_truth_cluster: set[str]
    ) -> tuple[float, float, float]:
        """
        Compute Precision, Recall, and F1 for entity clustering.

        Returns:
            tuple of (precision, recall, f1)
        """
        if not predicted_cluster or not ground_truth_cluster:
            return (0.0, 0.0, 0.0)

        intersection = len(predicted_cluster & ground_truth_cluster)
        precision = intersection / len(predicted_cluster)
        recall = intersection / len(ground_truth_cluster)

        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        return (precision, recall, f1)

    @staticmethod
    def pairwise_f1(
        predicted_pairs: set[tuple[str, str]], ground_truth_pairs: set[tuple[str, str]]
    ) -> tuple[float, float, float]:
        """
        Compute pairwise F1 for entity resolution.

        Each pair represents two mentions that should be linked.
        """
        if not predicted_pairs or not ground_truth_pairs:
            return (0.0, 0.0, 0.0)

        # Normalize pairs (order-independent)
        pred_normalized = {tuple(sorted(p)) for p in predicted_pairs}
        gt_normalized = {tuple(sorted(p)) for p in ground_truth_pairs}

        intersection = len(pred_normalized & gt_normalized)
        precision = intersection / len(pred_normalized) if pred_normalized else 0.0
        recall = intersection / len(gt_normalized) if gt_normalized else 0.0

        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        return (precision, recall, f1)


class SupplyChainMetrics:
    """Supply chain traversal evaluation metrics."""

    @staticmethod
    def path_accuracy(
        predicted_path: list[str],
        ground_truth_path: list[str],
        alternative_paths: list[list[str]] | None = None,
    ) -> float:
        """
        Compute path accuracy.

        Returns 1.0 if predicted path matches ground truth or any alternative.
        """
        all_valid_paths = [ground_truth_path]
        if alternative_paths:
            all_valid_paths.extend(alternative_paths)

        for valid_path in all_valid_paths:
            if predicted_path == valid_path:
                return 1.0

        return 0.0

    @staticmethod
    def partial_path_score(predicted_path: list[str], ground_truth_path: list[str]) -> float:
        """
        Compute partial credit for partially correct paths.

        Uses longest common subsequence normalized by ground truth length.
        """
        if not ground_truth_path:
            return 0.0

        # LCS dynamic programming
        m, n = len(predicted_path), len(ground_truth_path)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if predicted_path[i - 1] == ground_truth_path[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return lcs_length / len(ground_truth_path)

    @staticmethod
    def endpoint_accuracy(predicted_path: list[str], start_entity: str, end_entity: str) -> float:
        """
        Check if path connects correct start and end entities.
        """
        if not predicted_path:
            return 0.0

        start_correct = predicted_path[0] == start_entity
        end_correct = predicted_path[-1] == end_entity

        if start_correct and end_correct:
            return 1.0
        elif start_correct or end_correct:
            return 0.5
        return 0.0


# =============================================================================
# Per-Category Evaluators
# =============================================================================


class CrossScaleEvaluator:
    """Evaluator for cross-scale retrieval tasks."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = RetrievalMetrics()

    def evaluate_item(
        self,
        item: CrossScaleRetrievalItem,
        retrieve_fn: Callable[[str], list[tuple[str, float]]],
    ) -> dict[str, Any]:
        """
        Evaluate a single cross-scale retrieval item.

        Args:
            item: Benchmark item
            retrieve_fn: Function that takes query and returns [(doc_id, score), ...]

        Returns:
            Dictionary of metrics for this item
        """
        # Get retrieval results
        results = retrieve_fn(item.get_query())
        retrieved_ids = [r[0] for r in results]
        relevant_ids = set(item.relevant_target_ids)

        # Compute metrics at each K
        metrics = {
            "item_id": item.item_id,
            "difficulty": item.difficulty.value,
            "source_scale": item.source_scale.value,
            "target_scale": item.target_scale.value,
        }

        for k in self.config.top_k_values:
            metrics[f"recall@{k}"] = self.metrics.recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"precision@{k}"] = self.metrics.precision_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"ndcg@{k}"] = self.metrics.ndcg_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"hit@{k}"] = self.metrics.hit_rate(retrieved_ids, relevant_ids, k)

        metrics["mrr"] = self.metrics.mrr(retrieved_ids, relevant_ids)

        # Scale bridging score: penalize if bridging concepts not captured
        if item.bridging_concepts:
            # This would need actual implementation with concept extraction
            metrics["bridging_score"] = metrics["mrr"]  # Placeholder

        return metrics


class CrossModalEvaluator:
    """Evaluator for cross-modal alignment tasks."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = RetrievalMetrics()

    def evaluate_item(
        self,
        item: CrossModalAlignmentItem,
        retrieve_fn: Callable[[Any, str], list[tuple[str, float]]],
    ) -> dict[str, Any]:
        """
        Evaluate a single cross-modal alignment item.

        Args:
            item: Benchmark item
            retrieve_fn: Function that takes (query_content, source_modality)
                        and returns [(doc_id, score), ...]
        """
        # Get retrieval results
        results = retrieve_fn(item.get_query(), item.source_modality.value)
        retrieved_ids = [r[0] for r in results]
        relevant_ids = set(item.relevant_target_ids)

        metrics = {
            "item_id": item.item_id,
            "difficulty": item.difficulty.value,
            "source_modality": item.source_modality.value,
            "target_modality": item.target_modality.value,
        }

        for k in self.config.top_k_values:
            metrics[f"recall@{k}"] = self.metrics.recall_at_k(retrieved_ids, relevant_ids, k)
            metrics[f"ndcg@{k}"] = self.metrics.ndcg_at_k(retrieved_ids, relevant_ids, k)

        metrics["mrr"] = self.metrics.mrr(retrieved_ids, relevant_ids)

        return metrics


class EntityResolutionEvaluator:
    """Evaluator for entity resolution tasks."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = EntityResolutionMetrics()

    def evaluate_item(
        self,
        item: EntityResolutionItem,
        resolve_fn: Callable[[str], set[str]],
    ) -> dict[str, Any]:
        """
        Evaluate a single entity resolution item.

        Args:
            item: Benchmark item
            resolve_fn: Function that takes entity name and returns set of matched aliases
        """
        # Get predicted cluster
        predicted_cluster = resolve_fn(item.canonical_entity)
        ground_truth_cluster = item.get_ground_truth()

        precision, recall, f1 = self.metrics.compute_f1(predicted_cluster, ground_truth_cluster)

        # Check negative examples (should NOT be matched)
        false_positives = sum(1 for neg in item.negative_examples if neg in predicted_cluster)

        metrics = {
            "item_id": item.item_id,
            "difficulty": item.difficulty.value,
            "entity_type": item.entity_type,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_count": false_positives,
            "num_aliases": len(item.aliases),
            "num_negatives": len(item.negative_examples),
        }

        return metrics


class SupplyChainEvaluator:
    """Evaluator for supply chain traversal tasks."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = SupplyChainMetrics()

    def evaluate_item(
        self,
        item: SupplyChainTraversalItem,
        traverse_fn: Callable[[str, str, str], list[str]],
    ) -> dict[str, Any]:
        """
        Evaluate a single supply chain traversal item.

        Args:
            item: Benchmark item
            traverse_fn: Function that takes (query, start_entity, end_entity)
                        and returns predicted path as list of entities
        """
        # Get predicted path
        predicted_path = traverse_fn(item.query_text, item.start_entity, item.end_entity)

        ground_truth = item.get_ground_truth()

        # Compute metrics
        path_acc = self.metrics.path_accuracy(
            predicted_path, ground_truth["primary_path"], ground_truth["alternatives"]
        )

        partial_score = self.metrics.partial_path_score(
            predicted_path, ground_truth["primary_path"]
        )

        endpoint_acc = self.metrics.endpoint_accuracy(
            predicted_path, item.start_entity, item.end_entity
        )

        metrics = {
            "item_id": item.item_id,
            "difficulty": item.difficulty.value,
            "path_accuracy": path_acc,
            "partial_path_score": partial_score,
            "endpoint_accuracy": endpoint_acc,
            "predicted_path_length": len(predicted_path),
            "expected_path_length": item.expected_path_length,
            "length_difference": abs(len(predicted_path) - item.expected_path_length),
        }

        return metrics


class TemporalEvaluator:
    """Evaluator for temporal consistency tasks."""

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate_item(
        self,
        item: TemporalConsistencyItem,
        query_fn: Callable[[str, str], str],
    ) -> dict[str, Any]:
        """
        Evaluate a single temporal consistency item.

        Args:
            item: Benchmark item
            query_fn: Function that takes (query, timestamp) and returns answer
        """
        query_text, query_timestamp = item.get_query()

        # Get answer for current timestamp
        predicted_answer = query_fn(query_text, query_timestamp)
        expected_answer = item.get_ground_truth()

        # Simple exact match (could be improved with semantic similarity)
        current_correct = self._answer_matches(predicted_answer, expected_answer)

        # Test historical answers
        historical_scores = []
        for hist_timestamp, hist_expected in item.expected_historical_answers.items():
            hist_predicted = query_fn(query_text, hist_timestamp)
            hist_correct = self._answer_matches(hist_predicted, hist_expected)
            historical_scores.append(hist_correct)

        metrics = {
            "item_id": item.item_id,
            "difficulty": item.difficulty.value,
            "current_timestamp_correct": current_correct,
            "historical_accuracy": np.mean(historical_scores) if historical_scores else 1.0,
            "num_time_points_tested": 1 + len(historical_scores),
        }

        return metrics

    def _answer_matches(self, predicted: str, expected: str) -> float:
        """Check if predicted answer matches expected (with some flexibility)."""
        # Normalize
        pred_lower = predicted.lower().strip()
        exp_lower = expected.lower().strip()

        # Exact match
        if pred_lower == exp_lower:
            return 1.0

        # Substring match (partial credit)
        if exp_lower in pred_lower or pred_lower in exp_lower:
            return 0.5

        return 0.0


# =============================================================================
# Results Container
# =============================================================================


@dataclass
class EvaluationResults:
    """Container for evaluation results."""

    model_name: str
    benchmark_version: str
    evaluation_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Aggregate scores
    overall_score: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)

    # Detailed metrics by category
    cross_scale_metrics: dict[str, float] = field(default_factory=dict)
    cross_modal_metrics: dict[str, float] = field(default_factory=dict)
    entity_resolution_metrics: dict[str, float] = field(default_factory=dict)
    supply_chain_metrics: dict[str, float] = field(default_factory=dict)
    temporal_metrics: dict[str, float] = field(default_factory=dict)

    # Per-item results
    per_item_results: list[dict[str, Any]] = field(default_factory=list)

    # Confidence intervals (if computed)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)

    def save_report(self, path: str):
        """Save evaluation report to JSON."""
        report = {
            "model_name": self.model_name,
            "benchmark_version": self.benchmark_version,
            "evaluation_timestamp": self.evaluation_timestamp,
            "summary": {
                "overall_score": self.overall_score,
                "category_scores": self.category_scores,
            },
            "detailed_metrics": {
                "cross_scale_retrieval": self.cross_scale_metrics,
                "cross_modal_alignment": self.cross_modal_metrics,
                "entity_resolution": self.entity_resolution_metrics,
                "supply_chain_traversal": self.supply_chain_metrics,
                "temporal_consistency": self.temporal_metrics,
            },
            "confidence_intervals": {
                k: {"lower": v[0], "upper": v[1]} for k, v in self.confidence_intervals.items()
            },
            "per_item_results": self.per_item_results,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved evaluation report to {path}")

    def print_summary(self):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("CMM EMBEDDING EVALUATION RESULTS")
        print(f"Model: {self.model_name}")
        print(f"Benchmark: {self.benchmark_version}")
        print("=" * 60)

        print(f"\nOVERALL SCORE: {self.overall_score:.4f}")

        print("\nCategory Scores:")
        for category, score in self.category_scores.items():
            print(f"  {category}: {score:.4f}")

        print("\nKey Metrics:")
        print(f"  Cross-Scale MRR: {self.cross_scale_metrics.get('mrr', 0):.4f}")
        print(f"  Cross-Modal MRR: {self.cross_modal_metrics.get('mrr', 0):.4f}")
        print(f"  Entity Resolution F1: {self.entity_resolution_metrics.get('f1', 0):.4f}")
        print(f"  Supply Chain Path Acc: {self.supply_chain_metrics.get('path_accuracy', 0):.4f}")
        print(
            f"  Temporal Consistency: {self.temporal_metrics.get('current_timestamp_correct', 0):.4f}"
        )

        print("=" * 60 + "\n")


# =============================================================================
# Main Benchmark Runner
# =============================================================================


class BenchmarkRunner:
    """
    Main class for running CMM benchmark evaluation.

    Usage:
        runner = BenchmarkRunner(embedding_model, config)
        results = runner.run_full_evaluation(benchmark_suite)
        results.print_summary()
        results.save_report("results.json")
    """

    def __init__(
        self,
        embedding_model: Any,
        config: EvaluationConfig = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            embedding_model: Model with methods:
                - embed(content, modality) -> embedding
                - retrieve(query, modality, top_k) -> [(id, score), ...]
                - resolve_entity(name) -> set[str]
                - traverse_supply_chain(query, start, end) -> list[str]
                - query_with_timestamp(query, timestamp) -> str
            config: Evaluation configuration
        """
        self.model = embedding_model
        self.config = config or EvaluationConfig()

        # Initialize evaluators
        self.cross_scale_eval = CrossScaleEvaluator(self.config)
        self.cross_modal_eval = CrossModalEvaluator(self.config)
        self.entity_eval = EntityResolutionEvaluator(self.config)
        self.supply_chain_eval = SupplyChainEvaluator(self.config)
        self.temporal_eval = TemporalEvaluator(self.config)

    def run_full_evaluation(
        self,
        benchmark: BenchmarkSuite,
        model_name: str = "unnamed_model",
    ) -> EvaluationResults:
        """
        Run full evaluation on benchmark suite.

        Args:
            benchmark: Benchmark suite to evaluate
            model_name: Name of the model being evaluated

        Returns:
            EvaluationResults with all metrics
        """
        logger.info(f"Starting evaluation of {model_name} on {benchmark.name}")

        results = EvaluationResults(
            model_name=model_name,
            benchmark_version=benchmark.version,
        )

        # Evaluate each category
        results.cross_scale_metrics, cross_scale_items = self._evaluate_cross_scale(
            benchmark.cross_scale_items
        )
        results.per_item_results.extend(cross_scale_items)

        results.cross_modal_metrics, cross_modal_items = self._evaluate_cross_modal(
            benchmark.cross_modal_items
        )
        results.per_item_results.extend(cross_modal_items)

        results.entity_resolution_metrics, entity_items = self._evaluate_entity_resolution(
            benchmark.entity_resolution_items
        )
        results.per_item_results.extend(entity_items)

        results.supply_chain_metrics, supply_chain_items = self._evaluate_supply_chain(
            benchmark.supply_chain_items
        )
        results.per_item_results.extend(supply_chain_items)

        results.temporal_metrics, temporal_items = self._evaluate_temporal(benchmark.temporal_items)
        results.per_item_results.extend(temporal_items)

        # Compute category scores
        results.category_scores = {
            "cross_scale_retrieval": results.cross_scale_metrics.get("mrr", 0),
            "cross_modal_alignment": results.cross_modal_metrics.get("mrr", 0),
            "entity_resolution": results.entity_resolution_metrics.get("f1", 0),
            "supply_chain_traversal": results.supply_chain_metrics.get("path_accuracy", 0),
            "temporal_consistency": results.temporal_metrics.get("current_timestamp_correct", 0),
        }

        # Compute overall score (weighted average)
        results.overall_score = sum(
            score * self.config.category_weights.get(cat, 0.2)
            for cat, score in results.category_scores.items()
        )

        # Compute confidence intervals if configured
        if self.config.compute_confidence_intervals:
            results.confidence_intervals = self._compute_confidence_intervals(
                results.per_item_results
            )

        logger.info(f"Evaluation complete. Overall score: {results.overall_score:.4f}")

        return results

    def _evaluate_cross_scale(
        self, items: list[CrossScaleRetrievalItem]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate cross-scale retrieval items."""
        if not items:
            return {}, []

        per_item_metrics = []
        for item in items:
            metrics = self.cross_scale_eval.evaluate_item(
                item, lambda q: self.model.retrieve(q, "text", max(self.config.top_k_values))
            )
            metrics["category"] = "cross_scale_retrieval"
            per_item_metrics.append(metrics)

        # Aggregate
        aggregate = self._aggregate_metrics(per_item_metrics)

        return aggregate, per_item_metrics

    def _evaluate_cross_modal(
        self, items: list[CrossModalAlignmentItem]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate cross-modal alignment items."""
        if not items:
            return {}, []

        per_item_metrics = []
        for item in items:
            metrics = self.cross_modal_eval.evaluate_item(
                item, lambda q, m: self.model.retrieve(q, m, max(self.config.top_k_values))
            )
            metrics["category"] = "cross_modal_alignment"
            per_item_metrics.append(metrics)

        aggregate = self._aggregate_metrics(per_item_metrics)

        return aggregate, per_item_metrics

    def _evaluate_entity_resolution(
        self, items: list[EntityResolutionItem]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate entity resolution items."""
        if not items:
            return {}, []

        per_item_metrics = []
        for item in items:
            metrics = self.entity_eval.evaluate_item(item, lambda n: self.model.resolve_entity(n))
            metrics["category"] = "entity_resolution"
            per_item_metrics.append(metrics)

        aggregate = self._aggregate_metrics(per_item_metrics)

        return aggregate, per_item_metrics

    def _evaluate_supply_chain(
        self, items: list[SupplyChainTraversalItem]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate supply chain traversal items."""
        if not items:
            return {}, []

        per_item_metrics = []
        for item in items:
            metrics = self.supply_chain_eval.evaluate_item(
                item, lambda q, s, e: self.model.traverse_supply_chain(q, s, e)
            )
            metrics["category"] = "supply_chain_traversal"
            per_item_metrics.append(metrics)

        aggregate = self._aggregate_metrics(per_item_metrics)

        return aggregate, per_item_metrics

    def _evaluate_temporal(
        self, items: list[TemporalConsistencyItem]
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Evaluate temporal consistency items."""
        if not items:
            return {}, []

        per_item_metrics = []
        for item in items:
            metrics = self.temporal_eval.evaluate_item(
                item, lambda q, t: self.model.query_with_timestamp(q, t)
            )
            metrics["category"] = "temporal_consistency"
            per_item_metrics.append(metrics)

        aggregate = self._aggregate_metrics(per_item_metrics)

        return aggregate, per_item_metrics

    def _aggregate_metrics(self, per_item_metrics: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate per-item metrics into category-level metrics."""
        if not per_item_metrics:
            return {}

        aggregate = {}
        numeric_keys = [
            k for k in per_item_metrics[0] if isinstance(per_item_metrics[0][k], (int, float))
        ]

        for key in numeric_keys:
            values = [m[key] for m in per_item_metrics if key in m]
            if values:
                aggregate[key] = np.mean(values)

        return aggregate

    def _compute_confidence_intervals(
        self, per_item_results: list[dict[str, Any]], alpha: float = 0.05
    ) -> dict[str, tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for key metrics.

        Returns:
            Dictionary mapping metric name to (lower, upper) bounds
        """
        if not per_item_results:
            return {}

        intervals = {}
        key_metrics = ["mrr", "f1", "path_accuracy", "current_timestamp_correct"]

        for metric in key_metrics:
            values = [r.get(metric) for r in per_item_results if metric in r]
            if not values:
                continue

            # Bootstrap
            bootstrap_means = []
            for _ in range(self.config.bootstrap_samples):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))

            lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

            intervals[metric] = (lower, upper)

        return intervals


# =============================================================================
# Convenience Functions
# =============================================================================


def run_quick_evaluation(
    model: Any,
    benchmark_path: str | None = None,
    output_path: str | None = None,
) -> EvaluationResults:
    """
    Run quick evaluation with default settings.

    Args:
        model: Embedding model to evaluate
        benchmark_path: Path to benchmark JSON (uses default if None)
        output_path: Path to save results (optional)

    Returns:
        EvaluationResults
    """
    from .cmm_benchmark_spec import create_full_benchmark_suite

    # Load or create benchmark
    if benchmark_path and Path(benchmark_path).exists():
        benchmark = BenchmarkSuite.load(benchmark_path)
    else:
        benchmark = create_full_benchmark_suite()

    # Run evaluation
    config = EvaluationConfig()
    runner = BenchmarkRunner(model, config)
    results = runner.run_full_evaluation(benchmark, model_name=getattr(model, "name", "model"))

    # Save if path provided
    if output_path:
        results.save_report(output_path)

    results.print_summary()

    return results


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CMM benchmark evaluation")
    parser.add_argument("--benchmark", "-b", help="Path to benchmark JSON")
    parser.add_argument("--output", "-o", default="evaluation_results.json")
    parser.add_argument("--model", "-m", help="Model identifier")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # This would need actual model loading
    print("To run evaluation, implement model loading and call run_quick_evaluation()")
    print(f"Benchmark: {args.benchmark}")
    print(f"Output: {args.output}")
