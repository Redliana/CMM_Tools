from __future__ import annotations

# embedding/evaluation/__init__.py
"""
CMM Embedding Evaluation Module

This package provides a comprehensive evaluation framework for CMM embeddings,
addressing the gap that no CMM-specific embedding benchmark exists.

Components:
    - cmm_benchmark_spec: Benchmark specification and test item definitions
    - benchmark_runner: Evaluation runner and metrics computation

Benchmark Categories:
    1. Cross-Scale Retrieval: Atomistic → Policy document linking
    2. Cross-Modal Alignment: Spectrum → Text retrieval
    3. Entity Resolution: Name variation matching
    4. Supply Chain Traversal: Multi-hop graph queries
    5. Temporal Consistency: Time-sensitive information handling

Usage:
    # Generate benchmark suite
    from embedding.evaluation import create_full_benchmark_suite

    suite = create_full_benchmark_suite()
    suite.save("cmm_benchmark_v1.json")

    # Run evaluation
    from embedding.evaluation import BenchmarkRunner, EvaluationConfig

    config = EvaluationConfig(top_k_values=[1, 5, 10, 20])
    runner = BenchmarkRunner(my_model, config)
    results = runner.run_full_evaluation(suite, model_name="my_model")

    results.print_summary()
    results.save_report("results.json")

Target Metrics (from CMM_RAG_Implementation_Guide.md Section 12.5):
    - Recall@10: >0.85
    - MRR: >0.7
    - Cross-Modal Precision: >0.8
    - Entity Resolution F1: >0.9
    - Path Accuracy: >0.75
    - Scale Bridging Score: >4.0/5.0
"""

from .benchmark_runner import (
    # Main runner
    BenchmarkRunner,
    CrossModalEvaluator,
    # Evaluators
    CrossScaleEvaluator,
    EntityResolutionEvaluator,
    EntityResolutionMetrics,
    # Config
    EvaluationConfig,
    # Results
    EvaluationResults,
    # Metrics
    RetrievalMetrics,
    SupplyChainEvaluator,
    SupplyChainMetrics,
    TemporalEvaluator,
    # Convenience
    run_quick_evaluation,
)
from .cmm_benchmark_spec import (
    # Enums
    BenchmarkCategory,
    # Data models
    BenchmarkItem,
    # Suite
    BenchmarkSuite,
    CrossModalAlignmentItem,
    CrossModalBenchmarkGenerator,
    # Generators
    CrossScaleBenchmarkGenerator,
    CrossScaleRetrievalItem,
    Difficulty,
    EntityResolutionBenchmarkGenerator,
    EntityResolutionItem,
    ModalityType,
    ScaleLevel,
    SupplyChainBenchmarkGenerator,
    SupplyChainTraversalItem,
    TemporalBenchmarkGenerator,
    TemporalConsistencyItem,
    # Factory
    create_full_benchmark_suite,
)

__all__ = [
    # Enums
    "BenchmarkCategory",
    # Data models
    "BenchmarkItem",
    # Main runner
    "BenchmarkRunner",
    # Suite
    "BenchmarkSuite",
    "CrossModalAlignmentItem",
    "CrossModalBenchmarkGenerator",
    "CrossModalEvaluator",
    # Generators
    "CrossScaleBenchmarkGenerator",
    # Evaluators
    "CrossScaleEvaluator",
    "CrossScaleRetrievalItem",
    "Difficulty",
    "EntityResolutionBenchmarkGenerator",
    "EntityResolutionEvaluator",
    "EntityResolutionItem",
    "EntityResolutionMetrics",
    # Config
    "EvaluationConfig",
    # Results
    "EvaluationResults",
    "ModalityType",
    # Metrics
    "RetrievalMetrics",
    "ScaleLevel",
    "SupplyChainBenchmarkGenerator",
    "SupplyChainEvaluator",
    "SupplyChainMetrics",
    "SupplyChainTraversalItem",
    "TemporalBenchmarkGenerator",
    "TemporalConsistencyItem",
    "TemporalEvaluator",
    # Factory
    "create_full_benchmark_suite",
    # Convenience
    "run_quick_evaluation",
]
