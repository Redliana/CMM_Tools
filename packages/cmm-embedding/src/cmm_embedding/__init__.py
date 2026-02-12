from __future__ import annotations

# embedding/__init__.py
"""
CMM Hierarchical Multi-Modal Embedding Package

This package implements the hierarchical multi-modal embedding architecture
for Critical Minerals and Materials (CMM) RAG systems, as specified in
CMM_RAG_Implementation_Guide.md Section 9.

Architecture Overview:
    Layer 1: Modality-Specific Encoders
        - SPECTER2 for scientific/policy text
        - CNN for spectral data (XRD/XRF)
        - GNN for molecular/crystal structures

    Layer 2: Cross-Modal Alignment
        - Contrastive learning on paired data
        - Shared semantic space projection

    Layer 3: Task Adapters
        - SPECTER2-style lightweight adapters
        - Task-specific embedding transformations

    Layer 4: Knowledge Graph Integration
        - Graph-based embedding retrofitting
        - Supply chain topology encoding

    Layer 5: Instruction Interface
        - E5-instruct style query prefixes
        - Dynamic embedding behavior

Subpackages:
    - training: Corpus building and contrastive training
    - evaluation: Benchmark specification and evaluation runner

Key Insight:
    No single embedding model can address CMM's scope spanning atomistic
    simulations (10⁻¹⁰ m) to global policy (10⁷ m). This package provides
    a modular architecture where specialized components work in concert.

Usage:
    # Build training corpus
    from embedding.training import CMMCorpusBuilder, CorpusBuilderConfig
    builder = CMMCorpusBuilder(config)
    corpus = await builder.build_corpus()

    # Train cross-modal alignment
    from embedding.training import CrossModalAlignmentTrainer
    trainer = CrossModalAlignmentTrainer(model, training_config)
    trainer.train(train_dataloader, val_dataloader)

    # Evaluate on benchmark
    from embedding.evaluation import BenchmarkRunner, create_full_benchmark_suite
    suite = create_full_benchmark_suite()
    results = runner.run_full_evaluation(suite)

Reference Documents:
    - CMM_RAG_Implementation_Guide.md Section 9
    - 260118_Claude_Opus_Embedding_Analysis.md
"""

__version__ = "0.1.0"

# Re-export key components from subpackages
from . import evaluation, training

__all__ = [
    "__version__",
    "evaluation",
    "training",
]
