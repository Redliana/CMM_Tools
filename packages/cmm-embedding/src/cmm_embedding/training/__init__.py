from __future__ import annotations

# embedding/training/__init__.py
"""
CMM Embedding Training Module

This package provides tools for building training corpora and training
the hierarchical multi-modal embedding system for Critical Minerals and Materials.

Main Components:
    - corpus_builder: Build cross-modal paired training data
    - paired_data_loader: PyTorch data loading for contrastive learning
    - alignment_training: Training loop for cross-modal alignment

Usage:
    # Build corpus
    from embedding.training.corpus_builder import CMMCorpusBuilder, CorpusBuilderConfig

    config = CorpusBuilderConfig(
        materials_project_api_key="your_key",
        usgs_data_dir="/path/to/usgs",
    )
    builder = CMMCorpusBuilder(config)
    corpus = await builder.build_corpus()
    corpus.save("cmm_corpus.jsonl")

    # Load for training
    from embedding.training.paired_data_loader import (
        CMMPairedDataset,
        create_contrastive_dataloader,
    )

    dataset = CMMPairedDataset("cmm_corpus.jsonl", min_confidence=0.3)
    dataloader = create_contrastive_dataloader(dataset, batch_size=32)

    # Train alignment
    from embedding.training.alignment_training import (
        CrossModalAlignmentTrainer,
        TrainingConfig,
    )

    config = TrainingConfig(learning_rate=1e-4, max_steps=50000)
    trainer = CrossModalAlignmentTrainer(model, config)
    trainer.train(train_dataloader, val_dataloader)
"""

from .alignment_training import (
    CrossModalAlignmentModel,
    CrossModalAlignmentTrainer,
    HardNegativeLoss,
    InfoNCELoss,
    TrainingConfig,
)
from .corpus_builder import (
    CMMCorpusBuilder,
    CorpusBuilderConfig,
    CrossModalPair,
    Modality,
    ModalityData,
    PairingMethod,
    TrainingCorpus,
)
from .paired_data_loader import (
    CMMPairedDataset,
    ConfidenceWeightedSampler,
    ContrastiveBatch,
    ContrastiveCollator,
    HardNegativeDataset,
    InBatchNegativeSampler,
    ModalityBalancedSampler,
    create_contrastive_dataloader,
    create_evaluation_dataloader,
)

__all__ = [
    # Corpus building
    "CMMCorpusBuilder",
    # Data loading
    "CMMPairedDataset",
    "ConfidenceWeightedSampler",
    "ContrastiveBatch",
    "ContrastiveCollator",
    "CorpusBuilderConfig",
    "CrossModalAlignmentModel",
    # Training
    "CrossModalAlignmentTrainer",
    "CrossModalPair",
    "HardNegativeDataset",
    "HardNegativeLoss",
    "InBatchNegativeSampler",
    "InfoNCELoss",
    "Modality",
    "ModalityBalancedSampler",
    "ModalityData",
    "PairingMethod",
    "TrainingConfig",
    "TrainingCorpus",
    "create_contrastive_dataloader",
    "create_evaluation_dataloader",
]
