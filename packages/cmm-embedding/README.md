# cmm-embedding

Domain-specific hierarchical multi-modal embedding training and evaluation for Critical
Minerals and Materials (CMM) RAG systems. Implements a five-layer architecture: modality
encoders, cross-modal alignment, task adapters, KG integration, and instruction interface.

## Installation

```bash
uv pip install -e packages/cmm-embedding
```

## Quick Start

```python
# Build a cross-modal training corpus
from cmm_embedding.training import CMMCorpusBuilder, CorpusBuilderConfig

config = CorpusBuilderConfig(
    materials_project_api_key="your_key",
    usgs_data_dir="/path/to/usgs",
)
builder = CMMCorpusBuilder(config)
corpus = await builder.build_corpus()
corpus.save("cmm_corpus.jsonl")

# Train cross-modal alignment
from cmm_embedding.training import (
    CMMPairedDataset,
    CrossModalAlignmentTrainer,
    TrainingConfig,
    create_contrastive_dataloader,
)

dataset = CMMPairedDataset("cmm_corpus.jsonl", min_confidence=0.3)
dataloader = create_contrastive_dataloader(dataset, batch_size=32)
trainer = CrossModalAlignmentTrainer(model, TrainingConfig(learning_rate=1e-4))
trainer.train(dataloader, val_dataloader)

# Evaluate on CMM benchmark
from cmm_embedding.evaluation import BenchmarkRunner, EvaluationConfig, create_full_benchmark_suite

suite = create_full_benchmark_suite()
runner = BenchmarkRunner(my_model, EvaluationConfig(top_k_values=[1, 5, 10, 20]))
results = runner.run_full_evaluation(suite, model_name="my_model")
results.print_summary()
```

## API Reference

### Training (`cmm_embedding.training`)

- `CMMCorpusBuilder(CorpusBuilderConfig)` -- Builds cross-modal paired training data.
- `CrossModalAlignmentTrainer(model, TrainingConfig)` -- Contrastive alignment training loop.
- `CrossModalAlignmentModel` -- Model with shared semantic space projection.
- `CMMPairedDataset` -- PyTorch dataset for contrastive learning pairs.
- `create_contrastive_dataloader(dataset, batch_size)` -- DataLoader with contrastive batching.
- `InfoNCELoss`, `HardNegativeLoss` -- Contrastive loss functions.
- `TrainingCorpus`, `CrossModalPair`, `Modality`, `ModalityData` -- Data structures.

### Evaluation (`cmm_embedding.evaluation`)

- `BenchmarkRunner` -- Runs evaluation across all benchmark categories.
- `EvaluationConfig` -- Evaluation settings (top-k values, thresholds).
- `EvaluationResults` -- Aggregated results with `print_summary()` and `save_report()`.
- `create_full_benchmark_suite()` -- Factory for the full CMM benchmark suite.
- `BenchmarkSuite`, `BenchmarkItem` -- Benchmark data structures.
- Evaluators: `CrossScaleEvaluator`, `CrossModalEvaluator`, `EntityResolutionEvaluator`, `SupplyChainEvaluator`, `TemporalEvaluator`.
- `run_quick_evaluation(model)` -- Convenience function for fast evaluation.

## Configuration

No environment variables required. All configuration is passed via constructor arguments.

## Dependencies

numpy, torch
