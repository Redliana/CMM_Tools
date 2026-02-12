# CMM Cross-Modal Embedding Training

This module provides tools for building training data and training the cross-modal alignment
component of the CMM hierarchical embedding architecture.

## Overview

The CMM embedding system requires training data that pairs items across modalities:
- Scientific text ↔ Crystal structures
- Policy documents ↔ Scientific text
- Spectral data (XRD/XRF) ↔ Text descriptions
- Molecular structures ↔ Property text

Since no such corpus exists for CMM, this module provides tools to construct one using:

1. **Direct database pairing** - Materials Project, ICSD, etc.
2. **LLM synthetic pairing** - Use LLMs to generate bridging descriptions
3. **Weak supervision** - Entity co-occurrence across modalities
4. **Metadata matching** - Same mineral/formula across modalities

## Installation

```bash
# Core dependencies
pip install torch numpy httpx

# For Materials Project access
pip install mp-api

# For training
pip install transformers accelerate wandb

# For spectral processing
pip install scipy

# For molecular processing
pip install torch-geometric rdkit
```

## Quick Start

### 1. Build Training Corpus

```python
import asyncio
from embedding.training import CMMCorpusBuilder, CorpusBuilderConfig

# Configure data sources
config = CorpusBuilderConfig(
    output_dir="./cmm_corpus",
    materials_project_api_key="your_mp_api_key",  # Get from materialsproject.org
    usgs_data_dir="/path/to/usgs/reports",        # Pre-downloaded USGS PDFs (converted to text)
    spectrum_data_dir="/path/to/spectra",         # XRD/XRF spectrum files
)

# Build corpus
builder = CMMCorpusBuilder(config)
corpus = asyncio.run(builder.build_corpus())

# Save
corpus.save("cmm_training_corpus.jsonl")

# Check statistics
print(corpus.get_statistics())
```

### 2. Load Data for Training

```python
from embedding.training import (
    CMMPairedDataset,
    create_contrastive_dataloader,
)

# Create dataset
dataset = CMMPairedDataset(
    "cmm_training_corpus.jsonl",
    min_confidence=0.3,  # Filter low-confidence pairs
    augment=True,        # Enable data augmentation
)

print(f"Dataset size: {len(dataset)}")
print(f"Modality distribution: {dataset.get_modality_distribution()}")

# Create dataloader with balanced sampling
dataloader = create_contrastive_dataloader(
    dataset,
    batch_size=32,
    sampler_type="balanced",  # Balance modality pairs
    num_workers=4,
)

# Iterate
for batch in dataloader:
    print(f"Batch modalities: {batch.modality_a_types}")
    print(f"Confidence scores: {batch.confidence_scores}")
    break
```

### 3. Train Cross-Modal Alignment

```python
from embedding.training import (
    CrossModalAlignmentTrainer,
    TrainingConfig,
)

# Configure training
config = TrainingConfig(
    embedding_dim=768,
    projection_dim=512,
    learning_rate=1e-4,
    batch_size=32,
    max_steps=50000,
    use_wandb=True,
    wandb_project="cmm-embedding",
)

# Create trainer (model must be created separately)
trainer = CrossModalAlignmentTrainer(model, config)

# Train
trainer.train(train_dataloader, val_dataloader)
```

## Data Sources

### Materials Project

The Materials Project provides DFT-computed properties for ~150,000 materials.

```python
from embedding.training.corpus_builder import MaterialsProjectConnector

connector = MaterialsProjectConnector(api_key="your_key")

async for item in connector.fetch_items(limit=1000):
    print(f"Modality: {item.modality}")
    print(f"Formula: {item.metadata.get('formula')}")
```

### USGS Mineral Reports

USGS publishes annual Mineral Commodity Summaries and technical reports.

1. Download PDFs from https://www.usgs.gov/centers/national-minerals-information-center
2. Convert to text using `pdftotext` or similar
3. Point `usgs_data_dir` to the text files

### Federal Register (Policy Documents)

Automatically fetched via public API - no setup required.

### Spectral Databases

For XRD/XRF/Raman spectra:
- RRUFF Database: https://rruff.info/
- Download `.txt` spectrum files
- Point `spectrum_data_dir` to the directory

## Pairing Strategies

### Entity Co-occurrence (Weak Supervision)

Pairs items that mention the same CMM entities (minerals, countries, companies).

```python
from embedding.training.corpus_builder import EntityCooccurrenceStrategy

strategy = EntityCooccurrenceStrategy(
    entity_list=["lithium", "cobalt", "DRC", "CATL"],
    min_overlap=2,  # At least 2 shared entities
)
```

**Confidence**: 0.3-0.8 (based on overlap ratio)

### LLM Synthetic Pairing

Uses an LLM to assess relationships and generate bridging descriptions.

```python
from embedding.training.corpus_builder import LLMSyntheticPairingStrategy
import anthropic

client = anthropic.AsyncAnthropic()
strategy = LLMSyntheticPairingStrategy(
    llm_client=client,
    model="claude-sonnet-4-20250514",
)
```

**Confidence**: 0.3-1.0 (from LLM assessment)

### Metadata Matching

Pairs items with matching metadata fields (formula, mineral name, etc.).

```python
from embedding.training.corpus_builder import MetadataMatchStrategy

strategy = MetadataMatchStrategy(
    match_fields=["formula", "mineral_name", "material_id"]
)
```

**Confidence**: 0.7-0.95 (based on match quality)

## Corpus Format

The corpus is stored as JSONL with one pair per line:

```json
{
  "pair_id": "eco_000001",
  "modality_a": {
    "modality": "text_scientific",
    "content": "LiCoO2 is a layered oxide...",
    "content_hash": "a1b2c3d4...",
    "source": "materials_project",
    "source_id": "mp-22526_text",
    "metadata": {"formula": "LiCoO2"}
  },
  "modality_b": {
    "modality": "crystal_structure",
    "content": {"lattice": {...}, "sites": [...]},
    "content_hash": "e5f6g7h8...",
    "source": "materials_project",
    "source_id": "mp-22526",
    "metadata": {"formula": "LiCoO2", "band_gap": 2.1}
  },
  "pairing_method": "metadata_match",
  "confidence_score": 0.95,
  "bridging_entities": ["LiCoO2", "lithium", "cobalt"],
  "bridging_text": null,
  "human_validated": false,
  "created_at": "2026-01-19T12:00:00"
}
```

## Training Details

### Contrastive Loss

Uses InfoNCE loss (same as CLIP):

```
L = -log(exp(sim(a,b)/τ) / Σ exp(sim(a,b_neg)/τ))
```

Where:
- `sim(a,b)` is cosine similarity
- `τ` is temperature (learnable, default 0.07)
- Negatives are other items in the batch

### Confidence Weighting

Higher-confidence pairs contribute more to the loss:

```python
loss = (loss_per_example * confidence_scores).mean()
```

### Hard Negative Mining

Optional: use entity overlap to find hard negatives:

```python
dataset = HardNegativeDataset(
    "corpus.jsonl",
    num_hard_negatives=5,
)
```

## Best Practices

### 1. Start with High-Confidence Data

```python
# Phase 1: Train on metadata matches (high precision)
dataset = CMMPairedDataset("corpus.jsonl", min_confidence=0.8)

# Phase 2: Add entity co-occurrence (more data)
dataset = CMMPairedDataset("corpus.jsonl", min_confidence=0.5)

# Phase 3: Include LLM pairs (most data)
dataset = CMMPairedDataset("corpus.jsonl", min_confidence=0.3)
```

### 2. Balance Modality Pairs

Use the balanced sampler to prevent overfitting to common pairs:

```python
dataloader = create_contrastive_dataloader(
    dataset,
    sampler_type="balanced",
)
```

### 3. Human Validation

For critical pairs, add human validation:

```python
# Load corpus
corpus = TrainingCorpus.load("corpus.jsonl")

# Mark validated pairs
for pair in corpus.pairs[:100]:  # Review top 100
    pair.human_validated = True
    pair.confidence_score = 1.0  # Human-validated = high confidence

corpus.save("corpus_validated.jsonl")
```

### 4. Incremental Corpus Building

Build corpus in stages:

```bash
# Stage 1: Materials Project only
python -m embedding.training.corpus_builder \
    --mp-key YOUR_KEY \
    -o corpus_mp.jsonl

# Stage 2: Add USGS
python -m embedding.training.corpus_builder \
    --mp-key YOUR_KEY \
    --usgs-dir /path/to/usgs \
    -o corpus_mp_usgs.jsonl

# Stage 3: Add spectra
python -m embedding.training.corpus_builder \
    --mp-key YOUR_KEY \
    --usgs-dir /path/to/usgs \
    --spectrum-dir /path/to/spectra \
    -o corpus_full.jsonl
```

## Evaluation

After training, evaluate on held-out data:

```python
from embedding.training import create_evaluation_dataloader

eval_dataloader = create_evaluation_dataloader(
    "corpus_test.jsonl",
    batch_size=32,
)

# Compute retrieval metrics
for batch in eval_dataloader:
    embeddings_a, embeddings_b = model(batch)
    # Compute recall@k, MRR, etc.
```

## File Structure

```
embedding/training/
├── __init__.py              # Package exports
├── README.md                # This file
├── corpus_builder.py        # Corpus construction tools
├── paired_data_loader.py    # PyTorch data loading
├── alignment_training.py    # Training loop
└── data/                    # Training data (gitignored)
    ├── cmm_text_corpus/
    ├── paired_data/
    └── kg_triples/
```

## Environment Variables

```bash
# Required for Materials Project
export MP_API_KEY="your_materials_project_key"

# Optional: paths to data
export USGS_DATA_DIR="/path/to/usgs"
export SPECTRUM_DATA_DIR="/path/to/spectra"

# Optional: for LLM pairing
export ANTHROPIC_API_KEY="your_anthropic_key"
```

## Troubleshooting

### "No pairs generated"

- Check that data sources are accessible
- Verify API keys
- Lower `min_overlap` in entity co-occurrence strategy

### "CUDA out of memory"

- Reduce `batch_size` in training config
- Enable gradient accumulation
- Use mixed precision (`use_mixed_precision=True`)

### "Low confidence scores"

- Add more CMM entities to the entity list
- Use LLM pairing for higher-quality pairs
- Manually curate high-value pairs
