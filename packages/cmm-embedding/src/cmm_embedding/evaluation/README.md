# CMM Embedding Evaluation Benchmark

This module provides a comprehensive evaluation framework for CMM (Critical Minerals and Materials) embeddings. Since no CMM-specific embedding benchmark exists, this framework was designed to evaluate the unique challenges of multi-scale, multi-modal embedding systems for supply chain intelligence.

## Overview

The benchmark evaluates five core capabilities:

| Category | Description | Key Challenge |
|----------|-------------|---------------|
| **Cross-Scale Retrieval** | Link atomistic data to policy documents | Bridging 10⁻¹⁰ m to 10⁷ m scales |
| **Cross-Modal Alignment** | Retrieve across modalities (spectra ↔ text) | Different data representations |
| **Entity Resolution** | Match entity mentions across variations | "TFM" = "Tenke Fungurume Mine" |
| **Supply Chain Traversal** | Multi-hop queries through supply chain | Mine → Refiner → Manufacturer → Product |
| **Temporal Consistency** | Handle time-sensitive information | Current vs historical policies |

## Quick Start

### Generate Benchmark Suite

```python
from embedding.evaluation import create_full_benchmark_suite

# Create the benchmark
suite = create_full_benchmark_suite()

# View statistics
print(suite.get_statistics())
# {
#     "total_items": 25,
#     "by_category": {
#         "cross_scale_retrieval": 4,
#         "cross_modal_alignment": 3,
#         "entity_resolution": 5,
#         ...
#     },
#     "by_difficulty": {"easy": 3, "medium": 10, "hard": 8, "expert": 4}
# }

# Save to JSON
suite.save("cmm_benchmark_v1.json")
```

### Run Evaluation

```python
from embedding.evaluation import BenchmarkRunner, EvaluationConfig

# Configure evaluation
config = EvaluationConfig(
    top_k_values=[1, 5, 10, 20, 50],
    compute_confidence_intervals=True,
)

# Create runner with your model
runner = BenchmarkRunner(my_embedding_model, config)

# Run evaluation
results = runner.run_full_evaluation(suite, model_name="specter2_cmm_v1")

# Print summary
results.print_summary()

# Save detailed report
results.save_report("evaluation_report.json")
```

## Benchmark Categories

### 1. Cross-Scale Retrieval

Tests the ability to link information across vastly different scales.

**Example:**
- **Query (Atomistic)**: "LiCoO2 DFT calculation with formation energy -2.31 eV/atom and band gap 2.1 eV..."
- **Expected Retrieval (National Policy)**: "Section 232 cobalt import restrictions...", "DOE Battery Materials Strategy..."

**Difficulty Levels:**
- **Medium**: Material property → Trade data
- **Hard**: DFT calculation → Export control policy
- **Expert**: Quantum chemistry → Defense procurement

**Metrics:**
- Recall@K, Precision@K, MRR, NDCG@K
- Scale Bridging Score (captures conceptual distance traversed)

### 2. Cross-Modal Alignment

Tests retrieval across different data modalities.

**Example:**
- **Query (XRD Spectrum)**: Cobaltite diffraction pattern with peaks at 28.5°, 32.1°, 36.2°...
- **Expected Retrieval (Text)**: "Cobaltite (CoAsS) occurs in hydrothermal vein deposits..."

**Supported Modality Pairs:**
- Spectrum (XRD/XRF/Raman) → Scientific text
- Crystal structure → Mining/processing text
- Molecular structure → Property text
- Tabular data → Policy documents

**Metrics:**
- Cross-modal Recall@K, MRR
- Modality transfer accuracy

### 3. Entity Resolution

Tests recognition of entity variations and aliases.

**Example:**
```python
{
    "canonical_entity": "Tenke Fungurume Mine",
    "aliases": ["TFM", "Tenke-Fungurume", "Tenke Mining Corp", "腾科丰谷鲁米矿"],
    "negative_examples": ["Mutanda Mining", "Kamoto Mine"]  # Should NOT match
}
```

**Entity Types:**
- Mines (Tenke Fungurume, Escondida, Bayan Obo)
- Companies (CMOC, Glencore, CATL)
- Minerals/Elements (Neodymium/Nd/钕)
- Countries and regions

**Metrics:**
- Precision, Recall, F1 for entity clustering
- False positive rate on negative examples

### 4. Supply Chain Traversal

Tests multi-hop reasoning through the supply chain graph.

**Example:**
- **Query**: "Trace cobalt from DRC mines to Tesla vehicles"
- **Expected Path**: Tenke Fungurume → CMOC → Huayou Cobalt → CATL → Tesla Gigafactory → Tesla Model 3

**Path Types:**
- Raw material → End product (4-6 hops)
- Mine → Defense application (complex routing)
- Regional source → Global consumer

**Metrics:**
- Exact path accuracy
- Partial path score (LCS-based)
- Endpoint accuracy

### 5. Temporal Consistency

Tests handling of time-sensitive information.

**Example:**
- **Query**: "What are US sanctions on Russian aluminum?"
- **At 2024-06**: "200% tariff under Section 232 (effective March 2024)"
- **At 2023-06**: "10% tariff under Section 232"
- **At 2021-06**: "No specific sanctions"

**Test Scenarios:**
- Sanction changes over time
- Export control evolution
- Policy supersession

**Metrics:**
- Current timestamp accuracy
- Historical answer accuracy
- Temporal confusion rate

## Target Performance

Based on CMM_RAG_Implementation_Guide.md Section 12.5:

| Metric | Target | Description |
|--------|--------|-------------|
| Recall@10 | >0.85 | 85% of relevant items in top-10 |
| MRR | >0.70 | First relevant item in top ~1.4 position |
| Cross-Modal Precision | >0.80 | 80% of cross-modal matches correct |
| Entity Resolution F1 | >0.90 | High precision and recall on entities |
| Path Accuracy | >0.75 | 75% exact supply chain paths |
| Scale Bridging Score | >4.0/5.0 | Human-evaluated semantic coherence |

## Data Sources for Ground Truth

| Source | Content | Size |
|--------|---------|------|
| USGS Mineral Commodity Summaries | Policy/statistics text | ~500 docs |
| DOE Technical Reports | Scientific/policy text | ~1000 docs |
| Materials Project | DFT calculations | ~10,000 entries |
| ICSD Crystal Database | Crystal structures | ~5,000 structures |
| Federal Register | Policy documents | ~2,000 docs |
| ACLED | Conflict data | ~50,000 events |
| Expert-curated Q&A | Ground truth pairs | ~500 pairs |

## Extending the Benchmark

### Add Custom Benchmark Items

```python
from embedding.evaluation import (
    CrossScaleRetrievalItem,
    Difficulty,
    ScaleLevel,
)

# Create custom item
custom_item = CrossScaleRetrievalItem(
    item_id="custom_001",
    difficulty=Difficulty.HARD,
    description="Link graphite anode properties to supply policy",
    source_scale=ScaleLevel.MATERIAL,
    target_scale=ScaleLevel.NATIONAL,
    source_content="Natural graphite anode with 360 mAh/g capacity...",
    relevant_target_ids=["policy_graphite_china_001"],
    relevant_target_contents=["China graphite export restrictions..."],
    bridging_concepts=["graphite", "anode", "battery", "export control"],
)

# Add to suite
suite.cross_scale_items.append(custom_item)
```

### Create Custom Evaluator

```python
from embedding.evaluation import EvaluationConfig

class MyCustomEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate_item(self, item, model_fn):
        # Custom evaluation logic
        results = model_fn(item.get_query())

        # Compute custom metrics
        return {
            "item_id": item.item_id,
            "custom_metric": compute_custom_metric(results, item),
        }
```

## File Structure

```
embedding/evaluation/
├── __init__.py              # Package exports
├── README.md                # This file
├── cmm_benchmark_spec.py    # Benchmark specification
├── benchmark_runner.py      # Evaluation runner
└── data/                    # Benchmark data (optional)
    ├── cmm_benchmark_v1.json
    ├── ground_truth/
    └── test_documents/
```

## Example Output

```
============================================================
CMM EMBEDDING EVALUATION RESULTS
Model: specter2_cmm_finetuned_v1
Benchmark: CMM Embedding Benchmark v1.0
============================================================

OVERALL SCORE: 0.7842

Category Scores:
  cross_scale_retrieval: 0.7125
  cross_modal_alignment: 0.7850
  entity_resolution: 0.9234
  supply_chain_traversal: 0.6800
  temporal_consistency: 0.8200

Key Metrics:
  Cross-Scale MRR: 0.7125
  Cross-Modal MRR: 0.7850
  Entity Resolution F1: 0.9234
  Supply Chain Path Acc: 0.6800
  Temporal Consistency: 0.8200
============================================================
```

## Model Interface Requirements

Your embedding model must implement these methods:

```python
class MyEmbeddingModel:
    def retrieve(
        self,
        query: Any,
        modality: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Retrieve documents for a query.

        Args:
            query: Query content (text, spectrum, structure, etc.)
            modality: Query modality type
            top_k: Number of results to return

        Returns:
            List of (document_id, score) tuples
        """
        pass

    def resolve_entity(self, entity_name: str) -> Set[str]:
        """
        Resolve entity to all known aliases.

        Returns:
            Set of matched entity names/aliases
        """
        pass

    def traverse_supply_chain(
        self,
        query: str,
        start_entity: str,
        end_entity: Optional[str]
    ) -> List[str]:
        """
        Find supply chain path from start to end entity.

        Returns:
            List of entities in the path
        """
        pass

    def query_with_timestamp(
        self,
        query: str,
        timestamp: str
    ) -> str:
        """
        Answer query with temporal context.

        Returns:
            Answer string
        """
        pass
```

## Contributing

To add new benchmark items:

1. Identify the category (cross-scale, cross-modal, etc.)
2. Create ground truth by expert curation or verified data sources
3. Add item using the appropriate dataclass
4. Include difficulty rating and bridging concepts
5. Add to the appropriate generator in `cmm_benchmark_spec.py`

## References

- CMM_RAG_Implementation_Guide.md Section 12.5
- 260118_Claude_Opus_Embedding_Analysis.md
- SciRepEval: A Multi-Format Benchmark for Scientific Document Representations
- BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models
