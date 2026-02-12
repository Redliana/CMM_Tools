# cmm-fine-tune

Phi-4 LoRA fine-tuning pipeline for Critical Minerals and Materials (CMM) domain
knowledge. Full workflow from data preparation through training, evaluation, and
interactive chat on Apple Silicon via MLX.

## Installation

```bash
uv pip install -e packages/cmm-fine-tune
```

## Quick Start

```bash
# 1. Prepare training data (CSV -> chat JSONL)
cmm-prepare --commodities lithium cobalt --output data/

# 2. Train LoRA adapter
cmm-train --config configs/default.yaml

# 3. Evaluate against gold Q&A set
cmm-evaluate --adapter adapters/phi4_lora --gold-qa gold_qa/

# 4. Interactive chat with the fine-tuned model
cmm-chat --adapter adapters/phi4_lora
```

### Python API

```python
from cmm_fine_tune.training.config import TrainingConfig

config = TrainingConfig.from_yaml(Path("configs/default.yaml"))
print(config.model)  # "mlx-community/phi-4-bf16"
print(config.lora_parameters.rank)  # 16
```

## CLI Tools

| Command | Entry Point | Description |
|---|---|---|
| `cmm-prepare` | `cmm_fine_tune.data.prepare:main` | Convert trade/USGS CSVs to chat JSONL for training |
| `cmm-train` | `cmm_fine_tune.training.train:main` | Run LoRA fine-tuning via mlx-lm |
| `cmm-evaluate` | `cmm_fine_tune.evaluation.evaluate:main` | Score model against gold Q&A pairs |
| `cmm-chat` | `cmm_fine_tune.inference.chat:main` | Interactive chat with a fine-tuned adapter |

## API Reference

### Data Models (`cmm_fine_tune.models`)

- `TradeFlowRecord` -- UN Comtrade trade flow record.
- `SalientRecord` -- USGS MCS salient statistics row.
- `WorldProductionRecord` -- USGS MCS world mine production row.
- `QAPair` -- Generated question-answer pair for training.
- `ChatMessage`, `ChatExample` -- Chat format for mlx-lm training.
- `GoldQAPair` -- Gold-standard Q&A pair for evaluation.
- `ScoreResult`, `EvaluationReport` -- Evaluation scoring results.

### Training Config (`cmm_fine_tune.training.config`)

- `TrainingConfig` -- Pydantic model with LoRA hyperparameters, YAML I/O, and mlx-lm export.
- `LoRAParameters` -- Rank, alpha, dropout, scale settings.
- `LRSchedule` -- Learning rate schedule (cosine decay with warmup).

## Configuration

| Variable | Description | Default |
|---|---|---|
| `CMM_FINETUNE_DATA_ROOT` | Root directory containing `API_Scripts/` and `fine_tuning/` | (none) |

Expected structure: `API_Scripts/{gold_qa_data,usgs_mcs_data}` and
`fine_tuning/{data,adapters,results,configs}`.

## Dependencies

mlx, mlx-lm, pandas, pydantic, pyyaml, rouge-score, nltk, scikit-learn, rich, jinja2
