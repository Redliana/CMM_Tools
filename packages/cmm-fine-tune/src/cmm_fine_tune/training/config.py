"""Pydantic training config model with YAML loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LoRAParameters(BaseModel):
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    scale: float = 2.0


class LRSchedule(BaseModel):
    name: str = "cosine_decay"
    warmup: int = 100
    arguments: list[float] = Field(default_factory=lambda: [1e-5, 1000, 1e-7])


class TrainingConfig(BaseModel):
    """Configuration for mlx_lm.lora fine-tuning."""

    model: str = "mlx-community/phi-4-bf16"
    data: str = "data"
    adapter_path: str = "adapters/phi4_lora"

    lora_layers: int = 16
    lora_parameters: LoRAParameters = Field(default_factory=LoRAParameters)

    batch_size: int = 4
    iters: int = 1000
    val_batches: int = 25
    steps_per_report: int = 10
    steps_per_eval: int = 100
    save_every: int = 200

    learning_rate: float = 1e-5
    lr_schedule: LRSchedule | None = None

    grad_checkpoint: bool = True
    max_seq_length: int = 2048
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_mlx_config_dict(self) -> dict[str, Any]:
        """Export as a flat dict suitable for mlx_lm.lora --config."""
        d = self.model_dump()
        # Flatten lora_parameters
        lp = d.pop("lora_parameters")
        d["lora_parameters"] = lp
        return d


def load_config(path: Path) -> TrainingConfig:
    return TrainingConfig.from_yaml(path)
