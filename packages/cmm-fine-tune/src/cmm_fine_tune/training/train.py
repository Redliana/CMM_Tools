"""CLI wrapper around mlx_lm.lora for CMM fine-tuning.

Usage: cmm-train --config configs/phi4_lora.yaml [--resume adapters/checkpoint/]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from cmm_fine_tune.training.config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Phi-4 with LoRA on CMM data")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (e.g. configs/phi4_lora.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to adapter checkpoint to resume training from",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config)
    logger.info("Loaded config: model=%s, adapter=%s", config.model, config.adapter_path)

    # Ensure adapter directory exists
    adapter_dir = Path(config.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Write a resolved YAML config for mlx_lm.lora
    resolved_config_path = adapter_dir / "training_config.yaml"
    config_dict = config.to_mlx_config_dict()

    if args.resume:
        config_dict["resume_adapter_file"] = str(args.resume)
        logger.info("Resuming from %s", args.resume)

    with open(resolved_config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info("Resolved config written to %s", resolved_config_path)

    # Launch mlx_lm.lora
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(resolved_config_path)]
    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True)
        logger.info("Training completed successfully (exit code %d)", result.returncode)
    except subprocess.CalledProcessError as e:
        logger.error("Training failed with exit code %d", e.returncode)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
