"""Load gold-standard Q&A pairs from JSONL for evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cmm_fine_tune.models import GoldQAPair

logger = logging.getLogger(__name__)


def load_gold_qa(path: Path) -> list[GoldQAPair]:
    """Load gold Q&A pairs from a JSONL file.

    Each line must be a JSON object matching the GoldQAPair schema.
    """
    pairs: list[GoldQAPair] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pairs.append(GoldQAPair(**data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Skipping invalid gold QA line %d: %s", i, e)
    logger.info("Loaded %d gold Q&A pairs from %s", len(pairs), path)
    return pairs
