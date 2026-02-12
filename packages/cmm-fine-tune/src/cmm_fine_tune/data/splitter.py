"""Stratified train/valid/test splitting of QAPairs."""

from __future__ import annotations

import logging

from sklearn.model_selection import train_test_split

from cmm_fine_tune.models import QAPair

logger = logging.getLogger(__name__)


def _stratification_key(pair: QAPair) -> str:
    """Create a composite stratification key from commodity + complexity_level."""
    return f"{pair.commodity}_{pair.complexity_level}"


def stratified_split(
    pairs: list[QAPair],
    train_ratio: float = 0.85,
    valid_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[QAPair], list[QAPair], list[QAPair]]:
    """Split QAPairs into train/valid/test with stratification.

    Stratifies on commodity + complexity_level to ensure each split has
    proportional representation of all categories.

    Returns (train, valid, test) tuple.
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    strat_keys = [_stratification_key(p) for p in pairs]

    # Check minimum samples per stratum -- need at least 2 per stratum for stratified split
    from collections import Counter

    counts = Counter(strat_keys)
    min_count = min(counts.values())
    can_stratify = min_count >= 2

    if not can_stratify:
        logger.warning(
            "Some strata have <2 samples (min=%d). Falling back to non-stratified split.",
            min_count,
        )
        strat_keys = None  # type: ignore[assignment]

    # First split: train vs (valid+test)
    holdout_ratio = valid_ratio + test_ratio
    train_pairs, holdout_pairs, _train_keys, _holdout_keys = train_test_split(
        pairs,
        strat_keys if strat_keys is not None else [None] * len(pairs),
        test_size=holdout_ratio,
        random_state=seed,
        stratify=strat_keys if can_stratify else None,
    )

    # Second split: valid vs test (from holdout)
    relative_test = test_ratio / holdout_ratio
    if can_stratify:
        holdout_strat = [_stratification_key(p) for p in holdout_pairs]
        holdout_counts = Counter(holdout_strat)
        can_stratify_holdout = min(holdout_counts.values()) >= 2
    else:
        can_stratify_holdout = False
        holdout_strat = None

    valid_pairs, test_pairs = train_test_split(
        holdout_pairs,
        test_size=relative_test,
        random_state=seed,
        stratify=holdout_strat if can_stratify_holdout else None,
    )

    logger.info(
        "Split %d pairs -> train=%d, valid=%d, test=%d",
        len(pairs),
        len(train_pairs),
        len(valid_pairs),
        len(test_pairs),
    )
    return train_pairs, valid_pairs, test_pairs
