"""CLI entry point: end-to-end data preparation pipeline.

Usage: cmm-prepare --output-dir data/ [--seed 42] [--train-ratio 0.85] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from cmm_fine_tune.data.csv_loader import load_all_data
from cmm_fine_tune.data.jsonl_formatter import write_jsonl
from cmm_fine_tune.data.qa_generator import generate_all_qa
from cmm_fine_tune.data.splitter import stratified_split

logger = logging.getLogger(__name__)


def run_pipeline(
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.85,
    valid_ratio: float = 0.10,
    test_ratio: float = 0.05,
    dry_run: bool = False,
) -> dict:
    """Execute the full data preparation pipeline.

    Returns a summary dict with statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load all raw data
    logger.info("Loading raw data...")
    all_data = load_all_data()
    trade_data = all_data["trade"]
    salient_data = all_data["salient"]
    world_data = all_data["world"]

    total_trade = sum(len(v) for v in trade_data.values())
    total_salient = sum(len(v) for v in salient_data.values())
    total_world = sum(len(v) for v in world_data.values())
    logger.info(
        "Loaded: %d trade, %d salient, %d world production records",
        total_trade,
        total_salient,
        total_world,
    )

    # 2. Generate Q&A pairs
    logger.info("Generating Q&A pairs...")
    all_pairs = generate_all_qa(trade_data, salient_data, world_data)

    # Compute statistics
    by_commodity = Counter(p.commodity for p in all_pairs)
    by_level = Counter(p.complexity_level for p in all_pairs)
    by_template = Counter(p.template_id for p in all_pairs)

    summary = {
        "total_pairs": len(all_pairs),
        "by_commodity": dict(by_commodity.most_common()),
        "by_complexity_level": dict(by_level.most_common()),
        "by_template": dict(by_template.most_common()),
        "source_records": {
            "trade": total_trade,
            "salient": total_salient,
            "world_production": total_world,
        },
        "seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "valid": valid_ratio,
            "test": test_ratio,
        },
    }

    if dry_run:
        logger.info("Dry run -- not writing files.")
        print(json.dumps(summary, indent=2))
        return summary

    # 3. Split
    logger.info("Splitting into train/valid/test...")
    train_pairs, valid_pairs, test_pairs = stratified_split(
        all_pairs,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    summary["split_counts"] = {
        "train": len(train_pairs),
        "valid": len(valid_pairs),
        "test": len(test_pairs),
    }

    # 4. Write JSONL files
    logger.info("Writing JSONL files...")
    write_jsonl(train_pairs, output_dir / "train.jsonl")
    write_jsonl(valid_pairs, output_dir / "valid.jsonl")
    write_jsonl(test_pairs, output_dir / "test.jsonl")

    # 5. Write summary
    summary_path = output_dir / "preparation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", summary_path)

    logger.info(
        "Done! Wrote %d train, %d valid, %d test examples to %s",
        len(train_pairs),
        len(valid_pairs),
        len(test_pairs),
        output_dir,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CMM training data from raw CSVs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write train/valid/test JSONL (default: data/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing files")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = run_pipeline(
            output_dir=args.output_dir,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            dry_run=args.dry_run,
        )
        print(f"\nTotal Q&A pairs: {summary['total_pairs']}")
        if "split_counts" in summary:
            sc = summary["split_counts"]
            print(f"  Train: {sc['train']}, Valid: {sc['valid']}, Test: {sc['test']}")
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
