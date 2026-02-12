"""CLI entry point: end-to-end evaluation pipeline.

Usage:
  cmm-evaluate --model mlx-community/phi-4-bf16 \
    [--adapter adapters/phi4_lora/] \
    --gold-qa gold_qa/gold_qa_pairs.jsonl \
    --output results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cmm_fine_tune.evaluation.gold_qa_loader import load_gold_qa
from cmm_fine_tune.evaluation.inference import run_inference
from cmm_fine_tune.evaluation.report import build_report, write_report
from cmm_fine_tune.evaluation.scorer import score_all

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CMM fine-tuned model")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/phi-4-bf16",
        help="MLX model ID or path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--gold-qa",
        type=Path,
        required=True,
        help="Path to gold Q&A JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for reports",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        # 1. Load gold Q&A
        golds = load_gold_qa(args.gold_qa)
        if not golds:
            logger.error("No gold Q&A pairs found in %s", args.gold_qa)
            sys.exit(1)

        # 2. Run inference
        logger.info("Running inference on %d questions...", len(golds))
        answers = run_inference(
            model_id=args.model,
            questions=golds,
            adapter_path=args.adapter,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # 3. Score
        logger.info("Scoring answers...")
        scores = score_all(golds, answers)

        # 4. Build and write report
        report = build_report(
            model_id=args.model,
            adapter_path=args.adapter or "",
            golds=golds,
            scores=scores,
        )
        write_report(report, args.output)

        # Print summary
        print("\nEvaluation complete:")
        print(f"  Total questions: {report.total_questions}")
        print(f"  Mean score: {report.mean_score:.3f}")
        print(f"  Mean ROUGE-L: {report.mean_rouge_l:.3f}")
        print(f"  Reports written to: {args.output}")

    except Exception:
        logger.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
