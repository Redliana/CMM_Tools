"""Aggregate evaluation metrics and generate reports."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from statistics import mean

from cmm_fine_tune.models import EvaluationReport, GoldQAPair, ScoreResult

logger = logging.getLogger(__name__)


def build_report(
    model_id: str,
    adapter_path: str,
    golds: list[GoldQAPair],
    scores: list[ScoreResult],
) -> EvaluationReport:
    """Aggregate individual scores into an EvaluationReport."""
    assert len(golds) == len(scores)

    # Group scores by various dimensions
    by_level: dict[str, list[float]] = defaultdict(list)
    by_subdomain: dict[str, list[float]] = defaultdict(list)
    by_commodity: dict[str, list[float]] = defaultdict(list)

    for gold, score in zip(golds, scores, strict=False):
        by_level[gold.complexity_level].append(score.score)
        by_subdomain[gold.subdomain].append(score.score)
        by_commodity[gold.commodity].append(score.score)

    all_scores = [s.score for s in scores]
    all_rouge = [s.rouge_l for s in scores]

    return EvaluationReport(
        model_id=model_id,
        adapter_path=adapter_path,
        total_questions=len(scores),
        mean_score=mean(all_scores) if all_scores else 0.0,
        scores_by_level={k: mean(v) for k, v in sorted(by_level.items())},
        scores_by_subdomain={k: mean(v) for k, v in sorted(by_subdomain.items())},
        scores_by_commodity={k: mean(v) for k, v in sorted(by_commodity.items())},
        mean_rouge_l=mean(all_rouge) if all_rouge else 0.0,
        individual_scores=scores,
    )


def write_report(report: EvaluationReport, output_dir: Path) -> None:
    """Write evaluation report as JSON and markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    logger.info("JSON report written to %s", json_path)

    # Markdown
    md_path = output_dir / "evaluation_report.md"
    md = _render_markdown(report)
    with open(md_path, "w") as f:
        f.write(md)
    logger.info("Markdown report written to %s", md_path)


def _render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# CMM Evaluation Report",
        "",
        f"**Model**: `{report.model_id}`",
    ]
    if report.adapter_path:
        lines.append(f"**Adapter**: `{report.adapter_path}`")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total questions | {report.total_questions} |",
            f"| Mean score | {report.mean_score:.3f} |",
            f"| Mean ROUGE-L | {report.mean_rouge_l:.3f} |",
            "",
        ]
    )

    if report.scores_by_level:
        lines.extend(
            [
                "## Scores by Complexity Level",
                "",
                "| Level | Mean Score |",
                "|-------|-----------|",
            ]
        )
        for level, score in report.scores_by_level.items():
            lines.append(f"| {level} | {score:.3f} |")
        lines.append("")

    if report.scores_by_commodity:
        lines.extend(
            [
                "## Scores by Commodity",
                "",
                "| Commodity | Mean Score |",
                "|-----------|-----------|",
            ]
        )
        for commodity, score in report.scores_by_commodity.items():
            lines.append(f"| {commodity} | {score:.3f} |")
        lines.append("")

    if report.scores_by_subdomain:
        lines.extend(
            [
                "## Scores by Subdomain",
                "",
                "| Subdomain | Mean Score |",
                "|-----------|-----------|",
            ]
        )
        for subdomain, score in report.scores_by_subdomain.items():
            lines.append(f"| {subdomain} | {score:.3f} |")
        lines.append("")

    return "\n".join(lines) + "\n"
