"""Criteria-based + ROUGE-L scoring for CMM evaluation.

Scoring rubric (per gold QA methodology):
  1.0  -- All required elements present, no errors
  0.75 -- Minor omission or imprecision (e.g. rounding)
  0.50 -- Partially correct; some required elements missing
  0.25 -- Minimal relevant content; major gaps
  0.0  -- Disqualifying error OR completely irrelevant
"""

from __future__ import annotations

import logging
import re

from rouge_score import rouge_scorer

from cmm_fine_tune.models import GoldQAPair, ScoreResult

logger = logging.getLogger(__name__)

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def score_answer(gold: GoldQAPair, generated: str) -> ScoreResult:
    """Score a generated answer against a gold-standard QA pair.

    1. Check for disqualifying errors (score = 0.0)
    2. Count required elements present
    3. Assign score based on coverage
    """
    generated_lower = generated.lower()
    reasoning_parts: list[str] = []

    # Step 1: Disqualifying errors
    for error in gold.disqualifying_errors:
        if error.lower() in generated_lower:
            reasoning_parts.append(f"Disqualifying error found: '{error}'")
            rouge_l = _compute_rouge_l(gold.reference_answer, generated)
            return ScoreResult(
                gold_id=gold.id,
                score=0.0,
                rouge_l=rouge_l,
                generated_answer=generated,
                reasoning="; ".join(reasoning_parts),
            )

    # Step 2: Count required elements
    total_elements = len(gold.required_elements)
    if total_elements == 0:
        # No specific elements to check -- use ROUGE-L as proxy
        rouge_l = _compute_rouge_l(gold.reference_answer, generated)
        score = _rouge_to_rubric(rouge_l)
        reasoning_parts.append(f"No required elements specified; using ROUGE-L={rouge_l:.3f}")
        return ScoreResult(
            gold_id=gold.id,
            score=score,
            rouge_l=rouge_l,
            generated_answer=generated,
            reasoning="; ".join(reasoning_parts),
        )

    matched = 0
    for element in gold.required_elements:
        if _element_present(element, generated):
            matched += 1
        else:
            reasoning_parts.append(f"Missing element: '{element}'")

    coverage = matched / total_elements
    reasoning_parts.insert(0, f"Elements matched: {matched}/{total_elements} ({coverage:.0%})")

    # Step 3: Map coverage to rubric score
    if coverage >= 0.95:
        score = 1.0
    elif coverage >= 0.70:
        score = 0.75
    elif coverage >= 0.40:
        score = 0.50
    elif coverage > 0:
        score = 0.25
    else:
        score = 0.0

    rouge_l = _compute_rouge_l(gold.reference_answer, generated)

    return ScoreResult(
        gold_id=gold.id,
        score=score,
        rouge_l=rouge_l,
        generated_answer=generated,
        reasoning="; ".join(reasoning_parts),
    )


def _element_present(element: str, text: str) -> bool:
    """Check if a required element is present in the generated text.

    Supports exact substring match (case-insensitive) and numeric tolerance.
    Elements like "130,000" will match "130000", "130,000", or "130 000".
    """
    element_lower = element.lower().strip()
    text_lower = text.lower()

    # Direct substring match
    if element_lower in text_lower:
        return True

    # Try numeric matching: extract numbers from element and check if any appear in text
    element_nums = re.findall(r"[\d,]+\.?\d*", element)
    for num_str in element_nums:
        canonical = num_str.replace(",", "")
        if canonical in text_lower.replace(",", ""):
            return True

    return False


def _compute_rouge_l(reference: str, generated: str) -> float:
    scores = _rouge.score(reference, generated)
    return scores["rougeL"].fmeasure


def _rouge_to_rubric(rouge_l: float) -> float:
    """Map ROUGE-L score to the 5-point rubric."""
    if rouge_l >= 0.7:
        return 1.0
    if rouge_l >= 0.5:
        return 0.75
    if rouge_l >= 0.3:
        return 0.50
    if rouge_l >= 0.1:
        return 0.25
    return 0.0


def score_all(golds: list[GoldQAPair], generated_answers: list[str]) -> list[ScoreResult]:
    """Score all generated answers against gold QA pairs."""
    assert len(golds) == len(generated_answers)
    return [score_answer(g, a) for g, a in zip(golds, generated_answers, strict=False)]
