"""Tests for evaluation scorer."""

from __future__ import annotations

import pytest

from cmm_fine_tune.evaluation.scorer import (
    _element_present,
    score_all,
    score_answer,
)
from cmm_fine_tune.models import GoldQAPair


@pytest.fixture
def gold_full():
    return GoldQAPair(
        id="test_001",
        question="What was US cobalt production in 2022?",
        reference_answer="US mine production of cobalt in 2022 was 800 metric tons.",
        complexity_level="L1",
        subdomain="production",
        commodity="cobalt",
        required_elements=["800", "metric tons", "2022", "cobalt"],
        disqualifying_errors=["lithium"],
    )


class TestElementPresent:
    def test_exact_match(self):
        assert _element_present("cobalt", "US cobalt production was high")

    def test_case_insensitive(self):
        assert _element_present("Cobalt", "us cobalt production")

    def test_numeric_match(self):
        assert _element_present("130,000", "The value was 130000 tons")

    def test_numeric_comma_format(self):
        assert _element_present("130000", "The value was 130,000 tons")

    def test_not_present(self):
        assert not _element_present("lithium", "cobalt production was 800 tons")


class TestScoreAnswer:
    def test_perfect_score(self, gold_full):
        answer = "US mine production of cobalt in 2022 was 800 metric tons."
        result = score_answer(gold_full, answer)
        assert result.score == 1.0
        assert result.gold_id == "test_001"

    def test_disqualifying_error(self, gold_full):
        answer = "Lithium production in 2022 was 800 metric tons of cobalt."
        result = score_answer(gold_full, answer)
        assert result.score == 0.0
        assert "Disqualifying" in result.reasoning

    def test_partial_score(self, gold_full):
        answer = "Cobalt production was significant in 2022."
        result = score_answer(gold_full, answer)
        assert 0.0 < result.score < 1.0

    def test_zero_elements_matched(self, gold_full):
        answer = "I don't know the answer to that question."
        result = score_answer(gold_full, answer)
        assert result.score == 0.0

    def test_no_required_elements_uses_rouge(self):
        gold = GoldQAPair(
            id="test_002",
            question="Describe cobalt supply chains.",
            reference_answer="Cobalt supply chains involve mining in DRC and processing in China.",
            complexity_level="L3",
            subdomain="supply_chain",
            commodity="cobalt",
            required_elements=[],
            disqualifying_errors=[],
        )
        result = score_answer(
            gold, "Cobalt supply chains involve mining in DRC and refining in China."
        )
        assert result.score > 0.0
        assert result.rouge_l > 0.0


class TestScoreAll:
    def test_score_multiple(self, gold_full):
        golds = [gold_full, gold_full]
        answers = [
            "US cobalt production in 2022 was 800 metric tons.",
            "I don't know.",
        ]
        results = score_all(golds, answers)
        assert len(results) == 2
        assert results[0].score > results[1].score
