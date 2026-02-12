"""Shared fixtures for cmm-fine-tune tests."""

from __future__ import annotations

from typing import Any

import pytest

from cmm_fine_tune.models import (
    ChatExample,
    ChatMessage,
    EvaluationReport,
    GoldQAPair,
    QAPair,
    SalientRecord,
    ScoreResult,
    TradeFlowRecord,
    WorldProductionRecord,
)

# ---------------------------------------------------------------------------
# TradeFlowRecord fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_trade_record() -> TradeFlowRecord:
    """Create a sample trade flow record for testing.

    Returns:
        A fully populated TradeFlowRecord for cobalt.
    """
    return TradeFlowRecord(
        commodity="cobalt",
        hs_code="8105",
        reporter_code="842",
        reporter_desc="United States",
        partner_code="156",
        partner_desc="China",
        flow_code="M",
        year=2023,
        primary_value=1_500_000.0,
        net_weight=50_000.0,
        quantity=50_000.0,
        qty_unit="kg",
    )


@pytest.fixture()
def sample_trade_records() -> list[TradeFlowRecord]:
    """Create a list of trade flow records spanning multiple years.

    Returns:
        A list of TradeFlowRecords for testing aggregation logic.
    """
    return [
        TradeFlowRecord(
            commodity="cobalt",
            hs_code="8105",
            reporter_code="842",
            reporter_desc="United States",
            partner_code="156",
            partner_desc="China",
            flow_code="M",
            year=2022,
            primary_value=1_200_000.0,
            net_weight=40_000.0,
        ),
        TradeFlowRecord(
            commodity="cobalt",
            hs_code="8105",
            reporter_code="842",
            reporter_desc="United States",
            partner_code="156",
            partner_desc="China",
            flow_code="M",
            year=2023,
            primary_value=1_500_000.0,
            net_weight=50_000.0,
        ),
        TradeFlowRecord(
            commodity="cobalt",
            hs_code="282200",
            reporter_code="842",
            reporter_desc="United States",
            partner_code="0",
            partner_desc="World",
            flow_code="X",
            year=2023,
            primary_value=800_000.0,
            net_weight=20_000.0,
        ),
    ]


# ---------------------------------------------------------------------------
# SalientRecord fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_salient_record() -> SalientRecord:
    """Create a sample USGS salient statistics record.

    Returns:
        A SalientRecord with typical fields.
    """
    return SalientRecord(
        data_source="MCS2023",
        commodity="cobalt",
        year=2022,
        fields={
            "USProd_t": 800.0,
            "ImportsForConsumption_t": 12_000.0,
            "ExportsTrade_t": 5_000.0,
            "NIR_pct": 76.0,
        },
    )


# ---------------------------------------------------------------------------
# WorldProductionRecord fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_world_production_record() -> WorldProductionRecord:
    """Create a sample world production record.

    Returns:
        A WorldProductionRecord for cobalt in the DRC.
    """
    return WorldProductionRecord(
        source="MCS2023",
        commodity="cobalt",
        country="Congo (Kinshasa)",
        production_type="Mine production, metric tons of contained cobalt",
        production_year1=130_000.0,
        production_year1_label="2021",
        production_year2=140_000.0,
        production_year2_label="2022 (est.)",
        reserves=3_500_000.0,
    )


# ---------------------------------------------------------------------------
# QAPair fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_qa_pair() -> QAPair:
    """Create a sample QA pair for testing.

    Returns:
        A QAPair with commodity, complexity level, and source data.
    """
    return QAPair(
        question="What is the total cobalt trade value?",
        answer="The total cobalt trade value was $1.5 million in 2023.",
        commodity="cobalt",
        complexity_level="L1",
        template_id="trade_total_value",
        source_data={"commodity": "cobalt", "year": 2023, "value_usd": 1_500_000},
    )


@pytest.fixture()
def sample_qa_pairs() -> list[QAPair]:
    """Create a list of QA pairs covering multiple commodities and levels.

    Returns:
        A list of QAPairs suitable for splitter and formatter tests.
    """
    pairs = []
    for commodity in ["cobalt", "lithium", "nickel"]:
        for level in ["L1", "L2"]:
            for i in range(4):
                pairs.append(
                    QAPair(
                        question=f"Question about {commodity} {level} #{i}?",
                        answer=f"Answer about {commodity} {level} #{i}.",
                        commodity=commodity,
                        complexity_level=level,
                        template_id=f"test_{commodity}_{level}_{i}",
                        source_data={"commodity": commodity},
                    )
                )
    return pairs


# ---------------------------------------------------------------------------
# GoldQAPair / ScoreResult fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_gold_qa_pair() -> GoldQAPair:
    """Create a sample gold-standard QA pair for evaluation tests.

    Returns:
        A GoldQAPair with required elements and disqualifying errors.
    """
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


@pytest.fixture()
def sample_score_result() -> ScoreResult:
    """Create a sample score result.

    Returns:
        A ScoreResult with a perfect score.
    """
    return ScoreResult(
        gold_id="test_001",
        score=1.0,
        rouge_l=0.95,
        generated_answer="US mine production of cobalt in 2022 was 800 metric tons.",
        reasoning="Elements matched: 4/4 (100%)",
    )


# ---------------------------------------------------------------------------
# ChatExample fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_chat_example() -> ChatExample:
    """Create a sample chat example for JSONL formatting tests.

    Returns:
        A ChatExample with system, user, and assistant messages.
    """
    return ChatExample(
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is cobalt used for?"),
            ChatMessage(
                role="assistant",
                content="Cobalt is used in lithium-ion batteries, superalloys, and catalysts.",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# EvaluationReport fixture factory
# ---------------------------------------------------------------------------


def _make_evaluation_report(
    model_id: str = "test-model",
    adapter_path: str = "adapters/test",
    total_questions: int = 10,
    mean_score: float = 0.75,
    mean_rouge_l: float = 0.65,
    scores_by_level: dict[str, float] | None = None,
    scores_by_commodity: dict[str, float] | None = None,
    scores_by_subdomain: dict[str, float] | None = None,
    individual_scores: list[ScoreResult] | None = None,
) -> EvaluationReport:
    """Factory for building EvaluationReport instances in tests.

    Args:
        model_id: Model identifier string.
        adapter_path: Path to adapter weights.
        total_questions: Number of questions evaluated.
        mean_score: Overall mean score.
        mean_rouge_l: Mean ROUGE-L score.
        scores_by_level: Score breakdown by complexity level.
        scores_by_commodity: Score breakdown by commodity.
        scores_by_subdomain: Score breakdown by subdomain.
        individual_scores: List of individual ScoreResult objects.

    Returns:
        Fully constructed EvaluationReport.
    """
    return EvaluationReport(
        model_id=model_id,
        adapter_path=adapter_path,
        total_questions=total_questions,
        mean_score=mean_score,
        mean_rouge_l=mean_rouge_l,
        scores_by_level=scores_by_level or {"L1": 0.8, "L2": 0.7},
        scores_by_commodity=scores_by_commodity or {"cobalt": 0.75, "lithium": 0.8},
        scores_by_subdomain=scores_by_subdomain or {"production": 0.8, "trade_flow": 0.7},
        individual_scores=individual_scores or [],
    )


@pytest.fixture()
def make_evaluation_report() -> Any:
    """Expose the _make_evaluation_report factory as a fixture.

    Returns:
        The _make_evaluation_report callable.
    """
    return _make_evaluation_report


@pytest.fixture()
def sample_evaluation_report() -> EvaluationReport:
    """Create a sample evaluation report.

    Returns:
        An EvaluationReport populated with test data.
    """
    return _make_evaluation_report()
