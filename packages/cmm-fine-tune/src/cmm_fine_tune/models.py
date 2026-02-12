"""Pydantic models for the CMM fine-tuning pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Input data models (normalized from heterogeneous CSVs)
# ---------------------------------------------------------------------------


class TradeFlowRecord(BaseModel):
    """A single UN Comtrade trade flow record."""

    commodity: str
    hs_code: str
    reporter_code: str
    reporter_desc: str = ""
    partner_code: str
    partner_desc: str = ""
    flow_code: str  # "M" or "X"
    year: int
    primary_value: float | None = None  # USD
    net_weight: float | None = None  # kg
    quantity: float | None = None
    qty_unit: str = ""


class SalientRecord(BaseModel):
    """A row from a USGS MCS salient statistics CSV.

    Because each commodity has a different schema, we store the variable
    columns in the ``fields`` dict.  The three fixed columns
    (DataSource, Commodity, Year) are promoted to typed attributes.
    """

    data_source: str  # e.g. "MCS2023"
    commodity: str
    year: int
    fields: dict[str, str | float | None] = Field(default_factory=dict)


class WorldProductionRecord(BaseModel):
    """A row from a USGS MCS world mine production CSV."""

    source: str  # e.g. "MCS2023"
    commodity: str  # derived from filename
    country: str
    production_type: str  # e.g. "Mine production, metric tons of contained cobalt"
    production_year1: float | None = None  # e.g. Prod_t_2021
    production_year1_label: str = ""  # e.g. "2021"
    production_year2: float | None = None  # e.g. Prod_t_est_2022
    production_year2_label: str = ""  # e.g. "2022 (est.)"
    production_notes: str = ""
    reserves: float | None = None
    reserves_notes: str = ""


# ---------------------------------------------------------------------------
# Q&A intermediate models
# ---------------------------------------------------------------------------


class QAPair(BaseModel):
    """A generated question-answer pair for training."""

    question: str
    answer: str
    commodity: str
    complexity_level: str  # "L1", "L2", "L3"
    template_id: str  # for provenance tracking
    source_data: dict = Field(default_factory=dict)  # raw values used


# ---------------------------------------------------------------------------
# Chat / JSONL output models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str  # "system", "user", "assistant"
    content: str


class ChatExample(BaseModel):
    """A full chat example for mlx-lm training."""

    messages: list[ChatMessage]


# ---------------------------------------------------------------------------
# Evaluation models
# ---------------------------------------------------------------------------


class GoldQAPair(BaseModel):
    """A gold-standard Q&A pair for evaluation."""

    id: str
    question: str
    reference_answer: str
    complexity_level: str  # "L1", "L2", "L3", "L4"
    subdomain: str  # e.g. "trade_flow", "production", "reserves"
    commodity: str
    temporal_stratum: str = ""  # "A" (historical) or "B" (recent)
    required_elements: list[str] = Field(default_factory=list)
    disqualifying_errors: list[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    """Scoring result for a single Q&A pair."""

    gold_id: str
    score: float  # 0.0, 0.25, 0.50, 0.75, 1.0
    rouge_l: float = 0.0
    generated_answer: str = ""
    reasoning: str = ""


class EvaluationReport(BaseModel):
    """Aggregated evaluation report."""

    model_id: str
    adapter_path: str = ""
    total_questions: int = 0
    mean_score: float = 0.0
    scores_by_level: dict[str, float] = Field(default_factory=dict)
    scores_by_subdomain: dict[str, float] = Field(default_factory=dict)
    scores_by_commodity: dict[str, float] = Field(default_factory=dict)
    mean_rouge_l: float = 0.0
    individual_scores: list[ScoreResult] = Field(default_factory=list)
