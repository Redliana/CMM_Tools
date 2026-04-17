"""Unit tests for fred_mcp.models."""

from __future__ import annotations

from fred_mcp.models import CMM_FRED_SERIES, Observation, SeriesMetadata


def test_cmm_series_nonempty():
    assert len(CMM_FRED_SERIES) > 10
    assert "PCOPPUSDM" in CMM_FRED_SERIES
    assert CMM_FRED_SERIES["PCOPPUSDM"]["category"] == "commodity_prices"


def test_observation_missing_value_none():
    obs = Observation(date="2022-01-01", value=None)
    assert obs.value is None
    assert obs.date == "2022-01-01"


def test_series_metadata_roundtrip():
    meta = SeriesMetadata(
        id="GDPC1",
        title="Real Gross Domestic Product",
        units="Billions of Chained 2017 Dollars",
        frequency="Quarterly",
    )
    assert meta.id == "GDPC1"
    assert meta.model_dump()["frequency"] == "Quarterly"
