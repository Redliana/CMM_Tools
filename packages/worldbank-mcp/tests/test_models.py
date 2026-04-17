"""Unit tests for worldbank_mcp.models."""

from __future__ import annotations

from worldbank_mcp.models import (
    CMM_KEY_ECONOMIES,
    CMM_WDI_INDICATORS,
    IndicatorObservation,
)


def test_cmm_indicators_nonempty():
    assert len(CMM_WDI_INDICATORS) > 0
    assert "NY.GDP.MKTP.CD" in CMM_WDI_INDICATORS


def test_cmm_key_economies_iso3():
    for code in CMM_KEY_ECONOMIES:
        assert len(code) == 3
        assert code.isupper()


def test_indicator_observation_roundtrip():
    obs = IndicatorObservation(
        country_code="USA",
        country_name="United States",
        indicator_code="NY.GDP.MKTP.CD",
        indicator_name="GDP (current US$)",
        year=2022,
        value=2.5e13,
    )
    assert obs.country_code == "USA"
    assert obs.year == 2022
    assert obs.model_dump()["value"] == 2.5e13
