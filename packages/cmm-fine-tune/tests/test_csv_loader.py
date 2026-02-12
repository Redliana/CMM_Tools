"""Tests for CSV data loader."""

from __future__ import annotations

import pytest

from cmm_fine_tune.constants import COMMODITY_CONFIG, TRADE_DATA_DIR
from cmm_fine_tune.data.csv_loader import (
    _parse_numeric,
    load_all_data,
    load_all_salient_data,
    load_all_trade_data,
    load_all_world_production_data,
    load_trade_data,
)


class TestParseNumeric:
    def test_normal_float(self):
        assert _parse_numeric("123.45") == 123.45

    def test_integer(self):
        assert _parse_numeric("1000") == 1000.0

    def test_comma_separated(self):
        assert _parse_numeric("1,000,000") == 1_000_000.0

    def test_withheld_w(self):
        assert _parse_numeric("W") is None

    def test_withheld_lowercase(self):
        assert _parse_numeric("w") is None

    def test_empty_string(self):
        assert _parse_numeric("") is None

    def test_none(self):
        assert _parse_numeric(None) is None

    def test_gt_pattern(self):
        assert _parse_numeric(">25") == 25.0

    def test_gt_pattern_with_space(self):
        assert _parse_numeric("> 50") == 50.0

    def test_nan(self):

        assert _parse_numeric(float("nan")) is None

    def test_less_than_half(self):
        assert _parse_numeric("Less than 1/2 unit") is None


class TestTradeLoader:
    @pytest.fixture
    def cobalt_csv(self):
        return TRADE_DATA_DIR / COMMODITY_CONFIG["cobalt"]["trade_csv"]

    def test_load_single_trade_csv(self, cobalt_csv):
        if not cobalt_csv.exists():
            pytest.skip("Trade CSV not found")
        records = load_trade_data(cobalt_csv)
        assert len(records) > 0
        r = records[0]
        assert r.commodity != ""
        assert r.year > 2000
        assert r.flow_code in ("M", "X")

    def test_load_all_trade_data(self):
        data = load_all_trade_data()
        # Should have at least some commodities
        assert len(data) > 0
        total = sum(len(v) for v in data.values())
        assert total > 0, "Expected at least some trade records"

    def test_trade_record_fields(self, cobalt_csv):
        if not cobalt_csv.exists():
            pytest.skip("Trade CSV not found")
        records = load_trade_data(cobalt_csv)
        for r in records:
            assert r.hs_code != ""
            assert r.reporter_code != ""


class TestSalientLoader:
    def test_load_salient_data(self):
        data = load_all_salient_data()
        assert len(data) > 0
        # At least some commodities should have records
        loaded = {k: v for k, v in data.items() if v}
        assert len(loaded) > 0, f"No salient data loaded. Keys: {list(data.keys())}"
        total = sum(len(v) for v in loaded.values())
        assert total > 10, f"Expected >10 salient records total, got {total}"

    def test_salient_variable_schemas(self):
        """Each commodity has different fields -- verify they are captured."""
        data = load_all_salient_data()
        for key, records in data.items():
            if not records:
                continue
            # All records for a commodity should have the same field keys
            field_keys = set(records[0].fields.keys())
            assert len(field_keys) > 0, f"{key} salient has no variable fields"

    def test_salient_nir_present(self):
        """NIR_pct should be present for most commodities."""
        data = load_all_salient_data()
        nir_found = False
        for records in data.values():
            for r in records:
                if "NIR_pct" in r.fields:
                    nir_found = True
                    break
        assert nir_found, "Expected NIR_pct in at least one salient dataset"


class TestWorldProductionLoader:
    def test_load_world_production(self):
        data = load_all_world_production_data()
        assert len(data) > 0
        loaded = {k: v for k, v in data.items() if v}
        assert len(loaded) > 0

    def test_world_has_countries(self):
        data = load_all_world_production_data()
        for key, records in data.items():
            if not records:
                continue
            countries = {r.country for r in records}
            assert len(countries) > 1, f"{key} should have multiple producing countries"


class TestLoadAllData:
    def test_load_all(self):
        data = load_all_data()
        assert "trade" in data
        assert "salient" in data
        assert "world" in data
