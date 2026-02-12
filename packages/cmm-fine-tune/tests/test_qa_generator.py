"""Tests for Q&A generator."""

from __future__ import annotations

import pytest

from cmm_fine_tune.data.csv_loader import load_all_data
from cmm_fine_tune.data.qa_generator import (
    _generate_salient_qa,
    _generate_trade_qa,
    _generate_world_production_qa,
    generate_all_qa,
)
from cmm_fine_tune.models import QAPair


@pytest.fixture(scope="module")
def all_data():
    return load_all_data()


@pytest.fixture(scope="module")
def all_qa_pairs(all_data):
    return generate_all_qa(all_data["trade"], all_data["salient"], all_data["world"])


class TestTradeQA:
    def test_generates_pairs(self, all_data):
        trade = all_data["trade"]
        for key, records in trade.items():
            if not records:
                continue
            pairs = _generate_trade_qa(key, records)
            assert len(pairs) > 0, f"Expected trade Q&A pairs for {key}"
            break  # test at least one

    def test_template_ids(self, all_qa_pairs):
        trade_pairs = [p for p in all_qa_pairs if p.template_id.startswith("trade_")]
        template_ids = {p.template_id for p in trade_pairs}
        assert "trade_total_value" in template_ids
        assert "trade_bilateral" in template_ids


class TestSalientQA:
    def test_generates_pairs(self, all_data):
        salient = all_data["salient"]
        for key, records in salient.items():
            if not records:
                continue
            pairs = _generate_salient_qa(key, records)
            assert len(pairs) > 0, f"Expected salient Q&A pairs for {key}"
            break

    def test_nir_template(self, all_qa_pairs):
        nir_pairs = [p for p in all_qa_pairs if p.template_id == "salient_nir"]
        assert len(nir_pairs) > 0, "Expected NIR Q&A pairs"


class TestWorldProductionQA:
    def test_generates_pairs(self, all_data):
        world = all_data["world"]
        for key, records in world.items():
            if not records:
                continue
            pairs = _generate_world_production_qa(key, records)
            assert len(pairs) > 0, f"Expected world production Q&A pairs for {key}"
            break

    def test_top_producers_template(self, all_qa_pairs):
        top_pairs = [p for p in all_qa_pairs if p.template_id == "world_top_producers"]
        assert len(top_pairs) > 0, "Expected top producers Q&A pairs"


class TestGenerateAll:
    def test_total_count(self, all_qa_pairs):
        """Should generate a substantial number of Q&A pairs."""
        assert len(all_qa_pairs) >= 500, f"Expected >= 500 pairs, got {len(all_qa_pairs)}"

    def test_all_commodities_represented(self, all_qa_pairs):
        commodities = {p.commodity for p in all_qa_pairs}
        assert len(commodities) >= 6, f"Expected >= 6 commodities, got {commodities}"

    def test_complexity_levels(self, all_qa_pairs):
        levels = {p.complexity_level for p in all_qa_pairs}
        assert "L1" in levels
        assert "L2" in levels

    def test_qa_pair_structure(self, all_qa_pairs):
        for pair in all_qa_pairs[:20]:
            assert isinstance(pair, QAPair)
            assert pair.question.endswith("?")
            assert len(pair.answer) > 20
            assert pair.commodity != ""
            assert pair.template_id != ""
            assert pair.source_data  # not empty

    def test_source_data_provenance(self, all_qa_pairs):
        """Each pair should have source_data for provenance."""
        for pair in all_qa_pairs[:50]:
            assert "commodity" in pair.source_data
