"""Tests for cmm_embedding.evaluation.cmm_benchmark_spec module."""

from __future__ import annotations

import pytest

from cmm_embedding.evaluation.cmm_benchmark_spec import (
    BenchmarkCategory,
    BenchmarkSuite,
    CrossModalAlignmentItem,
    CrossModalBenchmarkGenerator,
    CrossScaleBenchmarkGenerator,
    CrossScaleRetrievalItem,
    Difficulty,
    EntityResolutionBenchmarkGenerator,
    EntityResolutionItem,
    ModalityType,
    ScaleLevel,
    SupplyChainBenchmarkGenerator,
    SupplyChainTraversalItem,
    TemporalBenchmarkGenerator,
    TemporalConsistencyItem,
    create_full_benchmark_suite,
)

# ============================================================================
# BenchmarkCategory enum
# ============================================================================


class TestBenchmarkCategory:
    """Tests for ``BenchmarkCategory`` enum values."""

    def test_expected_members(self) -> None:
        """All five benchmark categories should be defined."""
        expected = {
            "CROSS_SCALE_RETRIEVAL",
            "CROSS_MODAL_ALIGNMENT",
            "ENTITY_RESOLUTION",
            "SUPPLY_CHAIN_TRAVERSAL",
            "TEMPORAL_CONSISTENCY",
        }
        actual = {m.name for m in BenchmarkCategory}
        assert actual == expected

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            (BenchmarkCategory.CROSS_SCALE_RETRIEVAL, "cross_scale_retrieval"),
            (BenchmarkCategory.CROSS_MODAL_ALIGNMENT, "cross_modal_alignment"),
            (BenchmarkCategory.ENTITY_RESOLUTION, "entity_resolution"),
            (BenchmarkCategory.SUPPLY_CHAIN_TRAVERSAL, "supply_chain_traversal"),
            (BenchmarkCategory.TEMPORAL_CONSISTENCY, "temporal_consistency"),
        ],
    )
    def test_string_values(self, member: BenchmarkCategory, value: str) -> None:
        """Each category's string value should match the specification."""
        assert member.value == value


class TestDifficultyEnum:
    """Tests for ``Difficulty`` enum."""

    def test_levels(self) -> None:
        """Difficulty should have exactly four levels."""
        assert {d.value for d in Difficulty} == {"easy", "medium", "hard", "expert"}


class TestScaleLevelEnum:
    """Tests for ``ScaleLevel`` enum."""

    def test_scale_hierarchy(self) -> None:
        """All six scale levels should be present."""
        expected = {"atomistic", "material", "facility", "regional", "national", "global"}
        assert {s.value for s in ScaleLevel} == expected


class TestModalityTypeEnum:
    """Tests for ``ModalityType`` enum."""

    def test_modality_types(self) -> None:
        """All expected modality types should be defined."""
        values = {m.value for m in ModalityType}
        for expected in [
            "text_scientific",
            "text_policy",
            "spectrum_xrd",
            "crystal_structure",
            "tabular_data",
        ]:
            assert expected in values


# ============================================================================
# CrossScaleRetrievalItem
# ============================================================================


class TestCrossScaleRetrievalItem:
    """Tests for ``CrossScaleRetrievalItem``."""

    def test_post_init_sets_category(self) -> None:
        """__post_init__ should force category to CROSS_SCALE_RETRIEVAL."""
        item = CrossScaleRetrievalItem(
            item_id="test_cs",
            category=BenchmarkCategory.ENTITY_RESOLUTION,  # intentionally wrong
            difficulty=Difficulty.MEDIUM,
            description="test",
            source_content="query text",
            relevant_target_ids=["t1"],
        )
        assert item.category == BenchmarkCategory.CROSS_SCALE_RETRIEVAL

    def test_get_query(self) -> None:
        """get_query should return the source_content."""
        item = CrossScaleRetrievalItem(
            item_id="cs_q",
            category=BenchmarkCategory.CROSS_SCALE_RETRIEVAL,
            difficulty=Difficulty.EASY,
            description="test",
            source_content="DFT calculation data",
        )
        assert item.get_query() == "DFT calculation data"

    def test_get_ground_truth(self) -> None:
        """get_ground_truth should return relevant_target_ids."""
        item = CrossScaleRetrievalItem(
            item_id="cs_gt",
            category=BenchmarkCategory.CROSS_SCALE_RETRIEVAL,
            difficulty=Difficulty.EASY,
            description="test",
            relevant_target_ids=["id1", "id2"],
        )
        assert item.get_ground_truth() == ["id1", "id2"]


# ============================================================================
# EntityResolutionItem
# ============================================================================


class TestEntityResolutionItem:
    """Tests for ``EntityResolutionItem``."""

    def test_get_ground_truth_includes_canonical(self) -> None:
        """Ground truth should include the canonical entity plus all aliases."""
        item = EntityResolutionItem(
            item_id="er_test",
            category=BenchmarkCategory.ENTITY_RESOLUTION,
            difficulty=Difficulty.MEDIUM,
            description="test",
            canonical_entity="Tenke Fungurume Mine",
            aliases=["TFM", "Tenke-Fungurume"],
        )
        gt = item.get_ground_truth()
        assert isinstance(gt, set)
        assert "Tenke Fungurume Mine" in gt
        assert "TFM" in gt
        assert "Tenke-Fungurume" in gt
        assert len(gt) == 3


# ============================================================================
# SupplyChainTraversalItem
# ============================================================================


class TestSupplyChainTraversalItem:
    """Tests for ``SupplyChainTraversalItem``."""

    def test_get_ground_truth_structure(self) -> None:
        """get_ground_truth should return dict with primary_path and alternatives."""
        item = SupplyChainTraversalItem(
            item_id="sc_test",
            category=BenchmarkCategory.SUPPLY_CHAIN_TRAVERSAL,
            difficulty=Difficulty.MEDIUM,
            description="test",
            expected_path_entities=["A", "B", "C"],
            expected_relationships=["R1", "R2"],
            alternative_valid_paths=[["A", "D", "C"]],
        )
        gt = item.get_ground_truth()
        assert gt["primary_path"] == ["A", "B", "C"]
        assert gt["relationships"] == ["R1", "R2"]
        assert gt["alternatives"] == [["A", "D", "C"]]


# ============================================================================
# TemporalConsistencyItem
# ============================================================================


class TestTemporalConsistencyItem:
    """Tests for ``TemporalConsistencyItem``."""

    def test_get_query_returns_tuple(self) -> None:
        """get_query should return (query_text, query_timestamp)."""
        item = TemporalConsistencyItem(
            item_id="tc_test",
            category=BenchmarkCategory.TEMPORAL_CONSISTENCY,
            difficulty=Difficulty.HARD,
            description="test",
            query_text="What are sanctions?",
            query_timestamp="2024-06-01T00:00:00Z",
        )
        q = item.get_query()
        assert isinstance(q, tuple)
        assert q[0] == "What are sanctions?"
        assert q[1] == "2024-06-01T00:00:00Z"

    def test_get_ground_truth(self) -> None:
        """get_ground_truth should return expected_current_answer."""
        item = TemporalConsistencyItem(
            item_id="tc_test2",
            category=BenchmarkCategory.TEMPORAL_CONSISTENCY,
            difficulty=Difficulty.MEDIUM,
            description="test",
            expected_current_answer="200% tariff",
        )
        assert item.get_ground_truth() == "200% tariff"


# ============================================================================
# Benchmark Generators
# ============================================================================


class TestBenchmarkGenerators:
    """Smoke tests for benchmark item generators."""

    def test_cross_scale_atomistic_to_policy(self) -> None:
        """Generator should produce a non-empty list of items."""
        items = CrossScaleBenchmarkGenerator.generate_atomistic_to_policy_items()
        assert len(items) >= 1
        assert all(isinstance(i, CrossScaleRetrievalItem) for i in items)

    def test_cross_scale_material_to_trade(self) -> None:
        """Material-to-trade generator should produce items."""
        items = CrossScaleBenchmarkGenerator.generate_material_to_trade_items()
        assert len(items) >= 1

    def test_cross_modal_spectrum_to_text(self) -> None:
        """Spectrum-to-text generator should produce items."""
        items = CrossModalBenchmarkGenerator.generate_spectrum_to_text_items()
        assert len(items) >= 1
        assert all(isinstance(i, CrossModalAlignmentItem) for i in items)

    def test_entity_mine_items(self) -> None:
        """Mine entity resolution generator should produce items."""
        items = EntityResolutionBenchmarkGenerator.generate_mine_entity_items()
        assert len(items) >= 1
        assert all(isinstance(i, EntityResolutionItem) for i in items)

    def test_entity_company_items(self) -> None:
        """Company entity resolution generator should produce items."""
        items = EntityResolutionBenchmarkGenerator.generate_company_entity_items()
        assert len(items) >= 1

    def test_entity_mineral_items(self) -> None:
        """Mineral entity resolution generator should produce items."""
        items = EntityResolutionBenchmarkGenerator.generate_mineral_entity_items()
        assert len(items) >= 1

    def test_supply_chain_items(self) -> None:
        """Supply chain generator should produce items."""
        items = SupplyChainBenchmarkGenerator.generate_items()
        assert len(items) >= 1
        assert all(isinstance(i, SupplyChainTraversalItem) for i in items)

    def test_temporal_items(self) -> None:
        """Temporal consistency generator should produce items."""
        items = TemporalBenchmarkGenerator.generate_items()
        assert len(items) >= 1
        assert all(isinstance(i, TemporalConsistencyItem) for i in items)


# ============================================================================
# Full Benchmark Suite
# ============================================================================


class TestCreateFullBenchmarkSuite:
    """Tests for the ``create_full_benchmark_suite`` factory."""

    def test_returns_benchmark_suite(self) -> None:
        """Factory should return a BenchmarkSuite instance."""
        suite = create_full_benchmark_suite()
        assert isinstance(suite, BenchmarkSuite)

    def test_all_categories_populated(self) -> None:
        """Every category list should have at least one item."""
        suite = create_full_benchmark_suite()
        assert len(suite.cross_scale_items) >= 1
        assert len(suite.cross_modal_items) >= 1
        assert len(suite.entity_resolution_items) >= 1
        assert len(suite.supply_chain_items) >= 1
        assert len(suite.temporal_items) >= 1

    def test_get_all_items(self) -> None:
        """get_all_items should return the union of all category lists."""
        suite = create_full_benchmark_suite()
        total = (
            len(suite.cross_scale_items)
            + len(suite.cross_modal_items)
            + len(suite.entity_resolution_items)
            + len(suite.supply_chain_items)
            + len(suite.temporal_items)
        )
        assert len(suite.get_all_items()) == total

    def test_get_statistics(self) -> None:
        """get_statistics should return a well-formed dict."""
        suite = create_full_benchmark_suite()
        stats = suite.get_statistics()
        assert "total_items" in stats
        assert stats["total_items"] > 0
        assert "by_category" in stats
        assert "by_difficulty" in stats

    def test_unique_item_ids(self) -> None:
        """All item IDs across the suite should be unique."""
        suite = create_full_benchmark_suite()
        ids = [item.item_id for item in suite.get_all_items()]
        assert len(ids) == len(set(ids)), "Duplicate item_id found"
