"""Tests for cmm_embedding.training.corpus_builder module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cmm_embedding.training.corpus_builder import (
    CrossModalPair,
    Modality,
    ModalityData,
    PairingMethod,
    TrainingCorpus,
)

# ============================================================================
# ModalityData
# ============================================================================


class TestModalityDataConstruction:
    """Tests for ``ModalityData`` construction and hashing."""

    def test_basic_construction(self, text_scientific_data: ModalityData) -> None:
        """ModalityData should store all fields correctly."""
        assert text_scientific_data.modality == Modality.TEXT_SCIENTIFIC
        assert "LiCoO2" in text_scientific_data.content
        assert text_scientific_data.source == "test_source"
        assert text_scientific_data.source_id == "test_001"

    def test_compute_hash_text(self, text_scientific_data: ModalityData) -> None:
        """_compute_hash should produce a 16-char hex string for text content."""
        h = text_scientific_data._compute_hash()
        assert isinstance(h, str)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_hash_dict(self, spectrum_xrd_data: ModalityData) -> None:
        """_compute_hash should handle dict content."""
        h = spectrum_xrd_data._compute_hash()
        assert isinstance(h, str)
        assert len(h) == 16

    def test_compute_hash_numpy(self, numpy_spectrum_data: ModalityData) -> None:
        """_compute_hash should handle numpy array content."""
        h = numpy_spectrum_data._compute_hash()
        assert isinstance(h, str)
        assert len(h) == 16

    def test_auto_hash_on_empty_string(self) -> None:
        """When content_hash is '' at construction, __post_init__ should compute it."""
        data = ModalityData(
            modality=Modality.TEXT_SCIENTIFIC,
            content="test content",
            content_hash="",
            source="s",
            source_id="id",
        )
        assert data.content_hash != ""
        assert len(data.content_hash) == 16

    def test_deterministic_hash(self) -> None:
        """The same content should always yield the same hash."""
        d1 = ModalityData(
            modality=Modality.TEXT_SCIENTIFIC,
            content="deterministic",
            content_hash="",
            source="s",
            source_id="id",
        )
        d2 = ModalityData(
            modality=Modality.TEXT_SCIENTIFIC,
            content="deterministic",
            content_hash="",
            source="s",
            source_id="id",
        )
        assert d1.content_hash == d2.content_hash

    def test_different_content_different_hash(self) -> None:
        """Different content should produce different hashes."""
        d1 = ModalityData(
            modality=Modality.TEXT_SCIENTIFIC,
            content="alpha",
            content_hash="",
            source="s",
            source_id="id",
        )
        d2 = ModalityData(
            modality=Modality.TEXT_SCIENTIFIC,
            content="beta",
            content_hash="",
            source="s",
            source_id="id",
        )
        assert d1.content_hash != d2.content_hash

    def test_metadata_default_is_empty_dict(self) -> None:
        """The default metadata should be an empty dict."""
        data = ModalityData(
            modality=Modality.TEXT_SCIENTIFIC,
            content="x",
            content_hash="abc",
            source="s",
            source_id="id",
        )
        assert data.metadata == {}

    def test_fallback_hash_for_unknown_type(self) -> None:
        """Content types other than str/dict/list/ndarray should use str() fallback."""
        data = ModalityData(
            modality=Modality.TEXT_SCIENTIFIC,
            content=12345,
            content_hash="",
            source="s",
            source_id="id",
        )
        assert len(data.content_hash) == 16


# ============================================================================
# CrossModalPair
# ============================================================================


class TestCrossModalPairToDict:
    """Tests for ``CrossModalPair.to_dict``."""

    def test_to_dict_keys(
        self,
        text_scientific_data: ModalityData,
        text_policy_data: ModalityData,
        make_pair: Any,
    ) -> None:
        """to_dict should return all expected top-level keys."""
        pair: CrossModalPair = make_pair("t001", text_scientific_data, text_policy_data)
        d = pair.to_dict()
        expected_keys = {
            "pair_id",
            "modality_a",
            "modality_b",
            "pairing_method",
            "confidence_score",
            "bridging_entities",
            "bridging_text",
            "human_validated",
            "created_at",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_modality_values(
        self,
        text_scientific_data: ModalityData,
        text_policy_data: ModalityData,
        make_pair: Any,
    ) -> None:
        """Modality enum values should be serialised as plain strings."""
        pair: CrossModalPair = make_pair("t002", text_scientific_data, text_policy_data)
        d = pair.to_dict()
        assert d["modality_a"]["modality"] == "text_scientific"
        assert d["modality_b"]["modality"] == "text_policy"

    def test_to_dict_pairing_method_string(
        self,
        text_scientific_data: ModalityData,
        text_policy_data: ModalityData,
        make_pair: Any,
    ) -> None:
        """pairing_method should be serialised as a string value."""
        pair: CrossModalPair = make_pair(
            "t003",
            text_scientific_data,
            text_policy_data,
            method=PairingMethod.LLM_SYNTHETIC,
        )
        d = pair.to_dict()
        assert d["pairing_method"] == "llm_synthetic"

    def test_to_dict_non_serialisable_content_becomes_none(
        self,
        numpy_spectrum_data: ModalityData,
        text_scientific_data: ModalityData,
        make_pair: Any,
    ) -> None:
        """numpy array content should become None in to_dict (not serialisable)."""
        pair: CrossModalPair = make_pair("t004", numpy_spectrum_data, text_scientific_data)
        d = pair.to_dict()
        assert d["modality_a"]["content"] is None

    def test_to_dict_is_json_serialisable(
        self,
        text_scientific_data: ModalityData,
        text_policy_data: ModalityData,
        make_pair: Any,
    ) -> None:
        """to_dict output should be safe to pass to json.dumps."""
        pair: CrossModalPair = make_pair("t005", text_scientific_data, text_policy_data)
        d = pair.to_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)


# ============================================================================
# TrainingCorpus -- filtering and statistics
# ============================================================================


class TestTrainingCorpusFilterByConfidence:
    """Tests for ``TrainingCorpus.filter_by_confidence``."""

    def test_returns_new_corpus(self, sample_corpus: TrainingCorpus) -> None:
        """filter_by_confidence should return a *new* TrainingCorpus instance."""
        filtered = sample_corpus.filter_by_confidence(0.5)
        assert filtered is not sample_corpus

    def test_high_threshold_excludes_low_confidence(self, sample_corpus: TrainingCorpus) -> None:
        """Only pairs meeting the threshold should survive."""
        filtered = sample_corpus.filter_by_confidence(0.8)
        assert len(filtered.pairs) == 1
        assert filtered.pairs[0].pair_id == "p001"

    def test_zero_threshold_keeps_all(self, sample_corpus: TrainingCorpus) -> None:
        """A zero threshold should keep every pair."""
        filtered = sample_corpus.filter_by_confidence(0.0)
        assert len(filtered.pairs) == len(sample_corpus.pairs)

    def test_one_threshold_keeps_none_or_exact(self, sample_corpus: TrainingCorpus) -> None:
        """Threshold of 1.0 should exclude everything below 1.0."""
        filtered = sample_corpus.filter_by_confidence(1.0)
        assert len(filtered.pairs) == 0

    def test_metadata_records_threshold(self, sample_corpus: TrainingCorpus) -> None:
        """The filtered corpus metadata should record the threshold."""
        filtered = sample_corpus.filter_by_confidence(0.6)
        assert filtered.metadata["filtered_min_confidence"] == 0.6


class TestTrainingCorpusFilterByModalities:
    """Tests for ``TrainingCorpus.filter_by_modalities``."""

    def test_filters_to_matching_pair(self, sample_corpus: TrainingCorpus) -> None:
        """Filtering for scientific+policy should return only that pair."""
        filtered = sample_corpus.filter_by_modalities(
            Modality.TEXT_SCIENTIFIC, Modality.TEXT_POLICY
        )
        assert len(filtered.pairs) == 1
        assert filtered.pairs[0].pair_id == "p001"

    def test_order_independent(self, sample_corpus: TrainingCorpus) -> None:
        """Filtering with swapped modality order should yield the same result."""
        f1 = sample_corpus.filter_by_modalities(Modality.TEXT_SCIENTIFIC, Modality.TEXT_POLICY)
        f2 = sample_corpus.filter_by_modalities(Modality.TEXT_POLICY, Modality.TEXT_SCIENTIFIC)
        assert len(f1.pairs) == len(f2.pairs)

    def test_no_match_returns_empty(self, sample_corpus: TrainingCorpus) -> None:
        """Non-existent modality pairing should return an empty corpus."""
        filtered = sample_corpus.filter_by_modalities(
            Modality.GEOSPATIAL, Modality.MOLECULAR_STRUCTURE
        )
        assert len(filtered.pairs) == 0

    def test_metadata_records_modalities(self, sample_corpus: TrainingCorpus) -> None:
        """Filtered corpus metadata should include the modality filter."""
        filtered = sample_corpus.filter_by_modalities(
            Modality.TEXT_SCIENTIFIC, Modality.SPECTRUM_XRD
        )
        assert "filtered_modalities" in filtered.metadata


class TestTrainingCorpusGetStatistics:
    """Tests for ``TrainingCorpus.get_statistics``."""

    def test_total_pairs(self, sample_corpus: TrainingCorpus) -> None:
        """total_pairs should equal the number of pairs added."""
        stats = sample_corpus.get_statistics()
        assert stats["total_pairs"] == 3

    def test_modality_distribution(self, sample_corpus: TrainingCorpus) -> None:
        """modality_distribution should count appearances of each modality."""
        stats = sample_corpus.get_statistics()
        dist = stats["modality_distribution"]
        # TEXT_SCIENTIFIC appears in p001 and p002 (twice each via modality_a or _b)
        assert dist["text_scientific"] >= 2

    def test_pairing_method_distribution(self, sample_corpus: TrainingCorpus) -> None:
        """pairing_method_distribution should tally each PairingMethod."""
        stats = sample_corpus.get_statistics()
        method_dist = stats["pairing_method_distribution"]
        assert "entity_cooccurrence" in method_dist
        assert "metadata_match" in method_dist

    def test_confidence_mean_in_expected_range(self, sample_corpus: TrainingCorpus) -> None:
        """Mean confidence should be between 0 and 1."""
        stats = sample_corpus.get_statistics()
        mean_conf = stats["confidence_mean"]
        assert 0.0 <= mean_conf <= 1.0

    def test_confidence_std_non_negative(self, sample_corpus: TrainingCorpus) -> None:
        """Standard deviation should be non-negative."""
        stats = sample_corpus.get_statistics()
        assert stats["confidence_std"] >= 0.0

    def test_human_validated_count(self, sample_corpus: TrainingCorpus) -> None:
        """human_validated_count should reflect the fixture data (1 validated)."""
        stats = sample_corpus.get_statistics()
        assert stats["human_validated_count"] == 1

    def test_empty_corpus_statistics(self) -> None:
        """An empty corpus should return zeroes without errors."""
        empty = TrainingCorpus()
        stats = empty.get_statistics()
        assert stats["total_pairs"] == 0
        assert stats["confidence_mean"] == 0
        assert stats["confidence_std"] == 0
        assert stats["human_validated_count"] == 0


# ============================================================================
# TrainingCorpus -- save and load round-trip
# ============================================================================


class TestTrainingCorpusSaveLoad:
    """Tests for save/load serialisation round-trip."""

    def test_save_creates_file(self, sample_corpus: TrainingCorpus, tmp_path: Path) -> None:
        """save() should write a JSONL file."""
        out = tmp_path / "corpus.jsonl"
        sample_corpus.save(str(out))
        assert out.exists()

    def test_load_recovers_pairs(self, sample_corpus: TrainingCorpus, tmp_path: Path) -> None:
        """load() should recover the same number of pairs that were saved."""
        out = tmp_path / "corpus.jsonl"
        sample_corpus.save(str(out))
        loaded = TrainingCorpus.load(str(out))
        assert len(loaded.pairs) == len(sample_corpus.pairs)

    def test_load_recovers_metadata(self, sample_corpus: TrainingCorpus, tmp_path: Path) -> None:
        """Metadata written by save() should be recovered by load()."""
        sample_corpus.metadata["custom_key"] = "custom_value"
        out = tmp_path / "corpus.jsonl"
        sample_corpus.save(str(out))
        loaded = TrainingCorpus.load(str(out))
        # The loaded metadata comes from the _metadata line which includes stats
        assert "statistics" in loaded.metadata

    def test_round_trip_pair_ids(self, sample_corpus: TrainingCorpus, tmp_path: Path) -> None:
        """Pair IDs should survive the save/load round-trip."""
        out = tmp_path / "corpus.jsonl"
        sample_corpus.save(str(out))
        loaded = TrainingCorpus.load(str(out))
        original_ids = {p.pair_id for p in sample_corpus.pairs}
        loaded_ids = {p.pair_id for p in loaded.pairs}
        assert original_ids == loaded_ids


# ============================================================================
# Modality and PairingMethod enums
# ============================================================================


class TestEnums:
    """Basic smoke tests for the Modality and PairingMethod enums."""

    def test_modality_values(self) -> None:
        """All expected modalities should be present."""
        values = {m.value for m in Modality}
        assert "text_scientific" in values
        assert "text_policy" in values
        assert "spectrum_xrd" in values
        assert "crystal_structure" in values
        assert "geospatial" in values

    def test_pairing_method_values(self) -> None:
        """All expected pairing methods should be present."""
        values = {m.value for m in PairingMethod}
        assert "direct_database" in values
        assert "llm_synthetic" in values
        assert "entity_cooccurrence" in values
        assert "human_curated" in values
