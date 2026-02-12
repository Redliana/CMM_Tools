"""Tests for cmm_embedding.training.paired_data_loader module.

Note: These tests only exercise the dataclass construction and non-IO
functionality.  The full ``CMMPairedDataset`` class requires a JSONL
corpus file and is tested separately where a temp file is created.
"""

from __future__ import annotations

import torch

from cmm_embedding.training.paired_data_loader import (
    ContrastiveBatch,
    PairedExample,
)

# ============================================================================
# ContrastiveBatch dataclass
# ============================================================================


class TestContrastiveBatchConstruction:
    """Tests for ``ContrastiveBatch`` dataclass construction."""

    def test_minimal_construction(self) -> None:
        """ContrastiveBatch should be constructable with just required lists."""
        batch = ContrastiveBatch(
            modality_a_types=["text_scientific"],
            modality_a_contents=["some text"],
            modality_b_types=["text_policy"],
            modality_b_contents=["policy text"],
        )
        assert batch.modality_a_types == ["text_scientific"]
        assert batch.modality_b_types == ["text_policy"]

    def test_optional_fields_default_none(self) -> None:
        """Optional tensor fields should default to None."""
        batch = ContrastiveBatch(
            modality_a_types=[],
            modality_a_contents=[],
            modality_b_types=[],
            modality_b_contents=[],
        )
        assert batch.modality_a_tensors is None
        assert batch.modality_b_tensors is None
        assert batch.labels is None
        assert batch.pair_ids is None
        assert batch.confidence_scores is None

    def test_with_tensors(self) -> None:
        """ContrastiveBatch should accept explicit tensor values."""
        a_tensors = torch.randn(2, 10)
        b_tensors = torch.randn(2, 10)
        labels = torch.arange(2)
        conf = torch.tensor([0.8, 0.6])

        batch = ContrastiveBatch(
            modality_a_types=["text_scientific", "text_policy"],
            modality_a_contents=["text1", "text2"],
            modality_a_tensors=a_tensors,
            modality_b_types=["spectrum_xrd", "spectrum_xrd"],
            modality_b_contents=[{"y": [1, 2]}, {"y": [3, 4]}],
            modality_b_tensors=b_tensors,
            labels=labels,
            pair_ids=["p1", "p2"],
            confidence_scores=conf,
        )
        assert batch.modality_a_tensors is not None
        assert batch.modality_a_tensors.shape == (2, 10)
        assert batch.labels is not None
        assert batch.labels.tolist() == [0, 1]

    def test_to_device_cpu(self) -> None:
        """to() should move tensors to the specified device."""
        batch = ContrastiveBatch(
            modality_a_types=["t"],
            modality_a_contents=["c"],
            modality_a_tensors=torch.tensor([1.0]),
            modality_b_types=["t"],
            modality_b_contents=["c"],
            modality_b_tensors=torch.tensor([2.0]),
            labels=torch.tensor([0]),
            confidence_scores=torch.tensor([0.5]),
        )
        moved = batch.to(torch.device("cpu"))
        assert moved.modality_a_tensors is not None
        assert moved.modality_b_tensors is not None
        assert moved.labels is not None
        assert moved.confidence_scores is not None

    def test_to_device_preserves_none_tensors(self) -> None:
        """to() should leave None tensors as None."""
        batch = ContrastiveBatch(
            modality_a_types=["t"],
            modality_a_contents=["c"],
            modality_b_types=["t"],
            modality_b_contents=["c"],
        )
        moved = batch.to(torch.device("cpu"))
        assert moved.modality_a_tensors is None
        assert moved.modality_b_tensors is None

    def test_to_preserves_non_tensor_fields(self) -> None:
        """to() should preserve non-tensor fields like pair_ids."""
        batch = ContrastiveBatch(
            modality_a_types=["t"],
            modality_a_contents=["c"],
            modality_b_types=["t"],
            modality_b_contents=["c"],
            pair_ids=["p1"],
        )
        moved = batch.to(torch.device("cpu"))
        assert moved.pair_ids == ["p1"]
        assert moved.modality_a_types == ["t"]


# ============================================================================
# PairedExample dataclass
# ============================================================================


class TestPairedExample:
    """Tests for ``PairedExample`` dataclass construction."""

    def test_construction(self) -> None:
        """PairedExample should store all fields."""
        ex = PairedExample(
            pair_id="test_001",
            modality_a_type="text_scientific",
            modality_a_content="LiCoO2 is a cathode",
            modality_a_metadata={"formula": "LiCoO2"},
            modality_b_type="text_policy",
            modality_b_content="Cobalt tariffs increased",
            modality_b_metadata={"topic": "trade"},
            confidence_score=0.85,
            bridging_entities=["cobalt"],
            bridging_text="Both relate to cobalt supply.",
        )
        assert ex.pair_id == "test_001"
        assert ex.confidence_score == 0.85
        assert ex.bridging_entities == ["cobalt"]
        assert ex.bridging_text is not None

    def test_default_bridging_text_none(self) -> None:
        """bridging_text should default to None."""
        ex = PairedExample(
            pair_id="t",
            modality_a_type="t",
            modality_a_content="c",
            modality_a_metadata={},
            modality_b_type="t",
            modality_b_content="c",
            modality_b_metadata={},
            confidence_score=0.5,
            bridging_entities=[],
        )
        assert ex.bridging_text is None
