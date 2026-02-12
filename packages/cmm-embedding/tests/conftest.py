"""Shared fixtures for cmm-embedding tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="cmm-embedding requires torch")

from cmm_embedding.training.corpus_builder import (
    CrossModalPair,
    Modality,
    ModalityData,
    PairingMethod,
    TrainingCorpus,
)

# ---------------------------------------------------------------------------
# ModalityData helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def text_scientific_data() -> ModalityData:
    """Return a ``ModalityData`` instance with TEXT_SCIENTIFIC modality.

    Returns:
        ModalityData populated with a short scientific text.
    """
    return ModalityData(
        modality=Modality.TEXT_SCIENTIFIC,
        content="LiCoO2 is a layered oxide cathode material for lithium-ion batteries.",
        content_hash="",
        source="test_source",
        source_id="test_001",
        metadata={"formula": "LiCoO2", "topic": "battery cathode"},
    )


@pytest.fixture()
def text_policy_data() -> ModalityData:
    """Return a ``ModalityData`` instance with TEXT_POLICY modality.

    Returns:
        ModalityData populated with a short policy text.
    """
    return ModalityData(
        modality=Modality.TEXT_POLICY,
        content="Section 232 tariffs on cobalt imports from the DRC were increased.",
        content_hash="",
        source="federal_register",
        source_id="policy_001",
        metadata={"topic": "trade policy", "country": "DRC"},
    )


@pytest.fixture()
def spectrum_xrd_data() -> ModalityData:
    """Return a ``ModalityData`` instance with SPECTRUM_XRD modality.

    Returns:
        ModalityData with a mock XRD spectrum as a dict of lists.
    """
    return ModalityData(
        modality=Modality.SPECTRUM_XRD,
        content={"x": [28.5, 32.1, 36.2], "y": [100, 45, 78]},
        content_hash="",
        source="spectrum_db",
        source_id="xrd_001",
        metadata={"mineral_name": "cobaltite"},
    )


@pytest.fixture()
def numpy_spectrum_data() -> ModalityData:
    """Return a ``ModalityData`` whose content is a numpy array.

    Returns:
        ModalityData wrapping a 1-D numpy array.
    """
    return ModalityData(
        modality=Modality.SPECTRUM_XRD,
        content=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        content_hash="",
        source="test_source",
        source_id="np_001",
    )


# ---------------------------------------------------------------------------
# Pair / Corpus helpers
# ---------------------------------------------------------------------------


def _make_pair(
    pair_id: str,
    modality_a: ModalityData,
    modality_b: ModalityData,
    confidence: float = 0.8,
    method: PairingMethod = PairingMethod.ENTITY_COOCCURRENCE,
    bridging_entities: list[str] | None = None,
    human_validated: bool = False,
) -> CrossModalPair:
    """Factory for building ``CrossModalPair`` instances in tests.

    Args:
        pair_id: Unique identifier for the pair.
        modality_a: First modality data.
        modality_b: Second modality data.
        confidence: Confidence score (0.0 - 1.0).
        method: Pairing method enum value.
        bridging_entities: Entities that bridge the two modalities.
        human_validated: Whether the pair was validated by a human.

    Returns:
        Fully constructed CrossModalPair.
    """
    return CrossModalPair(
        pair_id=pair_id,
        modality_a=modality_a,
        modality_b=modality_b,
        pairing_method=method,
        confidence_score=confidence,
        bridging_entities=bridging_entities or [],
        human_validated=human_validated,
    )


@pytest.fixture()
def make_pair() -> Any:
    """Expose the _make_pair factory as a fixture.

    Returns:
        The _make_pair callable.
    """
    return _make_pair


@pytest.fixture()
def sample_corpus(
    text_scientific_data: ModalityData,
    text_policy_data: ModalityData,
    spectrum_xrd_data: ModalityData,
) -> TrainingCorpus:
    """Return a small ``TrainingCorpus`` with three pairs.

    Returns:
        TrainingCorpus containing a mix of modalities and confidence levels.
    """
    corpus = TrainingCorpus()
    corpus.add_pair(
        _make_pair(
            "p001",
            text_scientific_data,
            text_policy_data,
            confidence=0.9,
            bridging_entities=["cobalt", "battery"],
            human_validated=True,
        )
    )
    corpus.add_pair(
        _make_pair(
            "p002",
            text_scientific_data,
            spectrum_xrd_data,
            confidence=0.5,
            bridging_entities=["cobalt"],
        )
    )
    corpus.add_pair(
        _make_pair(
            "p003",
            text_policy_data,
            spectrum_xrd_data,
            confidence=0.3,
            method=PairingMethod.METADATA_MATCH,
        )
    )
    return corpus
