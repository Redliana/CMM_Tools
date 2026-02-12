"""Tests for lazy loader imports via cmm_data.__init__.__getattr__."""

from __future__ import annotations

import pytest

import cmm_data


class TestLazyImports:
    """Smoke tests ensuring all loader classes are importable via the top-level package."""

    @pytest.mark.parametrize(
        "loader_name",
        [
            "USGSCommodityLoader",
            "USGSOreDepositsLoader",
            "OSTIDocumentsLoader",
            "PreprocessedCorpusLoader",
            "GAChronostratigraphicLoader",
            "NETLREECoalLoader",
            "OECDSupplyChainLoader",
            "MindatLoader",
        ],
    )
    def test_loader_importable(self, loader_name: str) -> None:
        """Each loader class should be accessible as cmm_data.<LoaderName>."""
        cls = getattr(cmm_data, loader_name)
        assert cls is not None
        # Verify it is actually a class
        assert isinstance(cls, type), f"{loader_name} is not a class"

    def test_unknown_attribute_raises(self) -> None:
        """Accessing an undefined attribute should raise AttributeError."""
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = cmm_data.NonExistentLoader  # type: ignore[attr-defined]

    def test_version_accessible(self) -> None:
        """__version__ should be a non-empty string."""
        assert isinstance(cmm_data.__version__, str)
        assert len(cmm_data.__version__) > 0
