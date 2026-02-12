"""Tests for cmm_docs.data_tools module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from cmm_docs.data_tools import DataManager


def _make_data_manager(schemas: dict[str, Any]) -> DataManager:
    """Create a DataManager with a given schemas dict, bypassing file I/O.

    Args:
        schemas: The schemas dictionary to inject.

    Returns:
        A DataManager instance with the provided schemas.
    """
    with patch.object(DataManager, "_load_schemas", return_value=schemas):
        return DataManager()


class TestListDatasets:
    """Tests for DataManager.list_datasets()."""

    def test_list_all_datasets(self, sample_schemas: dict[str, Any]) -> None:
        """Should list all datasets across all categories."""
        dm = _make_data_manager(sample_schemas)
        results = dm.list_datasets()
        assert len(results) == 3  # 2 LISA_Model + 1 USGS_Ore_Deposits

    def test_list_datasets_returns_expected_keys(self, sample_schemas: dict[str, Any]) -> None:
        """Each dataset entry should contain expected fields."""
        dm = _make_data_manager(sample_schemas)
        results = dm.list_datasets()
        for entry in results:
            assert "category" in entry
            assert "file" in entry
            assert "row_count" in entry
            assert "column_count" in entry
            assert "path" in entry

    def test_list_datasets_filter_by_category(self, sample_schemas: dict[str, Any]) -> None:
        """Filtering by category should only return matching datasets."""
        dm = _make_data_manager(sample_schemas)
        results = dm.list_datasets(category="LISA")
        assert len(results) == 2
        assert all("LISA" in r["category"] for r in results)

    def test_list_datasets_filter_case_insensitive(self, sample_schemas: dict[str, Any]) -> None:
        """Category filtering should be case-insensitive."""
        dm = _make_data_manager(sample_schemas)
        results = dm.list_datasets(category="usgs")
        assert len(results) == 1
        assert results[0]["file"] == "deposits_us.csv"

    def test_list_datasets_no_match(self, sample_schemas: dict[str, Any]) -> None:
        """Filtering by a non-existent category should return empty list."""
        dm = _make_data_manager(sample_schemas)
        results = dm.list_datasets(category="nonexistent")
        assert results == []

    def test_list_datasets_empty_schemas(self) -> None:
        """Empty schemas should return empty list."""
        dm = _make_data_manager({})
        results = dm.list_datasets()
        assert results == []

    def test_column_count_matches_columns(self, sample_schemas: dict[str, Any]) -> None:
        """column_count should match the number of columns in the schema."""
        dm = _make_data_manager(sample_schemas)
        results = dm.list_datasets()
        chem1 = next(r for r in results if r["file"] == "ChemData1.csv")
        assert chem1["column_count"] == 3
        deposits = next(r for r in results if r["file"] == "deposits_us.csv")
        assert deposits["column_count"] == 3


class TestGetSchema:
    """Tests for DataManager.get_schema()."""

    def test_get_schema_found(self, sample_schemas: dict[str, Any]) -> None:
        """Should return schema info for an existing dataset file."""
        dm = _make_data_manager(sample_schemas)
        result = dm.get_schema("ChemData1.csv")
        assert result is not None
        assert result["file"] == "ChemData1.csv"
        assert result["category"] == "LISA_Model"
        assert result["row_count"] == 500
        assert len(result["columns"]) == 3

    def test_get_schema_not_found(self, sample_schemas: dict[str, Any]) -> None:
        """Should return an error dict for a non-existent dataset."""
        dm = _make_data_manager(sample_schemas)
        result = dm.get_schema("nonexistent.csv")
        assert result is not None
        assert "error" in result

    def test_get_schema_includes_path(self, sample_schemas: dict[str, Any]) -> None:
        """Schema result should include the file path."""
        dm = _make_data_manager(sample_schemas)
        result = dm.get_schema("deposits_us.csv")
        assert "path" in result
        assert result["path"] == "/fake/path/USGS/deposits_us.csv"

    def test_get_schema_empty_schemas(self) -> None:
        """With no schemas loaded, any lookup should return error."""
        dm = _make_data_manager({})
        result = dm.get_schema("anything.csv")
        assert "error" in result


class TestFindDatasetPath:
    """Tests for DataManager.find_dataset_path()."""

    def test_find_existing_dataset(self, sample_schemas: dict[str, Any]) -> None:
        """Should return a Path for a known dataset."""
        dm = _make_data_manager(sample_schemas)
        path = dm.find_dataset_path("ChemData1.csv")
        assert path is not None
        assert isinstance(path, Path)
        assert path.name == "ChemData1.csv"

    def test_find_nonexistent_dataset(self, sample_schemas: dict[str, Any]) -> None:
        """Should return None for an unknown dataset."""
        dm = _make_data_manager(sample_schemas)
        path = dm.find_dataset_path("nonexistent.csv")
        assert path is None


class TestGetStatistics:
    """Tests for DataManager.get_statistics()."""

    def test_statistics_structure(self, sample_schemas: dict[str, Any]) -> None:
        """Statistics should contain total counts and per-category breakdowns."""
        dm = _make_data_manager(sample_schemas)
        stats = dm.get_statistics()
        assert "total_datasets" in stats
        assert "total_rows" in stats
        assert "by_category" in stats

    def test_statistics_totals(self, sample_schemas: dict[str, Any]) -> None:
        """Total datasets and rows should match the sum across categories."""
        dm = _make_data_manager(sample_schemas)
        stats = dm.get_statistics()
        assert stats["total_datasets"] == 3
        assert stats["total_rows"] == 500 + 300 + 1200

    def test_statistics_by_category(self, sample_schemas: dict[str, Any]) -> None:
        """Per-category stats should include file counts and row totals."""
        dm = _make_data_manager(sample_schemas)
        stats = dm.get_statistics()
        lisa = stats["by_category"]["LISA_Model"]
        assert lisa["file_count"] == 2
        assert lisa["total_rows"] == 800

        usgs = stats["by_category"]["USGS_Ore_Deposits"]
        assert usgs["file_count"] == 1
        assert usgs["total_rows"] == 1200

    def test_statistics_empty_schemas(self) -> None:
        """With no schemas, all counts should be zero."""
        dm = _make_data_manager({})
        stats = dm.get_statistics()
        assert stats["total_datasets"] == 0
        assert stats["total_rows"] == 0
        assert stats["by_category"] == {}
