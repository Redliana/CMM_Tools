"""Shared fixtures for cmm-docs-mcp tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def sample_catalog() -> list[dict[str, Any]]:
    """Provide a sample OSTI document catalog for testing.

    Returns:
        A list of document metadata dictionaries.
    """
    return [
        {
            "osti_id": "3004920",
            "title": "Lithium Recovery from Coal Combustion Residuals",
            "authors": ["Smith, John", "Doe, Jane", "Johnson, Alice"],
            "publication_date": "2024-03-15",
            "commodity_category": "LI",
            "product_type": "Journal Article",
            "doi": "10.1234/test.2024.001",
            "journal_name": "Environmental Science & Technology",
            "journal_volume": "58",
            "research_orgs": ["National Energy Technology Laboratory"],
            "abstract": "A study on lithium recovery methods.",
        },
        {
            "osti_id": "3005100",
            "title": "Cobalt Extraction from Mine Tailings",
            "authors": ["Brown, Bob"],
            "publication_date": "2023-11-20",
            "commodity_category": "CO",
            "product_type": "Technical Report",
            "doi": "",
            "research_orgs": ["Pacific Northwest National Laboratory"],
            "abstract": "A technical report on cobalt extraction.",
        },
        {
            "osti_id": "3005200",
            "title": "Rare Earth Element Characterization in Appalachian Coal",
            "authors": ["Williams, Carol", "Davis, Eve"],
            "publication_date": "2024-01-10",
            "commodity_category": "HREE",
            "product_type": "Journal Article",
            "doi": "10.1234/test.2024.002",
            "journal_name": "Fuel",
            "journal_volume": "360",
            "research_orgs": [],
            "abstract": "Characterization of REE in Appalachian coal.",
        },
        {
            "osti_id": "3005300",
            "title": "Lithium Market Analysis and Supply Chain",
            "authors": ["Taylor, Frank"],
            "publication_date": "2023-08-01",
            "commodity_category": "LI",
            "product_type": "Technical Report",
            "doi": "",
            "research_orgs": ["Argonne National Laboratory"],
            "abstract": "Analysis of the lithium supply chain.",
        },
    ]


@pytest.fixture()
def sample_schemas() -> dict[str, Any]:
    """Provide a sample schemas dict mimicking all_schemas.json structure.

    Returns:
        A dictionary with category keys and schemas sub-dicts.
    """
    return {
        "LISA_Model": {
            "schemas": [
                {
                    "file": "ChemData1.csv",
                    "path": "/fake/path/LISA_Model/ChemData1.csv",
                    "row_count": 500,
                    "columns": [
                        {"name": "SampleID", "dtype": "object"},
                        {"name": "Element", "dtype": "object"},
                        {"name": "Concentration", "dtype": "float64"},
                    ],
                },
                {
                    "file": "ChemData2.csv",
                    "path": "/fake/path/LISA_Model/ChemData2.csv",
                    "row_count": 300,
                    "columns": [
                        {"name": "SampleID", "dtype": "object"},
                        {"name": "pH", "dtype": "float64"},
                    ],
                },
            ]
        },
        "USGS_Ore_Deposits": {
            "schemas": [
                {
                    "file": "deposits_us.csv",
                    "path": "/fake/path/USGS/deposits_us.csv",
                    "row_count": 1200,
                    "columns": [
                        {"name": "DepositName", "dtype": "object"},
                        {"name": "Latitude", "dtype": "float64"},
                        {"name": "Longitude", "dtype": "float64"},
                    ],
                },
            ]
        },
    }


@pytest.fixture()
def catalog_json_file(tmp_path: Path, sample_catalog: list[dict[str, Any]]) -> Path:
    """Write sample catalog to a temporary JSON file.

    Args:
        tmp_path: Pytest's tmp_path fixture.
        sample_catalog: The sample catalog data.

    Returns:
        Path to the temporary catalog JSON file.
    """
    catalog_path = tmp_path / "document_catalog.json"
    catalog_path.write_text(json.dumps(sample_catalog), encoding="utf-8")
    return catalog_path


@pytest.fixture()
def schemas_json_file(tmp_path: Path, sample_schemas: dict[str, Any]) -> Path:
    """Write sample schemas to a temporary JSON file.

    Args:
        tmp_path: Pytest's tmp_path fixture.
        sample_schemas: The sample schemas data.

    Returns:
        Path to the temporary schemas JSON file.
    """
    schemas_path = tmp_path / "all_schemas.json"
    schemas_path.write_text(json.dumps(sample_schemas), encoding="utf-8")
    return schemas_path
