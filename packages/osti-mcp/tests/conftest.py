"""Shared fixtures for osti-mcp tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osti_mcp.client import OSTIClient, OSTIDocument


@pytest.fixture()
def sample_catalog_data() -> list[dict[str, Any]]:
    """Create sample document catalog data matching the real JSON structure.

    Returns:
        A list of dicts representing OSTI document records as they appear
        in document_catalog.json.
    """
    return [
        {
            "osti_id": "2342032",
            "title": "Recovery of Rare Earth Elements from Coal Fly Ash",
            "authors": ["Smith, John A.", "Doe, Jane B."],
            "publication_date": "2023-06-15",
            "description": "This report investigates methods for extracting rare earth elements from coal fly ash using acid leaching.",
            "subjects": ["rare earth elements", "coal fly ash", "extraction"],
            "commodity_category": "HREE",
            "doi": "10.2172/2342032",
            "product_type": "Technical Report",
            "research_orgs": ["Pacific Northwest National Laboratory"],
            "sponsor_orgs": ["USDOE Office of Fossil Energy"],
        },
        {
            "osti_id": "2342033",
            "title": "Lithium Brine Extraction Optimization",
            "authors": ["Johnson, Alice"],
            "publication_date": "2023-09-01",
            "description": "Optimization study for lithium extraction from geothermal brines.",
            "subjects": ["lithium", "brine", "geothermal"],
            "commodity_category": "LI",
            "doi": "10.2172/2342033",
            "product_type": "Technical Report",
            "research_orgs": ["Argonne National Laboratory"],
            "sponsor_orgs": ["USDOE Office of Energy Efficiency"],
        },
        {
            "osti_id": "2342034",
            "title": "Cobalt Supply Chain Risk Assessment",
            "authors": ["Williams, Robert", "Garcia, Maria"],
            "publication_date": "2022-03-10",
            "description": "Assessment of supply chain risks for cobalt used in battery manufacturing.",
            "subjects": ["cobalt", "supply chain", "battery"],
            "commodity_category": "CO",
            "doi": "10.2172/2342034",
            "product_type": "Journal Article",
            "research_orgs": ["National Renewable Energy Laboratory"],
            "sponsor_orgs": ["USDOE"],
        },
        {
            "osti_id": "2342035",
            "title": "Graphite Processing Methods for Battery Applications",
            "authors": ["Chen, Wei"],
            "publication_date": "2024-01-20",
            "description": "Review of graphite processing methods for use in lithium-ion battery anodes.",
            "subjects": ["graphite", "battery", "processing"],
            "commodity_category": "GR",
            "doi": "10.2172/2342035",
            "product_type": "Technical Report",
            "research_orgs": ["Oak Ridge National Laboratory"],
            "sponsor_orgs": ["USDOE Office of Science"],
        },
        {
            "osti_id": "2342036",
            "title": "Nickel Laterite Hydrometallurgy Advances",
            "authors": ["Brown, David"],
            "publication_date": "2021-11-05",
            "description": "Recent advances in hydrometallurgical processing of nickel laterite ores.",
            "subjects": ["nickel", "laterite", "hydrometallurgy"],
            "commodity_category": "NI",
            "doi": "10.2172/2342036",
            "product_type": "Technical Report",
            "research_orgs": ["Idaho National Laboratory"],
            "sponsor_orgs": ["USDOE"],
        },
        {
            "osti_id": "2342037",
            "title": "Gallium Arsenide Recycling from Electronic Waste",
            "authors": ["Kim, Sung-Ho"],
            "publication_date": "2020-07-12",
            "description": "Methods for recovering gallium from electronic waste streams.",
            "subjects": ["gallium", "recycling", "electronic waste"],
            "commodity_category": "GA",
            "doi": "10.2172/2342037",
            "product_type": "Journal Article",
            "research_orgs": ["Sandia National Laboratories"],
            "sponsor_orgs": ["USDOE"],
        },
    ]


@pytest.fixture()
def tmp_catalog_dir(tmp_path: Path, sample_catalog_data: list[dict[str, Any]]) -> Path:
    """Create a temporary directory with a document_catalog.json file.

    Args:
        tmp_path: pytest built-in temporary path fixture.
        sample_catalog_data: The sample catalog records.

    Returns:
        Path to the temporary directory containing the catalog file.
    """
    catalog_file = tmp_path / "document_catalog.json"
    catalog_file.write_text(json.dumps(sample_catalog_data))
    return tmp_path


@pytest.fixture()
def client(tmp_catalog_dir: Path) -> OSTIClient:
    """Create an OSTIClient pointed at a temporary catalog.

    Args:
        tmp_catalog_dir: Path to directory with test catalog data.

    Returns:
        An OSTIClient configured to use the test catalog.
    """
    return OSTIClient(data_path=str(tmp_catalog_dir))


@pytest.fixture()
def sample_osti_document() -> OSTIDocument:
    """Create a sample OSTIDocument for model testing.

    Returns:
        A fully populated OSTIDocument instance.
    """
    return OSTIDocument(
        osti_id="2342032",
        title="Recovery of Rare Earth Elements from Coal Fly Ash",
        authors=["Smith, John A.", "Doe, Jane B."],
        publication_date="2023-06-15",
        description="This report investigates methods for extracting rare earth elements.",
        subjects=["rare earth elements", "coal fly ash", "extraction"],
        commodity_category="HREE",
        doi="10.2172/2342032",
        product_type="Technical Report",
        research_orgs=["Pacific Northwest National Laboratory"],
        sponsor_orgs=["USDOE Office of Fossil Energy"],
    )
