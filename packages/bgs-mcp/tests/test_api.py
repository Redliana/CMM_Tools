"""Tests for bgs_mcp.api module.

Tests cover the FastAPI application's Pydantic response models, the endpoint
behavior (via TestClient), and smoke-test verification that routes are
correctly registered.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from bgs_mcp.api import (
    CommodityList,
    CountryInfo,
    OpenAIFunction,
    ProductionRecord,
    RankedCountry,
    TimeSeriesPoint,
    app,
)
from bgs_mcp.bgs_client import MineralRecord

# ---------------------------------------------------------------------------
# Pydantic response model tests
# ---------------------------------------------------------------------------


class TestCommodityListModel:
    """Tests for the CommodityList response model."""

    def test_construction(self) -> None:
        """Verify CommodityList can be constructed with expected fields."""
        model = CommodityList(
            total=2,
            commodities=["lithium minerals", "cobalt, mine"],
            categories={"battery": ["lithium minerals", "cobalt, mine"]},
        )
        assert model.total == 2
        assert len(model.commodities) == 2
        assert "battery" in model.categories  # type: ignore[operator]

    def test_categories_optional(self) -> None:
        """Verify CommodityList categories default to None."""
        model = CommodityList(total=0, commodities=[])
        assert model.categories is None


class TestCountryInfoModel:
    """Tests for the CountryInfo response model."""

    def test_construction(self) -> None:
        """Verify CountryInfo construction with all fields."""
        info = CountryInfo(name="Australia", iso2="AU", iso3="AUS")
        assert info.name == "Australia"
        assert info.iso2 == "AU"
        assert info.iso3 == "AUS"

    def test_iso_codes_accept_none(self) -> None:
        """Verify ISO codes accept None values."""
        info = CountryInfo(name="Unknown Territory", iso2=None, iso3=None)
        assert info.iso2 is None
        assert info.iso3 is None


class TestProductionRecordModel:
    """Tests for the ProductionRecord response model."""

    def test_construction(self) -> None:
        """Verify ProductionRecord construction."""
        record = ProductionRecord(
            commodity="lithium minerals",
            country="Australia",
            country_iso3="AUS",
            year=2022,
            quantity=61000.0,
            units="Tonnes (metric)",
        )
        assert record.commodity == "lithium minerals"
        assert record.year == 2022

    def test_fields_accept_none(self) -> None:
        """Verify nullable fields accept None values."""
        record = ProductionRecord(
            commodity="test",
            country="test",
            country_iso3=None,
            year=None,
            quantity=None,
            units=None,
        )
        assert record.country_iso3 is None
        assert record.year is None
        assert record.quantity is None
        assert record.units is None


class TestRankedCountryModel:
    """Tests for the RankedCountry response model."""

    def test_construction(self) -> None:
        """Verify RankedCountry construction."""
        ranked = RankedCountry(
            rank=1,
            country="Australia",
            country_iso3="AUS",
            quantity=61000.0,
            share_percent=62.24,
        )
        assert ranked.rank == 1
        assert ranked.share_percent == 62.24


class TestTimeSeriesPointModel:
    """Tests for the TimeSeriesPoint response model."""

    def test_construction_with_yoy(self) -> None:
        """Verify TimeSeriesPoint with year-over-year change."""
        point = TimeSeriesPoint(year=2022, quantity=61000.0, yoy_change_percent=10.5)
        assert point.yoy_change_percent == 10.5

    def test_yoy_defaults_to_none(self) -> None:
        """Verify yoy_change_percent defaults to None."""
        point = TimeSeriesPoint(year=2022, quantity=61000.0)
        assert point.yoy_change_percent is None


class TestOpenAIFunctionModel:
    """Tests for the OpenAIFunction response model."""

    def test_construction(self) -> None:
        """Verify OpenAIFunction construction."""
        func = OpenAIFunction(
            name="search_mineral_production",
            description="Search for mineral production data",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        assert func.name == "search_mineral_production"
        assert isinstance(func.parameters, dict)


# ---------------------------------------------------------------------------
# FastAPI TestClient endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client() -> TestClient:
    """Create a FastAPI TestClient for endpoint testing."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    def test_root_returns_api_info(self, test_client: TestClient) -> None:
        """Verify the root endpoint returns API metadata."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "BGS World Mineral Statistics API"
        assert "version" in data
        assert "endpoints" in data

    def test_root_includes_endpoint_listing(self, test_client: TestClient) -> None:
        """Verify the root endpoint lists all available endpoints."""
        response = test_client.get("/")
        data = response.json()
        endpoints = data["endpoints"]
        assert "commodities" in endpoints
        assert "countries" in endpoints
        assert "production" in endpoints


class TestCommoditiesEndpoint:
    """Tests for the /commodities endpoint."""

    def test_critical_only_returns_predefined_list(self, test_client: TestClient) -> None:
        """Verify critical_only=true returns the predefined mineral list without HTTP."""
        response = test_client.get("/commodities", params={"critical_only": True})
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0
        assert isinstance(data["commodities"], list)
        assert "lithium minerals" in data["commodities"]

    def test_critical_only_with_categorize(self, test_client: TestClient) -> None:
        """Verify categorization groups minerals by type."""
        response = test_client.get(
            "/commodities",
            params={"critical_only": True, "categorize": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["categories"] is not None
        # Should have battery minerals
        assert "battery" in data["categories"]
        assert len(data["categories"]["battery"]) > 0

    def test_critical_only_without_categorize(self, test_client: TestClient) -> None:
        """Verify no categories when categorize=false."""
        response = test_client.get(
            "/commodities",
            params={"critical_only": True, "categorize": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["categories"] is None


class TestProductionSearchEndpoint:
    """Tests for the /production/search endpoint with mocked BGS client."""

    def test_search_with_mocked_client(self, test_client: TestClient) -> None:
        """Verify /production/search returns formatted records from mocked data."""
        mock_records = [
            MineralRecord(
                commodity="lithium minerals",
                statistic_type="Production",
                country="Australia",
                country_iso3="AUS",
                year=2022,
                quantity=61000.0,
                units="Tonnes (metric)",
            ),
        ]

        with patch("bgs_mcp.api.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.search_production.return_value = mock_records
            mock_get_client.return_value = mock_client

            response = test_client.get(
                "/production/search",
                params={"commodity": "lithium minerals"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["records"][0]["commodity"] == "lithium minerals"
        assert data["records"][0]["country"] == "Australia"

    def test_search_requires_commodity(self, test_client: TestClient) -> None:
        """Verify /production/search returns 422 when commodity is missing."""
        response = test_client.get("/production/search")
        assert response.status_code == 422


class TestProductionRankingEndpoint:
    """Tests for the /production/ranking endpoint with mocked BGS client."""

    def test_ranking_with_mocked_client(self, test_client: TestClient) -> None:
        """Verify /production/ranking returns ranked countries from mocked data."""
        mock_ranked = [
            {
                "country": "Australia",
                "country_iso3": "AUS",
                "quantity": 61000.0,
                "units": "Tonnes (metric)",
                "year": 2022,
            },
            {
                "country": "Chile",
                "country_iso3": "CHL",
                "quantity": 39000.0,
                "units": "Tonnes (metric)",
                "year": 2022,
            },
        ]

        with patch("bgs_mcp.api.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_commodity_by_country.return_value = mock_ranked
            mock_get_client.return_value = mock_client

            response = test_client.get(
                "/production/ranking",
                params={"commodity": "lithium minerals"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["commodity"] == "lithium minerals"
        assert len(data["rankings"]) == 2
        assert data["rankings"][0]["rank"] == 1
        assert data["rankings"][0]["country"] == "Australia"
        assert data["total_quantity"] == 100000.0

    def test_ranking_404_when_no_data(self, test_client: TestClient) -> None:
        """Verify /production/ranking returns 404 for unknown commodity."""
        with patch("bgs_mcp.api.get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get_commodity_by_country.return_value = []
            mock_get_client.return_value = mock_client

            response = test_client.get(
                "/production/ranking",
                params={"commodity": "unobtanium"},
            )

        assert response.status_code == 404


class TestOpenAIFunctionsEndpoint:
    """Tests for the /openai/functions endpoint."""

    def test_returns_function_definitions(self, test_client: TestClient) -> None:
        """Verify /openai/functions returns a list of function schemas."""
        response = test_client.get("/openai/functions")
        assert response.status_code == 200
        functions = response.json()
        assert isinstance(functions, list)
        assert len(functions) > 0

    def test_functions_have_expected_keys(self, test_client: TestClient) -> None:
        """Verify each function has name, description, and parameters."""
        response = test_client.get("/openai/functions")
        functions = response.json()
        for func in functions:
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_includes_search_function(self, test_client: TestClient) -> None:
        """Verify the search_mineral_production function is registered."""
        response = test_client.get("/openai/functions")
        functions = response.json()
        names = [f["name"] for f in functions]
        assert "search_mineral_production" in names

    def test_includes_critical_minerals_function(self, test_client: TestClient) -> None:
        """Verify the list_critical_minerals function is registered."""
        response = test_client.get("/openai/functions")
        functions = response.json()
        names = [f["name"] for f in functions]
        assert "list_critical_minerals" in names


# ---------------------------------------------------------------------------
# Route registration smoke tests
# ---------------------------------------------------------------------------


class TestRouteRegistration:
    """Smoke tests to verify all expected routes are registered."""

    def test_docs_endpoint_exists(self, test_client: TestClient) -> None:
        """Verify the Swagger docs endpoint is accessible."""
        response = test_client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_endpoint(self, test_client: TestClient) -> None:
        """Verify the OpenAPI JSON schema is accessible."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/commodities" in schema["paths"]
        assert "/production/search" in schema["paths"]
        assert "/production/ranking" in schema["paths"]
        assert "/production/timeseries" in schema["paths"]
        assert "/production/compare" in schema["paths"]
