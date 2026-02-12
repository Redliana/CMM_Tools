"""Shared fixtures for claimm-mcp tests."""

from __future__ import annotations

from typing import Any

import pytest

from claimm_mcp.config import Settings


@pytest.fixture()
def env_vars(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set minimal required environment variables for Settings construction.

    Returns:
        Dictionary of the environment variable names and values that were set.
    """
    vars_map = {
        "EDX_API_KEY": "test-edx-api-key-123",
        "ANTHROPIC_API_KEY": "test-anthropic-key-456",
    }
    for key, value in vars_map.items():
        monkeypatch.setenv(key, value)
    return vars_map


@pytest.fixture()
def settings(env_vars: dict[str, str]) -> Settings:
    """Create a Settings instance with test environment variables.

    Args:
        env_vars: Fixture that sets required env vars.

    Returns:
        A fully-constructed Settings object using test values.
    """
    return Settings()  # type: ignore[call-arg]


@pytest.fixture()
def settings_with_reset(settings: Settings) -> Settings:
    """Provide settings and reset the global singleton afterward.

    Args:
        settings: A Settings instance for testing.

    Returns:
        The same Settings instance.
    """
    from claimm_mcp import config as config_module

    original = config_module._settings
    config_module._settings = settings
    yield settings
    config_module._settings = original


@pytest.fixture()
def sample_csv_content() -> str:
    """Provide a realistic multi-line CSV string for header detection tests.

    Returns:
        A CSV string with headers and several data rows.
    """
    return (
        "sample_id,element,concentration_ppm,date_collected,is_valid\n"
        "S001,Li,45.2,2024-01-15,true\n"
        "S002,Co,12.8,2024-02-20,false\n"
        "S003,Li,67.1,2024-03-10,true\n"
        "S004,Ni,,2024-04-05,true\n"
        "S005,Co,23.4,2024-05-12,false\n"
    )


@pytest.fixture()
def sample_tsv_content() -> str:
    """Provide a tab-separated CSV string.

    Returns:
        A TSV string with headers and data rows.
    """
    return "name\tvalue\tcategory\nalpha\t100\tA\nbeta\t200\tB\ngamma\t300\tA\n"


@pytest.fixture()
def mock_edx_response() -> dict[str, Any]:
    """Provide a mock CKAN/EDX API success response.

    Returns:
        A dictionary mimicking a successful EDX API response.
    """
    return {
        "success": True,
        "result": {
            "id": "test-resource-id",
            "name": "test-resource.csv",
            "description": "A test resource",
            "format": "CSV",
            "size": 1024,
            "url": "https://edx.netl.doe.gov/resource/test-resource-id/download",
            "created": "2024-01-01T00:00:00",
            "last_modified": "2024-06-01T00:00:00",
            "package_id": "test-package-id",
        },
    }
