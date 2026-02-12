"""Basic tests for UN Comtrade client."""

from __future__ import annotations

import asyncio
import os

from uncomtrade_mcp.client import ComtradeClient
from uncomtrade_mcp.models import CRITICAL_MINERAL_HS_CODES


def test_client_initialization():
    """Test client can be initialized."""
    client = ComtradeClient()
    assert client is not None


def test_critical_minerals_defined():
    """Test critical minerals HS codes are defined."""
    assert "lithium" in CRITICAL_MINERAL_HS_CODES
    assert "cobalt" in CRITICAL_MINERAL_HS_CODES
    assert "rare_earth" in CRITICAL_MINERAL_HS_CODES
    assert "graphite" in CRITICAL_MINERAL_HS_CODES
    assert len(CRITICAL_MINERAL_HS_CODES) >= 9


def test_api_key_from_env():
    """Test API key can be loaded from environment."""
    # Set a test key
    os.environ["UNCOMTRADE_API_KEY"] = "test-key"
    client = ComtradeClient()
    assert client.is_available()
    assert client.api_key == "test-key"

    # Clean up
    del os.environ["UNCOMTRADE_API_KEY"]


async def check_api_status():
    """Check API connectivity (requires network)."""
    client = ComtradeClient()
    status = await client.check_status()
    print(f"API Status: {status}")
    return status


if __name__ == "__main__":
    # Run a quick connectivity check
    result = asyncio.run(check_api_status())
    print(f"Result: {result}")
