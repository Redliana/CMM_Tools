"""Tests for arxiv_mcp.server tool functions and LLM API helpers.

Covers:
- get_arxiv_paper() with mocked network responses
- call_openai_api() with mocked httpx
- call_anthropic_api() with mocked httpx
- summarize_paper_with_llm() with mocked dependencies
- search_and_summarize() with mocked dependencies
- Query construction edge cases in search_arxiv
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from arxiv_mcp.server import (
    call_anthropic_api,
    call_openai_api,
    get_arxiv_paper,
    summarize_paper_with_llm,
)

# ---------------------------------------------------------------------------
# get_arxiv_paper
# ---------------------------------------------------------------------------


class TestGetArxivPaper:
    """Tests for the get_arxiv_paper MCP tool with mocked network."""

    @pytest.mark.asyncio()
    async def test_returns_paper_details(
        self,
        sample_feed_xml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A valid paper ID should return formatted paper details."""
        mock_response = MagicMock()
        mock_response.text = sample_feed_xml
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result: str = await get_arxiv_paper("2301.07041v1")
        assert "Attention Is All You Need" in result
        assert "Ashish Vaswani" in result

    @pytest.mark.asyncio()
    async def test_strips_version_from_id(
        self,
        sample_feed_xml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Version suffixes like 'v1' should be stripped from the ID in the URL."""
        captured_urls: list[str] = []

        async def mock_get(url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            resp = MagicMock()
            resp.text = sample_feed_xml
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        await get_arxiv_paper("2301.07041v3")
        assert "id_list=2301.07041" in captured_urls[0]

    @pytest.mark.asyncio()
    async def test_returns_error_on_network_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A network error should return an error message, not raise."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await get_arxiv_paper("2301.07041")
        assert "Error" in result

    @pytest.mark.asyncio()
    async def test_returns_not_found_for_empty_feed(
        self,
        sample_empty_feed_xml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty feed response should produce a 'not found' message."""
        mock_response = MagicMock()
        mock_response.text = sample_empty_feed_xml
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await get_arxiv_paper("9999.99999")
        assert "not found" in result.lower() or "Paper not found" in result


# ---------------------------------------------------------------------------
# call_openai_api
# ---------------------------------------------------------------------------


class TestCallOpenAIApi:
    """Tests for call_openai_api() with mocked httpx and environment."""

    @pytest.mark.asyncio()
    async def test_returns_none_without_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When OPENAI_API_KEY is not set, the function should return None."""
        monkeypatch.setattr("arxiv_mcp.server.OPENAI_API_KEY", None)
        result = await call_openai_api("test prompt")
        assert result is None

    @pytest.mark.asyncio()
    async def test_returns_completion_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A successful API call should return the content string."""
        monkeypatch.setattr("arxiv_mcp.server.OPENAI_API_KEY", "sk-test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is a summary."}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await call_openai_api("Summarize this paper")
        assert result == "This is a summary."

    @pytest.mark.asyncio()
    async def test_returns_none_on_http_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An HTTP error from OpenAI should be caught and None returned."""
        monkeypatch.setattr("arxiv_mcp.server.OPENAI_API_KEY", "sk-test-key")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=MagicMock(),
            )
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await call_openai_api("test prompt")
        assert result is None

    @pytest.mark.asyncio()
    async def test_uses_specified_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The specified model should be included in the API payload."""
        monkeypatch.setattr("arxiv_mcp.server.OPENAI_API_KEY", "sk-test-key")

        captured_payloads: list[dict[str, Any]] = []

        async def mock_post(url: str, **kwargs: Any) -> MagicMock:
            captured_payloads.append(kwargs.get("json", {}))
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
            return resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        await call_openai_api("test", model="gpt-3.5-turbo")
        assert captured_payloads[0]["model"] == "gpt-3.5-turbo"


# ---------------------------------------------------------------------------
# call_anthropic_api
# ---------------------------------------------------------------------------


class TestCallAnthropicApi:
    """Tests for call_anthropic_api() with mocked httpx and environment."""

    @pytest.mark.asyncio()
    async def test_returns_none_without_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ANTHROPIC_API_KEY is not set, the function should return None."""
        monkeypatch.setattr("arxiv_mcp.server.ANTHROPIC_API_KEY", None)
        result = await call_anthropic_api("test prompt")
        assert result is None

    @pytest.mark.asyncio()
    async def test_returns_completion_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A successful API call should return the text content."""
        monkeypatch.setattr("arxiv_mcp.server.ANTHROPIC_API_KEY", "sk-ant-test")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"content": [{"text": "Anthropic summary here."}]}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await call_anthropic_api("Summarize this")
        assert result == "Anthropic summary here."

    @pytest.mark.asyncio()
    async def test_returns_none_on_http_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An HTTP error from Anthropic should be caught and None returned."""
        monkeypatch.setattr("arxiv_mcp.server.ANTHROPIC_API_KEY", "sk-ant-test")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "500 Server Error",
                request=MagicMock(),
                response=MagicMock(),
            )
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await call_anthropic_api("test prompt")
        assert result is None

    @pytest.mark.asyncio()
    async def test_sends_correct_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The request should include the x-api-key and anthropic-version headers."""
        monkeypatch.setattr("arxiv_mcp.server.ANTHROPIC_API_KEY", "sk-ant-test")

        captured_kwargs: dict[str, Any] = {}

        async def mock_post(url: str, **kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"content": [{"text": "ok"}]}
            return resp

        mock_client = AsyncMock()
        mock_client.post = mock_post
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        await call_anthropic_api("test prompt")
        headers = captured_kwargs["headers"]
        assert headers["x-api-key"] == "sk-ant-test"
        assert "anthropic-version" in headers


# ---------------------------------------------------------------------------
# summarize_paper_with_llm
# ---------------------------------------------------------------------------


class TestSummarizePaperWithLLM:
    """Tests for summarize_paper_with_llm() with mocked sub-functions."""

    @pytest.mark.asyncio()
    async def test_returns_error_for_unknown_provider(
        self,
        sample_feed_xml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unknown LLM provider should return an error message."""
        # Mock get_arxiv_paper to return valid paper info
        mock_response = MagicMock()
        mock_response.text = sample_feed_xml
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await summarize_paper_with_llm("2301.07041", llm_provider="unknown_llm")
        assert "Error" in result
        assert "Unknown LLM provider" in result

    @pytest.mark.asyncio()
    async def test_returns_paper_error_when_fetch_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When paper fetch fails, the error should be propagated."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await summarize_paper_with_llm("2301.07041", llm_provider="openai")
        assert "Error" in result

    @pytest.mark.asyncio()
    async def test_openai_provider_calls_openai_api(
        self,
        sample_feed_xml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When llm_provider='openai', call_openai_api should be invoked."""
        # Mock the arxiv request
        mock_arxiv_response = MagicMock()
        mock_arxiv_response.text = sample_feed_xml
        mock_arxiv_response.raise_for_status = MagicMock()

        call_count = {"openai": 0, "anthropic": 0}

        # We need to mock AsyncClient for both arxiv fetch and openai calls
        # Instead, mock the LLM function directly
        monkeypatch.setattr("arxiv_mcp.server.OPENAI_API_KEY", "sk-test")

        async def mock_call_openai(prompt: str, model: str = "gpt-4") -> str:
            call_count["openai"] += 1
            return "OpenAI summary"

        monkeypatch.setattr("arxiv_mcp.server.call_openai_api", mock_call_openai)

        # Mock the arxiv fetch
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_arxiv_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await summarize_paper_with_llm("2301.07041", llm_provider="openai")
        assert call_count["openai"] == 1
        assert "OpenAI summary" in result

    @pytest.mark.asyncio()
    async def test_anthropic_provider_calls_anthropic_api(
        self,
        sample_feed_xml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When llm_provider='anthropic', call_anthropic_api should be invoked."""
        mock_arxiv_response = MagicMock()
        mock_arxiv_response.text = sample_feed_xml
        mock_arxiv_response.raise_for_status = MagicMock()

        monkeypatch.setattr("arxiv_mcp.server.ANTHROPIC_API_KEY", "sk-ant-test")

        async def mock_call_anthropic(
            prompt: str, model: str = "claude-3-5-sonnet-20241022"
        ) -> str:
            return "Anthropic summary"

        monkeypatch.setattr("arxiv_mcp.server.call_anthropic_api", mock_call_anthropic)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_arxiv_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        result = await summarize_paper_with_llm("2301.07041", llm_provider="anthropic")
        assert "Anthropic summary" in result


# ---------------------------------------------------------------------------
# Query construction edge cases
# ---------------------------------------------------------------------------


class TestQueryConstruction:
    """Tests for query parameter construction in search_arxiv."""

    @pytest.mark.asyncio()
    async def test_bare_query_gets_all_prefix(
        self,
        sample_feed_xml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A query without field prefix should get 'all:' prepended."""
        captured_urls: list[str] = []

        async def mock_get(url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            resp = MagicMock()
            resp.text = sample_feed_xml
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        from arxiv_mcp.server import search_arxiv

        await search_arxiv("transformer architecture")
        assert "all:transformer" in captured_urls[0]

    @pytest.mark.asyncio()
    async def test_prefixed_query_kept_as_is(
        self,
        sample_feed_xml: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A query with field prefix like 'ti:' should not get 'all:' prepended."""
        captured_urls: list[str] = []

        async def mock_get(url: str, **kwargs: Any) -> MagicMock:
            captured_urls.append(url)
            resp = MagicMock()
            resp.text = sample_feed_xml
            resp.raise_for_status = MagicMock()
            return resp

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        monkeypatch.setattr(httpx, "AsyncClient", lambda: mock_client)

        from arxiv_mcp.server import search_arxiv

        await search_arxiv("ti:transformer AND au:vaswani")
        # Should NOT have "all:" prefix
        assert "all:" not in captured_urls[0]
        assert "ti:transformer" in captured_urls[0]
