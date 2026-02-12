"""Tests for scholar.tools module.

Covers:
- TOOL_DEFINITIONS structure and contents
- get_tool_schemas() generic format
- get_openai_tools() OpenAI function-calling format
- get_anthropic_tools() Anthropic tool-use format
- execute_tool() with unknown tool name
- Module import smoke test
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scholar.tools import (
    _TOOL_FUNCTIONS,
    TOOL_DEFINITIONS,
    execute_tool,
    get_anthropic_tools,
    get_openai_tools,
    get_tool_schemas,
)

# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Verify the module and its public symbols are importable."""

    def test_import_tools_module(self) -> None:
        """Importing scholar.tools should succeed without errors."""
        import scholar.tools  # noqa: F401

    def test_tool_definitions_is_list(self) -> None:
        """TOOL_DEFINITIONS should be a non-empty list."""
        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) > 0

    def test_tool_functions_is_dict(self) -> None:
        """_TOOL_FUNCTIONS should be a dict mapping names to callables."""
        assert isinstance(_TOOL_FUNCTIONS, dict)
        assert len(_TOOL_FUNCTIONS) > 0


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS structure
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    """Tests for the TOOL_DEFINITIONS constant."""

    EXPECTED_TOOL_NAMES = {
        "search_scholar",
        "search_author",
        "get_author_profile",
        "get_paper_citations",
    }

    def test_contains_expected_tools(self) -> None:
        """All expected tool names should be present in TOOL_DEFINITIONS."""
        actual_names = {str(t["name"]) for t in TOOL_DEFINITIONS}
        assert actual_names == self.EXPECTED_TOOL_NAMES

    def test_each_tool_has_required_keys(self) -> None:
        """Each tool definition should have name, description, parameters, and function."""
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool, f"Missing 'name' key in tool: {tool}"
            assert "description" in tool, f"Missing 'description' in tool: {tool['name']}"
            assert "parameters" in tool, f"Missing 'parameters' in tool: {tool['name']}"
            assert "function" in tool, f"Missing 'function' in tool: {tool['name']}"

    def test_parameters_structure(self) -> None:
        """Each tool's parameters should be a JSON Schema object with properties."""
        for tool in TOOL_DEFINITIONS:
            params = tool["parameters"]
            assert params["type"] == "object", f"Tool {tool['name']} params type != object"
            assert "properties" in params, f"Tool {tool['name']} missing 'properties'"
            assert "required" in params, f"Tool {tool['name']} missing 'required'"

    def test_function_is_callable(self) -> None:
        """The function in each tool definition should be callable."""
        for tool in TOOL_DEFINITIONS:
            assert callable(tool["function"]), f"Tool {tool['name']} function not callable"

    def test_search_scholar_has_query_param(self) -> None:
        """The search_scholar tool should require a 'query' parameter."""
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "search_scholar")
        assert "query" in tool["parameters"]["properties"]
        assert "query" in tool["parameters"]["required"]

    def test_search_scholar_has_optional_params(self) -> None:
        """The search_scholar tool should have optional year and num_results params."""
        tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "search_scholar")
        props = tool["parameters"]["properties"]
        assert "year_from" in props
        assert "year_to" in props
        assert "num_results" in props


# ---------------------------------------------------------------------------
# get_tool_schemas
# ---------------------------------------------------------------------------


class TestGetToolSchemas:
    """Tests for get_tool_schemas()."""

    def test_returns_list(self) -> None:
        """get_tool_schemas() should return a list."""
        schemas = get_tool_schemas()
        assert isinstance(schemas, list)

    def test_length_matches_definitions(self) -> None:
        """The number of schemas should match TOOL_DEFINITIONS."""
        schemas = get_tool_schemas()
        assert len(schemas) == len(TOOL_DEFINITIONS)

    def test_schema_keys(self) -> None:
        """Each schema should have name, description, parameters (no function)."""
        for schema in get_tool_schemas():
            assert set(schema.keys()) == {"name", "description", "parameters"}

    def test_no_function_key_leaked(self) -> None:
        """Schemas should not leak the internal 'function' key."""
        for schema in get_tool_schemas():
            assert "function" not in schema


# ---------------------------------------------------------------------------
# get_openai_tools
# ---------------------------------------------------------------------------


class TestGetOpenaiTools:
    """Tests for get_openai_tools()."""

    def test_returns_list(self) -> None:
        """get_openai_tools() should return a list."""
        tools = get_openai_tools()
        assert isinstance(tools, list)

    def test_openai_format_structure(self) -> None:
        """Each item should have 'type': 'function' and a 'function' dict."""
        for tool in get_openai_tools():
            assert tool["type"] == "function"
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_no_extra_top_level_keys(self) -> None:
        """OpenAI tools should only have 'type' and 'function' at top level."""
        for tool in get_openai_tools():
            assert set(tool.keys()) == {"type", "function"}


# ---------------------------------------------------------------------------
# get_anthropic_tools
# ---------------------------------------------------------------------------


class TestGetAnthropicTools:
    """Tests for get_anthropic_tools()."""

    def test_returns_list(self) -> None:
        """get_anthropic_tools() should return a list."""
        tools = get_anthropic_tools()
        assert isinstance(tools, list)

    def test_anthropic_format_structure(self) -> None:
        """Each item should have name, description, and input_schema."""
        for tool in get_anthropic_tools():
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_input_schema_matches_parameters(self) -> None:
        """input_schema should be the same as the original parameters."""
        anthropic_tools = get_anthropic_tools()
        for i, tool in enumerate(anthropic_tools):
            assert tool["input_schema"] == TOOL_DEFINITIONS[i]["parameters"]

    def test_no_function_key(self) -> None:
        """Anthropic tool dicts should not contain a 'function' key."""
        for tool in get_anthropic_tools():
            assert "function" not in tool


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------


class TestExecuteTool:
    """Tests for execute_tool()."""

    def test_unknown_tool_raises_value_error(self) -> None:
        """Calling execute_tool with an unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            execute_tool("nonexistent_tool", {})

    def test_unknown_tool_error_lists_available(self) -> None:
        """The ValueError message should list available tool names."""
        with pytest.raises(ValueError, match="search_scholar"):
            execute_tool("bad_tool_name", {})

    @patch("scholar.tools._TOOL_FUNCTIONS", {"mock_tool": MagicMock()})
    def test_execute_calls_function_and_returns_dict(self) -> None:
        """execute_tool should call the function and return its to_dict() result."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"key": "value"}

        with patch.dict(
            "scholar.tools._TOOL_FUNCTIONS",
            {"mock_tool": MagicMock(return_value=mock_result)},
        ):
            result: dict[str, Any] = execute_tool("mock_tool", {"arg1": "val1"})
            assert result == {"key": "value"}

    @patch("scholar.tools._TOOL_FUNCTIONS")
    def test_execute_passes_arguments(self, mock_functions: MagicMock) -> None:
        """execute_tool should pass keyword arguments to the underlying function."""
        mock_func = MagicMock()
        mock_func.return_value.to_dict.return_value = {}
        mock_functions.__contains__ = MagicMock(return_value=True)
        mock_functions.__getitem__ = MagicMock(return_value=mock_func)

        execute_tool("search_scholar", {"query": "test", "num_results": 5})
        mock_func.assert_called_once_with(query="test", num_results=5)


# ---------------------------------------------------------------------------
# Cross-format consistency
# ---------------------------------------------------------------------------


class TestCrossFormatConsistency:
    """Ensure all format generators produce consistent tool sets."""

    def test_same_tool_names_across_formats(self) -> None:
        """All three format functions should produce the same set of tool names."""
        generic_names = {s["name"] for s in get_tool_schemas()}
        openai_names = {t["function"]["name"] for t in get_openai_tools()}
        anthropic_names = {t["name"] for t in get_anthropic_tools()}

        assert generic_names == openai_names == anthropic_names

    def test_same_count_across_formats(self) -> None:
        """All three format functions should return the same number of tools."""
        assert len(get_tool_schemas()) == len(get_openai_tools()) == len(get_anthropic_tools())
