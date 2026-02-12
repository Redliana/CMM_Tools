"""
LLM Tool schema generators and execution helpers.

Provides tool definitions compatible with OpenAI, Anthropic, and other LLMs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .search import (
    get_author_profile,
    get_paper_citations,
    search_author,
    search_scholar,
)

# Tool definitions in a provider-agnostic format
TOOL_DEFINITIONS = [
    {
        "name": "search_scholar",
        "description": """Search Google Scholar for academic papers across all publication types.

Searches comprehensively across:
- Peer-reviewed journal articles
- Conference proceedings (NeurIPS, ICML, ACL, CVPR, AAAI, etc.)
- Preprints (arXiv, bioRxiv, medRxiv, SSRN, etc.)
- Technical reports, theses, and books

Search tips:
- Add "arxiv" to query to find preprints
- Add conference names like "NeurIPS 2023" for proceedings
- Recent preprints may have low citations but cutting-edge research""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'retrieval augmented generation', 'transformer arxiv')",
                },
                "year_from": {
                    "type": "integer",
                    "description": "Filter papers from this year (inclusive)",
                },
                "year_to": {
                    "type": "integer",
                    "description": "Filter papers until this year (inclusive)",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum results to return (1-20, default 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
        "function": search_scholar,
    },
    {
        "name": "search_author",
        "description": "Search for an author on Google Scholar to find their author ID and basic info.",
        "parameters": {
            "type": "object",
            "properties": {
                "author_name": {
                    "type": "string",
                    "description": "Name of the author (e.g., 'Geoffrey Hinton')",
                },
            },
            "required": ["author_name"],
        },
        "function": search_author,
    },
    {
        "name": "get_author_profile",
        "description": "Get detailed author profile including h-index, citations, and publications using their Google Scholar author ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "author_id": {
                    "type": "string",
                    "description": "Google Scholar author ID (e.g., 'JicYPdAAAAAJ')",
                },
            },
            "required": ["author_id"],
        },
        "function": get_author_profile,
    },
    {
        "name": "get_paper_citations",
        "description": "Get papers that cite a given paper using its citation ID from search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "citation_id": {
                    "type": "string",
                    "description": "Citation ID from a previous search result",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum citing papers to return (1-20, default 10)",
                    "default": 10,
                },
            },
            "required": ["citation_id"],
        },
        "function": get_paper_citations,
    },
]

# Map tool names to functions
_TOOL_FUNCTIONS: dict[str, Callable] = {tool["name"]: tool["function"] for tool in TOOL_DEFINITIONS}


def get_tool_schemas() -> list[dict]:
    """
    Get tool definitions in a generic format.

    Returns a list of tool definitions that can be adapted to any LLM.
    """
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"],
        }
        for tool in TOOL_DEFINITIONS
    ]


def get_openai_tools() -> list[dict]:
    """
    Get tool definitions in OpenAI function calling format.

    Use with: client.chat.completions.create(tools=get_openai_tools(), ...)
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in TOOL_DEFINITIONS
    ]


def get_anthropic_tools() -> list[dict]:
    """
    Get tool definitions in Anthropic Claude format.

    Use with: client.messages.create(tools=get_anthropic_tools(), ...)
    """
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["parameters"],
        }
        for tool in TOOL_DEFINITIONS
    ]


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict:
    """
    Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments for the tool

    Returns:
        Tool result as a dictionary

    Raises:
        ValueError: If tool_name is not recognized
    """
    if tool_name not in _TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {tool_name}. Available: {list(_TOOL_FUNCTIONS.keys())}")

    func = _TOOL_FUNCTIONS[tool_name]
    result = func(**arguments)
    return result.to_dict()


def execute_tool_json(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a tool and return JSON string result.

    Convenience wrapper for LLM integrations that expect string responses.
    """
    result = execute_tool(tool_name, arguments)
    return json.dumps(result, indent=2)


# Convenience function for processing OpenAI tool calls
def process_openai_tool_call(tool_call) -> dict:
    """
    Process an OpenAI tool call and return the result.

    Args:
        tool_call: OpenAI tool call object from response.choices[0].message.tool_calls

    Returns:
        Dictionary with tool_call_id and result for the response
    """
    name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    result = execute_tool_json(name, arguments)

    return {
        "tool_call_id": tool_call.id,
        "role": "tool",
        "content": result,
    }


# Convenience function for processing Anthropic tool use
def process_anthropic_tool_use(tool_use_block) -> dict:
    """
    Process an Anthropic tool use block and return the result.

    Args:
        tool_use_block: Anthropic tool use content block

    Returns:
        Dictionary formatted for tool_result response
    """
    name = tool_use_block.name
    arguments = tool_use_block.input
    result = execute_tool_json(name, arguments)

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_block.id,
        "content": result,
    }
