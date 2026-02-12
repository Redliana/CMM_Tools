# google-scholar-api

Standalone Google Scholar search library designed for use with any LLM. Provides
search functions for papers, authors, and citations, plus ready-made tool schemas
for OpenAI, Anthropic, and generic LLM tool-calling APIs.

## Installation

```bash
uv pip install -e packages/scholar-api
uv pip install -e "packages/scholar-api[all]"   # openai + anthropic + ollama
```

## Quick Start

```python
from scholar import search_scholar, set_api_key

set_api_key("your-serpapi-key")  # or set SERPAPI_KEY env var

results = search_scholar("critical minerals supply chain", year_from=2022)
for paper in results.papers:
    print(f"{paper.title} ({paper.year}) - {paper.citations} citations")
```

### LLM Integration

```python
# OpenAI
from scholar import get_openai_tools, process_openai_tool_call
response = client.chat.completions.create(
    model="gpt-4", messages=messages, tools=get_openai_tools(),
)
for tool_call in response.choices[0].message.tool_calls:
    messages.append(process_openai_tool_call(tool_call))

# Anthropic
from scholar import get_anthropic_tools, process_anthropic_tool_use
response = client.messages.create(
    model="claude-sonnet-4-20250514", messages=messages, tools=get_anthropic_tools(),
)
for block in response.content:
    if block.type == "tool_use":
        result = process_anthropic_tool_use(block)
```

## API Reference

### Search Functions

- `search_scholar(query, year_from, year_to, num_results)` -- Search papers. Returns `ScholarResult`.
- `search_author(author_name)` -- Find authors by name. Returns `AuthorResult`.
- `get_author_profile(author_id)` -- Profile by Scholar ID (h-index, publications). Returns `AuthorResult`.
- `get_paper_citations(citation_id, num_results)` -- Citing papers. Returns `CitationResult`.
- `set_api_key(key)` -- Set the SerpAPI key for all requests.

### LLM Tool Helpers

- `get_openai_tools()` -- Tool definitions in OpenAI function calling format.
- `get_anthropic_tools()` -- Tool definitions in Anthropic Claude format.
- `get_tool_schemas()` -- Provider-agnostic tool definitions.
- `process_openai_tool_call(tool_call)` -- Execute and format an OpenAI tool call.
- `process_anthropic_tool_use(tool_use_block)` -- Execute and format an Anthropic tool use block.
- `execute_tool(tool_name, arguments)` -- Run any tool by name, returns dict.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `SERPAPI_KEY` | SerpAPI key for Google Scholar queries | (none -- required) |

Get a free key at [https://serpapi.com](https://serpapi.com). Also accepts `.env` files
or `set_api_key()` at runtime.

## Dependencies

google-search-results, python-dotenv. Optional: openai, anthropic, ollama.
