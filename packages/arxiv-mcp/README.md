# arxiv-mcp

MCP server for searching ArXiv papers and generating LLM-powered summaries.
Queries the ArXiv API for academic paper metadata and optionally summarizes
results using OpenAI or Anthropic models.

## Tools

| Tool | Description |
|------|-------------|
| `search_arxiv` | Search ArXiv for papers matching a query string |
| `get_arxiv_paper` | Get detailed metadata for a specific paper by ArXiv ID |
| `summarize_paper_with_llm` | Fetch a paper and generate an LLM summary (OpenAI or Anthropic) |
| `search_and_summarize` | Search ArXiv and auto-summarize the top results in one step |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Optional | OpenAI API key for GPT-based summaries |
| `ANTHROPIC_API_KEY` | Optional | Anthropic API key for Claude-based summaries |

At least one LLM key is needed for the summarization tools. Search works without any keys.

## Usage

```sh
uv run arxiv-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "arxiv-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "arxiv-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```
