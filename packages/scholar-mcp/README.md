# google-scholar-mcp

MCP server for searching Google Scholar via the SerpAPI service. Provides
tools for finding academic papers, retrieving citations, and looking up
author profiles across all publication types.

## Tools

| Tool | Description |
|------|-------------|
| `search_scholar` | Search Google Scholar for papers, preprints, and proceedings |
| `get_paper_citations` | Get papers that cite a given paper by citation ID |
| `search_author` | Search for an author by name to find their Scholar ID |
| `get_author_profile` | Get an author's full profile, publications, and metrics |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `SERPAPI_KEY` | Optional | SerpAPI key for Google Scholar access |

The server will start without a key but all tool calls will fail until one
is configured.

## Usage

```sh
uv run google-scholar-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "google-scholar-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "google-scholar-mcp"],
      "env": {
        "SERPAPI_KEY": "your-serpapi-key"
      }
    }
  }
}
```
