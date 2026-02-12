# bgs-mcp

MCP server for the British Geological Survey (BGS) World Mineral Statistics API.
Provides access to global mineral production, import, and export data from 1970
to present, covering critical minerals and base metals.

## Tools

| Tool | Description |
|------|-------------|
| `list_commodities` | List available mineral commodities (optionally critical minerals only) |
| `list_countries` | List countries with mineral data, optionally filtered by commodity |
| `search_production` | Search for mineral production or trade data with filters |
| `get_commodity_ranking` | Get top producing countries for a commodity in a given year |
| `get_time_series` | Get historical time-series data for a commodity |
| `compare_countries` | Compare mineral production across multiple countries |
| `get_country_profile` | Get a country's full mineral production profile |
| `get_api_info` | Get BGS API documentation and data coverage details |

## Configuration

No environment variables are required. The BGS API is publicly accessible.

## Usage

Run as an MCP server (STDIO transport):

```sh
uv run bgs-mcp
```

Run as a REST API server:

```sh
uv run bgs-api
```

### Claude Desktop

```json
{
  "mcpServers": {
    "bgs-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "bgs-mcp"],
      "env": {}
    }
  }
}
```
