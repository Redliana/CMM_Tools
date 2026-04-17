# worldbank-mcp

MCP server for the World Bank World Development Indicators (WDI) API. Provides
macroeconomic, trade, and governance context relevant to critical-minerals
supply-chain analysis.

## Tools

| Tool | Description |
|------|-------------|
| `get_api_status` | Check WDI API connectivity |
| `list_cmm_indicators` | List curated CMM-relevant WDI indicators |
| `list_cmm_key_economies` | List ISO3 codes of key CMM supply-chain economies |
| `search_indicators` | Search the WDI catalogue by keyword |
| `list_countries` | List World Bank countries with region/income classification |
| `get_indicator` | Fetch indicator observations for country/years |
| `get_cmm_profile` | Fetch the curated CMM bundle for a single country |
| `compare_countries` | Pivot table of one indicator across multiple countries (markdown) |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `WORLDBANK_API_KEY` | No | WDI is public; reserved for future WITS authenticated endpoints |

## Usage

```sh
uv run worldbank-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "worldbank-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "worldbank-mcp"]
    }
  }
}
```

## Scope Notes

- **WDI coverage only in v0.1.0.** WITS (World Integrated Trade Solution) endpoints
  return SDMX XML and have a distinct rate/auth model; a `wits_*` tool family is
  deferred to v0.2.0.
- Observations with `value = None` represent legitimate missing data (country did
  not report or indicator unavailable) rather than errors.
- The CMM indicator bundle (`list_cmm_indicators`) is a curated subset; browse the
  full catalogue via `search_indicators`.
