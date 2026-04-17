# fred-mcp

MCP server for the Federal Reserve Economic Data (FRED) API. Provides
macroeconomic and commodity-price time series relevant to critical-minerals
supply-chain analysis.

## Tools

| Tool | Description |
|------|-------------|
| `get_api_status` | Check FRED API connectivity and key validity |
| `list_cmm_series` | List curated CMM-relevant series (commodity prices, production, FX, etc.) |
| `search_series` | Search the FRED series catalogue by keyword |
| `get_series_metadata` | Fetch metadata (units, frequency, revision dates) for a series |
| `get_observations` | Fetch observations for any FRED series by ID |
| `get_cmm_dashboard` | Latest observation for every curated CMM series |
| `list_categories` | Browse FRED category taxonomy |
| `list_category_series` | List series under a specific category |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `FRED_API_KEY` | **Yes** | Obtain at https://fred.stlouisfed.org/docs/api/api_key.html |

## Usage

```sh
uv run fred-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "fred-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "fred-mcp"],
      "env": {
        "FRED_API_KEY": "your-fred-key"
      }
    }
  }
}
```

## Curated CMM Series

The `list_cmm_series` tool exposes a curated set spanning:

- **Commodity prices**: PALLFNFINDEXM, PMETAINDEXM, PCOPPUSDM, PNICKUSDM, PALUMINUMUSDM, PIORECRUSDM
- **Production**: INDPRO, IPMINE, IPMANSICS
- **Trade prices**: IR (import PI), IQ (export PI)
- **Logistics**: WPU301 (transportation PPI)
- **FX**: DEXCHUS (CNY/USD), DEXUSEU (USD/EUR)
- **Rates**: DGS10 (10Y Treasury)

Edit `CMM_FRED_SERIES` in `models.py` to extend.

## Scope Notes

- FRED observations with value `"."` (missing) are normalized to `None`.
- FRED imposes a rate limit of 120 requests per 60-second window. The
  `get_cmm_dashboard` tool paces requests with `asyncio.sleep(0.1)` but still
  issues ~15 requests; adjust if you extend the curated set substantially.
- ALFRED (archival / real-time vintages) is not yet exposed; open an issue if needed.
