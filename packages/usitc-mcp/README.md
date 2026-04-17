# usitc-mcp

MCP server for the US International Trade Commission (USITC) DataWeb API.
Provides US-specific import and export statistics at HTS (Harmonized Tariff
Schedule of the United States) granularity.

## Relationship to `uncomtrade-mcp`

| Aspect | `uncomtrade-mcp` | `usitc-mcp` |
|---|---|---|
| Reporting perspective | All UN members | United States only |
| Classification granularity | HS (2/4/6-digit) | HTS (up to 10-digit) |
| Trade-measure varieties | Imports / Exports | General imports, Imports for consumption, Domestic exports, Total exports |
| Use for CMM | Global bilateral flows | US supply-chain specifics (FTZ, stockpile, tariff exposure) |

Use both together for cross-validation and disaggregated US views.

## Tools

| Tool | Description |
|------|-------------|
| `get_api_status` | Check API connectivity and token validity |
| `list_cmm_hts_codes` | Curated HTS codes per critical mineral |
| `search_hts` | Search HTS classification by keyword or code prefix |
| `get_trade_data` | Fetch trade data by HTS, year, partner, data type |
| `get_critical_mineral_trade` | Query US trade for a curated CMM mineral |
| `compare_import_types` | General imports vs. imports for consumption side-by-side |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `USITC_API_TOKEN` | **Yes** | Bearer token from https://dataweb.usitc.gov → Account → API Access |

## Usage

```sh
uv run usitc-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "usitc-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "usitc-mcp"],
      "env": {
        "USITC_API_TOKEN": "your-bearer-token"
      }
    }
  }
}
```

## Important Caveat — Schema Validation Required

USITC DataWeb's v2 API does **not** publish a canonical OpenAPI / JSON-schema
document. The request/response shapes implemented in `client.py` reflect the
publicly observed structure as of 2026-Q1. Before relying on query results in
scientific outputs, Nancy should:

1. Obtain the authenticated token
2. Issue a known-good query (e.g., HTS `2836.91.00` imports from Chile, 2023)
3. Inspect the actual response envelope
4. Reconcile field names in `client.py::_post` parsing (`year`, `hts_code`,
   `value`, `quantity_1`, etc.) with the observed payload

Adjust `TradeRecord` field mappings accordingly. The scaffold is structured
to make this reconciliation localized — parsing lives in `get_trade_data`.

## Scope Notes

- v0.1.0 does **not** cover USITC's tariff-schedule lookups (MFN, column-2,
  special-program rates). A `get_tariff_rate` tool is a natural v0.2 addition.
- Tariff-line-level data is often subject to BIS/Census confidentiality
  suppressions at low-volume partners; expect gaps below ~$250,000/year per
  HTS×country×month cell.
