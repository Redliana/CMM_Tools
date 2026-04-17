# census-aes-mcp

MCP server for US Census Bureau International Trade data (AES-derived exports
and Customs-derived imports). Monthly HS-granular US trade statistics.

## Scope Clarification

AES (Automated Export System) is the filing interface through which US
exporters submit Electronic Export Information (EEI). AES itself is not a
query API. Publicly queryable AES-derived data are served via the **Census
International Trade time-series API** (`api.census.gov/data/timeseries/intltrade/`),
which is what this server wraps.

The server is named `census-aes-mcp` to reflect the AES provenance of the
export data, per user convention. It also exposes the paired import data
(which is *not* AES-derived but comes from Customs entry filings) because CMM
analysis typically requires both flows.

## Tools

| Tool | Description |
|------|-------------|
| `get_api_status` | Check API connectivity and key validity |
| `list_cmm_hs_codes` | Curated HS codes per critical mineral |
| `get_exports` | US exports by HS, year, month, destination |
| `get_imports` | US imports by HS, year, month, origin |
| `get_cmm_mineral_trade` | Curated CMM mineral trade (exports or imports) |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `CENSUS_API_KEY` | **Yes** | Obtain at https://api.census.gov/data/key_signup.html |

## Usage

```sh
uv run census-aes-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "census-aes-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "census-aes-mcp"],
      "env": {
        "CENSUS_API_KEY": "your-census-key"
      }
    }
  }
}
```

## Comparison to Sibling MCP Servers

| Server | Reporter | Partner detail | Classification | Frequency | Authoritative for |
|---|---|---|---|---|---|
| `uncomtrade-mcp` | All UN members | Yes (bilateral) | HS | Annual | Global bilateral flows |
| `usitc-mcp` | USA only | Yes | HTS (2–10 digit) | Annual (monthly via DataWeb queries) | US tariff-line detail, import-type distinction |
| `census-aes-mcp` | USA only | Yes | HS (6/10 digit) | Monthly | US monthly timeliness; AES-derived export disclosures |

For CMM analyses requiring cross-validation, run the same query via
`uncomtrade-mcp` (USA reporter) and `census-aes-mcp` and compare — systematic
discrepancies often reflect timing adjustments, revision lags, or the UN's
in-kind aid inclusions.

## Scope Notes

- Confidentiality suppressions apply at low-volume HS × country × month cells;
  expect missing values. The `_safe_float` parser normalizes these to `None`.
- The Census API uses HS codes without dots ("283691"), whereas USITC uses
  dotted HTS ("2836.91.00"). Codes returned by this server are un-dotted.
- This v0.1.0 does not query `porths` (port-level) or `statenaics` (state-level);
  add those endpoints in a v0.2.0 if state-sourced export analysis is needed.
- Schedule C (country) codes differ from ISO3; a `list_schedule_c_countries`
  reference tool is a sensible v0.2.0 addition.
