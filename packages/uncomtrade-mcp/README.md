# uncomtrade-mcp

MCP server for the UN Comtrade international trade database. Specializes in
critical minerals trade data (lithium, cobalt, rare earths, graphite, nickel,
manganese, gallium, germanium) but can query any HS-coded commodity.

## Tools

| Tool | Description |
|------|-------------|
| `get_api_status` | Check API connectivity and key validity |
| `list_critical_minerals` | List pre-configured critical minerals with HS codes |
| `list_reporters` | List available reporter countries with codes |
| `list_partners` | List available partner countries/areas |
| `list_commodities` | List HS commodity codes filtered by level or search term |
| `get_trade_data` | Get trade data for a reporter, commodity, and partner |
| `get_critical_mineral_trade` | Get trade data for a critical mineral using preset HS codes |
| `get_commodity_trade_summary` | Get a trade summary across major economies |
| `get_country_trade_profile` | Get a country's critical minerals import/export profile |

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `UNCOMTRADE_API_KEY` | Optional | UN Comtrade API subscription key for higher rate limits |

The server works without an API key but may hit rate limits on the free tier.

## Usage

```sh
uv run uncomtrade-mcp
```

### Claude Desktop

```json
{
  "mcpServers": {
    "uncomtrade-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/CMM_Tools", "run", "uncomtrade-mcp"],
      "env": {
        "UNCOMTRADE_API_KEY": "your-comtrade-key"
      }
    }
  }
}
```
