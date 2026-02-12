# cmm-data

Unified data access library for Critical Minerals Modeling (CMM). Provides loaders for
USGS commodity statistics, ore deposits, OSTI documents, OECD supply chain data, NETL
REE coal data, Mindat minerals, and Geoscience Australia chronostratigraphic models.

## Installation

```bash
uv pip install -e packages/cmm-data
# With optional geospatial and visualization extras:
uv pip install -e "packages/cmm-data[geo,viz]"
```

## Quick Start

```python
import cmm_data

# Configure data root (or set CMM_DATA_PATH env var)
cmm_data.configure(data_root="/path/to/Globus_Sharing")

# Load USGS world production data for lithium
df = cmm_data.load_usgs_commodity("lithi", data_type="world")

# Load ore deposits
ore_df = cmm_data.load_ore_deposits(table="all")

# Search OSTI technical documents
docs = cmm_data.search_documents("rare earth elements")

# Browse available datasets
catalog = cmm_data.get_data_catalog()
print(catalog[["name", "source", "available"]])

# List DOE critical minerals
minerals = cmm_data.list_critical_minerals()
```

## API Reference

### Convenience Functions

- `load_usgs_commodity(commodity, data_type="world")` -- Load USGS commodity data as a DataFrame.
- `load_ore_deposits(table="all")` -- Load USGS ore deposits database.
- `search_documents(query, **kwargs)` -- Search OSTI technical documents.
- `iter_corpus_documents(**kwargs)` -- Iterate over preprocessed corpus documents.
- `get_data_catalog()` -- DataFrame listing all datasets with availability status.
- `list_commodities()` -- All available USGS commodity codes.
- `list_critical_minerals()` -- DOE critical mineral codes.
- `configure(data_root=None, cache_enabled=None, ...)` -- Set package configuration.
- `get_config()` -- Return current `CMMDataConfig` instance.

### Loader Classes

- `USGSCommodityLoader` -- World production and salient statistics for 80+ commodities.
- `USGSOreDepositsLoader` -- Geochemical analyses from ore deposits (356 fields).
- `OSTIDocumentsLoader` -- OSTI/DOE technical reports on critical minerals.
- `PreprocessedCorpusLoader` -- Unified JSONL corpus (3,298 documents).
- `GAChronostratigraphicLoader` -- Geoscience Australia 3D chronostratigraphic surfaces.
- `NETLREECoalLoader` -- NETL REE data from coal and coal-related resources.
- `OECDSupplyChainLoader` -- OECD/IEA export restrictions and supply chain data.
- `MindatLoader` -- Mindat mineral database with critical element groupings.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `CMM_DATA_PATH` | Root directory containing CMM data (Globus_Sharing) | Auto-detected |

If `CMM_DATA_PATH` is not set, the package searches parent directories for a
`Globus_Sharing` folder and common installation paths.

## Dependencies

- **Required:** pandas, numpy
- **Optional (geo):** geopandas, rasterio, fiona
- **Optional (viz):** matplotlib, plotly
