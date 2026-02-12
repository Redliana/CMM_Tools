"""Paths, commodity configurations, model IDs, and system prompts."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repo root LLM_Fine_Tuning/Exp_Design)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]  # Exp_Design/

TRADE_DATA_DIR = REPO_ROOT / "API_Scripts" / "gold_qa_data"
USGS_DATA_DIR = REPO_ROOT / "API_Scripts" / "usgs_mcs_data" / "cmm_extracted"

FINE_TUNING_DIR = REPO_ROOT / "fine_tuning"
OUTPUT_DATA_DIR = FINE_TUNING_DIR / "data"
GOLD_QA_DIR = FINE_TUNING_DIR / "gold_qa"
ADAPTERS_DIR = FINE_TUNING_DIR / "adapters"
RESULTS_DIR = FINE_TUNING_DIR / "results"
CONFIGS_DIR = FINE_TUNING_DIR / "configs"

# ---------------------------------------------------------------------------
# Commodity configurations
# ---------------------------------------------------------------------------
# Maps commodity key -> metadata used across the pipeline
COMMODITY_CONFIG: dict[str, dict] = {
    "cobalt": {
        "display_name": "Cobalt",
        "symbol": "Co",
        "trade_csv": "cobalt_trade_data.csv",
        "hs_codes": ["8105", "282200"],
        "salient_csvs": [
            "2023/cobalt_mcs2023-cobal_salient.csv",
            "2024/cobalt_mcs2024-cobal_salient.csv",
        ],
        "world_csvs": [
            "2023/cobalt_mcs2023-cobal_world.csv",
            "2024/cobalt_mcs2024-cobal_world.csv",
        ],
    },
    "copper": {
        "display_name": "Copper",
        "symbol": "Cu",
        "trade_csv": "copper_trade_data.csv",
        "hs_codes": ["7402", "7403"],
        "salient_csvs": [
            "2023/copper_mcs2023-coppe_salient.csv",
            "2024/copper_mcs2024-coppe_salient.csv",
        ],
        "world_csvs": [
            "2023/copper_mcs2023-coppe_world.csv",
            "2024/copper_mcs2024-coppe_world.csv",
        ],
    },
    "gallium": {
        "display_name": "Gallium",
        "symbol": "Ga",
        "trade_csv": "gallium_trade_data.csv",
        "hs_codes": ["811292"],
        "salient_csvs": [
            "2023/gallium_mcs2023-galli_salient.csv",
            "2024/gallium_mcs2024-galli_salient.csv",
        ],
        "world_csvs": [
            "2023/gallium_mcs2023-galli_world.csv",
            "2024/gallium_mcs2024-galli_world.csv",
        ],
    },
    "graphite": {
        "display_name": "Graphite",
        "symbol": "Gr",
        "trade_csv": "graphite_trade_data.csv",
        "hs_codes": ["250410", "250490"],
        "salient_csvs": [
            "2023/graphite_mcs2023-graph_salient.csv",
            "2024/graphite_mcs2024-graph_salient.csv",
        ],
        "world_csvs": [
            "2023/graphite_mcs2023-graph_world.csv",
            "2024/graphite_mcs2024-graph_world.csv",
        ],
    },
    "heavy_ree": {
        "display_name": "Heavy Rare Earth Elements",
        "symbol": "HREE",
        "trade_csv": "heavy_ree_trade_data.csv",
        "hs_codes": ["284690"],
        "salient_csvs": [
            "2023/heavy_ree_mcs2023-raree_salient.csv",
            "2024/heavy_ree_mcs2024-raree_salient.csv",
        ],
        "world_csvs": [
            "2023/heavy_ree_mcs2023-raree_world.csv",
            "2024/heavy_ree_mcs2024-raree_world.csv",
        ],
    },
    "light_ree": {
        "display_name": "Light Rare Earth Elements",
        "symbol": "LREE",
        "trade_csv": "light_ree_trade_data.csv",
        "hs_codes": ["284610"],
        "salient_csvs": [
            "2023/light_ree_mcs2023-raree_salient.csv",
            "2024/light_ree_mcs2024-raree_salient.csv",
        ],
        "world_csvs": [
            "2023/light_ree_mcs2023-raree_world.csv",
            "2024/light_ree_mcs2024-raree_world.csv",
        ],
    },
    "lithium": {
        "display_name": "Lithium",
        "symbol": "Li",
        "trade_csv": "lithium_trade_data.csv",
        "hs_codes": ["282520", "283691", "253090"],
        "salient_csvs": [
            "2023/lithium_mcs2023-lithi_salient.csv",
            "2024/lithium_mcs2024-lithi_salient.csv",
        ],
        "world_csvs": [
            "2023/lithium_mcs2023-lithi_world.csv",
            "2024/lithium_mcs2024-lithi_world.csv",
        ],
    },
    "nickel": {
        "display_name": "Nickel",
        "symbol": "Ni",
        "trade_csv": "nickel_trade_data.csv",
        "hs_codes": ["7501", "7502", "281122"],
        "salient_csvs": [
            "2023/nickel_mcs2023-nicke_salient.csv",
            "2024/nickel_mcs2024-nicke_salient.csv",
        ],
        "world_csvs": [
            "2023/nickel_mcs2023-nicke_world.csv",
            "2024/nickel_mcs2024-nicke_world.csv",
        ],
    },
}

# All commodity keys for iteration
COMMODITIES = list(COMMODITY_CONFIG.keys())

# ---------------------------------------------------------------------------
# Country code mappings  (UN M49 -> display name)
# ---------------------------------------------------------------------------
COUNTRY_CODES: dict[str, str] = {
    "842": "United States",
    "156": "China",
    "276": "Germany",
    "392": "Japan",
    "410": "South Korea",
    "36": "Australia",
    "124": "Canada",
    "826": "United Kingdom",
    "250": "France",
    "699": "India",
    "180": "Democratic Republic of the Congo",
    "360": "Indonesia",
    "76": "Brazil",
    "380": "Italy",
    "528": "Netherlands",
    "56": "Belgium",
    "710": "South Africa",
    "152": "Chile",
    "32": "Argentina",
    "608": "Philippines",
    "643": "Russia",
    "0": "World",
}

# Reverse mapping: ISO3 -> M49 code
ISO3_TO_M49: dict[str, str] = {
    "USA": "842",
    "CHN": "156",
    "DEU": "276",
    "JPN": "392",
    "KOR": "410",
    "AUS": "36",
    "CAN": "124",
    "GBR": "826",
    "FRA": "250",
    "IND": "699",
    "COD": "180",
    "IDN": "360",
    "BRA": "76",
    "ITA": "380",
    "NLD": "528",
    "BEL": "56",
    "ZAF": "710",
    "CHL": "152",
    "ARG": "32",
    "PHL": "608",
    "RUS": "643",
}

# Flow codes
FLOW_CODES: dict[str, str] = {
    "M": "Imports",
    "X": "Exports",
}

# ---------------------------------------------------------------------------
# MLX model IDs
# ---------------------------------------------------------------------------
MLX_MODELS: dict[str, str] = {
    "bf16": "mlx-community/phi-4-bf16",
    "8bit": "mlx-community/phi-4-8bit",
    "4bit": "mlx-community/phi-4-4bit",
}

DEFAULT_MODEL = MLX_MODELS["bf16"]

# ---------------------------------------------------------------------------
# System prompt for CMM fine-tuning
# ---------------------------------------------------------------------------
CMM_SYSTEM_PROMPT = (
    "You are an expert analyst specializing in critical minerals and materials (CMM) "
    "supply chains. You provide accurate, data-driven answers about international trade "
    "flows, production statistics, reserves, prices, and supply chain dependencies for "
    "minerals including lithium, cobalt, nickel, rare earth elements, graphite, gallium, "
    "germanium, and copper. When citing statistics, include the year, units, and source "
    "context. If data is unavailable or withheld, state that clearly."
)
