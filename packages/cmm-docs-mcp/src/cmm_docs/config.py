"""
Configuration for CMM MCP Server
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

# API Keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Base data directory
DATA_ROOT = Path(
    "/Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/LLM_Fine_Tuning/Claude"
)

# Data source directories
OSTI_DIR = DATA_ROOT / "OSTI_retrieval"
OSTI_PDFS_DIR = OSTI_DIR / "pdfs"
OSTI_CATALOG = OSTI_DIR / "document_catalog.json"

SCHEMAS_DIR = DATA_ROOT / "schemas"
SCHEMAS_JSON = SCHEMAS_DIR / "all_schemas.json"

USGS_ORE_DIR = DATA_ROOT / "USGS_Ore_Deposits"
USGS_DATA_DIR = DATA_ROOT / "USGS_Data"
LISA_MODEL_DIR = DATA_ROOT / "LISA_Model" / "extracted_data"
NETL_REE_DIR = DATA_ROOT / "NETL_REE_Coal" / "geodatabase_csv"

# Search index location
INDEX_DIR = DATA_ROOT / "cmm_mcp_server" / "index"
SEARCH_DB = INDEX_DIR / "search_index.db"
TFIDF_VECTORS = INDEX_DIR / "tfidf_vectors.pkl"

# Limits
MAX_CSV_ROWS = 100  # Maximum rows to return from CSV queries
MAX_PDF_CHARS = 50000  # Maximum characters to extract from PDF
MAX_SEARCH_RESULTS = 20  # Maximum search results to return

# Commodity categories
COMMODITIES = {
    "HREE": "Heavy Rare Earth Elements (Dy, Tb)",
    "LREE": "Light Rare Earth Elements (Nd, Pr)",
    "CO": "Cobalt",
    "LI": "Lithium",
    "GA": "Gallium",
    "GR": "Graphite",
    "NI": "Nickel",
    "CU": "Copper",
    "GE": "Germanium",
    "OTH": "Other (Mn, Ti, PGM, W)",
}

# Sub-domains
SUBDOMAINS = {
    "T-EC": "Extraction Chemistry",
    "T-PM": "Processing Metallurgy",
    "T-GO": "Geological Occurrence",
    "S-ST": "Supply Chain Topology",
    "G-PR": "Policy/Regulatory",
}

# Data categories for CSV files
DATA_CATEGORIES = {
    "LISA_Model": LISA_MODEL_DIR,
    "USGS_Ore_Deposits": USGS_ORE_DIR,
    "USGS_Mineral_Commodities_Salient": USGS_DATA_DIR / "salient",
    "USGS_Mineral_Commodities_World": USGS_DATA_DIR / "world",
    "USGS_Industry_Trends": USGS_DATA_DIR,
    "NETL_REE_Coal_Geodatabase": NETL_REE_DIR,
}
