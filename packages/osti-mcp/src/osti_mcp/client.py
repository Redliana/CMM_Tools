"""OSTI data client for accessing document catalog and metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class OSTIDocument(BaseModel):
    """OSTI document metadata."""

    osti_id: str
    title: str
    authors: list[str] = []
    publication_date: str | None = None
    description: str | None = None
    subjects: list[str] = []
    commodity_category: str | None = None
    doi: str | None = None
    product_type: str | None = None
    research_orgs: list[str] = []
    sponsor_orgs: list[str] = []


class OSTIClient:
    """Client for accessing OSTI document data."""

    # CMM commodity categories
    COMMODITIES = {
        "HREE": "Heavy Rare Earth Elements",
        "LREE": "Light Rare Earth Elements",
        "CO": "Cobalt",
        "LI": "Lithium",
        "GA": "Gallium",
        "GR": "Graphite",
        "NI": "Nickel",
        "CU": "Copper",
        "GE": "Germanium",
        "OTH": "Other Critical Materials",
    }

    def __init__(self, data_path: str | None = None):
        """Initialize OSTI client.

        Args:
            data_path: Path to OSTI_retrieval directory. If not provided,
                      attempts to find it relative to project structure.
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Try to find the data path from environment or default locations
            env_path = os.environ.get("OSTI_DATA_PATH")
            if env_path:
                self.data_path = Path(env_path)
            else:
                # Default relative to typical project structure
                self.data_path = Path(__file__).parents[4] / "Globus_Sharing" / "OSTI_retrieval"

        self._catalog: pd.DataFrame | None = None

    @property
    def catalog(self) -> pd.DataFrame:
        """Load and cache the document catalog."""
        if self._catalog is None:
            catalog_path = self.data_path / "document_catalog.json"
            if not catalog_path.exists():
                raise FileNotFoundError(
                    f"Document catalog not found at {catalog_path}. "
                    "Set OSTI_DATA_PATH environment variable or provide data_path."
                )
            with open(catalog_path) as f:
                data = json.load(f)
            self._catalog = pd.DataFrame(data)
        return self._catalog

    def get_statistics(self) -> dict:
        """Get collection statistics."""
        df = self.catalog

        stats = {
            "total_documents": len(df),
            "commodities": {},
            "product_types": {},
            "year_range": {},
        }

        # Count by commodity
        if "commodity_category" in df.columns:
            commodity_counts = df["commodity_category"].value_counts().to_dict()
            stats["commodities"] = {
                k: {"count": v, "name": self.COMMODITIES.get(k, k)}
                for k, v in commodity_counts.items()
            }

        # Count by product type
        if "product_type" in df.columns:
            stats["product_types"] = df["product_type"].value_counts().to_dict()

        # Year range
        if "publication_date" in df.columns:
            dates = pd.to_datetime(df["publication_date"], errors="coerce")
            years = dates.dt.year.dropna()
            if len(years) > 0:
                stats["year_range"] = {
                    "min": int(years.min()),
                    "max": int(years.max()),
                }

        return stats

    def search_documents(
        self,
        query: str | None = None,
        commodity: str | None = None,
        product_type: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        limit: int = 50,
    ) -> list[OSTIDocument]:
        """Search documents with filters.

        Args:
            query: Text search in title/description
            commodity: Filter by commodity category (HREE, LREE, CO, LI, etc.)
            product_type: Filter by product type (Technical Report, Journal Article)
            year_from: Minimum publication year
            year_to: Maximum publication year
            limit: Maximum results to return

        Returns:
            List of matching OSTIDocument objects
        """
        df = self.catalog.copy()

        # Text search
        if query:
            query_lower = query.lower()
            mask = (
                df["title"].str.lower().str.contains(query_lower, na=False)
                | df["description"].fillna("").str.lower().str.contains(query_lower, na=False)
                | df["subjects"].astype(str).str.lower().str.contains(query_lower, na=False)
            )
            df = df[mask]

        # Commodity filter
        if commodity:
            commodity_upper = commodity.upper()
            df = df[df["commodity_category"] == commodity_upper]

        # Product type filter
        if product_type:
            df = df[df["product_type"].str.lower() == product_type.lower()]

        # Year filters
        if year_from or year_to:
            dates = pd.to_datetime(df["publication_date"], errors="coerce")
            years = dates.dt.year
            if year_from:
                df = df[years >= year_from]
            if year_to:
                df = df[years <= year_to]

        # Limit results
        df = df.head(limit)

        # Convert to OSTIDocument objects
        documents = []
        for _, row in df.iterrows():
            doc = OSTIDocument(
                osti_id=str(row.get("osti_id", "")),
                title=row.get("title", ""),
                authors=row.get("authors", []) if isinstance(row.get("authors"), list) else [],
                publication_date=row.get("publication_date"),
                description=row.get("description"),
                subjects=row.get("subjects", []) if isinstance(row.get("subjects"), list) else [],
                commodity_category=row.get("commodity_category"),
                doi=row.get("doi"),
                product_type=row.get("product_type"),
                research_orgs=row.get("research_orgs", [])
                if isinstance(row.get("research_orgs"), list)
                else [],
                sponsor_orgs=row.get("sponsor_orgs", [])
                if isinstance(row.get("sponsor_orgs"), list)
                else [],
            )
            documents.append(doc)

        return documents

    def get_document(self, osti_id: str) -> OSTIDocument | None:
        """Get a specific document by OSTI ID.

        Args:
            osti_id: The OSTI document identifier

        Returns:
            OSTIDocument if found, None otherwise
        """
        df = self.catalog
        match = df[df["osti_id"].astype(str) == str(osti_id)]

        if match.empty:
            return None

        row = match.iloc[0]
        return OSTIDocument(
            osti_id=str(row.get("osti_id", "")),
            title=row.get("title", ""),
            authors=row.get("authors", []) if isinstance(row.get("authors"), list) else [],
            publication_date=row.get("publication_date"),
            description=row.get("description"),
            subjects=row.get("subjects", []) if isinstance(row.get("subjects"), list) else [],
            commodity_category=row.get("commodity_category"),
            doi=row.get("doi"),
            product_type=row.get("product_type"),
            research_orgs=row.get("research_orgs", [])
            if isinstance(row.get("research_orgs"), list)
            else [],
            sponsor_orgs=row.get("sponsor_orgs", [])
            if isinstance(row.get("sponsor_orgs"), list)
            else [],
        )

    def list_commodities(self) -> dict[str, str]:
        """List available commodity categories with descriptions."""
        return self.COMMODITIES.copy()

    def get_documents_by_commodity(self, commodity: str, limit: int = 100) -> list[OSTIDocument]:
        """Get documents for a specific commodity category.

        Args:
            commodity: Commodity code (HREE, LREE, CO, LI, GA, GR, NI, CU, GE, OTH)
            limit: Maximum results to return

        Returns:
            List of OSTIDocument objects
        """
        return self.search_documents(commodity=commodity, limit=limit)

    def get_recent_documents(self, limit: int = 20) -> list[OSTIDocument]:
        """Get most recently published documents.

        Args:
            limit: Maximum results to return

        Returns:
            List of OSTIDocument objects sorted by publication date (newest first)
        """
        df = self.catalog.copy()
        df["_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
        df = df.sort_values("_date", ascending=False).head(limit)

        documents = []
        for _, row in df.iterrows():
            doc = OSTIDocument(
                osti_id=str(row.get("osti_id", "")),
                title=row.get("title", ""),
                authors=row.get("authors", []) if isinstance(row.get("authors"), list) else [],
                publication_date=row.get("publication_date"),
                description=row.get("description"),
                subjects=row.get("subjects", []) if isinstance(row.get("subjects"), list) else [],
                commodity_category=row.get("commodity_category"),
                doi=row.get("doi"),
                product_type=row.get("product_type"),
                research_orgs=row.get("research_orgs", [])
                if isinstance(row.get("research_orgs"), list)
                else [],
                sponsor_orgs=row.get("sponsor_orgs", [])
                if isinstance(row.get("sponsor_orgs"), list)
                else [],
            )
            documents.append(doc)

        return documents
