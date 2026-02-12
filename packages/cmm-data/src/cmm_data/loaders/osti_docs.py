"""OSTI document retrieval loader."""

from __future__ import annotations

import json

import pandas as pd

from ..exceptions import DataNotFoundError
from .base import BaseLoader


class OSTIDocumentsLoader(BaseLoader):
    """
    Loader for OSTI (Office of Scientific and Technical Information) documents.

    Provides access to technical reports and publications related to
    critical minerals and materials science.
    """

    dataset_name = "osti"

    def list_available(self) -> list[str]:
        """List available document collections/files."""
        if not self.data_path.exists():
            return []

        items = []
        # JSON metadata files
        for f in self.data_path.glob("*.json"):
            items.append(f.stem)
        # Text document directories
        for d in self.data_path.iterdir():
            if d.is_dir():
                items.append(d.name)

        return sorted(items)

    def load(self, collection: str | None = None) -> pd.DataFrame:
        """
        Load OSTI document metadata.

        Args:
            collection: Optional collection name to load

        Returns:
            DataFrame with document metadata
        """
        cache_key = self._cache_key("docs", collection)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._validate_path(self.data_path, "OSTI directory")

        if collection:
            json_path = self.data_path / f"{collection}.json"
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
            else:
                raise DataNotFoundError(f"Collection '{collection}' not found")
        else:
            # Load all JSON metadata
            dfs = []
            for json_file in self.data_path.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
                    df["_source_file"] = json_file.name
                    dfs.append(df)
                except (json.JSONDecodeError, OSError):
                    continue

            if dfs:
                df = pd.concat(dfs, ignore_index=True)
            else:
                # Create empty DataFrame with expected columns
                df = pd.DataFrame(
                    columns=[
                        "osti_id",
                        "title",
                        "authors",
                        "publication_date",
                        "abstract",
                        "doi",
                        "url",
                    ]
                )

        self._set_cached(cache_key, df)
        return df

    def search_documents(
        self, query: str, fields: list[str] | None = None, limit: int = 100
    ) -> pd.DataFrame:
        """
        Search documents by keyword.

        Args:
            query: Search query string
            fields: Fields to search (default: title, abstract, keywords)
            limit: Maximum results to return

        Returns:
            DataFrame with matching documents
        """
        df = self.load()

        if df.empty:
            return df

        if fields is None:
            fields = ["title", "abstract", "keywords", "subjects"]

        # Search across specified fields
        mask = pd.Series([False] * len(df))
        for field in fields:
            if field in df.columns:
                field_mask = df[field].astype(str).str.contains(query, case=False, na=False)
                mask = mask | field_mask

        results = df[mask].head(limit)
        return results

    def get_document_text(self, doc_id: str) -> str | None:
        """
        Get full text of a document.

        Args:
            doc_id: Document identifier (OSTI ID)

        Returns:
            Document text content or None
        """
        # Look for text files in subdirectories
        for txt_path in self.data_path.rglob(f"*{doc_id}*.txt"):
            return txt_path.read_text(errors="replace")

        # Check for document in JSON
        df = self.load()
        if "osti_id" in df.columns:
            match = df[df["osti_id"].astype(str) == str(doc_id)]
            if not match.empty and "full_text" in match.columns:
                return match.iloc[0]["full_text"]

        return None

    def get_documents_by_year(self, year: int) -> pd.DataFrame:
        """
        Get documents published in a specific year.

        Args:
            year: Publication year

        Returns:
            DataFrame with documents from that year
        """
        df = self.load()

        date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower()]
        for col in date_cols:
            try:
                # Try to extract year
                years = pd.to_datetime(df[col], errors="coerce").dt.year
                mask = years == year
                if mask.any():
                    return df[mask]
            except (ValueError, TypeError):
                continue

        return pd.DataFrame()

    def get_statistics(self) -> dict:
        """Get document collection statistics."""
        df = self.load()

        stats = {
            "total_documents": len(df),
            "fields": list(df.columns),
        }

        # Count by year if possible
        date_cols = [c for c in df.columns if "date" in c.lower()]
        for col in date_cols:
            try:
                years = pd.to_datetime(df[col], errors="coerce").dt.year
                stats["year_distribution"] = years.value_counts().to_dict()
                break
            except (ValueError, TypeError):
                continue

        return stats

    def describe(self) -> dict:
        """Describe the OSTI documents dataset."""
        base = super().describe()
        try:
            stats = self.get_statistics()
            base.update(stats)
        except (OSError, ValueError):
            pass
        return base
