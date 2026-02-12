"""
Data tools for CMM MCP Server
Handles CSV files and schema queries
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cmm_docs.config import MAX_CSV_ROWS, SCHEMAS_JSON


class DataManager:
    """Manages CSV data and schemas"""

    def __init__(self):
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> dict:
        """Load schema definitions from JSON"""
        if not SCHEMAS_JSON.exists():
            return {}
        with open(SCHEMAS_JSON) as f:
            return json.load(f)

    def list_datasets(self, category: str | None = None) -> list[dict]:
        """
        list available CSV datasets.

        Args:
            category: Optional category to filter by

        Returns:
            list of dataset summaries
        """
        results = []

        for cat_name, cat_data in self.schemas.items():
            if category and category.lower() not in cat_name.lower():
                continue

            for schema in cat_data.get("schemas", []):
                results.append(
                    {
                        "category": cat_name,
                        "file": schema.get("file"),
                        "row_count": schema.get("row_count", 0),
                        "column_count": len(schema.get("columns", [])),
                        "path": schema.get("path"),
                    }
                )

        return results

    def get_schema(self, dataset: str) -> dict | None:
        """
        Get schema/column information for a specific dataset.

        Args:
            dataset: Dataset filename (e.g., "ChemData1.csv")

        Returns:
            Schema information including column names, types, and samples
        """
        for cat_name, cat_data in self.schemas.items():
            for schema in cat_data.get("schemas", []):
                if schema.get("file") == dataset:
                    return {
                        "category": cat_name,
                        "file": schema.get("file"),
                        "path": schema.get("path"),
                        "row_count": schema.get("row_count", 0),
                        "columns": schema.get("columns", []),
                    }

        return {"error": f"Dataset not found: {dataset}"}

    def find_dataset_path(self, dataset: str) -> Path | None:
        """Find the full path to a dataset file"""
        for _cat_name, cat_data in self.schemas.items():
            for schema in cat_data.get("schemas", []):
                if schema.get("file") == dataset:
                    return Path(schema.get("path"))
        return None

    def query_csv(
        self,
        dataset: str,
        filters: dict | None = None,
        columns: list[str] | None = None,
        limit: int = MAX_CSV_ROWS,
    ) -> dict:
        """
        Query a CSV file with optional filters.

        Args:
            dataset: Dataset filename
            filters: Dictionary of column:value filters
            columns: list of columns to return (None = all)
            limit: Maximum rows to return

        Returns:
            Query results with matching rows
        """
        path = self.find_dataset_path(dataset)
        if not path or not path.exists():
            return {"error": f"Dataset not found: {dataset}"}

        try:
            # Read CSV
            df = pd.read_csv(path, low_memory=False)

            # Apply filters
            if filters:
                for col, value in filters.items():
                    if col in df.columns:
                        # Handle different filter types
                        if isinstance(value, str) and value.startswith(">"):
                            df = df[df[col] > float(value[1:])]
                        elif isinstance(value, str) and value.startswith("<"):
                            df = df[df[col] < float(value[1:])]
                        elif isinstance(value, str) and value.startswith("~"):
                            # Contains search
                            df = df[
                                df[col].astype(str).str.contains(value[1:], case=False, na=False)
                            ]
                        else:
                            df = df[df[col] == value]

            # Select columns
            if columns:
                available_cols = [c for c in columns if c in df.columns]
                if available_cols:
                    df = df[available_cols]

            # Limit rows
            total_rows = len(df)
            df = df.head(limit)

            # Convert to records
            records = df.to_dict(orient="records")

            return {
                "dataset": dataset,
                "total_matching_rows": total_rows,
                "returned_rows": len(records),
                "columns": list(df.columns),
                "data": records,
            }

        except (OSError, ValueError, KeyError) as e:
            return {"error": f"Error reading dataset: {e!s}"}

    def read_csv_sample(self, dataset: str, n_rows: int = 10) -> dict:
        """
        Read first N rows from a CSV file.

        Args:
            dataset: Dataset filename
            n_rows: Number of rows to read

        Returns:
            Sample data from the dataset
        """
        path = self.find_dataset_path(dataset)
        if not path or not path.exists():
            return {"error": f"Dataset not found: {dataset}"}

        try:
            df = pd.read_csv(path, nrows=n_rows, low_memory=False)

            return {
                "dataset": dataset,
                "columns": list(df.columns),
                "sample_rows": n_rows,
                "data": df.to_dict(orient="records"),
            }

        except (OSError, ValueError) as e:
            return {"error": f"Error reading dataset: {e!s}"}

    def get_statistics(self) -> dict:
        """Get data collection statistics"""
        stats = {
            "total_datasets": 0,
            "total_rows": 0,
            "by_category": {},
        }

        for cat_name, cat_data in self.schemas.items():
            schemas = cat_data.get("schemas", [])
            cat_rows = sum(s.get("row_count", 0) for s in schemas)

            stats["by_category"][cat_name] = {
                "file_count": len(schemas),
                "total_rows": cat_rows,
            }
            stats["total_datasets"] += len(schemas)
            stats["total_rows"] += cat_rows

        return stats


# Singleton instance
_data_manager = None


def get_data_manager() -> DataManager:
    """Get or create DataManager singleton"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager
