"""Preprocessed corpus loader for LLM training data."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..exceptions import DataNotFoundError
from .base import BaseLoader

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


class PreprocessedCorpusLoader(BaseLoader):
    """
    Loader for preprocessed document corpus (JSONL format).

    Provides access to unified corpus of critical minerals documents
    prepared for LLM training and analysis.
    """

    dataset_name = "preprocessed"

    def list_available(self) -> list[str]:
        """List available corpus files."""
        if not self.data_path.exists():
            return []

        return [f.name for f in self.data_path.glob("*.jsonl")]

    def load(self, corpus_file: str = "unified_corpus.jsonl") -> pd.DataFrame:
        """
        Load corpus as DataFrame.

        Args:
            corpus_file: Name of JSONL file to load

        Returns:
            DataFrame with document records
        """
        cache_key = self._cache_key("corpus", corpus_file)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._validate_path(self.data_path, "Preprocessed directory")

        file_path = self.data_path / corpus_file
        if not file_path.exists():
            available = self.list_available()
            raise DataNotFoundError(
                f"Corpus file '{corpus_file}' not found. Available: {available}"
            )

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue

        df = pd.DataFrame(records)

        self._set_cached(cache_key, df)
        return df

    def iter_documents(
        self, corpus_file: str = "unified_corpus.jsonl", batch_size: int | None = None
    ) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over documents in the corpus.

        Args:
            corpus_file: Name of JSONL file
            batch_size: If set, yield batches of documents

        Yields:
            Document dictionaries (or lists if batch_size set)
        """
        file_path = self.data_path / corpus_file
        self._validate_path(file_path, f"Corpus file {corpus_file}")

        batch = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if batch_size:
                        batch.append(record)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    else:
                        yield record
                except json.JSONDecodeError:
                    continue

        # Yield remaining batch
        if batch_size and batch:
            yield batch

    def get_corpus_stats(self, corpus_file: str = "unified_corpus.jsonl") -> dict:
        """
        Get statistics about the corpus.

        Args:
            corpus_file: Name of JSONL file

        Returns:
            Dictionary with corpus statistics
        """
        df = self.load(corpus_file)

        stats = {
            "total_documents": len(df),
            "columns": list(df.columns),
        }

        # Text length statistics
        if "text" in df.columns:
            text_lengths = df["text"].str.len()
            stats["text_stats"] = {
                "total_chars": text_lengths.sum(),
                "mean_length": text_lengths.mean(),
                "median_length": text_lengths.median(),
                "min_length": text_lengths.min(),
                "max_length": text_lengths.max(),
            }

        # Source distribution
        if "source" in df.columns:
            stats["source_distribution"] = df["source"].value_counts().to_dict()

        # Document type distribution
        if "doc_type" in df.columns:
            stats["type_distribution"] = df["doc_type"].value_counts().to_dict()

        return stats

    def search(self, query: str, fields: list[str] | None = None, limit: int = 100) -> pd.DataFrame:
        """
        Search documents in the corpus.

        Args:
            query: Search query
            fields: Fields to search (default: text, title)
            limit: Maximum results

        Returns:
            DataFrame with matching documents
        """
        df = self.load()

        if fields is None:
            fields = ["text", "title", "content", "abstract"]

        mask = pd.Series([False] * len(df))
        for field in fields:
            if field in df.columns:
                field_mask = df[field].astype(str).str.contains(query, case=False, na=False)
                mask = mask | field_mask

        return df[mask].head(limit)

    def filter_by_source(self, source: str) -> pd.DataFrame:
        """
        Filter corpus by source.

        Args:
            source: Source name to filter

        Returns:
            Filtered DataFrame
        """
        df = self.load()
        if "source" in df.columns:
            return df[df["source"].str.contains(source, case=False, na=False)]
        return df

    def export_for_training(
        self, output_path: Path, text_column: str = "text", format: str = "jsonl"
    ) -> int:
        """
        Export corpus for LLM training.

        Args:
            output_path: Output file path
            text_column: Column containing text
            format: Output format ('jsonl' or 'txt')

        Returns:
            Number of documents exported
        """
        df = self.load()

        if text_column not in df.columns:
            raise DataNotFoundError(f"Column '{text_column}' not found in corpus")

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                text = row[text_column]
                if pd.isna(text) or not str(text).strip():
                    continue

                if format == "jsonl":
                    record = {"text": str(text)}
                    if "id" in row:
                        record["id"] = row["id"]
                    f.write(json.dumps(record) + "\n")
                else:
                    f.write(str(text) + "\n\n")

                count += 1

        return count

    def describe(self) -> dict:
        """Describe the preprocessed corpus."""
        base = super().describe()
        try:
            stats = self.get_corpus_stats()
            base.update(stats)
        except (OSError, ValueError):
            pass
        return base
