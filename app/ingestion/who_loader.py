"""
Structured-data loader.

Reads ALL CSV files from the configured data directory (via glob),
merges them into a single Pandas DataFrame, and exposes query helpers
used by the Structured Retriever agent.  This supports heterogeneous
datasets — just drop any CSV into ``data/`` and restart.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pandas as pd

from app.logging_cfg import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class WHODataStore:
    """In-memory store backed by Pandas DataFrames loaded from CSV files."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._df: pd.DataFrame | None = None
        self._source_map: dict[str, list[str]] = {}  # filename → column list

    async def load(self) -> None:
        """Glob all CSVs from the data directory and merge them."""
        logger.info("[Ingestion] Scanning for CSV files in %s", self._data_dir)
        self._df, self._source_map = await asyncio.to_thread(self._read_all_csvs)

        if self._df is not None and not self._df.empty:
            logger.info(
                "[Ingestion] Loaded %d rows from %d CSV files, columns=%s",
                len(self._df),
                len(self._source_map),
                list(self._df.columns),
            )
        else:
            logger.warning("[Ingestion] No CSV data found in %s", self._data_dir)
            self._df = pd.DataFrame()

    def _read_all_csvs(self) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Read and merge all CSVs from the data directory (runs in a thread)."""
        csv_files = sorted(self._data_dir.glob("*.csv"))
        if not csv_files:
            logger.warning("[Ingestion] No .csv files found in %s", self._data_dir)
            return pd.DataFrame(), {}

        frames: list[pd.DataFrame] = []
        source_map: dict[str, list[str]] = {}

        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                # Normalise column names to lower_snake_case
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                df.dropna(how="all", inplace=True)
                # Tag each row with its source file for traceability
                df["_source_file"] = csv_path.name
                frames.append(df)
                source_map[csv_path.name] = list(df.columns)
                logger.info(
                    "[Ingestion] Loaded %s — %d rows, %d columns",
                    csv_path.name,
                    len(df),
                    len(df.columns) - 1,  # exclude _source_file
                )
            except Exception:
                logger.exception("[Ingestion] Failed to read %s — skipping", csv_path.name)

        if not frames:
            return pd.DataFrame(), source_map

        merged = pd.concat(frames, ignore_index=True, sort=False)
        return merged, source_map

    @property
    def df(self) -> pd.DataFrame:
        """Return the loaded DataFrame (raises if not yet loaded)."""
        if self._df is None:
            raise RuntimeError("Structured data not loaded — call .load() first")
        return self._df

    @property
    def columns(self) -> list[str]:
        """Column names for schema-aware routing."""
        cols = list(self.df.columns)
        return [c for c in cols if c != "_source_file"]

    @property
    def source_files(self) -> dict[str, list[str]]:
        """Map of loaded CSV filenames to their column lists."""
        return self._source_map

    @property
    def summary(self) -> str:
        """Short human-readable summary of the dataset."""
        return (
            f"Structured data: {len(self.df)} rows from "
            f"{len(self._source_map)} CSV file(s), columns={self.columns}"
        )

    async def query(
        self,
        filters: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        limit: int = 20,
    ) -> str:
        """Run a filter/select query and return a CSV string.

        Args:
            filters: Column→value equality filters (case-insensitive match
                     for string columns).
            columns: Subset of columns to return.
            limit: Maximum rows.

        Returns:
            CSV-formatted string of matching rows.
        """
        df = self.df.copy()
        if filters:
            for col, val in filters.items():
                col_lower = col.lower()
                if col_lower not in df.columns:
                    continue
                if df[col_lower].dtype == object:
                    df = df[df[col_lower].str.lower() == str(val).lower()]
                else:
                    df = df[df[col_lower] == val]
        if columns:
            valid = [c for c in columns if c in df.columns]
            if valid:
                df = df[valid]
        df = df.drop(columns=["_source_file"], errors="ignore").head(limit)
        return df.to_csv(index=False)

    async def text_search(self, query: str, limit: int = 10) -> str:
        """Word-level full-text search across all string columns.

        Splits the query into individual words and matches rows where
        ANY word appears in ANY string column.  Stop words are filtered
        out to reduce noise.
        """
        stop_words = {
            "the", "a", "an", "is", "of", "in", "for", "and", "or", "to",
            "on", "at", "by", "it", "its", "be", "as", "do", "has", "have",
            "was", "were", "with", "what", "which", "how", "from", "that",
        }
        words = [
            w for w in query.lower().split()
            if w not in stop_words and len(w) > 1
        ]
        if not words:
            words = query.lower().split()

        df = self.df
        mask = pd.Series(False, index=df.index)
        str_cols = df.select_dtypes(include=["object"]).columns
        str_cols = [c for c in str_cols if c != "_source_file"]
        for word in words:
            for col in str_cols:
                mask = mask | df[col].str.lower().str.contains(word, na=False)
        result = df[mask].drop(columns=["_source_file"], errors="ignore").head(limit)
        if result.empty:
            return "No matching rows found."
        return result.to_csv(index=False)

    async def aggregate(
        self,
        group_by: str,
        agg_column: str,
        agg_func: str = "mean",
        limit: int = 20,
    ) -> str:
        """Group-by aggregation on a numeric column.

        Args:
            group_by: Column to group by.
            agg_column: Numeric column to aggregate.
            agg_func: One of mean, sum, min, max, count.
            limit: Max rows in output.
        """
        df = self.df
        if group_by not in df.columns or agg_column not in df.columns:
            return f"Column not found. Available: {self.columns}"
        allowed = {"mean", "sum", "min", "max", "count"}
        if agg_func not in allowed:
            return f"agg_func must be one of {allowed}"
        result = df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
        result = result.sort_values(agg_column, ascending=False).head(limit)
        return result.to_csv(index=False)

    async def get_entity_data(self, entity: str) -> str:
        """Retrieve all rows related to a named entity (e.g. a country).

        Used by the entity traversal agent to link structured data
        with vector search results for cross-source reasoning.

        Args:
            entity: Entity name to search for (case-insensitive).

        Returns:
            CSV-formatted string of all matching rows.
        """
        df = self.df
        mask = pd.Series(False, index=df.index)
        entity_lower = entity.lower()
        for col in df.select_dtypes(include=["object"]).columns:
            if col == "_source_file":
                continue
            mask = mask | df[col].str.lower().str.contains(entity_lower, na=False)
        result = df[mask].drop(columns=["_source_file"], errors="ignore")
        if result.empty:
            return ""
        return result.to_csv(index=False)
