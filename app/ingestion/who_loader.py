"""
Structured-data loader.

Reads ALL CSV files from the configured data directory (via glob),
merges them into a single Pandas DataFrame, and exposes query helpers
used by the Structured Retriever agent.  This supports heterogeneous
datasets — just drop any CSV into ``data/`` and restart.
"""

from __future__ import annotations

import asyncio
import re
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

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalise free text for matching against schema values."""
        lowered = text.lower().strip()
        lowered = re.sub(r"['’]s\b", "", lowered)
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def build_search_terms(self, query: str) -> list[str]:
        """Tokenise user text into normalised search terms."""
        stop_words = {
            "the", "a", "an", "is", "of", "in", "for", "and", "or", "to",
            "on", "at", "by", "it", "its", "be", "as", "do", "has", "have",
            "was", "were", "with", "what", "which", "how", "from", "that",
        }

        terms: list[str] = []
        for token in self.normalize_text(query).split():
            if token in stop_words or len(token) <= 1:
                continue
            terms.append(token)
            # Recover singular forms from simple plurals / possessives like "germanys".
            if token.endswith("s") and len(token) > 4:
                singular = token[:-1]
                if singular not in stop_words and len(singular) > 1:
                    terms.append(singular)

        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped

    @property
    def country_values(self) -> list[str]:
        """Return canonical country values from the dataset."""
        if "country" not in self.df.columns:
            return []
        values = self.df["country"].dropna().astype(str).tolist()
        return list(dict.fromkeys(values))

    def find_countries(self, text: str) -> list[str]:
        """Return dataset countries mentioned in ``text``."""
        normalized_text = self.normalize_text(text)
        search_terms = set(self.build_search_terms(text))
        matches: list[tuple[int, str]] = []

        for country in self.country_values:
            alias = self.normalize_text(country)
            if not alias:
                continue
            if " " in alias:
                pos = normalized_text.find(alias)
                if pos >= 0:
                    matches.append((pos, country))
            elif alias in search_terms:
                pos = normalized_text.find(alias)
                matches.append((pos if pos >= 0 else len(normalized_text), country))

        matches.sort(key=lambda item: (item[0], item[1]))
        ordered: list[str] = []
        seen: set[str] = set()
        for _, country in matches:
            if country in seen:
                continue
            seen.add(country)
            ordered.append(country)
        return ordered

    def canonicalize_value(self, column: str, value: Any) -> Any:
        """Return the canonical dataset value for a string field when possible."""
        if value is None or column not in self.df.columns:
            return value

        series = self.df[column]
        if series.dtype != object:
            return value

        if column == "country":
            countries = self.find_countries(str(value))
            if countries:
                return countries[0]

        normalized = self.normalize_text(str(value))
        value_map = {
            self.normalize_text(str(item)): str(item)
            for item in series.dropna().astype(str).unique().tolist()
        }

        if normalized in value_map:
            return value_map[normalized]
        if normalized.endswith("s") and normalized[:-1] in value_map:
            return value_map[normalized[:-1]]
        return value

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
            invalid_filters = [col for col in filters if col.lower() not in df.columns]
            if invalid_filters:
                raise ValueError(f"Invalid filter columns: {invalid_filters}")

            for col, val in filters.items():
                col_lower = col.lower()
                series = df[col_lower]
                if isinstance(val, (list, tuple, set)):
                    values = list(val)
                    if not values:
                        return "No matching rows found."
                    if series.dtype == object:
                        normalized_values = {
                            str(self.canonicalize_value(col_lower, item)).lower()
                            for item in values
                        }
                        df = df[series.astype(str).str.lower().isin(normalized_values)]
                    else:
                        df = df[series.isin(values)]
                elif series.dtype == object:
                    canonical = self.canonicalize_value(col_lower, val)
                    df = df[series.astype(str).str.lower() == str(canonical).lower()]
                else:
                    df = df[series == val]
        if columns:
            valid = [c for c in columns if c in df.columns]
            if not valid:
                raise ValueError(f"No valid columns selected from: {columns}")
            if valid:
                df = df[valid]
        df = df.drop(columns=["_source_file"], errors="ignore").head(limit)
        if df.empty:
            return "No matching rows found."
        return df.to_csv(index=False)

    async def text_search(self, query: str, limit: int = 10) -> str:
        """Word-level full-text search across all string columns.

        Splits the query into individual words and matches rows where
        ANY word appears in ANY string column.  Stop words are filtered
        out to reduce noise.
        """
        words = self.build_search_terms(query)
        if not words:
            words = self.normalize_text(query).split()

        df = self.df
        mask = pd.Series(False, index=df.index)
        str_cols = df.select_dtypes(include=["object"]).columns
        str_cols = [c for c in str_cols if c != "_source_file"]

        normalized_cols = {
            col: df[col]
            .astype(str)
            .map(self.normalize_text)
            for col in str_cols
        }

        for word in words:
            pattern = rf"\b{re.escape(word)}\b"
            for col, series in normalized_cols.items():
                mask = mask | series.str.contains(pattern, na=False, regex=True)
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
        entity_lower = self.normalize_text(entity)
        for col in df.select_dtypes(include=["object"]).columns:
            if col == "_source_file":
                continue
            normalized = df[col].astype(str).map(self.normalize_text)
            mask = mask | normalized.str.contains(rf"\b{re.escape(entity_lower)}\b", na=False, regex=True)
        result = df[mask].drop(columns=["_source_file"], errors="ignore")
        if result.empty:
            return ""
        return result.to_csv(index=False)
