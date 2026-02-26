from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List

# Callback signature: (phase, current, total, message)
ProgressCb = Callable[[str, int, int, str], None]


@dataclass
class HeaderRoles:
    """Map header rows to semantic fields for metric columns.
    Row indexes are 0-based; None means not used.
    """
    base_name_row: Optional[int] = 1
    source_row: Optional[int] = 2
    unit_row: Optional[int] = 3
    type_row: Optional[int] = 4
    # If set, header cells can be split by this delimiter and mapped to
    # fields according to `header_split_order`. Example: header_split_delim='-' and
    # header_split_order=['base','source','unit','type']
    header_split_delim: Optional[str] = None
    header_split_order: Optional[List[str]] = None


@dataclass
class ImportOptions:
    # lineage
    system_name: str = "DefaultSystem"
    dataset_name: str = "DefaultDataset"
    duplicate_policy: str = "cancel"  # cancel | replace

    # Structure
    excel_header_rows: int = 4
    csv_header_rows: int = 1
    header_roles: HeaderRoles = field(default_factory=HeaderRoles)

    # Timestamp handling
    date_column: Optional[str] = None
    auto_detect_datetime: bool = True
    assume_dayfirst: bool = True

    # New (optional) explicit patterns and toggles
    # If provided, we will try these formats before falling back to heuristics.
    # Examples: ["%d/%m/%Y %H.%M", "%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M"]
    datetime_formats: Optional[List[str]] = None
    # Treat "9.00" as "9:00" etc. When False we won't rewrite dot-separated times.
    dot_time_as_colon: bool = True

    # CSV reader hints
    csv_has_header: bool = True
    csv_delimiter: Optional[str] = None
    csv_decimal: Optional[str] = None
    csv_encoding: Optional[str] = None
    # Use DuckDB CSV import (auto-enabled for large files when None)
    use_duckdb_csv_import: Optional[bool] = None

    # Note: header splitting configuration now lives in HeaderRoles.

    # Explicit list of header labels that should always be treated as meta columns
    # during parsing, even if their values look numeric.
    force_meta_columns: Optional[List[str]] = None

    # Optional list of prefixes; any column whose header starts with one of these
    # prefixes (case-insensitive) will be skipped entirely during import.
    ignore_column_prefixes: Optional[List[str]] = None

    # Insert batching and progress callback
    insert_chunk_rows: int = 200_000
    progress_cb: Optional[Callable[[str, int, int, str], None]] = None

    insert_workers: int = 1            # >1 to enable threaded inserts
