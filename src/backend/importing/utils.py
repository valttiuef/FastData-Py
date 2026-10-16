
from __future__ import annotations
import datetime
import hashlib
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

import re
import math
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

import warnings

from core.datetime_utils import ensure_series_naive

_EXCEL_EPOCH = pd.Timestamp("1899-12-30")  # Excel serial origin (Windows)


def _is_excel_serial_number(x) -> bool:
    """Heuristic: treat as Excel serial if numeric and within sane calendar range."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return False
    try:
        v = float(x)
    except Exception:
        return False
    # Excel dates after ~1900: serial > 1; upper bound ~ far future
    return 1 <= v <= 600000  # ~2600-04-07 — generous bound


def _from_excel_serial(x) -> Optional[pd.Timestamp]:
    try:
        return _EXCEL_EPOCH + pd.to_timedelta(float(x), unit="D")
    except Exception:
        return None


_DOT_TIME_RE = re.compile(
    r"""
    ^\s*
    (?P<datepart>.+?)      # anything up to a space separates date/time
    \s+
    (?P<h>\d{1,2})
    \.(?P<m>\d{2})
    (?:\.(?P<s>\d{2}))?
    \s*$
    """,
    re.VERBOSE,
)

# Also support pure time like "9.00" without date (we'll pass through and let parser combine).
_PURE_DOT_TIME_RE = re.compile(r"^\s*(?P<h>\d{1,2})\.(?P<m>\d{2})(?:\.(?P<s>\d{2}))?\s*$")


def _rewrite_dot_time_to_colon(s: str) -> str:
    """
    Convert Finnish/European style 'HH.MM' or 'H.MM.SS' into 'HH:MM[:SS]'.
    Works whether a date part precedes it or not.
    """
    if not isinstance(s, str):
        return s

    m = _DOT_TIME_RE.match(s)
    if m:
        h = m.group("h")
        mnt = m.group("m")
        sec = m.group("s")
        time_part = f"{int(h):02d}:{int(mnt):02d}" + (f":{int(sec):02d}" if sec else "")
        return f"{m.group('datepart')} {time_part}"

    pm = _PURE_DOT_TIME_RE.match(s)
    if pm:
        h = pm.group("h")
        mnt = pm.group("m")
        sec = pm.group("s")
        return f"{int(h):02d}:{int(mnt):02d}" + (f":{int(sec):02d}" if sec else "")

    return s


def _try_parse_with_formats(s: pd.Series, fmts: Iterable[str], *, dayfirst: bool) -> pd.Series:
    out = pd.Series(pd.NaT, index=s.index, dtype="object")
    for f in fmts:
        if f == "epoch_s":
            mask = out.isna()
            cand = pd.to_datetime(s[mask], unit="s", errors="coerce")
            out.loc[mask] = cand
            continue
        if f == "epoch_ms":
            mask = out.isna()
            cand = pd.to_datetime(s[mask], unit="ms", errors="coerce")
            out.loc[mask] = cand
            continue
        mask = out.isna()
        cand = pd.to_datetime(s[mask], errors="coerce", format=f, dayfirst=dayfirst)
        out.loc[mask] = cand
    return out


_ISO_LEADING_DATE_RE = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\D|$)")


def _parse_single_datetime_mixed(x, *, dayfirst: bool):
    """Parse one datetime-like value with robust day/month handling."""
    if x is None:
        return pd.NaT
    if isinstance(x, float) and pd.isna(x):
        return pd.NaT

    # For leading YYYY-MM-DD style values, prefer year-first semantics.
    if isinstance(x, str) and _ISO_LEADING_DATE_RE.match(x):
        p = pd.to_datetime(
            x,
            errors="coerce",
            dayfirst=False,
            format="mixed",
            utc=False,
        )
        if not pd.isna(p):
            return p

    p = pd.to_datetime(
        x,
        errors="coerce",
        dayfirst=dayfirst,
        format="mixed",
        utc=False,
    )
    if pd.isna(p):
        p = pd.to_datetime(
            x,
            errors="coerce",
            dayfirst=not dayfirst,
            format="mixed",
            utc=False,
        )
    return p


def _normalize_ts_series(
    ts_col: pd.Series,
    dayfirst: bool = True,
    *,
    dot_time_as_colon: bool = True,
    explicit_formats: Optional[Iterable[str]] = None,
) -> pd.Series:
    """
    Normalize timestamps to a timezone-naive datetime64[ns] Series without altering
    the original wall-clock times.
    """
    # @ai(gpt-6, codex, bugfix, 2026-10-16)
    s = ts_col.copy()

    # --- 0) Already datetime-like -> drop any timezone info without conversion ---
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return ensure_series_naive(s)

    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return pd.to_datetime(s, errors="coerce")

    # --- 1) Excel serials (vectorized heuristic) ---
    nn = s.dropna()
    numeric = pd.to_numeric(nn, errors="coerce")
    for unit, low, high in (("s", 10**9, 2_000_000_000), ("ms", 10**12, 2_000_000_000_000)):
        if not numeric.empty and numeric.between(low, high).mean() > 0.5:
            return pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit=unit, errors="coerce")
    excel_mask = nn.map(_is_excel_serial_number)
    if not nn.empty and excel_mask.mean() > 0.5:
        # Parse serials and drop any timezone information without conversion
        out = s.map(lambda x: _from_excel_serial(x) if _is_excel_serial_number(x) else pd.NaT)
        out = pd.to_datetime(out, errors="coerce")
        return ensure_series_naive(out)

    # --- 2) Strings: optionally rewrite dot-separated times ---
    s2 = s.astype("object")
    if dot_time_as_colon:
        s2 = s2.map(lambda x: _rewrite_dot_time_to_colon(x) if isinstance(x, str) else x)

    # --- 3) Try explicit formats first (fast/strict) ---
    if explicit_formats:
        parsed = _try_parse_with_formats(s2, explicit_formats, dayfirst=dayfirst)
        remaining = parsed.isna()
    else:
        parsed = pd.Series(pd.NaT, index=s2.index, dtype="object")
        remaining = pd.Series(True, index=s2.index)

    # --- 4) Generic parser attempts (progressively looser) ---
    if remaining.any():
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Could not infer format, so each element will be parsed individually",
                    category=UserWarning,
                )
                # Prefer stated day order; parse without altering wall-clock times
                p1 = pd.to_datetime(
                    s2[remaining],
                    errors="coerce",
                    dayfirst=dayfirst,
                    format="mixed",
                    utc=False,
                )
                iso = s2[remaining].map(lambda x: isinstance(x, str) and bool(_ISO_LEADING_DATE_RE.match(x)))
                if iso.any():
                    p1.loc[iso] = pd.to_datetime(s2[remaining][iso], errors="coerce", format="mixed", dayfirst=False)
        except Exception:
            # Fall back to per-value parsing if vectorized parsing fails
            # (e.g. mixed timezone offsets / mixed aware+naive values).
            p1 = s2[remaining].map(
                lambda x: _parse_single_datetime_mixed(x, dayfirst=dayfirst)
            )
        parsed.loc[remaining] = p1
        remaining = parsed.isna()

    if remaining.any():
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Could not infer format, so each element will be parsed individually",
                    category=UserWarning,
                )
                # Flip dayfirst as a fallback without introducing localization
                p2 = pd.to_datetime(
                    s2[remaining],
                    errors="coerce",
                    dayfirst=not dayfirst,
                    format="mixed",
                    utc=False,
                )
        except Exception:
            p2 = s2[remaining].map(
                lambda x: _parse_single_datetime_mixed(x, dayfirst=not dayfirst)
            )
        parsed.loc[remaining] = p2
        remaining = parsed.isna()

    # --- 5) Final cleanup: ensure tz-naive datetime64[ns] without localization ---
    return ensure_series_naive(parsed)


def as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)

def clean_cell(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    if s.startswith("Unnamed"):
        return None
    return s

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def chunks(n: int, size: int) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < n:
        j = min(n, i + size)
        yield i, j
        i = j

def detect_first_data_row(raw: pd.DataFrame, search_cols: int = 6, *, dayfirst: bool = True, dot_time_as_colon: bool = True) -> Tuple[int, int]:
    """Return (first_data_row_idx, ts_col_idx) by scanning first few columns for a datetime.
    
    Args:
        raw: DataFrame with header=None (raw data)
        search_cols: Number of columns to check for datetime
        dayfirst: Whether to interpret ambiguous dates as DD/MM
        dot_time_as_colon: Whether to rewrite dot-separated times (HH.MM -> HH:MM)
    """
    # @ai(gpt-6, codex, bugfix, 2026-10-16)
    best = (None, None)
    for c in range(min(search_cols, raw.shape[1])):
        col_data = raw.iloc[:, c].copy()
        # Header numbers, measurements and times alone are not calendar dates.
        col_data = col_data.where(col_data.map(_is_calendar_value))
        
        # Apply dot-time normalization if enabled
        if dot_time_as_colon:
            col_data = col_data.map(lambda x: _rewrite_dot_time_to_colon(x) if isinstance(x, str) else x)
        
        try:
            s = _normalize_ts_series(col_data, dayfirst=dayfirst, dot_time_as_colon=dot_time_as_colon)
        except Exception:
            s = col_data.map(
                lambda x: _parse_single_datetime_mixed(x, dayfirst=dayfirst)
            )
            s = ensure_series_naive(pd.Series(s, index=col_data.index))
        idx = s.first_valid_index()
        if idx is not None:
            r = int(idx)
            if best[0] is None or r < best[0]:
                best = (r, c)
    if best[0] is None:
        raise ValueError("Could not find a timestamp column in the first few columns.")
    return best


def _is_calendar_value(value) -> bool:
    if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp, np.datetime64)):
        return not pd.isna(value)
    if isinstance(value, str):
        if value.strip().isdigit():
            number = int(value.strip())
            return 10**9 <= number <= 2_000_000_000 or 10**12 <= number <= 2_000_000_000_000
        return bool(re.search(r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}|[A-Za-z]{3,}\s+\d{1,2},?\s+\d{4})", value))
    if isinstance(value, (int, float, np.number)) and not pd.isna(value):
        return 20000 <= value <= 80000 or 10**9 <= value <= 2_000_000_000 or 10**12 <= value <= 2_000_000_000_000
    return False


def _normalize_time_series(series: pd.Series, *, dot_time_as_colon: bool = True) -> pd.Series:
    """Parse clock times as offsets from midnight, never using today's date."""
    # @ai(gpt-6, codex, implementation, 2026-10-16)
    def parse(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, datetime.time):
            return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second, microseconds=value.microsecond)
        if isinstance(value, (float, int, np.number)):
            if 0 <= value < 1:
                return pd.Timedelta(seconds=round(float(value) * 86400, 6))
            value = str(value)
            if dot_time_as_colon and re.fullmatch(r"\d{1,2}\.\d", value):
                value += "0"
        text = str(value).strip()
        if dot_time_as_colon:
            text = _rewrite_dot_time_to_colon(text)
        match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2})(\.\d{1,6})?)?", text)
        if not match:
            return pd.NaT
        h, m, s = (int(match.group(i) or 0) for i in (1, 2, 3))
        if h >= 24 or m >= 60 or s >= 60:
            return pd.NaT
        return pd.Timedelta(hours=h, minutes=m, seconds=s + float(match.group(4) or 0))
    return pd.Series([parse(value) for value in series], index=series.index, dtype="timedelta64[ns]")

def header_tuple_for_col(raw: pd.DataFrame, col_idx: int, header_rows: int) -> tuple:
    """Return a fixed-length 4-tuple (row0..row3) of header cells for a column."""
    cells = []
    for r in range(header_rows):
        cells.append(clean_cell(raw.iat[r, col_idx]) if col_idx < raw.shape[1] else None)
    while len(cells) < 4:
        cells.append(None)
    return tuple(cells[:header_rows])

def harvest_group_value_from_header(raw: pd.DataFrame, header_rows: int, key_name: Optional[str]) -> Optional[str]:
    if not key_name:
        return None
    # Look for a cell equal to key_name in the top-left 10x(header_rows) block, then take right-neighbor.
    rows = min(header_rows, raw.shape[0])
    cols = min(10, raw.shape[1])
    key_name_norm = str(key_name).strip().lower()
    for r in range(rows):
        for c in range(cols):
            cell = clean_cell(raw.iat[r, c])
            if cell and cell.strip().lower() == key_name_norm:
                right = clean_cell(raw.iat[r, c+1]) if (c + 1) < raw.shape[1] else None
                return right
    return None

def coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None
