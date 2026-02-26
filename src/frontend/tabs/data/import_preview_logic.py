
from __future__ import annotations
import datetime
import os
import re
import warnings
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from backend.importing import _get_encoding_candidates

_EPOCH_S_LO = 10**9
_EPOCH_S_HI = 2_000_000_000
_EPOCH_MS_LO = 10**12
_EPOCH_MS_HI = 2_000_000_000_000

_PATTERNS = [
    (re.compile(r"^\s*\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)?\s*$"), "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"^\s*\d{4}/\d{1,2}/\d{1,2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\s*$"), "%Y/%m/%d %H:%M:%S"),
    (re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}(?:[ T]\d{2}[:\.]?\d{2}(?:[:\.]?\d{2})?)?\s*$"), "%d/%m/%Y %H:%M:%S"),
    (re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}(?:[ T]\d{2}[:\.]?\d{2}(?:[:\.]?\d{2})?)?\s*$"), "%d.%m.%Y %H:%M:%S"),
    (re.compile(r"^\s*\d{1,2}-\d{1,2}-\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\s*$"), "%d-%m-%Y %H:%M:%S"),
    (re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$"), "%d/%m/%Y"),
    (re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$"), "%d.%m.%Y"),
]

_TIME_DOT_RE = re.compile(r"\b\d{1,2}\.\d{2}(?:\.\d{2})?\b")
_MAX_PREVIEW_ROWS = 8
_MAX_PREVIEW_COLS = 32
_MAX_DATETIME_SCAN_VALUES = 24


def _normalize_incomplete_time(value: str) -> str:
    value = re.sub(r"(\d{1,2})\.(\d{2})(?![\d.])", r"\1.\2.00", value)
    value = re.sub(r"(\d{1,2})\.(\d{2})\.(\d{2})", r"\1:\2:\3", value)
    return value


def _looks_epoch(value) -> Optional[str]:
    try:
        if pd.isna(value):
            return None
        parsed = float(str(value).strip())
        if _EPOCH_S_LO <= parsed <= _EPOCH_S_HI:
            return "s"
        if _EPOCH_MS_LO <= parsed <= _EPOCH_MS_HI:
            return "ms"
    except Exception:
        return None
    return None


def _infer_dayfirst_from_samples(samples: list[str]) -> bool:
    dayfirst_votes = 0
    total_votes = 0
    for sample in samples:
        sample_text = sample if isinstance(sample, str) else str(sample)
        match = re.match(r"^\s*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})", sample_text)
        if not match:
            continue
        a_val, b_val = int(match.group(1)), int(match.group(2))
        if a_val > 12 and b_val <= 12:
            dayfirst_votes += 1
            total_votes += 1
        elif b_val > 12 and a_val <= 12:
            total_votes += 1
    return dayfirst_votes >= max(1, total_votes)


def _infer_common_format(samples: list[str], dayfirst_guess: bool) -> tuple[list[str], bool]:
    hits = Counter()
    dot_time_seen = False
    for sample in samples:
        if sample is None or (isinstance(sample, float) and np.isnan(sample)):
            continue
        sample_text = str(sample).strip()
        if not sample_text:
            continue
        if _TIME_DOT_RE.search(sample_text):
            dot_time_seen = True
        for pattern, fmt in _PATTERNS:
            if pattern.match(sample_text):
                hits[fmt] += 1
                break

    formats = [fmt for fmt, _count in hits.most_common()]
    if any("%d/%m/%Y" in fmt for fmt in formats) and not dayfirst_guess:
        formats = [fmt.replace("%d/%m/%Y", "%m/%d/%Y") for fmt in formats]

    result: list[str] = []
    for fmt in formats[:3]:
        if " %H:%M:%S" in fmt:
            result.append(fmt)
            result.append(fmt.replace(" %H:%M:%S", ""))
        else:
            result.append(fmt)
    if not result:
        result = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    return result[:3], dot_time_seen


def _guess_datetime_info_from_df(
    frame: pd.DataFrame,
    sample_rows: int = 50,
    max_columns: int = _MAX_PREVIEW_COLS,
) -> tuple[Optional[int], Optional[str], list[str], bool, bool]:
    best_idx = None
    best_score = -1.0
    best_dayfirst = False
    best_dot_time = False
    best_formats: list[str] = []
    best_name: Optional[str] = None

    min_datetime_values = 5
    min_success_percentage = 70

    def _trim_leading_nulls(series: pd.Series) -> pd.Series:
        start_idx = 0
        for idx, value in enumerate(series.tolist()):
            if not pd.isna(value):
                start_idx = idx
                break
        return series.iloc[start_idx:]

    def _datetime_series_default_formats(series: pd.Series) -> list[str]:
        try:
            parsed = pd.to_datetime(series, errors="coerce").dropna()
        except Exception:
            parsed = pd.Series(dtype="datetime64[ns]")
        if parsed.empty:
            return []
        has_time = False
        for ts in parsed.head(16):
            try:
                stamp = pd.Timestamp(ts)
                if stamp.hour or stamp.minute or stamp.second or stamp.microsecond:
                    has_time = True
                    break
            except Exception:
                continue
        return ["%Y-%m-%d %H:%M:%S"] if has_time else ["%Y-%m-%d"]

    columns = list(frame.columns)[: max(1, int(max_columns))]
    for col_idx, col in enumerate(columns):
        sampled = frame[col].head(sample_rows)
        sampled = _trim_leading_nulls(sampled)
        series = sampled.dropna()
        if series.empty:
            continue
        total = len(series)
        if total < min_datetime_values:
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            score = 100.0
            if score > best_score:
                best_score = score
                best_idx = col_idx
                best_name = str(col)
                best_dayfirst = False
                best_dot_time = False
                best_formats = _datetime_series_default_formats(series)
            continue

        epoch_votes = sum(1 for value in series if _looks_epoch(value))
        if epoch_votes >= max(3, int(0.6 * total)):
            score = (epoch_votes / total) * 100.0
            if score > best_score:
                best_score = score
                best_idx = col_idx
                best_name = str(col)
                best_dayfirst = False
                best_dot_time = False
                first_kind = _looks_epoch(series.iloc[0])
                best_formats = ["epoch_ms" if first_kind == "ms" else "epoch_s"]
            continue

        raw_samples = [str(value) for value in series.tolist()]
        normalized_samples = [_normalize_incomplete_time(sample) for sample in raw_samples]
        has_date_structure = any(re.search(r"[/\-\.\s]", sample) for sample in raw_samples)
        if not has_date_structure:
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Could not infer format, so each element will be parsed individually",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Parsing dates in .* format when dayfirst=.* was specified",
                category=UserWarning,
            )
            parsed_true = pd.to_datetime(pd.Series(normalized_samples), errors="coerce", dayfirst=True)
            parsed_false = pd.to_datetime(pd.Series(normalized_samples), errors="coerce", dayfirst=False)

        ok_true = int(parsed_true.notna().sum())
        ok_false = int(parsed_false.notna().sum())
        ok_count = max(ok_true, ok_false)
        if ok_count < min_datetime_values:
            continue

        success_pct = (ok_count / total) * 100.0
        if success_pct < min_success_percentage:
            continue

        dayfirst_guess = ok_true >= ok_false
        if ok_true == ok_false:
            dayfirst_guess = _infer_dayfirst_from_samples(raw_samples)

        dot_time = any(_TIME_DOT_RE.search(sample) for sample in raw_samples)
        formats, _ = _infer_common_format(normalized_samples, dayfirst_guess)
        if success_pct > best_score:
            best_score = success_pct
            best_idx = col_idx
            best_name = str(col)
            best_dayfirst = bool(dayfirst_guess)
            best_dot_time = bool(dot_time)
            best_formats = formats

    return best_idx, best_name, best_formats, best_dayfirst, best_dot_time


def _read_first_line(path: str, encoding_candidates: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252")) -> str:
    for encoding in encoding_candidates:
        try:
            with open(path, "r", encoding=encoding, errors="replace") as handle:
                return handle.readline().rstrip("\n\r")
        except Exception:
            continue
    return ""


def _read_csv_with_encoding_fallback(path: str, user_encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in _get_encoding_candidates(user_encoding):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    return pd.read_csv(path, **kwargs)


def _quick_guess_delimiter_from_line(line: str) -> Optional[str]:
    if not line:
        return None
    candidates = ["\t", ",", ";", "|", ":"]
    counts = {candidate: line.count(candidate) for candidate in candidates}
    if max(counts.values(), default=0) == 0:
        return None
    priority = {candidate: idx for idx, candidate in enumerate(candidates)}
    return sorted(counts.items(), key=lambda item: (-item[1], priority[item[0]]))[0][0]


def _is_excel(path: str) -> bool:
    lowered = path.lower()
    return lowered.endswith(".xlsx") or lowered.endswith(".xls")


def _is_csv_like(path: str) -> bool:
    lowered = path.lower()
    return any(lowered.endswith(ext) for ext in (".csv", ".tsv", ".txt"))


def _resolve_preview_limits(nrows: int, ncolumns: int) -> tuple[int, int]:
    rows = max(4, min(int(nrows), _MAX_PREVIEW_ROWS))
    cols = max(8, min(int(ncolumns), _MAX_PREVIEW_COLS))
    return rows, cols


def _limited_usecols(ncolumns: int) -> list[int]:
    return list(range(max(1, int(ncolumns))))


def _read_preview_df(
    *,
    file_path: str,
    user_encoding: Optional[str],
    header: Optional[int],
    nrows: int,
    ncolumns: int,
    kwargs: dict[str, object],
) -> pd.DataFrame:
    usecols = _limited_usecols(ncolumns)
    if _is_excel(file_path):
        try:
            return pd.read_excel(file_path, header=header, nrows=nrows, usecols=usecols)
        except Exception:
            return pd.read_excel(file_path, header=header, nrows=nrows)
    try:
        return _read_csv_with_encoding_fallback(
            file_path,
            user_encoding=user_encoding,
            header=header,
            nrows=nrows,
            usecols=usecols,
            **kwargs,
        )
    except Exception:
        return _read_csv_with_encoding_fallback(
            file_path,
            user_encoding=user_encoding,
            header=header,
            nrows=nrows,
            **kwargs,
        )


def _count_datetime_like(values: pd.Series, max_values: int = _MAX_DATETIME_SCAN_VALUES) -> int:
    count = 0
    checked = 0
    for value in values:
        if checked >= max_values:
            break
        if pd.isna(value):
            continue
        checked += 1
        if isinstance(value, (pd.Timestamp, datetime.datetime, np.datetime64)):
            count += 1
            continue
        text = str(value).strip()
        if not text:
            continue
        if _looks_epoch(text):
            count += 1
            continue
        matched = False
        for pattern, _fmt in _PATTERNS:
            if pattern.match(text):
                count += 1
                matched = True
                break
        if matched:
            continue
        if re.search(r"[/\-\.\s:]", text) and re.search(r"\d{2,4}", text):
            count += 1
    return count


def build_import_preview_payload(
    *,
    file_path: str,
    csv_delimiter: Optional[str],
    csv_decimal: Optional[str],
    csv_encoding: Optional[str],
    base_header_index: Optional[int],
    guess: bool = True,
    nrows: int = _MAX_PREVIEW_ROWS,
    ncolumns: int = _MAX_PREVIEW_COLS,
) -> dict[str, object]:
    path = str(file_path)
    payload: dict[str, object] = {
        "display_df": pd.DataFrame(),
        "header_rows": None,
        "base_header_index": base_header_index,
        "date_column": None,
        "datetime_formats": [],
        "assume_dayfirst": None,
        "dot_time_as_colon": None,
        "csv_delimiter_guess": None,
        "error": None,
    }

    try:
        preview_rows, preview_cols = _resolve_preview_limits(nrows, ncolumns)
        kwargs: dict[str, object] = {}
        user_sep = (csv_delimiter or "").strip()
        delim_guess = None
        if user_sep:
            kwargs["sep"] = user_sep
        elif guess and _is_csv_like(path):
            delim_guess = _quick_guess_delimiter_from_line(_read_first_line(path))
            if delim_guess:
                kwargs["sep"] = delim_guess
        payload["csv_delimiter_guess"] = delim_guess

        if (csv_decimal or "").strip():
            kwargs["decimal"] = csv_decimal.strip()
        user_encoding = csv_encoding.strip() if (csv_encoding or "").strip() else None

        header_rows = None
        resolved_base_index = int(base_header_index or 0)

        if guess:
            raw_df = _read_preview_df(
                file_path=path,
                user_encoding=user_encoding,
                header=None,
                nrows=preview_rows,
                ncolumns=preview_cols,
                kwargs=kwargs,
            )

            ok_per_row: list[int] = []
            for row_idx in range(min(len(raw_df), preview_rows)):
                row = raw_df.iloc[row_idx]
                ok = _count_datetime_like(row, max_values=min(preview_cols, _MAX_DATETIME_SCAN_VALUES))
                ok_per_row.append(ok)

            first_row = next((idx for idx, count in enumerate(ok_per_row) if count > 0), None)
            total_parseable = sum(ok_per_row)
            if first_row is not None and total_parseable >= 3:
                full_header_rows = int(first_row)
                header_rows = min(full_header_rows, 4)
                if full_header_rows <= 1:
                    resolved_base_index = 0
                elif full_header_rows > 4:
                    resolved_base_index = 1
                else:
                    resolved_base_index = 0
                if header_rows is not None and resolved_base_index >= int(header_rows):
                    resolved_base_index = max(0, int(header_rows) - 1)

        if header_rows is not None and header_rows > 0:
            df = _read_preview_df(
                file_path=path,
                user_encoding=user_encoding,
                header=resolved_base_index,
                nrows=preview_rows,
                ncolumns=preview_cols,
                kwargs=kwargs,
            )

            idx, name, formats, dayfirst, dot_time = _guess_datetime_info_from_df(
                df,
                sample_rows=preview_rows,
                max_columns=preview_cols,
            )
            if idx is not None and name and int(header_rows) == 1:
                payload["date_column"] = name
            if formats:
                payload["datetime_formats"] = [fmt for fmt in formats if not fmt.startswith("epoch_")]
            payload["assume_dayfirst"] = bool(dayfirst)
            if dot_time:
                payload["dot_time_as_colon"] = True
        else:
            df = _read_preview_df(
                file_path=path,
                user_encoding=user_encoding,
                header=resolved_base_index,
                nrows=preview_rows,
                ncolumns=preview_cols,
                kwargs=kwargs,
            )

        rows = min(preview_rows, len(df.index))
        cols = min(preview_cols, len(df.columns))
        payload["display_df"] = df.head(rows).iloc[:, :cols]
        payload["header_rows"] = header_rows
        payload["base_header_index"] = resolved_base_index
        return payload
    except Exception as exc:
        payload["error"] = str(exc)
        payload["display_df"] = pd.DataFrame([{"Preview error": str(exc)}])
        return payload


def should_enable_duckdb_csv_import(file_path: str, threshold_bytes: int) -> bool:
    if not _is_csv_like(file_path):
        return False
    try:
        return os.path.getsize(file_path) >= int(threshold_bytes)
    except Exception:
        return False
