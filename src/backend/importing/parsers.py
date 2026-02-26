from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import pandas as pd
import numpy as np
import logging

from ..models import ImportOptions, HeaderRoles
from .utils import (
    _normalize_ts_series,
    detect_first_data_row,
    header_tuple_for_col,
    clean_cell,
)

import os
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

def _prepare_header_meta_for_csv(file_path: Path, options: ImportOptions, read_kwargs: dict, unit_callback=None) -> dict:
    # a tiny head read for header/ts inference
    head_kwargs = dict(read_kwargs)
    head_kwargs.pop("chunksize", None)
    head_kwargs["nrows"] = 2_048  # enough for header sniff + a bit of data
    raw_head = pd.read_csv(file_path, **head_kwargs)

    # reuse your inference (works on a small frame)
    header_rows, ts_col = _find_ts_column(raw_head, options, file_type="csv")
    if ts_col is None or ts_col < 0 or ts_col >= int(raw_head.shape[1]):  # safety
        ts_col = 0

    # build header tuples once from header rows only
    header_only = raw_head.iloc[:header_rows] if header_rows > 0 else raw_head.iloc[0:0]
    ncols = int(raw_head.shape[1])
    header_tuples = {c: header_tuple_for_col(header_only, c, header_rows) for c in range(ncols)}

    # precompute meta names for *all* non-ts columns so names stay stable across chunks
    roles: HeaderRoles = options.header_roles
    DEFAULT_META_PREFIX = "meta"
    used_names: Set[str] = set({"ts", "value", "base_name", "source", "unit", "type"})

    def _cell_at(tup, idx):
        return tup[idx] if (idx is not None and idx < len(tup)) else None

    def _cleanish_name_from_tuple(cidx: int) -> str:
        tup = header_tuples.get(cidx, ())
        first_non_empty = None
        for x in tup:
            if x is not None and str(x).strip():
                first_non_empty = str(x).strip()
                break
        base = clean_cell(first_non_empty) if first_non_empty else ""
        if not base:
            base = f"{DEFAULT_META_PREFIX}_{cidx+1}"
        name0 = base
        k = 1
        while base in used_names:
            base = f"{name0}_{cidx+1 if k == 1 else f'{cidx+1}_{k}'}"
            k += 1
        used_names.add(base)
        return base

    meta_name_map = {c: _cleanish_name_from_tuple(c) for c in range(ncols) if c != ts_col}

    # prepare split config once
    hdelim = getattr(roles, "header_split_delim", None)
    horder = getattr(roles, "header_split_order", None)
    do_split = bool(hdelim and isinstance(horder, list) and len(horder) > 0)

    forced_meta_cols = _resolve_forced_meta_columns(
        header_tuples,
        getattr(options, "force_meta_columns", None) or [],
        ts_col=ts_col,
    )

    ignored_cols = _resolve_ignored_columns(
        header_tuples,
        getattr(options, "ignore_column_prefixes", None) or [],
        ts_col=ts_col,
    )

    sample_data = raw_head.iloc[header_rows:] if header_rows > 0 else raw_head
    metric_meta = _infer_csv_metric_columns(
        sample_data,
        {
            "ts_col": ts_col,
            "ncols": ncols,
            "roles": roles,
            "header_tuples": header_tuples,
            "do_split": do_split,
            "hdelim": hdelim,
            "horder": horder,
            "forced_meta_cols": forced_meta_cols,
            "ignored_cols": ignored_cols,
        },
        options,
    )

    if unit_callback:
        try:
            unit_callback(f"Parsed column {ts_col+1} (timestamp)")
            unit_callback(f"Parse phase: {ncols}")
        except Exception:
            logger.warning("Unit callback failed while preparing CSV header metadata", exc_info=True)

    return {
        "ncols": ncols,
        "header_rows": header_rows,
        "ts_col": ts_col,
        "header_tuples": header_tuples,
        "meta_name_map": meta_name_map,
        "roles": roles,
        "do_split": do_split,
        "hdelim": hdelim,
        "horder": horder,
        "forced_meta_cols": forced_meta_cols,
        "ignored_cols": ignored_cols,
        "metric_meta": metric_meta,
    }

def _split_header_parts(
    source: Optional[object],
    *,
    delimiter: Optional[str],
    order: Optional[List[str]],
) -> Dict[str, str]:
    """Split a header text into named parts only when delimiter is present."""
    if source is None:
        return {}
    if not delimiter or not isinstance(order, list) or not order:
        return {}

    text = str(source).strip()
    if not text or delimiter not in text:
        return {}

    parts = [p.strip() for p in text.split(delimiter)]
    mapping: Dict[str, str] = {}
    for i, key in enumerate(order):
        if i < len(parts) and parts[i]:
            mapping[str(key).lower()] = parts[i]
    return mapping


def _infer_csv_metric_columns(
    data: pd.DataFrame,
    header_meta: dict,
    options: ImportOptions,
) -> List[Dict[str, Optional[str]]]:
    """Infer metric columns from a CSV sample using header metadata."""
    if data is None or data.empty:
        return []

    ts_col = header_meta["ts_col"]
    ncols = header_meta["ncols"]
    roles = header_meta["roles"]
    header_tuples = header_meta["header_tuples"]
    do_split = header_meta.get("do_split")
    hdelim = header_meta.get("hdelim")
    horder = header_meta.get("horder")
    forced_meta_cols = set(header_meta.get("forced_meta_cols") or [])
    ignored_cols = set(header_meta.get("ignored_cols") or [])

    metric_cols: List[int] = []
    decimal_hint = options.csv_decimal

    for col in range(ncols):
        if col >= data.shape[1]:
            continue
        if col == ts_col or col in ignored_cols:
            continue
        s = data.iloc[:, col]
        s_num = _coerce_metric_series(s, decimal_hint=decimal_hint)
        if s_num.notna().any():
            metric_cols.append(col)

    if forced_meta_cols:
        forced_meta_cols = {c for c in forced_meta_cols if c < data.shape[1] and c != ts_col and c not in ignored_cols}
        if forced_meta_cols:
            metric_cols = [c for c in metric_cols if c not in forced_meta_cols]

    if not metric_cols:
        return []

    def _cell_at(tup, idx: Optional[int]):
        return tup[idx] if (idx is not None and idx < len(tup)) else None

    metric_meta: List[Dict[str, Optional[str]]] = []
    for col in metric_cols:
        tup = header_tuples.get(col, ())
        base_name = _cell_at(tup, roles.base_name_row)
        source = _cell_at(tup, roles.source_row)
        unit = _cell_at(tup, roles.unit_row)
        type = _cell_at(tup, roles.type_row)

        base_name_raw = str(base_name) if base_name is not None else None
        if do_split and (base_name_raw or (len(tup) > 0 and tup[0])):
            src = base_name_raw if base_name_raw else tup[0]
            mapping = _split_header_parts(src, delimiter=hdelim, order=horder)
            if mapping:
                base_name = mapping.get("base", base_name)
                source = mapping.get("source", source)
                unit = mapping.get("unit", unit)
                type = mapping.get("type", type)

        if not base_name and len(tup) > 0 and tup[0] and str(tup[0]).strip():
            base_name = str(tup[0]).strip()
        if not base_name:
            for x in tup:
                if x is not None and str(x).strip():
                    base_name = str(x).strip()
                    break
        if not base_name:
            for cand in (source, type, unit):
                if cand:
                    base_name = str(cand)
                    break
        if not base_name:
            base_name = f"col_{col+1}"

        base_name = str(base_name)
        source_s = str(source) if source is not None else None
        unit_s = str(unit) if unit is not None else None
        type_s = str(type) if type is not None else None

        metric_meta.append(
            {
                "column_index": int(col),
                "base_name": base_name,
                "source": source_s,
                "unit": unit_s,
                "type": type_s,
            }
        )

    return metric_meta


def _resolve_forced_meta_columns(
    header_tuples: Dict[int, Tuple[Any, ...]],
    forced_names: Iterable[str],
    *,
    ts_col: Optional[int] = None,
) -> Set[int]:
    """Determine which column indexes should always be treated as meta columns."""
    normalized: Set[str] = set()
    for name in forced_names or []:
        if name is None:
            continue
        s = str(name).strip()
        if not s:
            continue
        normalized.add(s.lower())

    if not normalized:
        return set()

    forced_cols: Set[int] = set()
    for col, tup in header_tuples.items():
        if ts_col is not None and col == ts_col:
            continue
        for cell in tup:
            if cell is None:
                continue
            cell_str = str(cell).strip().lower()
            if cell_str and cell_str in normalized:
                forced_cols.add(col)
                break
    return forced_cols


def _resolve_ignored_columns(
    header_tuples: Dict[int, Tuple[Any, ...]],
    prefixes: Iterable[str],
    *,
    ts_col: Optional[int] = None,
) -> Set[int]:
    """Return column indexes whose headers should be ignored."""

    norm_prefixes = []
    for pref in prefixes or []:
        if pref is None:
            continue
        s = str(pref).strip().lower()
        if not s:
            continue
        norm_prefixes.append(s)

    if not norm_prefixes:
        return set()

    ignored: Set[int] = set()
    for col, tup in header_tuples.items():
        if ts_col is not None and col == ts_col:
            continue
        for cell in tup:
            if cell is None:
                continue
            text = str(cell).strip().lower()
            if not text:
                continue
            if any(text.startswith(pref) for pref in norm_prefixes):
                ignored.add(col)
                break
    return ignored


def _build_tall_from_chunk(chunk: pd.DataFrame, H: dict, options: ImportOptions, unit_callback=None) -> pd.DataFrame:
    """Transform one CSV data chunk to tall using header/meta prepared up front."""
    if chunk is None or chunk.empty:
        return pd.DataFrame(columns=["ts","value","base_name","source","unit","type"])

    ts_col = H["ts_col"]; ncols = H["ncols"]
    roles = H["roles"]; header_tuples = H["header_tuples"]
    meta_name_map = H["meta_name_map"]
    do_split = H["do_split"]; hdelim = H["hdelim"]; horder = H["horder"]
    forced_meta_cols = set(H.get("forced_meta_cols") or [])
    ignored_cols = set(H.get("ignored_cols") or [])
    predefined_metric_meta = list(H.get("metric_meta") or [])

    # normalize ts for this chunk
    data = chunk  # chunk has no header rows
    ts = _normalize_ts_series(
        data.iloc[:, ts_col],
        dayfirst=options.assume_dayfirst,
        dot_time_as_colon=options.dot_time_as_colon,
        explicit_formats=options.datetime_formats,
    )

    # classify columns per chunk
    metric_cols: List[int] = []
    meta_cols:   List[int] = []
    coerced_numeric: Dict[int, pd.Series] = {}
    decimal_hint = options.csv_decimal

    if predefined_metric_meta:
        predefined_by_col = {
            int(m["column_index"]): m
            for m in predefined_metric_meta
            if isinstance(m, dict) and "column_index" in m
        }
        for col in range(ncols):
            if col >= data.shape[1]:
                continue  # ragged chunk edge
            if col == ts_col or col in ignored_cols:
                continue
            if col in predefined_by_col:
                s_num = _coerce_metric_series(data.iloc[:, col], decimal_hint=decimal_hint)
                metric_cols.append(col)
                coerced_numeric[col] = s_num
                if unit_callback:
                    try:
                        unit_callback(f"Parsed column {col+1}")
                    except Exception:
                        logger.warning("Unit callback failed while parsing column %s", col + 1, exc_info=True)
            else:
                meta_cols.append(col)
    else:
        for col in range(ncols):
            if col >= data.shape[1]:
                continue  # ragged chunk edge
            if col == ts_col or col in ignored_cols:
                continue
            s = data.iloc[:, col]
            s_num = _coerce_metric_series(s, decimal_hint=decimal_hint)
            if s_num.notna().any():
                metric_cols.append(col)
                coerced_numeric[col] = s_num
                if unit_callback:
                    try:
                        unit_callback(f"Parsed column {col+1}")
                    except Exception:
                        logger.warning("Unit callback failed while parsing column %s", col + 1, exc_info=True)
            else:
                meta_cols.append(col)

    if ignored_cols:
        ignored_cols = {c for c in ignored_cols if c < data.shape[1] and c != ts_col}
        if ignored_cols:
            metric_cols = [c for c in metric_cols if c not in ignored_cols]
            meta_cols = [c for c in meta_cols if c not in ignored_cols]
            for c in ignored_cols:
                coerced_numeric.pop(c, None)

    if forced_meta_cols:
        # Only keep forced columns that actually exist in this chunk
        forced_meta_cols = {c for c in forced_meta_cols if c < data.shape[1] and c != ts_col and c not in ignored_cols}
        if forced_meta_cols:
            metric_cols = [c for c in metric_cols if c not in forced_meta_cols]
            for c in forced_meta_cols:
                coerced_numeric.pop(c, None)

    if not metric_cols:
        return pd.DataFrame(columns=["ts","value","base_name","source","unit","type"])

    if forced_meta_cols:
        meta_union = set(meta_cols)
        meta_union.update(forced_meta_cols)
        meta_cols = [
            c for c in range(min(ncols, data.shape[1]))
            if c != ts_col and c in meta_union and c not in ignored_cols
        ]

    # derive per-metric metadata from header tuples (constant across chunks)
    def cell_at(tup, idx: Optional[int]):
        return tup[idx] if (idx is not None and idx < len(tup)) else None

    metric_meta: List[Dict[str, Optional[str]]] = []
    if predefined_metric_meta:
        predefined_by_col = {
            int(m["column_index"]): m
            for m in predefined_metric_meta
            if isinstance(m, dict) and "column_index" in m
        }
        for col in metric_cols:
            meta = predefined_by_col.get(col)
            if not meta:
                continue
            metric_meta.append(
                {
                    "feature_order": int(col),
                    "base_name": str(meta.get("base_name") or f"col_{col+1}"),
                    "source": str(meta.get("source")) if meta.get("source") is not None else None,
                    "unit": str(meta.get("unit")) if meta.get("unit") is not None else None,
                    "type": str(meta.get("type")) if meta.get("type") is not None else None,
                }
            )
    else:
        for col in metric_cols:
            tup = header_tuples.get(col, ())
            base_name = cell_at(tup, roles.base_name_row)
            source    = cell_at(tup, roles.source_row)
            unit      = cell_at(tup, roles.unit_row)
            type = cell_at(tup, roles.type_row)

            base_name_raw = str(base_name) if base_name is not None else None

            if do_split and (base_name_raw or (len(tup) > 0 and tup[0])):
                src = base_name_raw if base_name_raw else tup[0]
                mapping = _split_header_parts(src, delimiter=hdelim, order=horder)
                if mapping:
                    base_name = mapping.get("base", base_name)
                    source = mapping.get("source", source)
                    unit = mapping.get("unit", unit)
                    type = mapping.get("type", type)

            # fallbacks consistent with your full-frame logic
            if not base_name and len(tup) > 0 and tup[0] and str(tup[0]).strip():
                base_name = str(tup[0]).strip()
            if not base_name:
                for x in tup:
                    if x is not None and str(x).strip():
                        base_name = str(x).strip()
                        break
            if not base_name:
                for cand in (source, type, unit):
                    if cand:
                        base_name = str(cand)
                        break
            if not base_name:
                base_name = f"col_{col+1}"

            base_name = str(base_name)
            source_s    = str(source) if source is not None else None
            unit_s      = str(unit) if unit is not None else None
            type_s = str(type) if type is not None else None

            metric_meta.append({
                "feature_order": int(col),
                "base_name": base_name,
                "source": source_s,
                "unit": unit_s,
                "type": type_s,
            })

    # vectorized tall build for this chunk
    n_rows = len(data); m = len(metric_cols)

    ts_np = ts.to_numpy()
    ts_values = np.tile(ts_np, m)
    value_values = np.concatenate([coerced_numeric[c].to_numpy() for c in metric_cols])

    def rep(values):
        return np.repeat(np.asarray(values, dtype=object), n_rows)

    base_name_values        = rep([mm["base_name"]        for mm in metric_meta])
    stream_values           = rep([mm["source"]           for mm in metric_meta])
    unit_values             = rep([mm["unit"]             for mm in metric_meta])
    qualifier_values        = rep([mm["type"]        for mm in metric_meta])
    feature_order_values    = rep([mm["feature_order"] for mm in metric_meta])
    data_dict = {
        "ts": ts_values,
        "value": value_values,
        "base_name": base_name_values,
        "source": stream_values,
        "unit": unit_values,
        "type": qualifier_values,
        "feature_order": feature_order_values,
    }

    # broadcast meta columns directly from the chunk using stable names
    for mc in meta_cols:
        if mc >= data.shape[1] or mc == ts_col:
            continue
        name = meta_name_map.get(mc)
        if not name:
            continue
        col_np = data.iloc[:, mc].to_numpy()
        data_dict[name] = np.tile(col_np, m)

    tall = pd.DataFrame(data_dict, copy=False)

    forced_meta_names = [
        meta_name_map.get(mc)
        for mc in forced_meta_cols
        if mc in meta_name_map and mc not in ignored_cols
    ]
    forced_meta_payload = sorted({name for name in forced_meta_names if name}) if forced_meta_names else []
    all_feature_defs_payload = []
    if predefined_metric_meta:
        all_feature_defs_payload = [
            {
                "feature_order": int(m["column_index"]),
                "base_name": str(m.get("base_name") or f"col_{int(m['column_index']) + 1}"),
                "source": str(m.get("source")) if m.get("source") is not None else None,
                "unit": str(m.get("unit")) if m.get("unit") is not None else None,
                "type": str(m.get("type")) if m.get("type") is not None else None,
            }
            for m in predefined_metric_meta
            if isinstance(m, dict) and "column_index" in m
        ]

    # drop NaT ts and reindex
    mask_ts = pd.notna(tall["ts"]).to_numpy()
    tall = tall.loc[mask_ts]
    tall.index = np.arange(mask_ts.sum(), dtype=np.int64)

    if getattr(options, "strings_as_category", False):
        for s in ("base_name", "source", "unit", "type"):
            if s in tall.columns:
                tall[s] = tall[s].astype("category")
        for mc in meta_cols:
            name = meta_name_map.get(mc)
            if name in tall.columns and pd.api.types.is_object_dtype(tall[name].dtype):
                tall[name] = tall[name].astype("category")

    if forced_meta_payload:
        tall.attrs["forced_meta_kinds"] = forced_meta_payload
    if all_feature_defs_payload:
        tall.attrs["all_feature_defs"] = all_feature_defs_payload

    return tall


# --- Helper: choose reasonable CSV chunk size based on file size ---
def _choose_csv_chunk_rows(file_path: Path, floor: int = 4096, ceil: int = 500_000, target_mb: int = 32) -> int:
    """
    Aim for ~target_mb per chunk. We don't know bytes/row, so estimate
    with a conservative 200 bytes/row when dtype=object and many columns.
    Clamp between [floor, ceil].
    """
    try:
        fsize = os.path.getsize(file_path)
    except OSError:
        return floor
    if fsize <= 0:
        return floor
    # rough bytes/row guess; tune if you like
    est_bytes_per_row = 1024
    rows = int((target_mb * 1024 * 1024) / est_bytes_per_row)
    return max(floor, min(ceil, rows))


# Common fallback encodings for handling non-UTF-8 CSV files
_CSV_ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1")


def _get_encoding_candidates(user_encoding: Optional[str]) -> List[str]:
    """
    Build list of encodings to try: user-specified first, then common fallbacks.
    This enables automatic encoding detection when a file contains non-UTF-8 characters.
    """
    encodings_to_try = []
    if user_encoding:
        encodings_to_try.append(user_encoding)
    for enc in _CSV_ENCODING_FALLBACKS:
        if enc not in encodings_to_try:
            encodings_to_try.append(enc)
    return encodings_to_try


def parse_file_to_tall(file_path: Path, options: ImportOptions, unit_callback=None) -> List[Tuple[str, pd.DataFrame, int]]:
    """Read a CSV/Excel and return list of (sheet_name, tall_df, header_rows)."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        sheets = read_raw_csv(file_path, options, unit_callback=unit_callback)
        file_type = "csv"
    else:
        sheets = read_raw_excel(file_path, unit_callback=unit_callback, suffix=suffix)
        file_type = "excel"
    out: List[Tuple[str, pd.DataFrame, int]] = []
    for name, raw in sheets.items():
        tall, header_rows = _build_tall_from_raw(raw, options, unit_callback=unit_callback, file_type=file_type)
        out.append((name, tall, header_rows))
    return out


def read_raw_csv(file_path: Path, options: ImportOptions, unit_callback=None) -> Dict[str, pd.DataFrame]:
    chunk_rows = _choose_csv_chunk_rows(file_path)  # stays aligned with estimator
    # pre-plan read units using the SAME chunk size
    planned = 1
    try:
        # fast line count like your DB helper
        size = 0
        with open(file_path, "rb", buffering=1024*1024) as fh:
            for chunk in iter(lambda: fh.read(1024*1024), b""):
                size += chunk.count(b"\n")
        planned = max(1, (int(size) + chunk_rows - 1) // chunk_rows)
    except Exception:
        planned = 1
    if unit_callback:
        try:
            unit_callback(f"Read phase: {planned} chunks planned")
        except Exception:
            logger.warning("Unit callback failed during read phase announcement", exc_info=True)

    # Build list of encodings to try using shared helper
    encodings_to_try = _get_encoding_candidates(options.csv_encoding)

    base_read_kwargs = dict(
        header=None,
        dtype=object,
        chunksize=chunk_rows,
        engine="c",
        memory_map=True,
        low_memory=False,
    )
    if options.csv_delimiter:
        base_read_kwargs["sep"] = options.csv_delimiter
    if options.csv_decimal:
        base_read_kwargs["decimal"] = options.csv_decimal

    frames: List[pd.DataFrame] = []
    last_error: Optional[Exception] = None

    # Try each encoding until one works
    for encoding in encodings_to_try:
        read_kwargs = dict(base_read_kwargs)
        read_kwargs["encoding"] = encoding
        frames = []
        try:
            for idx, chunk in enumerate(pd.read_csv(file_path, **read_kwargs), start=1):
                frames.append(chunk)
                if unit_callback:
                    try:
                        unit_callback(f"Read CSV chunk {idx} of {planned}")
                    except Exception:
                        logger.warning(
                            "Unit callback failed while reading CSV chunk %s of %s",
                            idx,
                            planned,
                            exc_info=True,
                        )
            # Successfully read all chunks with this encoding
            df = pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame()
            return {"-": df}
        except UnicodeDecodeError as e:
            # This encoding failed, try the next one
            last_error = e
            continue
        except TypeError:
            # fallback: full read without chunking
            read_kwargs.pop("chunksize", None)
            try:
                df = pd.read_csv(file_path, **read_kwargs)
                return {"-": df}
            except UnicodeDecodeError as e:
                last_error = e
                continue
        except Exception as e:
            # Other error - don't try other encodings for non-encoding errors
            raise e

    # If we exhausted all encodings, raise the last error
    if last_error:
        raise last_error

    # Fallback - shouldn't normally reach here, but return empty df for safety
    return {"-": pd.DataFrame()}


def _pick_excel_engine(suffix: str) -> str:
    # Prefer explicit engines to skip sniffing overhead and get best perf
    if suffix == ".xlsx":
        return "openpyxl"
    if suffix == ".xlsb":
        return "pyxlsb"
    if suffix == ".xls":
        return "xlrd"
    # Fall back to pandas default if unknown
    return None  # type: ignore[return-value]


def read_raw_excel(file_path: Path, unit_callback=None, suffix: str = ".xlsx") -> Dict[str, pd.DataFrame]:
    """Read all sheets as raw dataframes (no header inference)."""
    engine = _pick_excel_engine(suffix)
    # ExcelFile avoids re-opening the workbook for each sheet
    try:
        with pd.ExcelFile(file_path, engine=engine) as xls:
            out: Dict[str, pd.DataFrame] = {}
            for sheet in xls.sheet_names:
                # dtype=object + header=None preserves current behavior
                # convert_float=False helps avoid silent float casting for some engines
                df = pd.read_excel(
                    xls,
                    sheet_name=sheet,
                    header=None,
                    dtype=object,
                    engine=engine,
                    convert_float=False if engine in ("openpyxl", "xlrd") else None,
                )
                out[sheet] = df
                if unit_callback:
                    try:
                        unit_callback(f"Read sheet {sheet}")
                    except Exception:
                        logger.warning("Unit callback failed while reading sheet %s", sheet, exc_info=True)
            return out
    except Exception:
        # Last-resort fallback if engine not installed; behavior unchanged
        out: Dict[str, pd.DataFrame] = {}
        xls = pd.ExcelFile(file_path)  # default engine sniff
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object, engine=None)
            out[sheet] = df
            if unit_callback:
                try:
                    unit_callback(f"Read sheet {sheet}")
                except Exception:
                    logger.warning("Unit callback failed while reading sheet %s", sheet, exc_info=True)
        return out


def _find_ts_column(raw: pd.DataFrame, options: ImportOptions, file_type: str = "excel") -> Tuple[int, Optional[int]]:
    """Return (header_rows, ts_col_idx). Uses options to determine behavior.
    
    Args:
        raw: DataFrame with no header row (header=None)
        options: ImportOptions with configuration
        file_type: 'csv' or 'excel' - determines which header row count to use
    """
    # Select appropriate header row count based on file type
    default_header_rows = options.csv_header_rows if file_type == "csv" else options.excel_header_rows
    
    if options.date_column:
        header_rows = default_header_rows
        for c in range(raw.shape[1]):
            tup = header_tuple_for_col(raw, c, header_rows)
            stacked = " ".join([x for x in tup if x])  # collapse
            if stacked and stacked.strip().lower() == options.date_column.strip().lower():
                return header_rows, c
        # fall through to detection

    if options.auto_detect_datetime:
        r, c = detect_first_data_row(raw, dayfirst=options.assume_dayfirst, dot_time_as_colon=options.dot_time_as_colon)
        header_rows = max(default_header_rows, r)
        return header_rows, c

    # Fallback: assume first column after header_rows is timestamp
    return default_header_rows, 0


def _coerce_metric_series(
    ser: pd.Series,
    *,
    decimal_hint: Optional[str],
) -> pd.Series:
    """
    Convert metric strings to numeric robustly, honoring decimal hints.
    If decimal is ',', convert '123,45' -> '123.45'.
    """
    s = ser.astype("object")

    if decimal_hint == ",":
        s = s.map(lambda x: (str(x).replace(",", ".")) if isinstance(x, str) else x)

    # Also strip spaces / NBSP
    s = s.map(lambda x: (str(x).replace("\xa0", "").strip()) if isinstance(x, str) else x)

    return pd.to_numeric(s, errors="coerce")


def _build_tall_from_raw(raw: pd.DataFrame, options: ImportOptions, unit_callback=None, file_type: str = "excel") -> Tuple[pd.DataFrame, int]:
    # --- small upfronts ---
    ncols = int(raw.shape[1]) if raw is not None else 0
    if unit_callback and ncols:
        try: unit_callback(f"Parse phase: {ncols}")
        except Exception:
            logger.warning("Unit callback failed during parse phase announcement", exc_info=True)

    header_rows, ts_col = _find_ts_column(raw, options, file_type=file_type)

    # Guard timestamp column
    if ts_col is None or ts_col < 0 or ts_col >= ncols:
        ts_col = 0

    if unit_callback:
        try: unit_callback(f"Parsed column {ts_col+1} (timestamp)")
        except Exception:
            logger.warning("Unit callback failed while reporting timestamp column", exc_info=True)

    # --- Slice data once (no reset_index copy) ---
    # Keep as a view; we’ll reindex cheaply later only when we need to.
    data = raw.iloc[header_rows:]
    # normalize ts once
    ts = _normalize_ts_series(
        data.iloc[:, ts_col],
        dayfirst=options.assume_dayfirst,
        dot_time_as_colon=options.dot_time_as_colon,
        explicit_formats=options.datetime_formats,
    )

    roles: HeaderRoles = options.header_roles

    DEFAULT_BASE_PREFIX = "col"
    DEFAULT_META_PREFIX = "meta"
    CORE_NAMES: Set[str] = {"ts", "value", "base_name", "source", "unit", "type"}

    # -------------------------
    # 1) classify metric vs meta (no header work yet)
    # -------------------------
    metric_cols: List[int] = []
    meta_cols:   List[int] = []
    coerced_numeric: Dict[int, pd.Series] = {}

    # localize option lookups
    decimal_hint = options.csv_decimal
    uc = unit_callback

    for col in range(ncols):
        if col == ts_col:
            continue
        s = data.iloc[:, col]
        s_num = _coerce_metric_series(s, decimal_hint=decimal_hint)
        # “metric” if any non-NaN after coercion
        if s_num.notna().any():
            metric_cols.append(col)
            coerced_numeric[col] = s_num  # cache for reuse
            if uc:
                try: uc(f"Parsed column {col+1}")
                except Exception:
                    logger.warning("Unit callback failed while parsing column %s", col + 1, exc_info=True)
        else:
            meta_cols.append(col)

    # Early out: no metrics -> return empty tall with expected columns
    if not metric_cols:
        tall_empty = pd.DataFrame(columns=["ts","value","base_name","source","unit","type"])
        return tall_empty, header_rows

    # -------------------------
    # 2) build header tuples only for needed cols (metric + meta)
    # -------------------------
    need_cols = set(metric_cols) | set(meta_cols)
    # cache only what we need (saves a lot on wide files)
    header_tuples: Dict[int, Tuple[Any, ...]] = {
        c: header_tuple_for_col(raw, c, header_rows) for c in need_cols
    }

    forced_meta_cols = _resolve_forced_meta_columns(
        header_tuples,
        getattr(options, "force_meta_columns", None) or [],
        ts_col=ts_col,
    )

    ignored_cols = _resolve_ignored_columns(
        header_tuples,
        getattr(options, "ignore_column_prefixes", None) or [],
        ts_col=ts_col,
    )

    if ignored_cols:
        ignored_cols = {c for c in ignored_cols if c < ncols and c != ts_col}
        if ignored_cols:
            metric_cols = [c for c in metric_cols if c not in ignored_cols]
            meta_cols = [c for c in meta_cols if c not in ignored_cols]
            for c in ignored_cols:
                coerced_numeric.pop(c, None)

    if forced_meta_cols:
        forced_meta_cols = {c for c in forced_meta_cols if c < ncols and c != ts_col and c not in ignored_cols}
        if forced_meta_cols:
            metric_cols = [c for c in metric_cols if c not in forced_meta_cols]
            for c in forced_meta_cols:
                coerced_numeric.pop(c, None)
            meta_union = set(meta_cols)
            meta_union.update(forced_meta_cols)
            meta_cols = [c for c in range(ncols) if c != ts_col and c in meta_union and c not in ignored_cols]
            if not metric_cols:
                tall_empty = pd.DataFrame(columns=["ts","value","base_name","source","unit","type"])
                return tall_empty, header_rows

    if not metric_cols:
        tall_empty = pd.DataFrame(columns=["ts","value","base_name","source","unit","type"])
        return tall_empty, header_rows

    # helpers
    def cell_at(tup, idx: Optional[int]):
        return tup[idx] if (idx is not None and idx < len(tup)) else None

    # prepare split config once
    hdelim = getattr(roles, "header_split_delim", None)
    horder = getattr(roles, "header_split_order", None)
    do_split = bool(hdelim and isinstance(horder, list) and len(horder) > 0)

    # -------------------------
    # 3) derive per-metric metadata (base/source/unit/type)
    # -------------------------
    metric_meta: List[Dict[str, Optional[str]]] = []
    for col in metric_cols:
        tup = header_tuples.get(col, ())
        base_name = cell_at(tup, roles.base_name_row)
        source    = cell_at(tup, roles.source_row)
        unit      = cell_at(tup, roles.unit_row)
        type = cell_at(tup, roles.type_row)

        base_name_raw = str(base_name) if base_name is not None else None

        # optional split mapping
        if do_split and (base_name_raw or (len(tup) > 0 and tup[0])):
            src = base_name_raw if base_name_raw else tup[0]
            mapping = _split_header_parts(src, delimiter=hdelim, order=horder)
            if mapping:
                base_name = mapping.get("base", base_name)
                source = mapping.get("source", source)
                unit = mapping.get("unit", unit)
                type = mapping.get("type", type)

        # fallback base_name
        if not base_name and len(tup) > 0 and tup[0] and str(tup[0]).strip():
            base_name = str(tup[0]).strip()
        if not base_name:
            for x in tup:
                if x is not None and str(x).strip():
                    base_name = str(x).strip()
                    break
        if not base_name:
            for cand in (source, type, unit):
                if cand:
                    base_name = str(cand)
                    break
        if not base_name:
            base_name = f"{DEFAULT_BASE_PREFIX}_{col+1}"

        base_name = str(base_name)
        source_s    = str(source) if source is not None else None
        unit_s      = str(unit) if unit is not None else None
        type_s = str(type) if type is not None else None

        metric_meta.append({
            "feature_order": int(col),
            "base_name": base_name,
            "source": source_s,
            "unit": unit_s,
            "type": type_s,
        })

    # -------------------------
    # 4) meta-column naming (no dtype churn / no DataFrame yet)
    # -------------------------
    used_names: Set[str] = set(CORE_NAMES)
    meta_name_map: Dict[int, str] = {}

    def _name_for_meta_col(cidx: int) -> str:
        tup = header_tuples.get(cidx, ())
        first_non_empty = None
        for x in tup:
            if x is not None and str(x).strip():
                first_non_empty = str(x)
                break
        base = clean_cell(first_non_empty) if first_non_empty else ""
        if not base:
            base = f"{DEFAULT_META_PREFIX}_{cidx+1}"
        name0 = base
        k = 1
        while base in used_names:
            base = f"{name0}_{cidx+1 if k == 1 else f'{cidx+1}_{k}'}"
            k += 1
        used_names.add(base)
        return base

    for cidx in meta_cols:
        meta_name_map[cidx] = _name_for_meta_col(cidx)

    # -------------------------
    # 5) Vectorized tall build (no intermediate frames)
    # -------------------------
    n_rows = len(data)
    m = len(metric_cols)

    # Core arrays
    ts_np = ts.to_numpy()                       # datetime64[ns] or NaT
    ts_values = np.tile(ts_np, m)

    value_values = np.concatenate([coerced_numeric[c].to_numpy() for c in metric_cols])

    def rep(values):
        return np.repeat(np.asarray(values, dtype=object), n_rows)

    base_name_values        = rep([mm["base_name"]        for mm in metric_meta])
    stream_values           = rep([mm["source"]           for mm in metric_meta])
    unit_values             = rep([mm["unit"]             for mm in metric_meta])
    qualifier_values        = rep([mm["type"]        for mm in metric_meta])
    feature_order_values    = rep([mm["feature_order"] for mm in metric_meta])
    data_dict = {
        "ts": ts_values,
        "value": value_values,
        "base_name": base_name_values,
        "source": stream_values,
        "unit": unit_values,
        "type": qualifier_values,
        "feature_order": feature_order_values,
    }

    # Broadcast meta columns directly from raw arrays (skip meta_block DataFrame)
    if meta_cols:
        for mc in meta_cols:
            name = meta_name_map[mc]
            # tile the column’s values m times
            col_np = data.iloc[:, mc].to_numpy()
            data_dict[name] = np.tile(col_np, m)

    tall = pd.DataFrame(data_dict, copy=False)

    forced_meta_names = [
        meta_name_map.get(mc)
        for mc in forced_meta_cols
        if mc in meta_name_map and mc not in ignored_cols
    ]
    if forced_meta_names:
        tall.attrs["forced_meta_kinds"] = sorted({name for name in forced_meta_names if name})

    # Drop NaT ts rows without creating a big copy; reindex in place
    mask_ts = pd.notna(tall["ts"]).to_numpy()
    tall = tall.loc[mask_ts]
    tall.index = np.arange(mask_ts.sum(), dtype=np.int64)

    # Optional: compress repeated strings
    if getattr(options, "strings_as_category", False):
        for s in ("base_name", "source", "unit", "type"):
            if s in tall.columns:
                tall[s] = tall[s].astype("category")
        for mc in meta_cols:
            name = meta_name_map[mc]
            if name in tall.columns and pd.api.types.is_object_dtype(tall[name].dtype):
                tall[name] = tall[name].astype("category")

    return tall, header_rows
