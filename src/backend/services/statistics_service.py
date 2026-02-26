
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from backend.data_db import Database
from core.datetime_utils import ensure_series_naive
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatisticsResult:
    """Container for statistics preview and metadata."""

    preview: pd.DataFrame
    statistics: Sequence[str]
    mode: str
    group_column: Optional[str]
    warnings: Sequence[str]
    separate_timeframes_used: bool


def _stat_mean(values: pd.Series, _times: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    return float(s.mean()) if not s.empty else np.nan


def _stat_min(values: pd.Series, _times: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    return float(s.min()) if not s.empty else np.nan


def _stat_max(values: pd.Series, _times: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    return float(s.max()) if not s.empty else np.nan


def _stat_median(values: pd.Series, _times: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    return float(s.median()) if not s.empty else np.nan


def _stat_std(values: pd.Series, _times: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.std())


def _stat_sum(values: pd.Series, _times: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    return float(s.sum()) if not s.empty else np.nan


def _stat_count(values: pd.Series, _times: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="coerce").count())


def _stat_iqr(values: pd.Series, _times: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return np.nan
    q75 = s.quantile(0.75)
    q25 = s.quantile(0.25)
    return float(q75 - q25)


def _outlier_count_and_percentage(values: pd.Series) -> tuple[int, int, float]:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if s.empty:
        return 0, 0, np.nan
    mean = s.mean()
    std = s.std()
    if not np.isfinite(std) or std == 0:
        return 0, int(s.count()), 0.0
    mask = (s > mean + 3 * std) | (s < mean - 3 * std)
    outlier_count = int(mask.sum())
    total_count = int(s.count())
    if total_count <= 0:
        return outlier_count, total_count, np.nan
    percentage = (float(outlier_count) / float(total_count)) * 100.0
    return outlier_count, total_count, percentage


def _stat_outliers(values: pd.Series, _times: pd.Series) -> float:
    _count, _total, pct = _outlier_count_and_percentage(values)
    return pct


def _stat_max_derivative(values: pd.Series, times: pd.Series) -> float:
    df = pd.DataFrame({"t": pd.to_datetime(times, errors="coerce"), "v": pd.to_numeric(values, errors="coerce")})
    df = df.dropna().sort_values("t")
    if len(df) < 2:
        return np.nan
    diffs = df["v"].diff().abs().dropna()
    if diffs.empty:
        return np.nan
    return float(diffs.max())


_STATISTICS: dict[str, Callable[[pd.Series, pd.Series], float]] = {
    "avg": _stat_mean,
    "mean": _stat_mean,
    "min": _stat_min,
    "max": _stat_max,
    "median": _stat_median,
    "std": _stat_std,
    "sum": _stat_sum,
    "count": _stat_count,
    "iqr": _stat_iqr,
    "outliers": _stat_outliers,
    "max_derivative": _stat_max_derivative,
}


def available_statistics() -> list[tuple[str, str]]:
    """Return available statistic identifiers and human readable labels."""

    labels = {
        "avg": "Average",
        "min": "Minimum",
        "max": "Maximum",
        "median": "Median",
        "std": "Std Dev",
        "sum": "Sum",
        "count": "Count",
        "iqr": "Interquartile Range",
        "outliers": "Outliers (3σ %)",
        "max_derivative": "Max derivative",
    }
    items: list[tuple[str, str]] = []
    for key in ["avg", "min", "max", "median", "std", "sum", "count", "iqr", "outliers", "max_derivative"]:
        if key in _STATISTICS:
            items.append((key, labels.get(key, key.title())))
    return items


class StatisticsService:
    """Statistics helpers over caller-provided, already-prepared dataframes."""
    _MAX_SEPARATE_TIMEFRAME_GROUPS = 300

    def __init__(self, database: Optional[Database] = None):
        self._db = database

    # ------------------------------------------------------------------
    def compute(
        self,
        *,
        frame: pd.DataFrame,
        months: Optional[Sequence[int]] = None,
        statistics: Sequence[str] = (),
        mode: str = "time",
        group_column: Optional[str] = None,
        preprocessing: Optional[dict] = None,
    ) -> StatisticsResult:
        stats = [s.lower() for s in statistics if s]
        stats = [s for s in stats if s in _STATISTICS]
        if not stats:
            raise ValueError("Select at least one statistic")
        warnings: list[str] = []
        requested_separate_timeframes = bool((preprocessing or {}).get("separate_timeframes", True))

        if frame is None:
            raise ValueError("Statistics input frame is required")
        df = frame.copy()
        if df is None or df.empty:
            return StatisticsResult(
                preview=pd.DataFrame(columns=self._preview_columns()),
                statistics=stats,
                mode=mode,
                group_column=group_column,
                warnings=warnings,
                separate_timeframes_used=requested_separate_timeframes,
            )
        if "t" not in df.columns:
            raise ValueError("Statistics input frame must contain a 't' timestamp column")
        if "v" not in df.columns:
            raise ValueError("Statistics input frame must contain a 'v' value column")

        df["t"] = ensure_series_naive(pd.to_datetime(df["t"], errors="coerce"))
        df["v"] = pd.to_numeric(df["v"], errors="coerce")
        df = df.dropna(subset=["t"])
        if months:
            months_set = {int(m) for m in months if m is not None}
            if months_set:
                df = df[df["t"].dt.month.isin(months_set)]
        if df.empty:
            return StatisticsResult(
                preview=pd.DataFrame(columns=self._preview_columns()),
                statistics=stats,
                mode=mode,
                group_column=group_column,
                warnings=warnings,
                separate_timeframes_used=requested_separate_timeframes,
            )

        params = preprocessing or {}
        freq = self._timestep_to_freq(params.get("timestep"))
        # Stats period is used for final statistics aggregation (separate from preprocessing timestep)
        stats_period = params.get("stats_period")
        stats_freq = self._timestep_to_freq(stats_period)
        separate_timeframes = bool(params.get("separate_timeframes", True))
        notes_text = self._build_notes_text(mode=mode, group_column=group_column, stats_period=stats_period)

        # Handle group: prefixed columns (database groups)
        _GROUP_PREFIX = "group:"
        actual_group_column = group_column
        effective_separate_timeframes = separate_timeframes
        if mode == "column":
            if not group_column:
                raise ValueError("Select a column to group by")
            if group_column.startswith(_GROUP_PREFIX):
                # It's a database group kind - join by saved group timeframes.
                group_kind = group_column[len(_GROUP_PREFIX):]
                if separate_timeframes:
                    timeframe_groups = self._estimate_timeframe_group_count(
                        group_kind=group_kind,
                        start=df["t"].min() if "t" in df.columns else None,
                        end=df["t"].max() if "t" in df.columns else None,
                    )
                    if timeframe_groups > int(self._MAX_SEPARATE_TIMEFRAME_GROUPS):
                        effective_separate_timeframes = False
                        warnings.append(
                            "Separate timeframes disabled for this run: "
                            f"{int(timeframe_groups)} groups exceeds limit "
                            f"{int(self._MAX_SEPARATE_TIMEFRAME_GROUPS)}."
                        )
                df = self._join_group_points(
                    df,
                    group_kind,
                    match_freq=freq,
                    separate_timeframes=effective_separate_timeframes,
                )
                actual_group_column = f"group_{group_kind}"
                if actual_group_column not in df.columns or df[actual_group_column].dropna().empty:
                    raise ValueError(f"No group data found for kind '{group_kind}'")
            elif group_column not in df.columns:
                raise ValueError(f"Column '{group_column}' is not available for grouping")
        else:
            actual_group_column = None

        group_keys = ["feature_id"]
        if "import_id" in df.columns:
            group_keys.append("import_id")
        if actual_group_column:
            group_keys.append(actual_group_column)

        # Prepare metadata columns to carry over
        metadata_cols = {
            "import_id",
            "feature_label",
            "system",
            "Dataset",
            "base_name",
            "source",
            "unit",
            "type",
        }
        if actual_group_column:
            metadata_cols.add(actual_group_column)

        # Ensure all needed columns exist in metadata
        available_cols = set(df.columns)
        metadata_cols_to_use = list({*group_keys, *metadata_cols} & available_cols)
        metadata = df.loc[:, metadata_cols_to_use].drop_duplicates().reset_index(drop=True)

        results: list[pd.DataFrame] = []
        grouped = df.groupby(group_keys, dropna=False)
        for keys, group in grouped:
            if group.empty:
                continue
            meta = metadata
            if not isinstance(keys, tuple):
                keys = (keys,)
            # Build mask to locate metadata row
            cond = pd.Series([True] * len(meta), index=meta.index)
            for col, value in zip(group_keys, keys):
                if col in meta.columns:
                    cond &= meta[col].fillna("__nan__").astype(str) == ("__nan__" if pd.isna(value) else str(value))
            meta_row = meta.loc[cond].head(1)
            if meta_row.empty:
                continue
            meta_dict = meta_row.iloc[0].to_dict()
            for stat_key in stats:
                func = _STATISTICS[stat_key]
                # When grouping by column (mode == "column"), aggregate ALL values
                # into a single statistic per group - do not produce multiple timestamps
                if mode == "column" and actual_group_column:
                    if stat_key == "outliers":
                        stat_df = self._aggregate_group_single_outliers(group)
                    else:
                        stat_df = self._aggregate_group_single(group, func=func)
                else:
                    # Time-based mode: aggregate by time period
                    if stat_key == "outliers":
                        stat_df = self._aggregate_group_outliers(group, freq=stats_freq)
                    else:
                        stat_df = self._aggregate_group(group, freq=stats_freq, func=func)
                if stat_df.empty:
                    continue
                stat_df = stat_df.rename(columns={"value": "preprocessed_value"})
                stat_df["statistic"] = stat_key
                for col, value in meta_dict.items():
                    stat_df[col] = value
                group_value = meta_dict.get(actual_group_column) if actual_group_column else None
                if actual_group_column:
                    stat_df["group_value"] = group_value
                stat_df["original_qualifier"] = meta_dict.get("type")
                qualifier_group_value = None
                stat_df["new_qualifier"] = stat_df.apply(
                    lambda row, gc=actual_group_column: self._build_qualifier(
                        row.get("original_qualifier"),
                        stat_key,
                        qualifier_group_value,
                    ),
                    axis=1,
                )
                stat_df["label"] = stat_df.apply(
                    lambda row: self._build_label(
                        meta_dict.get("feature_label"),
                        meta_dict.get("base_name"),
                        row.get("new_qualifier"),
                    ),
                    axis=1,
                )
                stat_df["notes"] = notes_text
                stat_df = stat_df.rename(columns={"preprocessed_value": "value"})
                stat_df = stat_df.reindex(columns=self._preview_columns())
                results.append(stat_df)

        if results:
            preview = pd.concat(results, ignore_index=True)
            preview = preview.sort_values(["base_name", "source", "t", "statistic"])
        else:
            preview = pd.DataFrame(columns=self._preview_columns())

        return StatisticsResult(
            preview=preview,
            statistics=stats,
            mode=mode,
            group_column=group_column,
            warnings=warnings,
            separate_timeframes_used=effective_separate_timeframes,
        )

    # ------------------------------------------------------------------
    def save(self, result: StatisticsResult) -> int:
        if self._db is None:
            raise RuntimeError("Database is not initialised")
        df = result.preview
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["t"] = pd.to_datetime(df["t"], errors="coerce")
        df = df.dropna(subset=["t"])
        if df.empty:
            return 0

        features_df = (
            df[["base_name", "source", "unit", "new_qualifier", "label"]]
            .drop_duplicates()
            .rename(columns={"new_qualifier": "type"})
        )

        inserted = 0
        with self._db.write_transaction() as con:
            feature_ids: dict[tuple[str, str, str], int] = {}
            for row in features_df.itertuples(index=False):
                key = (str(row.base_name), (row.source or ""), (row.type or ""))
                fid = self._ensure_feature(
                    con,
                    base_name=str(row.base_name),
                    source=None if row.source in (None, "") else str(row.source),
                    unit=None if pd.isna(row.unit) else row.unit,
                    type=None if row.type in (None, "") else str(row.type),
                    label=None if pd.isna(row.label) else str(row.label),
                )
                feature_ids[key] = fid

            records = []
            for row in df.itertuples(index=False):
                key = (str(row.base_name), (row.source or ""), (row.new_qualifier or ""))
                feature_id = feature_ids.get(key)
                if feature_id is None:
                    continue
                value = row.value
                if pd.isna(value):
                    value = None
                import_id = getattr(row, "import_id", None)
                if pd.isna(import_id):
                    import_id = None
                else:
                    try:
                        import_id = int(import_id)
                    except Exception:
                        import_id = None
                records.append((pd.Timestamp(row.t), int(feature_id), value, import_id))
            if records:
                con.executemany(
                    "INSERT INTO measurements (ts, feature_id, value, import_id) VALUES (?, ?, ?, ?)",
                    records,
                )
                inserted = len(records)
        return inserted

    # ------------------------------------------------------------------
    @staticmethod
    def _preview_columns() -> list[str]:
        return [
            "t",
            "value",
            "import_id",
            "statistic",
            "system",
            "Dataset",
            "base_name",
            "source",
            "unit",
            "original_qualifier",
            "new_qualifier",
            "label",
            "notes",
            "sample_count",
            "outlier_count",
            "group_value",
        ]

    @staticmethod
    def _build_notes_text(
        *,
        mode: str,
        group_column: Optional[str],
        stats_period: Optional[object],
    ) -> str:
        agg = str(mode or "time")
        if agg == "column":
            group = str(group_column or "").strip() or "unknown"
            return f"aggregation={agg}, group_column={group}"
        if stats_period is None or str(stats_period).strip() == "":
            period = "default"
        else:
            period = str(stats_period)
        return f"aggregation={agg}, period={period}"

    @staticmethod
    def _apply_value_filters_long(df: pd.DataFrame, value_filters: Sequence[dict]) -> pd.DataFrame:
        if df is None or df.empty or not value_filters:
            return df
        if "feature_id" not in df.columns or "v" not in df.columns:
            return df
        out = df.copy()
        for flt in value_filters:
            try:
                fid = int(flt.get("feature_id"))
            except Exception:
                continue
            min_value = flt.get("min_value")
            max_value = flt.get("max_value")
            apply_globally = bool(flt.get("apply_globally"))
            mask = out["feature_id"] == fid
            if not mask.any():
                continue
            series = pd.to_numeric(out.loc[mask, "v"], errors="coerce")
            allowed = pd.Series(True, index=series.index)
            if min_value is not None:
                try:
                    allowed &= series >= float(min_value)
                except Exception:
                    logger.warning("Exception in _apply_value_filters_long", exc_info=True)
            if max_value is not None:
                try:
                    allowed &= series <= float(max_value)
                except Exception:
                    logger.warning("Exception in _apply_value_filters_long", exc_info=True)
            if apply_globally:
                bad_index = series.index[~allowed]
                bad_times = out.loc[bad_index, "t"]
                if not bad_times.empty:
                    out = out[~out["t"].isin(bad_times)]
            else:
                out.loc[series.index[~allowed], "v"] = np.nan
        return out

    @staticmethod
    def _timestep_to_freq(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            seconds = int(value)
            if seconds <= 0:
                return None
            return f"{seconds}s"
        text = str(value).strip()
        if not text:
            return None
        try:
            pd.to_timedelta(text)
            return text
        except Exception:
            return None

    @staticmethod
    def _resolve_moving_average_window(
        moving_average: object,
        freq: Optional[str],
    ) -> Optional[str]:
        if moving_average in (None, "", 0):
            return None

        if isinstance(moving_average, str):
            text = moving_average.strip().lower()
            if text in ("", "none"):
                return None
            if text == "auto":
                if not freq:
                    return None
                try:
                    secs = int(pd.to_timedelta(freq).total_seconds())
                except Exception:
                    return None
                if secs <= 0:
                    return None
                return f"{secs * 5}s"
            return moving_average

        if isinstance(moving_average, (int, float)):
            secs = int(moving_average)
            if secs <= 0:
                return None
            return f"{secs}s"

        if isinstance(moving_average, pd.Timedelta):
            return str(moving_average)

        return str(moving_average)

    def _aggregate_group(
        self,
        group: pd.DataFrame,
        *,
        freq: Optional[str],
        func: Callable[[pd.Series, pd.Series], float],
    ) -> pd.DataFrame:
        data = group[["t", "v"]].copy()
        data["t"] = pd.to_datetime(data["t"], errors="coerce")
        data["v"] = pd.to_numeric(data["v"], errors="coerce")
        data = data.dropna(subset=["t"])
        if data.empty:
            return pd.DataFrame(columns=["t", "value", "sample_count"])
        if freq:
            bucket = data["t"].dt.floor(freq)
        else:
            bucket = data["t"]
        data = data.assign(bucket=bucket)
        # Use include_groups=False to avoid FutureWarning in pandas>=2.1
        # but fall back gracefully for older pandas versions
        try:
            agg = data.groupby("bucket", dropna=False).apply(
                lambda frame: func(frame["v"], frame["t"]),
                include_groups=False,
            )
        except TypeError:
            # Older pandas versions don't support include_groups parameter
            agg = data.groupby("bucket", dropna=False).apply(
                lambda frame: func(frame["v"], frame["t"]),
            )
        agg = agg.reset_index().rename(columns={0: "value", "bucket": "t"})
        counts = (
            data.groupby("bucket", dropna=False)["v"]
            .count()
            .reset_index(name="sample_count")
            .rename(columns={"bucket": "t"})
        )
        return agg.merge(counts, on="t", how="left")

    def _aggregate_group_single(
        self,
        group: pd.DataFrame,
        *,
        func: Callable[[pd.Series, pd.Series], float],
    ) -> pd.DataFrame:
        """Aggregate ALL values in the group into a single statistic.
        
        This is used when grouping by column (mode="column") where we want
        one statistic per group rather than multiple timestamps.
        
        Args:
            group: DataFrame with 't' and 'v' columns
            func: Statistic function taking (values, times) -> float
        
        Returns:
            DataFrame with single row containing ['t', 'value', 'sample_count']
            The 't' column contains the earliest timestamp from the group
            as a representative timestamp.
        """
        data = group[["t", "v"]].copy()
        data["t"] = pd.to_datetime(data["t"], errors="coerce")
        data["v"] = pd.to_numeric(data["v"], errors="coerce")
        data = data.dropna(subset=["t"])
        if data.empty:
            return pd.DataFrame(columns=["t", "value", "sample_count"])
        
        # Compute the statistic over ALL values in the group
        stat_value = func(data["v"], data["t"])
        
        # Use the earliest timestamp as the representative time
        representative_time = data["t"].min()
        sample_count = int(data["v"].count())
        
        return pd.DataFrame({
            "t": [representative_time],
            "value": [stat_value],
            "sample_count": [sample_count],
        })

    def _aggregate_group_outliers(
        self,
        group: pd.DataFrame,
        *,
        freq: Optional[str],
    ) -> pd.DataFrame:
        data = group[["t", "v"]].copy()
        data["t"] = pd.to_datetime(data["t"], errors="coerce")
        data["v"] = pd.to_numeric(data["v"], errors="coerce")
        data = data.dropna(subset=["t"])
        if data.empty:
            return pd.DataFrame(columns=["t", "value", "sample_count", "outlier_count"])

        if freq:
            data = data.assign(bucket=data["t"].dt.floor(freq))
        else:
            data = data.assign(bucket=data["t"])

        rows: list[dict[str, object]] = []
        for bucket, frame in data.groupby("bucket", dropna=False):
            outlier_count, sample_count, percentage = _outlier_count_and_percentage(frame["v"])
            rows.append(
                {
                    "t": bucket,
                    "value": percentage,
                    "sample_count": sample_count,
                    "outlier_count": outlier_count,
                }
            )
        return pd.DataFrame(rows, columns=["t", "value", "sample_count", "outlier_count"])

    def _aggregate_group_single_outliers(self, group: pd.DataFrame) -> pd.DataFrame:
        data = group[["t", "v"]].copy()
        data["t"] = pd.to_datetime(data["t"], errors="coerce")
        data["v"] = pd.to_numeric(data["v"], errors="coerce")
        data = data.dropna(subset=["t"])
        if data.empty:
            return pd.DataFrame(columns=["t", "value", "sample_count", "outlier_count"])

        outlier_count, sample_count, percentage = _outlier_count_and_percentage(data["v"])
        representative_time = data["t"].min()
        return pd.DataFrame(
            {
                "t": [representative_time],
                "value": [percentage],
                "sample_count": [sample_count],
                "outlier_count": [outlier_count],
            }
        )

    def _apply_postprocess(
        self,
        df: pd.DataFrame,
        *,
        freq: Optional[str],
        fill: str,
        moving_average,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        d = df.copy()
        d["t"] = ensure_series_naive(pd.to_datetime(d["t"], errors="coerce"))
        d = d.dropna(subset=["t"]).sort_values("t")
        if d.empty:
            return d
        idx = d["t"]
        series = d.set_index("t")["value"].astype(float)
        if freq:
            full_idx = pd.date_range(idx.min().floor(freq), idx.max().ceil(freq), freq=freq)
            series = series.reindex(full_idx)
        if fill == "prev":
            series = series.ffill().bfill()
        elif fill == "next":
            series = series.bfill().ffill()
        elif fill == "zero":
            series = series.fillna(0.0)
        # moving average
        window = self._resolve_moving_average_window(moving_average, freq)
        if window is not None:
            try:
                series = series.rolling(window, min_periods=1).mean()
            except Exception:
                logger.warning("Exception in _apply_postprocess", exc_info=True)
        out = series.reset_index().rename(columns={"index": "t", 0: "value"})
        out["t"] = ensure_series_naive(out["t"])
        return out

    def _preprocess_group(
        self,
        group: pd.DataFrame,
        *,
        freq: Optional[str],
        fill: str,
        moving_average,
    ) -> pd.DataFrame:
        """Apply preprocessing (resample, fill, moving average) to a group before statistics calculation."""
        if group is None or group.empty:
            return pd.DataFrame(columns=["t", "v"])
        
        data = group[["t", "v"]].copy()
        data["t"] = ensure_series_naive(pd.to_datetime(data["t"], errors="coerce"))
        data = data.dropna(subset=["t"]).sort_values("t")
        if data.empty:
            return pd.DataFrame(columns=["t", "v"])

        ma_window = self._resolve_moving_average_window(moving_average, freq)

        # If no preprocessing is needed, return as-is
        if freq is None and fill == "none" and ma_window is None:
            return data
        
        idx = data["t"]
        series = data.set_index("t")["v"].astype(float)
        
        # Resample to target frequency
        if freq:
            # Aggregate values within each time bucket using mean
            series = series.resample(freq).mean()
        
        # Fill empty values
        if fill == "prev":
            series = series.ffill().bfill()
        elif fill == "next":
            series = series.bfill().ffill()
        elif fill == "zero":
            series = series.fillna(0.0)

        # Apply moving average
        if ma_window is not None:
            try:
                series = series.rolling(ma_window, min_periods=1).mean()
            except Exception:
                logger.warning("Exception in _preprocess_group", exc_info=True)
        
        out = series.reset_index().rename(columns={"index": "t", series.name: "v"})
        out["t"] = ensure_series_naive(out["t"])
        return out

    def _join_group_points(
        self,
        df: pd.DataFrame,
        group_kind: str,
        match_freq: Optional[str] = None,
        separate_timeframes: bool = True,
    ) -> pd.DataFrame:
        """Join measurement data with saved group timeframes for a specific group kind."""
        if df is None or df.empty:
            return df
        if self._db is None:
            return df
        
        # Get group labels for the specified kind
        group_labels = self._db.list_group_labels(kind=group_kind)
        if group_labels is None or group_labels.empty:
            # Fallback: tolerate whitespace/case mismatches in kind values.
            all_labels = self._db.list_group_labels(kind=None)
            if all_labels is not None and not all_labels.empty and "kind" in all_labels.columns:
                wanted = str(group_kind or "").strip().casefold()
                kinds = all_labels["kind"].astype(str).str.strip().str.casefold()
                group_labels = all_labels.loc[kinds == wanted].copy()
        if group_labels.empty:
            return df
        
        group_ids = group_labels["group_id"].tolist()
        
        # Get group timeframes
        start_ts = df["t"].min() if "t" in df.columns else None
        end_ts = df["t"].max() if "t" in df.columns else None
        group_points = self._db.group_points(
            group_ids,
            start=start_ts,
            end=end_ts,
        )

        if group_points.empty:
            # Retry without time bounds as a defensive fallback for
            # boundary-matching or mixed-granularity timestamp cases.
            group_points = self._db.group_points(group_ids)
            if group_points.empty:
                return df

        # Create label mapping
        label_map = dict(zip(group_labels["group_id"], group_labels["label"]))
        group_points = group_points.copy()
        group_points["group_label"] = group_points["group_id"].map(label_map)
        group_points["start_ts"] = ensure_series_naive(pd.to_datetime(group_points["start_ts"], errors="coerce"))
        group_points["end_ts"] = ensure_series_naive(pd.to_datetime(group_points["end_ts"], errors="coerce"))
        group_points = group_points.dropna(subset=["start_ts", "end_ts", "group_label"])
        group_points = group_points[group_points["end_ts"] >= group_points["start_ts"]]
        if group_points.empty:
            return df

        df = df.copy()
        df["t"] = ensure_series_naive(pd.to_datetime(df["t"], errors="coerce"))
        df = df.dropna(subset=["t"])
        if df.empty:
            return df

        group_col_name = f"group_{group_kind}"
        display_values = self._assign_group_timeframe_values(
            ts=df["t"],
            ranges=group_points,
            match_freq=match_freq,
            separate_timeframes=separate_timeframes,
        )
        out = df.copy()
        out[group_col_name] = display_values
        return out

    def _estimate_timeframe_group_count(
        self,
        *,
        group_kind: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> int:
        if self._db is None:
            return 0
        kind = str(group_kind or "").strip()
        if not kind:
            return 0
        try:
            group_labels = self._db.list_group_labels(kind=kind)
            if group_labels is None or group_labels.empty:
                return 0
            ids = [int(v) for v in group_labels.get("group_id", pd.Series(dtype=int)).tolist() if v is not None]
            if not ids:
                return 0
            points = self._db.group_points(ids, start=start, end=end)
            if points is None or points.empty:
                return 0
            work = points.copy()
            work["start_ts"] = ensure_series_naive(pd.to_datetime(work["start_ts"], errors="coerce"))
            work["end_ts"] = ensure_series_naive(pd.to_datetime(work["end_ts"], errors="coerce"))
            work = work.dropna(subset=["group_id", "start_ts", "end_ts"])
            work = work[work["end_ts"] >= work["start_ts"]]
            if work.empty:
                return 0
            return int(work[["group_id", "start_ts", "end_ts"]].drop_duplicates().shape[0])
        except Exception:
            logger.warning("Exception in _estimate_timeframe_group_count", exc_info=True)
            return 0

    def _apply_group_ids_filter_long(
        self,
        df: pd.DataFrame,
        *,
        group_ids: Sequence[int],
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        match_freq: Optional[str] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty or not group_ids:
            return df
        if self._db is None:
            return df
        try:
            ids = [int(g) for g in group_ids if g is not None]
        except Exception:
            return df
        if not ids:
            return df
        group_points = self._db.group_points(ids, start=start, end=end)
        if group_points is None or group_points.empty:
            return df.iloc[0:0]
        ranges = group_points.copy()
        ranges["start_ts"] = ensure_series_naive(pd.to_datetime(ranges["start_ts"], errors="coerce"))
        ranges["end_ts"] = ensure_series_naive(pd.to_datetime(ranges["end_ts"], errors="coerce"))
        ranges = ranges.dropna(subset=["start_ts", "end_ts"])
        ranges = ranges[ranges["end_ts"] >= ranges["start_ts"]]
        if ranges.empty:
            return df.iloc[0:0]

        out = df.copy()
        out["t"] = ensure_series_naive(pd.to_datetime(out["t"], errors="coerce"))
        out = out.dropna(subset=["t"])
        if out.empty:
            return out
        assigned = self._assign_group_timeframe_values(
            ts=out["t"],
            ranges=ranges,
            match_freq=match_freq,
        )
        return out[assigned.notna()]

    @staticmethod
    def _assign_group_timeframe_values(
        *,
        ts: pd.Series,
        ranges: pd.DataFrame,
        match_freq: Optional[str] = None,
        separate_timeframes: bool = True,
    ) -> pd.Series:
        if ts is None or ts.empty or ranges is None or ranges.empty:
            return pd.Series([pd.NA] * len(ts), index=ts.index if ts is not None else None, dtype="object")

        range_df = ranges.copy().sort_values(["start_ts", "end_ts"]).reset_index(drop=True)
        if match_freq:
            left_ts = pd.to_datetime(ts, errors="coerce").dt.floor(match_freq)
            range_df["start_norm"] = pd.to_datetime(range_df["start_ts"], errors="coerce").dt.floor(match_freq)
            range_df["end_norm"] = pd.to_datetime(range_df["end_ts"], errors="coerce").dt.floor(match_freq)
        else:
            left_ts = pd.to_datetime(ts, errors="coerce")
            range_df["start_norm"] = pd.to_datetime(range_df["start_ts"], errors="coerce")
            range_df["end_norm"] = pd.to_datetime(range_df["end_ts"], errors="coerce")

        labels = range_df.get("group_label", pd.Series(index=range_df.index, dtype=object)).astype(str)
        starts = range_df["start_norm"].to_numpy(dtype="datetime64[ns]")
        ends = range_df["end_norm"].to_numpy(dtype="datetime64[ns]")
        values = left_ts.to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(starts, values, side="right") - 1

        out = pd.Series([pd.NA] * len(left_ts), index=left_ts.index, dtype="object")
        for i, range_idx in enumerate(idx):
            if range_idx < 0:
                continue
            value_ts = values[i]
            if np.isnat(value_ts):
                continue
            match_idx = -1
            j = int(range_idx)
            # Walk backwards to handle overlapping/adjacent ranges from different labels.
            # We pick the most recent range start that still contains the timestamp.
            while j >= 0:
                end_ts = ends[j]
                if np.isnat(end_ts):
                    j -= 1
                    continue
                if value_ts <= end_ts:
                    match_idx = j
                    break
                j -= 1
            if match_idx < 0:
                continue
            label = str(labels.iloc[match_idx] or "").strip()
            if not label:
                continue
            raw_start = pd.Timestamp(range_df.iloc[match_idx]["start_ts"])
            raw_end = pd.Timestamp(range_df.iloc[match_idx]["end_ts"])
            if separate_timeframes:
                timeframe = StatisticsService._format_timeframe_display(raw_start, raw_end)
                out.iat[i] = f"{label} ({timeframe})"
            else:
                out.iat[i] = label
        return out

    @staticmethod
    def _format_timeframe_display(start: pd.Timestamp, end: pd.Timestamp) -> str:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        is_midnight_span = (
            start_ts.hour == 0
            and start_ts.minute == 0
            and start_ts.second == 0
            and start_ts.microsecond == 0
            and end_ts.hour == 0
            and end_ts.minute == 0
            and end_ts.second == 0
            and end_ts.microsecond == 0
        )
        if is_midnight_span:
            return f"{start_ts:%Y-%m-%d} - {end_ts:%Y-%m-%d}"
        return f"{start_ts:%Y-%m-%d %H:%M} - {end_ts:%Y-%m-%d %H:%M}"

    @staticmethod
    def _slug(value: Optional[object]) -> str:
        import re

        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "NA"
        text = str(value).strip()
        if not text:
            return "NA"
        slug = re.sub(r"[^0-9A-Za-z]+", "_", text)
        slug = slug.strip("_")
        return slug.upper() or "NA"

    def _build_qualifier(self, original: Optional[str], stat: str, group_value: Optional[object]) -> str:
        parts: list[str] = []
        if original:
            parts.append(str(original))
        if group_value is not None and not (isinstance(group_value, float) and np.isnan(group_value)):
            slug = self._slug(group_value)
            if slug:
                parts.append(slug)
        parts.append(stat.upper())
        return "_".join(parts)

    @staticmethod
    def _build_label(original_label: Optional[str], base_name: Optional[str], type: Optional[str]) -> str:
        label_base = original_label or base_name or "Feature"
        qual = type or ""
        return f"{label_base} {qual}".strip()

    def _ensure_feature(
        self,
        con,
        *,
        base_name: str,
        source: Optional[str],
        unit: Optional[str],
        type: Optional[str],
        label: Optional[str],
    ) -> int:
        stream_key = source or ""
        qual_key = type or ""
        cols_df = con.execute("PRAGMA table_info('features')").df()
        columns = {str(c).lower() for c in cols_df.get("name", pd.Series(dtype=str)).tolist()}
        name_col = "name" if "name" in columns else "base_name"
        notes_col = "notes" if "notes" in columns else ("label" if "label" in columns else None)
        has_unit_col = "unit" in columns
        has_type_col = "type" in columns

        where = [f"{name_col} = ?", "COALESCE(source,'') = ?"]
        params: list[object] = [base_name, stream_key]
        if has_type_col:
            where.append("COALESCE(type,'') = ?")
            params.append(qual_key)

        select_cols = ["id"]
        select_cols.append("unit" if has_unit_col else "NULL AS unit")
        select_cols.append(f"{notes_col} AS label" if notes_col else "NULL AS label")
        row = con.execute(
            f"SELECT {', '.join(select_cols)} FROM features WHERE {' AND '.join(where)}",
            params,
        ).fetchone()
        if row:
            fid, existing_unit, existing_label = row
            update_parts: list[str] = []
            update_params: list[object] = []
            if has_unit_col and existing_unit is None and unit is not None:
                update_parts.append("unit = COALESCE(unit, ?)")
                update_params.append(unit)
            if notes_col and existing_label is None and label is not None:
                update_parts.append(f"{notes_col} = COALESCE({notes_col}, ?)")
                update_params.append(label)
            if update_parts:
                update_params.append(fid)
                con.execute(
                    f"UPDATE features SET {', '.join(update_parts)} WHERE id = ?",
                    update_params,
                )
            return int(fid)

        insert_cols = ["id", name_col, "source"]
        insert_vals = ["nextval('features_id_seq')", "?", "NULLIF(?, '')"]
        insert_params: list[object] = [base_name, stream_key]
        if has_unit_col:
            insert_cols.append("unit")
            insert_vals.append("?")
            insert_params.append(unit)
        if has_type_col:
            insert_cols.append("type")
            insert_vals.append("NULLIF(?, '')")
            insert_params.append(qual_key)
        if notes_col:
            insert_cols.append(notes_col)
            insert_vals.append("?")
            insert_params.append(label)

        inserted = con.execute(
            f"INSERT INTO features ({', '.join(insert_cols)}) VALUES ({', '.join(insert_vals)}) RETURNING id",
            insert_params,
        ).fetchone()
        return int(inserted[0])


__all__ = [
    "StatisticsService",
    "StatisticsResult",
    "available_statistics",
]

