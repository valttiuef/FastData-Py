from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
"""Shared preprocessing and filtering utilities.

This module provides common functions for data preprocessing and filtering
that can be reused across the preprocessing service, hybrid pandas model,
and other components that need consistent data transformation.
"""


from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from core.datetime_utils import ensure_series_naive


@dataclass
class FilterConfig:
    """Configuration for filtering raw data."""

    systems: Optional[Sequence[str]] = None
    Datasets: Optional[Sequence[str]] = None
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    months: Optional[Sequence[int]] = None
    feature_ids: Optional[Sequence[int]] = None


@dataclass
class StatisticsConfig:
    """Configuration for data preparation before statistics."""

    timestep: Optional[str] = None  # e.g., "60s", "1h", "1d"
    fill: str = "none"  # "none", "prev", "next", "zero"
    moving_average: Optional[int | str] = None  # seconds or offset string
    agg: str = "avg"  # aggregation function for resampling


# ---------------------------------------------------------------------------
# Timestamp / Frequency Utilities
# ---------------------------------------------------------------------------

def timestep_to_freq(value) -> Optional[str]:
    """Convert various timestep representations to pandas frequency string.
    
    Args:
        value: int (seconds), float (seconds), or string (e.g., "60s", "1h")
    
    Returns:
        Pandas-compatible frequency string or None
    """
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


def freq_to_seconds(freq: Optional[str]) -> Optional[int]:
    """Convert a pandas frequency string to seconds.
    
    Args:
        freq: Pandas-compatible frequency string (e.g., "60s", "1h", "1d")
    
    Returns:
        Number of seconds or None
    """
    if freq is None:
        return None
    try:
        td = pd.to_timedelta(freq)
        return int(td.total_seconds())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Filtering Functions
# ---------------------------------------------------------------------------

def filter_dataframe(
    df: pd.DataFrame,
    config: FilterConfig,
) -> pd.DataFrame:
    """Apply filter configuration to a dataframe.
    
    Args:
        df: Input dataframe with columns like 't', 'system', 'Dataset', 'feature_id'
        config: FilterConfig with filter criteria
    
    Returns:
        Filtered dataframe
    """
    if df is None or df.empty:
        return df
    
    result = df.copy()
    
    # Ensure 't' column is datetime
    if "t" in result.columns:
        result["t"] = pd.to_datetime(result["t"], errors="coerce")
        result = result.dropna(subset=["t"])
    
    # Apply time range filters
    if config.start is not None and "t" in result.columns:
        result = result[result["t"] >= config.start]
    if config.end is not None and "t" in result.columns:
        result = result[result["t"] <= config.end]
    
    # Apply month filter
    if config.months and "t" in result.columns:
        months_set = {int(m) for m in config.months if m is not None}
        if months_set:
            result = result[result["t"].dt.month.isin(months_set)]
    
    # Apply system filter
    if config.systems and "system" in result.columns:
        result = result[result["system"].isin(config.systems)]
    
    # Apply Dataset filter
    if config.Datasets and "Dataset" in result.columns:
        result = result[result["Dataset"].isin(config.Datasets)]
    
    # Apply feature_id filter
    if config.feature_ids and "feature_id" in result.columns:
        result = result[result["feature_id"].isin(config.feature_ids)]
    
    return result


# ---------------------------------------------------------------------------
# Preprocessing Functions
# ---------------------------------------------------------------------------

def prepare_series_for_statistics(
    df: pd.DataFrame,
    config: StatisticsConfig,
    time_col: str = "t",
    value_col: str = "v",
) -> pd.DataFrame:
    """Apply preparation to a time series dataframe.
    
    Args:
        df: Input dataframe with time and value columns
        config: StatisticsConfig with preparation parameters
        time_col: Name of the timestamp column
        value_col: Name of the value column
    
    Returns:
        Preprocessed dataframe with [time_col, value_col] columns
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[time_col, value_col])
    
    data = df[[time_col, value_col]].copy()
    data[time_col] = ensure_series_naive(pd.to_datetime(data[time_col], errors="coerce"))
    data = data.dropna(subset=[time_col]).sort_values(time_col)
    
    if data.empty:
        return pd.DataFrame(columns=[time_col, value_col])
    
    # Convert value to numeric
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    
    freq = timestep_to_freq(config.timestep)
    
    # If no preprocessing needed, return as-is
    if freq is None and config.fill == "none" and config.moving_average in (None, "", 0):
        return data
    
    series = data.set_index(time_col)[value_col].astype(float)
    
    # Resample to target frequency
    if freq:
        agg_map = {
            "avg": "mean", "mean": "mean",
            "min": "min", "max": "max",
            "sum": "sum", "count": "count",
            "first": "first", "last": "last",
            "median": "median",
        }
        agg_fn = agg_map.get((config.agg or "avg").lower(), "mean")
        series = series.resample(freq).agg(agg_fn)
    
    # Fill empty values
    if config.fill == "prev":
        series = series.ffill().bfill()
    elif config.fill == "next":
        series = series.bfill().ffill()
    elif config.fill == "zero":
        series = series.fillna(0.0)
    
    # Apply moving average
    if config.moving_average not in (None, "", 0):
        try:
            if isinstance(config.moving_average, (int, float)):
                window = f"{int(config.moving_average)}s"
            else:
                window = str(config.moving_average)
            series = series.rolling(window, min_periods=1).mean()
        except Exception:
            logger.warning("Exception in prepare_series_for_statistics", exc_info=True)
    
    out = series.reset_index().rename(columns={"index": time_col, series.name: value_col})
    out[time_col] = ensure_series_naive(out[time_col])
    return out


# ---------------------------------------------------------------------------
# Aggregation Functions for Statistics
# ---------------------------------------------------------------------------

def aggregate_by_time(
    df: pd.DataFrame,
    freq: Optional[str],
    func: Callable[[pd.Series, pd.Series], float],
    time_col: str = "t",
    value_col: str = "v",
) -> pd.DataFrame:
    """Aggregate values by time buckets.
    
    Args:
        df: Input dataframe with time and value columns
        freq: Pandas frequency string for bucketing (None = no aggregation)
        func: Function that takes (values, times) and returns a single value
        time_col: Name of the timestamp column
        value_col: Name of the value column
    
    Returns:
        Dataframe with ['t', 'value'] columns
    """
    data = df[[time_col, value_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col])
    
    if data.empty:
        return pd.DataFrame(columns=["t", "value"])
    
    if freq:
        bucket = data[time_col].dt.floor(freq)
    else:
        bucket = data[time_col]
    
    data = data.assign(bucket=bucket)
    # Use include_groups=False to avoid FutureWarning in pandas>=2.1
    # but fall back gracefully for older pandas versions
    try:
        agg = data.groupby("bucket", dropna=False).apply(
            lambda frame: func(frame[value_col], frame[time_col]),
            include_groups=False,
        )
    except TypeError:
        # Older pandas versions don't support include_groups parameter
        agg = data.groupby("bucket", dropna=False).apply(
            lambda frame: func(frame[value_col], frame[time_col]),
        )
    agg = agg.reset_index().rename(columns={0: "value", "bucket": "t"})
    return agg


def aggregate_all(
    df: pd.DataFrame,
    func: Callable[[pd.Series, pd.Series], float],
    time_col: str = "t",
    value_col: str = "v",
) -> tuple[float, Optional[pd.Timestamp]]:
    """Aggregate all values into a single statistic.
    
    Args:
        df: Input dataframe with time and value columns
        func: Function that takes (values, times) and returns a single value
        time_col: Name of the timestamp column
        value_col: Name of the value column
    
    Returns:
        Tuple of (aggregated_value, representative_timestamp)
        The representative timestamp is the min timestamp in the data
    """
    if df is None or df.empty:
        return np.nan, None
    
    data = df[[time_col, value_col]].copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col])
    
    if data.empty:
        return np.nan, None
    
    values = data[value_col]
    times = data[time_col]
    
    stat_value = func(values, times)
    # Use the minimum timestamp as representative
    representative_time = times.min()
    
    return stat_value, representative_time


__all__ = [
    "FilterConfig",
    "StatisticsConfig",
    "timestep_to_freq",
    "freq_to_seconds",
    "filter_dataframe",
    "prepare_series_for_statistics",
    "aggregate_by_time",
    "aggregate_all",
]

