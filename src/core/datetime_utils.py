
from __future__ import annotations
from datetime import datetime

import pandas as pd


def drop_timezone_preserving_wall(value):
    """Return ``value`` without any timezone information, preserving wall time."""
    if value is None or value is pd.NaT:
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return pd.Timestamp(value.to_pydatetime().replace(tzinfo=None))
        return value
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value
    return value


def ensure_series_naive(series: pd.Series) -> pd.Series:
    """Ensure a Series of datetimes has no timezone information."""
    values = [drop_timezone_preserving_wall(v) for v in series]
    converted = pd.to_datetime(pd.Series(values, index=series.index), errors="coerce")
    return converted


__all__ = ["drop_timezone_preserving_wall", "ensure_series_naive"]
