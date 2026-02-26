
from __future__ import annotations
from typing import Any, Iterable

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_for_json(v) for v in value]
    return value


def frame_to_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    cleaned = frame.copy()
    cleaned = cleaned.replace({np.nan: None})
    for col in cleaned.columns:
        series = cleaned[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            cleaned[col] = series.dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            cleaned[col] = series.apply(normalize_for_json)
    return cleaned.to_dict(orient="records")


def records_to_frame(records: Iterable[dict[str, Any]] | None) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(list(records))
    if "t" in frame.columns:
        try:
            frame["t"] = pd.to_datetime(frame["t"], errors="coerce")
        except Exception:
            logger.warning("Exception in records_to_frame", exc_info=True)
    return frame
