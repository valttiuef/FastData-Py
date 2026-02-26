
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from PySide6.QtCore import QObject, Signal

from ...models.hybrid_pandas_model import (
    DataFilters,
    FeatureSelection,
    HybridPandasModel,
)
from ...threading.runner import run_in_thread
from ...threading.utils import run_in_main_thread


@dataclass
class ChartQuery:
    """Simple container describing the data request for a chart."""

    features: List[FeatureSelection]
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    systems: Optional[List[str]]
    datasets: Optional[List[str]]
    group_ids: Optional[List[int]]


@dataclass
class CorrelationEntry:
    feature: FeatureSelection
    correlation: float


@dataclass
class CorrelationSearchResult:
    target_feature: FeatureSelection
    top10: list[CorrelationEntry]


class ChartsViewModel(QObject):
    """Lightweight coordinator that reuses :class:`HybridPandasModel` for chart data."""

    add_chart_requested = Signal()
    remove_chart_requested = Signal()
    chart_configuration_changed = Signal(object)
    correlation_search_finished = Signal(object)
    correlation_search_failed = Signal(str)

    def __init__(
        self,
        model: HybridPandasModel,
        parent: Optional[QObject] = None,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Shared HybridPandasModel instance that owns DB access and
            selection state.
        """
        super().__init__(parent)
        self._model: HybridPandasModel = model

        self._correlation_running = False

    # ------------------------------------------------------------------
    @property
    def hybrid_model(self) -> HybridPandasModel:
        """Expose the underlying HybridPandasModel (read-only usage in views)."""
        return self._model

    # ------------------------------------------------------------------
    def close(self) -> None:
        """
        Called when the tab/view is closing.

        We deliberately do NOT close the underlying HybridPandasModel here,
        because it is typically shared between multiple view models/tabs and
        managed at a higher level.
        """
        # If you ever want to reset local state, do it here.
        pass

    # ------------------------------------------------------------------
    # UI coordination helpers -------------------------------------------------

    def request_add_chart(self) -> None:
        self.add_chart_requested.emit()

    def request_remove_chart(self) -> None:
        self.remove_chart_requested.emit()

    def notify_chart_configuration_changed(self, card) -> None:
        self.chart_configuration_changed.emit(card)

    def build_feature_items(
        self,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, FeatureSelection]]:
        """Return [(display_name, FeatureSelection)] from the data model."""
        try:
            df = self._model.features_for_systems_datasets(
                systems=systems or None,
                datasets=datasets or None,
                tags=tags,
            )
        except Exception:
            df = pd.DataFrame(
                columns=["feature_id", "name", "source", "unit", "type", "notes"]
            )

        def _display_name_without_label(selection: FeatureSelection) -> str:
            parts: list[str] = []
            for piece in (
                selection.base_name,
                selection.source,
                selection.unit,
                selection.type,
            ):
                if piece and piece not in parts:
                    parts.append(str(piece))
            if not parts:
                if selection.feature_id is not None:
                    return f"Feature {selection.feature_id}"
                return "Feature"
            return " / ".join(parts)

        items: List[Tuple[str, FeatureSelection]] = []
        for row in df.itertuples():
            feature_selection = FeatureSelection(
                feature_id=getattr(row, "feature_id", None),
                label=getattr(row, "notes", None),
                base_name=getattr(row, "name", None),
                source=getattr(row, "source", None),
                unit=getattr(row, "unit", None),
                type=getattr(row, "type", None),
                lag_seconds=getattr(row, "lag_seconds", None),
            )
            items.append((_display_name_without_label(feature_selection), feature_selection))
        return items

    # ------------------------------------------------------------------
    def build_filters(self, query: ChartQuery) -> Optional[DataFilters]:
        """Convert ChartQuery into DataFilters for the HybridPandasModel."""
        features = query.features or []
        if not features:
            return None
        return DataFilters(
            features=list(features),
            start=query.start,
            end=query.end,
            systems=list(query.systems) if query.systems else None,
            datasets=list(query.datasets) if query.datasets else None,
            group_ids=list(query.group_ids) if query.group_ids else None,
        )

    # ------------------------------------------------------------------
    def base_dataframe(
        self,
        filters: DataFilters,
        params: dict,
        *,
        data_frame: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return postprocessed BASE cache as a DataFrame."""
        if isinstance(data_frame, pd.DataFrame):
            return data_frame.copy()
        raise RuntimeError("base_dataframe requires a pre-fetched DataFrame from DataSelectorWidget.")

    def time_series_dataframe(
        self,
        filters: DataFilters,
        params: dict,
        *,
        data_frame: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return postprocessed time-series data for direct plotting."""
        if isinstance(data_frame, pd.DataFrame):
            return data_frame.copy()
        raise RuntimeError("time_series_dataframe requires a pre-fetched DataFrame from DataSelectorWidget.")

    # ------------------------------------------------------------------
    def start_correlation_search(
        self,
        *,
        target_feature: FeatureSelection,
        available_features: Sequence[FeatureSelection],
        data_frame: pd.DataFrame,
    ) -> bool:
        if self._correlation_running:
            return False
        self._correlation_running = True

        def _compute() -> CorrelationSearchResult:
            if data_frame is None or data_frame.empty:
                raise RuntimeError("No data for the selected filters.")
            features = self._ordered_unique_features(target_feature, available_features)
            if len(features) < 2:
                raise RuntimeError("At least 2 features are required for correlation analysis.")
            frame = data_frame.copy()

            data_cols = [col for col in frame.columns if col != "t"]
            if len(data_cols) < 2:
                raise RuntimeError("No comparable feature columns were found.")

            feature_by_col = self._feature_map_for_columns(data_cols, features)
            target_col = self._target_column(data_cols, feature_by_col, target_feature)
            if not target_col:
                raise RuntimeError("Target feature data is unavailable for current filters.")

            numeric = frame.loc[:, data_cols].apply(pd.to_numeric, errors="coerce")
            target_series = numeric[target_col]
            valid_target = target_series.notna()
            if int(valid_target.sum()) < 3:
                raise RuntimeError("Target feature has insufficient data for correlation analysis.")

            target_aligned = target_series.loc[valid_target]
            matrix = numeric.loc[valid_target]
            if target_aligned.nunique(dropna=True) < 2:
                raise RuntimeError("Target feature has no variation for correlation analysis.")

            # Guard against zero-variance / undersampled columns to avoid
            # NumPy divide-by-zero warnings during Pearson correlation.
            pair_counts = matrix.notna().sum(axis=0)
            variable_mask = matrix.nunique(dropna=True) >= 2
            matrix = matrix.loc[:, (pair_counts >= 3) & variable_mask]
            if matrix.empty:
                raise RuntimeError("No comparable varying features were found.")

            corr_series = matrix.corrwith(target_aligned, axis=0, drop=True)
            pair_counts = matrix.notna().sum(axis=0)

            corr_series = corr_series.drop(labels=[target_col], errors="ignore")
            corr_series = corr_series[pair_counts.reindex(corr_series.index).fillna(0) >= 3]
            corr_series = corr_series[pd.notna(corr_series)]
            corr_series = corr_series[~corr_series.isin([float("inf"), float("-inf")])]

            entries: list[CorrelationEntry] = [
                CorrelationEntry(
                    feature=feature_by_col.get(str(col), FeatureSelection(base_name=str(col))),
                    correlation=float(corr),
                )
                for col, corr in corr_series.items()
            ]

            if not entries:
                raise RuntimeError("Unable to calculate correlations for this feature.")

            entries.sort(key=lambda entry: abs(entry.correlation), reverse=True)
            top10 = entries[:10]
            return CorrelationSearchResult(
                target_feature=target_feature,
                top10=top10,
            )

        run_in_thread(
            _compute,
            on_result=lambda result: run_in_main_thread(self._on_correlation_search_success, result),
            on_error=lambda message: run_in_main_thread(self._on_correlation_search_error, message),
            owner=self,
            key="charts_correlation_search",
            cancel_previous=True,
        )
        return True

    def _on_correlation_search_success(self, result: CorrelationSearchResult) -> None:
        self._correlation_running = False
        self.correlation_search_finished.emit(result)

    def _on_correlation_search_error(self, message: str) -> None:
        self._correlation_running = False
        self.correlation_search_failed.emit(str(message or "Correlation analysis failed."))

    @staticmethod
    def _ordered_unique_features(
        target_feature: FeatureSelection,
        available_features: Sequence[FeatureSelection],
    ) -> list[FeatureSelection]:
        ordered: list[FeatureSelection] = []
        seen: set[tuple] = set()
        for item in [target_feature] + list(available_features):
            key = item.identity_key()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
        return ordered

    @staticmethod
    def _feature_map_for_columns(
        columns: Sequence[str],
        features: Sequence[FeatureSelection],
    ) -> dict[str, FeatureSelection]:
        mapping: dict[str, FeatureSelection] = {}
        for idx, name in enumerate(columns):
            if idx < len(features):
                mapping[str(name)] = features[idx]
            else:
                mapping[str(name)] = FeatureSelection(base_name=str(name))
        return mapping

    @staticmethod
    def _target_column(
        columns: Sequence[str],
        mapping: dict[str, FeatureSelection],
        target_feature: FeatureSelection,
    ) -> str | None:
        target_key = target_feature.identity_key()
        for name in columns:
            selection = mapping.get(str(name))
            if selection is not None and selection.identity_key() == target_key:
                return str(name)
        return None
