"""Backend service helpers with lazy imports to avoid circular dependencies."""

from importlib import import_module
from typing import Any, Dict

__all__ = [
    "ClassificationService",
    "ClusteringInputs",
    "ClusteringMethodSpec",
    "ClusteringService",
    "DimensionalityReductionService",
    "ForecastingService",
    "FeatureClusteringResult",
    "FilterConfig",
    "NeuronClusteringResult",
    "StatisticsConfig",
    "StatisticsResult",
    "StatisticsService",
    "RegressionService",
    "SOMService",
    "ScoreType",
    "SomResult",
    "aggregate_all",
    "aggregate_by_time",
    "available_statistics",
    "filter_dataframe",
    "freq_to_seconds",
    "prepare_series_for_statistics",
    "timestep_to_freq",
]

_MODULE_MAP: Dict[str, str] = {
    "ClassificationService": ".classification_service",
    "ClusteringInputs": ".som_service",
    "ClusteringMethodSpec": ".clustering",
    "ClusteringService": ".clustering",
    "DimensionalityReductionService": ".dimensionality_reduction_service",
    "FeatureClusteringResult": ".clustering",
    "FilterConfig": ".statistics_utils",
    "NeuronClusteringResult": ".clustering",
    "StatisticsConfig": ".statistics_utils",
    "StatisticsResult": ".statistics_service",
    "StatisticsService": ".statistics_service",
    "ForecastingService": ".forecasting_service",
    "RegressionService": ".regression_service",
    "SOMService": ".som_service",
    "ScoreType": ".clustering",
    "SomResult": ".som_service",
    "aggregate_all": ".statistics_utils",
    "aggregate_by_time": ".statistics_utils",
    "available_statistics": ".statistics_service",
    "filter_dataframe": ".statistics_utils",
    "freq_to_seconds": ".statistics_utils",
    "prepare_series_for_statistics": ".statistics_utils",
    "timestep_to_freq": ".statistics_utils",
}


def __getattr__(name: str) -> Any:
    module_name = _MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - convenience
    return sorted(set(__all__ + list(globals().keys())))
