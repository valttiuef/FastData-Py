from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

from backend.services.forecasting_service import ForecastingService


class StubDatabase:
    def __init__(self, irregular=False):
        self.irregular = irregular
        self._df = self._build_df()

    def _build_df(self):
        if self.irregular:
            np.random.seed(42)
            base_time = pd.Timestamp("2023-01-01")
            intervals = np.random.exponential(30, 120)
            times = [base_time + pd.Timedelta(minutes=float(t)) for t in np.cumsum(intervals)]
        else:
            times = pd.date_range("2023-01-01", periods=120, freq="h")

        dfs = []
        for fid in [1, 2, 3]:
            values = np.sin(np.linspace(0, 6 * np.pi, len(times))) + np.random.normal(scale=0.2, size=len(times))
            dfs.append(pd.DataFrame({"t": times, "v": values, "feature_id": fid}))
        return pd.concat(dfs, ignore_index=True)

    def query_raw(self, **kwargs):
        feature_ids = kwargs.get("feature_ids")
        if feature_ids is None:
            return self._df.copy()
        return self._df[self._df["feature_id"].isin(feature_ids)].copy()


def test_forecasting_service_basic_run_linear_default():
    service = ForecastingService(StubDatabase())
    summary = service.run_forecasts(
        features=[{"feature_id": 1, "label": "Signal"}],
        models=["linear_regression"],
        forecast_horizon=6,
    )
    assert summary.runs
    run = summary.runs[0]
    assert run.model_key == "linear_regression"
    assert "rmse" in run.metrics
    assert not run.forecast_frame.empty
    test_data = run.forecast_frame[run.forecast_frame["Period"] == "Testing"]
    assert not test_data.empty
    assert test_data["Forecast"].notna().any()


def test_forecasting_service_multiple_models_and_features():
    service = ForecastingService(StubDatabase())
    summary = service.run_forecasts(
        features=[
            {"feature_id": 1, "label": "Signal 1"},
            {"feature_id": 2, "label": "Signal 2"},
        ],
        models=["linear_regression", "ridge", "lasso"],
        forecast_horizon=8,
    )
    assert len(summary.runs) == 6


def test_forecasting_service_window_strategy_expanding():
    service = ForecastingService(StubDatabase(irregular=True))
    summary = service.run_forecasts(
        features=[{"feature_id": 1, "label": "Irregular Signal"}],
        models=["ridge"],
        forecast_horizon=5,
        window_strategy="expanding",
    )
    assert summary.runs
    run = summary.runs[0]
    assert run.metrics["rmse"] >= 0
    assert (run.forecast_frame["Period"] == "Forecast").any()


def test_forecasting_available_models_are_sklearn_based():
    service = ForecastingService(StubDatabase())
    keys = [k for k, _label, _defaults in service.available_models()]
    assert "linear_regression" in keys
    assert "ridge" in keys
    assert "lasso" in keys
