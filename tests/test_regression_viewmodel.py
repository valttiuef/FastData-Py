import os
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from backend.services.regression_service import RegressionRunResult, RegressionSummary
from frontend.tabs.regression.viewmodel import RegressionViewModel


def _build_run(key: str) -> RegressionRunResult:
    timeline = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-01-01 00:00:00"]),
            "Actual": [1.0],
            "Prediction (test)": [1.1],
        }
    )
    return RegressionRunResult(
        key=key,
        model_key="linear_regression",
        model_label="Linear Regression",
        selector_key="none",
        selector_label="None",
        reducer_key="none",
        reducer_label="None",
        metrics={},
        cv_scores={},
        progress_frame=pd.DataFrame(),
        timeline_frame=timeline,
        scatter_frame=pd.DataFrame(),
    )


def _make_viewmodel_stub(service):
    vm = RegressionViewModel.__new__(RegressionViewModel)
    vm._service = service
    return vm


def test_viewmodel_run_regressions_continues_when_one_target_fails() -> None:
    run_ok = _build_run("ok-1")

    class _Service:
        def run_regressions(self, **kwargs):
            target = dict(kwargs.get("target_feature") or {})
            if int(target.get("feature_id") or 0) == 2:
                raise ValueError("target has no usable rows")
            return RegressionSummary(runs=[run_ok])

    vm = _make_viewmodel_stub(_Service())
    statuses: list[str] = []

    summary = RegressionViewModel.run_regressions(
        vm,
        data_frame=pd.DataFrame({"t": pd.to_datetime(["2026-01-01"]), "x": [1.0]}),
        input_features=[{"feature_id": 1, "name": "Input"}],
        target_features=[
            {"feature_id": 2, "name": "Bad target"},
            {"feature_id": 3, "name": "Good target"},
        ],
        selectors=["none"],
        models=["linear_regression"],
        status_callback=statuses.append,
    )

    assert len(summary.runs) == 1
    assert summary.runs[0].key == "ok-1"
    assert any(text.startswith("WARNING: Skipped target Bad target:") for text in statuses)


def test_viewmodel_run_regressions_keeps_partial_runs_when_target_fails_late() -> None:
    run_partial = _build_run("partial-1")

    class _Service:
        def run_regressions(self, **kwargs):
            callback = kwargs.get("result_callback")
            if callable(callback):
                callback(run_partial)
            raise RuntimeError("late target failure")

    vm = _make_viewmodel_stub(_Service())
    statuses: list[str] = []

    summary = RegressionViewModel.run_regressions(
        vm,
        data_frame=pd.DataFrame({"t": pd.to_datetime(["2026-01-01"]), "x": [1.0]}),
        input_features=[{"feature_id": 1, "name": "Input"}],
        target_features=[{"feature_id": 2, "name": "Target A"}],
        selectors=["none"],
        models=["linear_regression"],
        status_callback=statuses.append,
    )

    assert len(summary.runs) == 1
    assert summary.runs[0].key == "partial-1"
    assert any(
        text.startswith("WARNING: Target Target A completed partially and then failed:")
        for text in statuses
    )
