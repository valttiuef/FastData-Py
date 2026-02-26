from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

from backend.services.feature_selection_service import FeatureSelectionService
from backend.services.regression_service import RegressionService


class StubDatabase:
    def __init__(self):
        self._df = self._build_df()

    def _build_df(self):
        times = pd.date_range("2023-01-01", periods=24, freq="h")
        df1 = pd.DataFrame({
            "t": times,
            "v": np.linspace(0, 23, 24),
            "feature_id": 1,
        })
        df2 = pd.DataFrame({
            "t": times,
            "v": np.linspace(0, 46, 24) + 5,
            "feature_id": 2,
        })
        return pd.concat([df1, df2], ignore_index=True)

    def query_raw(self, **kwargs):
        feature_ids = kwargs.get("feature_ids")
        if feature_ids is None:
            return self._df.copy()
        return self._df[self._df["feature_id"].isin(feature_ids)].copy()


def test_regression_service_basic_run():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "label": "Input"}],
        target_feature={"feature_id": 2, "label": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        test_size=0.25,
        cv_strategy="kfold",
        cv_folds=3,
    )

    assert summary.runs, "Expected at least one regression run"
    run = summary.runs[0]
    assert "r2_train" in run.metrics
    assert not run.timeline_frame.empty
    assert not run.scatter_frame.empty


def test_available_models_list():
    """Test that all expected regression models are available."""
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    models = service.available_models()
    model_keys = {key for key, _label, _defaults in models}

    expected_models = {
        "linear_regression",
        "ridge",
        "lasso",
        "elastic_net",
        "polynomial_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "adaboost",
        "svr",
        "knn",
        "decision_tree",
    }

    assert expected_models.issubset(model_keys), f"Missing models: {expected_models - model_keys}"


def test_available_feature_selectors_list():
    """Test that all expected feature selectors are available."""
    service = FeatureSelectionService()
    selectors = service.available_feature_selectors()
    selector_keys = {key for key, _label, _defaults in selectors}

    expected_selectors = {
        "none",
        "variance_threshold",
        "select_k_best",
        "mutual_info",
        "random_forest_importance",
        "extra_trees_importance",
        "gradient_boosting_importance",
        "rfe_ridge",
    }

    assert expected_selectors.issubset(selector_keys), f"Missing selectors: {expected_selectors - selector_keys}"


def test_available_dimensionality_reducers_list():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    reducers = service.available_dimensionality_reducers()
    reducer_keys = {key for key, _label, _defaults in reducers}

    expected_reducers = {
        "none",
        "pca",
        "pls_regression",
        "fast_ica",
        "factor_analysis",
        "truncated_svd",
    }
    assert expected_reducers.issubset(reducer_keys), f"Missing reducers: {expected_reducers - reducer_keys}"


def test_gradient_boosting_regression():
    """Test gradient boosting regression model."""
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "label": "Input"}],
        target_feature={"feature_id": 2, "label": "Target"},
        selectors=["none"],
        models=["gradient_boosting"],
        test_size=0.25,
        cv_strategy="none",
    )

    assert summary.runs, "Expected at least one regression run"
    run = summary.runs[0]
    assert run.model_key == "gradient_boosting"
    assert "r2_train" in run.metrics


def test_polynomial_regression():
    """Test polynomial regression model."""
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "label": "Input"}],
        target_feature={"feature_id": 2, "label": "Target"},
        selectors=["none"],
        models=["polynomial_regression"],
        model_params={"polynomial_regression": {"degree": 2}},
        test_size=0.25,
        cv_strategy="none",
    )

    assert summary.runs, "Expected at least one regression run"
    run = summary.runs[0]
    assert run.model_key == "polynomial_regression"
    assert "r2_train" in run.metrics


def test_svr_regression():
    """Test support vector regression model."""
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "label": "Input"}],
        target_feature={"feature_id": 2, "label": "Target"},
        selectors=["none"],
        models=["svr"],
        test_size=0.25,
        cv_strategy="none",
    )

    assert summary.runs, "Expected at least one regression run"
    run = summary.runs[0]
    assert run.model_key == "svr"
    assert "r2_train" in run.metrics


def test_mutual_info_feature_selection():
    """Test mutual information feature selection."""
    fss = FeatureSelectionService()
    label, selector = fss.build("mutual_info", {"k": "all"})

    assert label == "K Best (Mutual Information)"
    assert selector is not None


def test_rfe_feature_selection():
    """Test RFE feature selection."""
    fss = FeatureSelectionService()
    label, selector = fss.build("rfe_ridge", {"n_features_to_select": 1, "step": 1})

    assert label == "Recursive Feature Elimination (Ridge)"
    assert selector is not None


def test_regression_with_multiple_reducers_multiplies_runs():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "label": "Input"}],
        target_feature={"feature_id": 2, "label": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        reducers=["pca", "truncated_svd"],
        reducer_params={
            "pca": {"n_components": 1},
            "truncated_svd": {"n_components": 1},
        },
        test_size=0.25,
        cv_strategy="none",
    )

    assert len(summary.runs) == 2
    reducer_keys = {run.reducer_key for run in summary.runs}
    assert reducer_keys == {"pca", "truncated_svd"}
