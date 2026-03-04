from pathlib import Path
import sys
import uuid

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

from backend.data_db.database import Database
from backend.models import HeaderRoles, ImportOptions
from backend.services.feature_selection_service import FeatureSelectionService
from backend.services.modeling_shared import prepare_wide_frame
from backend.services.regression_service import RegressionService

TEST_DIR = Path(__file__).parent
TEST_IMPORTS_DIR = TEST_DIR / "test_data" / "imports"
TEST_TEMP_DIR = TEST_DIR / "temp"


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


def _build_grouped_regression_context(*, csv_name: str, system_name: str, dataset_name: str) -> dict[str, object]:
    TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    db_path = TEST_TEMP_DIR / f"regression_cv_{uuid.uuid4().hex}.duckdb"
    db = Database(db_path)
    try:
        opts = ImportOptions(
            system_name=system_name,
            dataset_name=dataset_name,
            csv_header_rows=1,
            auto_detect_datetime=False,
            date_column="Time",
            csv_delimiter=",",
            use_duckdb_csv_import=True,
            force_meta_columns=["Material"],
        )
        import_ids = db.import_file(TEST_IMPORTS_DIR / csv_name, opts)
        assert len(import_ids) == 1

        features = db.list_features(systems=[system_name], datasets=[dataset_name])
        assert not features.empty
        feature_map = {
            str(row["name"]): int(row["feature_id"])
            for _, row in features.iterrows()
        }
        assert "Input" in feature_map
        assert "Target" in feature_map

        return {
            "db": db,
            "db_path": db_path,
            "service": RegressionService(db, feature_selection_service=FeatureSelectionService()),
            "system_name": system_name,
            "dataset_name": dataset_name,
            "input_feature_id": int(feature_map["Input"]),
            "target_feature_id": int(feature_map["Target"]),
        }
    except Exception:
        db.close()
        try:
            for candidate in TEST_TEMP_DIR.glob(f"{db_path.name}*"):
                if candidate.is_file():
                    candidate.unlink()
        except Exception:
            pass
        raise


def _dispose_grouped_regression_context(context: dict[str, object]) -> None:
    db = context.get("db")
    if isinstance(db, Database):
        db.close()
    db_path = context.get("db_path")
    if isinstance(db_path, Path):
        for candidate in TEST_TEMP_DIR.glob(f"{db_path.name}*"):
            if candidate.is_file():
                candidate.unlink()


def _build_pipeline_dataframe() -> pd.DataFrame:
    start = pd.Timestamp("2026-03-01 00:00:00")
    rows: list[dict[str, object]] = []
    for idx in range(44):
        ts = start + pd.Timedelta(hours=idx)
        input_value = round(20.0 + idx * 0.7, 6)
        noise = ((idx % 5) - 2) * 0.05
        target_value = round((2.9 * input_value) + noise, 6)
        run_id = f"Run{(idx // 6) + 1}"
        material = "C" if idx in {5, 17, 29, 41} else ("A" if idx % 2 == 0 else "B")
        rows.append(
            {
                "Time": ts,
                "Input": input_value,
                "Target": target_value,
                "RunID": run_id,
                "Material": material,
            }
        )
    return pd.DataFrame(rows)


def _build_imported_pipeline_context(*, source_kind: str) -> dict[str, object]:
    TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    source_frame = _build_pipeline_dataframe()
    if source_kind == "excel":
        source_path = TEST_TEMP_DIR / f"regression_pipeline_{token}.xlsx"
        source_frame.to_excel(source_path, index=False)
        import_options = ImportOptions(
            system_name=f"SysRegressionPipelineExcel_{token}",
            dataset_name=f"DataRegressionPipelineExcel_{token}",
            excel_header_rows=1,
            auto_detect_datetime=False,
            date_column="Time",
            force_meta_columns=["RunID", "Material"],
            header_roles=HeaderRoles(
                base_name_row=0,
                source_row=None,
                unit_row=None,
                type_row=None,
            ),
        )
    else:
        source_path = TEST_TEMP_DIR / f"regression_pipeline_{token}.csv"
        source_frame.to_csv(source_path, index=False)
        import_options = ImportOptions(
            system_name=f"SysRegressionPipelineCsv_{token}",
            dataset_name=f"DataRegressionPipelineCsv_{token}",
            csv_header_rows=1,
            auto_detect_datetime=False,
            date_column="Time",
            csv_delimiter=",",
            use_duckdb_csv_import=True,
            force_meta_columns=["RunID", "Material"],
        )

    db_path = TEST_TEMP_DIR / f"regression_pipeline_{token}.duckdb"
    db = Database(db_path)
    try:
        import_ids = db.import_file(source_path, import_options)
        assert len(import_ids) == 1

        features = db.list_features(
            systems=[import_options.system_name],
            datasets=[import_options.dataset_name],
        )
        assert not features.empty
        feature_map = {str(row["name"]): int(row["feature_id"]) for _, row in features.iterrows()}
        assert "Input" in feature_map
        assert "Target" in feature_map
        assert db.list_group_labels(kind="RunID").shape[0] >= 6
        assert db.list_group_labels(kind="Material").shape[0] >= 3

        return {
            "db": db,
            "db_path": db_path,
            "source_path": source_path,
            "source_frame": source_frame,
            "service": RegressionService(db, feature_selection_service=FeatureSelectionService()),
            "system_name": import_options.system_name,
            "dataset_name": import_options.dataset_name,
            "input_feature_id": int(feature_map["Input"]),
            "target_feature_id": int(feature_map["Target"]),
        }
    except Exception:
        db.close()
        for candidate in TEST_TEMP_DIR.glob(f"{db_path.name}*"):
            if candidate.is_file():
                candidate.unlink()
        if source_path.exists():
            source_path.unlink()
        raise


def _dispose_imported_pipeline_context(context: dict[str, object]) -> None:
    db = context.get("db")
    if isinstance(db, Database):
        db.close()
    db_path = context.get("db_path")
    if isinstance(db_path, Path):
        for candidate in TEST_TEMP_DIR.glob(f"{db_path.name}*"):
            if candidate.is_file():
                candidate.unlink()
    source_path = context.get("source_path")
    if isinstance(source_path, Path) and source_path.exists():
        try:
            source_path.unlink()
        except Exception:
            pass


@pytest.fixture
def grouped_regression_context_complete():
    context = _build_grouped_regression_context(
        csv_name="csv_regression_groups_complete.csv",
        system_name="SysRegressionCVComplete",
        dataset_name="DataRegressionCVComplete",
    )
    try:
        yield context
    finally:
        _dispose_grouped_regression_context(context)


@pytest.fixture
def grouped_regression_context_partial():
    context = _build_grouped_regression_context(
        csv_name="csv_regression_groups.csv",
        system_name="SysRegressionCVPartial",
        dataset_name="DataRegressionCVPartial",
    )
    try:
        yield context
    finally:
        _dispose_grouped_regression_context(context)


@pytest.fixture
def pipeline_context_csv():
    context = _build_imported_pipeline_context(source_kind="csv")
    try:
        yield context
    finally:
        _dispose_imported_pipeline_context(context)


@pytest.fixture
def pipeline_context_excel():
    context = _build_imported_pipeline_context(source_kind="excel")
    try:
        yield context
    finally:
        _dispose_imported_pipeline_context(context)


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


def test_regression_drops_sparse_input_features_and_reports_warning():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    times = pd.date_range("2023-01-01", periods=40, freq="h")
    data_frame = pd.DataFrame(
        {
            "t": times,
            "Input Dense": np.linspace(0.0, 39.0, 40),
            "Input Sparse": [np.nan] * 39 + [1.0],
            "Target": np.linspace(10.0, 49.0, 40),
        }
    )
    status_messages: list[str] = []

    summary = service.run_regressions(
        input_features=[
            {"feature_id": 1, "base_name": "Input Dense"},
            {"feature_id": 2, "base_name": "Input Sparse"},
        ],
        target_feature={"feature_id": 3, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        data_frame=data_frame,
        cv_strategy="none",
        status_callback=status_messages.append,
    )

    assert summary.runs
    run = summary.runs[0]
    dropped = run.metadata.get("inputs_dropped_sparse_names") or []
    assert dropped == ["Input Sparse"]
    assert run.metadata.get("inputs_total") == 1
    assert any("Dropped sparse input features" in msg for msg in status_messages)


def test_regression_fails_when_all_inputs_are_sparse():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    times = pd.date_range("2023-01-01", periods=30, freq="h")
    data_frame = pd.DataFrame(
        {
            "t": times,
            "Input Sparse A": [np.nan] * 29 + [1.0],
            "Input Sparse B": [np.nan] * 29 + [2.0],
            "Target": np.linspace(0.0, 29.0, 30),
        }
    )

    try:
        service.run_regressions(
            input_features=[
                {"feature_id": 1, "base_name": "Input Sparse A"},
                {"feature_id": 2, "base_name": "Input Sparse B"},
            ],
            target_feature={"feature_id": 3, "base_name": "Target"},
            selectors=["none"],
            models=["linear_regression"],
            data_frame=data_frame,
            cv_strategy="none",
        )
    except ValueError as exc:
        assert "All input features were dropped by training rules" in str(exc)
    else:
        raise AssertionError("Expected ValueError when all input features exceed sparse threshold")


def test_regression_drops_static_input_features_and_reports_warning():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    times = pd.date_range("2023-01-01", periods=40, freq="h")
    data_frame = pd.DataFrame(
        {
            "t": times,
            "Input Variable": np.linspace(0.0, 39.0, 40),
            "Input Static": [7.0] * 40,
            "Target": np.linspace(10.0, 49.0, 40),
        }
    )
    status_messages: list[str] = []

    summary = service.run_regressions(
        input_features=[
            {"feature_id": 1, "base_name": "Input Variable"},
            {"feature_id": 2, "base_name": "Input Static"},
        ],
        target_feature={"feature_id": 3, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        data_frame=data_frame,
        cv_strategy="none",
        status_callback=status_messages.append,
    )

    assert summary.runs
    run = summary.runs[0]
    dropped_static = run.metadata.get("inputs_dropped_static_names") or []
    assert dropped_static == ["Input Static"]
    assert run.metadata.get("inputs_total") == 1
    assert any("Dropped static input features" in msg for msg in status_messages)


def test_time_series_cv_works_without_cross_val_predict_partition_errors():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "label": "Input"}],
        target_feature={"feature_id": 2, "label": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        test_size=0.25,
        cv_strategy="time_series",
        cv_folds=3,
    )

    assert summary.runs, "Expected at least one regression run with time-series CV"


def test_regression_uses_group_kind_column_from_preprocessed_frame_for_stratified_split():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    times = pd.date_range("2023-01-01", periods=60, freq="h")
    data_frame = pd.DataFrame(
        {
            "t": times,
            "Input": np.linspace(1.0, 60.0, 60),
            "Target": np.linspace(2.0, 120.0, 60),
            "Material": ["A", "B", "C"] * 20,
        }
    )

    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "base_name": "Input"}],
        target_feature={"feature_id": 2, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        data_frame=data_frame,
        cv_strategy="none",
        test_strategy="stratified",
        test_size=0.25,
        stratify_feature={"group_kind": "material_kind", "label": "Material"},
        stratify_bins=3,
    )

    assert summary.runs, "Expected at least one regression run with group-kind stratification column"


def test_regression_group_kind_stratify_with_missing_values_does_not_drop_all_rows():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    times = pd.date_range("2023-01-01", periods=60, freq="h")
    materials = (["A", "B", "C", np.nan] * 15)[:60]
    data_frame = pd.DataFrame(
        {
            "t": times,
            "Input": np.linspace(1.0, 60.0, 60),
            "Target": np.linspace(2.0, 120.0, 60),
            "Material": materials,
        }
    )

    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "base_name": "Input"}],
        target_feature={"feature_id": 2, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        data_frame=data_frame,
        cv_strategy="none",
        test_strategy="stratified",
        test_size=0.25,
        stratify_feature={"group_kind": "material_kind", "label": "Material"},
        stratify_bins=3,
    )

    assert summary.runs, "Expected regression run even when group-kind stratify has missing values"


def test_group_kfold_can_use_group_column_from_preprocessed_frame():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    times = pd.date_range("2023-01-01", periods=60, freq="h")
    data_frame = pd.DataFrame(
        {
            "t": times,
            "Input": np.linspace(1.0, 60.0, 60),
            "Target": np.linspace(2.0, 120.0, 60),
            "Material": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
        }
    )

    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "base_name": "Input"}],
        target_feature={"feature_id": 2, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        data_frame=data_frame,
        cv_strategy="group_kfold",
        cv_group_kind="Material",
        cv_folds=3,
        test_size=0.0,
    )

    assert summary.runs, "Expected at least one regression run with column-based Group K-Fold"


def test_group_kfold_with_database_groups_runs_without_fallback_warning(grouped_regression_context_complete):
    context = grouped_regression_context_complete
    service = context["service"]
    status_messages: list[str] = []

    summary = service.run_regressions(
        input_features=[{"feature_id": context["input_feature_id"], "base_name": "Input"}],
        target_feature={"feature_id": context["target_feature_id"], "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        systems=[context["system_name"]],
        Datasets=[context["dataset_name"]],
        cv_strategy="group_kfold",
        cv_group_kind="Material",
        cv_folds=3,
        test_size=0.0,
        status_callback=status_messages.append,
    )

    assert summary.runs, "Expected at least one regression run with DB-backed Group K-Fold"
    assert not any("Group K-Fold" in msg and "falling back" in msg for msg in status_messages)


def test_stratified_kfold_with_group_kind_works_even_when_prefetched_group_column_is_empty(
    grouped_regression_context_complete,
):
    context = grouped_regression_context_complete
    service = context["service"]
    db = context["db"]
    status_messages: list[str] = []

    raw = db.query_raw(
        systems=[context["system_name"]],
        datasets=[context["dataset_name"]],
        feature_ids=[context["input_feature_id"], context["target_feature_id"]],
    )
    frame = prepare_wide_frame(
        raw,
        [
            {"feature_id": context["input_feature_id"], "base_name": "Input"},
            {"feature_id": context["target_feature_id"], "base_name": "Target"},
        ],
    )
    frame["Material"] = np.nan

    summary = service.run_regressions(
        input_features=[{"feature_id": context["input_feature_id"], "base_name": "Input"}],
        target_feature={"feature_id": context["target_feature_id"], "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        data_frame=frame,
        cv_strategy="stratified_kfold",
        cv_folds=3,
        test_size=0.0,
        stratify_feature={"group_kind": "Material", "label": "Material"},
        status_callback=status_messages.append,
    )

    assert summary.runs
    assert not any("Stratified K-Fold" in msg and "falling back" in msg for msg in status_messages)


def test_stratified_kfold_with_partial_group_assignments_falls_back_with_warning(
    grouped_regression_context_partial,
):
    context = grouped_regression_context_partial
    service = context["service"]
    status_messages: list[str] = []

    summary = service.run_regressions(
        input_features=[{"feature_id": context["input_feature_id"], "base_name": "Input"}],
        target_feature={"feature_id": context["target_feature_id"], "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        systems=[context["system_name"]],
        Datasets=[context["dataset_name"]],
        cv_strategy="stratified_kfold",
        cv_folds=4,
        test_size=0.0,
        stratify_feature={"group_kind": "Material", "label": "Material"},
        status_callback=status_messages.append,
    )

    assert summary.runs, "Expected fallback to K-Fold when stratify labels are partially missing"
    assert any(
        msg.startswith("WARNING: Stratified K-Fold has rows without stratification labels")
        for msg in status_messages
    )


def test_group_kfold_without_assignments_emits_warning_and_falls_back():
    service = RegressionService(StubDatabase(), feature_selection_service=FeatureSelectionService())
    times = pd.date_range("2023-01-01", periods=60, freq="h")
    data_frame = pd.DataFrame(
        {
            "t": times,
            "Input": np.linspace(1.0, 60.0, 60),
            "Target": np.linspace(2.0, 120.0, 60),
            "Material": [np.nan] * 60,
        }
    )
    status_messages: list[str] = []

    summary = service.run_regressions(
        input_features=[{"feature_id": 1, "base_name": "Input"}],
        target_feature={"feature_id": 2, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        data_frame=data_frame,
        cv_strategy="group_kfold",
        cv_group_kind="Material",
        cv_folds=3,
        test_size=0.0,
        status_callback=status_messages.append,
    )

    assert summary.runs
    assert any(
        msg.startswith("WARNING: Group K-Fold has no group assignments; falling back to K-Fold.")
        or msg.startswith("WARNING: Group K-Fold requires a valid group; falling back to K-Fold.")
        for msg in status_messages
    )


def _assert_pipeline_regression_modes(context: dict[str, object]) -> None:
    service = context["service"]
    source_frame = context["source_frame"]
    input_feature_id = int(context["input_feature_id"])
    target_feature_id = int(context["target_feature_id"])
    systems = [context["system_name"]]
    datasets = [context["dataset_name"]]

    status_cv_strat: list[str] = []
    summary_cv_strat = service.run_regressions(
        input_features=[{"feature_id": input_feature_id, "base_name": "Input"}],
        target_feature={"feature_id": target_feature_id, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        systems=systems,
        Datasets=datasets,
        cv_strategy="stratified_kfold",
        cv_folds=3,
        test_size=0.0,
        stratify_feature={"group_kind": "Material", "label": "Material"},
        status_callback=status_cv_strat.append,
    )
    assert summary_cv_strat.runs
    assert not any("Stratified K-Fold" in msg and "falling back" in msg for msg in status_cv_strat)

    status_cv_group: list[str] = []
    summary_cv_group = service.run_regressions(
        input_features=[{"feature_id": input_feature_id, "base_name": "Input"}],
        target_feature={"feature_id": target_feature_id, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        systems=systems,
        Datasets=datasets,
        cv_strategy="group_kfold",
        cv_folds=4,
        cv_group_kind="RunID",
        test_size=0.0,
        status_callback=status_cv_group.append,
    )
    assert summary_cv_group.runs
    assert not any("Group K-Fold" in msg and "falling back" in msg for msg in status_cv_group)

    status_test_split: list[str] = []
    summary_test_split = service.run_regressions(
        input_features=[{"feature_id": input_feature_id, "base_name": "Input"}],
        target_feature={"feature_id": target_feature_id, "base_name": "Target"},
        selectors=["none"],
        models=["linear_regression"],
        systems=systems,
        Datasets=datasets,
        cv_strategy="none",
        test_size=0.25,
        test_strategy="stratified",
        stratify_feature={"group_kind": "Material", "label": "Material"},
        status_callback=status_test_split.append,
    )
    assert summary_test_split.runs
    assert not any("Stratified test split" in msg and "falling back" in msg for msg in status_test_split)

    run = summary_test_split.runs[0]
    timeline = run.timeline_frame
    assert not timeline.empty
    assert "Prediction (test)" in timeline.columns
    assert "Prediction (train)" in timeline.columns

    material_by_ts = pd.Series(
        source_frame["Material"].astype(str).to_numpy(),
        index=pd.to_datetime(source_frame["Time"], errors="coerce"),
    ).to_dict()
    test_labels = (
        timeline.loc[timeline["Prediction (test)"].notna(), "t"]
        .map(material_by_ts)
        .dropna()
        .astype(str)
        .tolist()
    )
    train_labels = (
        timeline.loc[timeline["Prediction (train)"].notna(), "t"]
        .map(material_by_ts)
        .dropna()
        .astype(str)
        .tolist()
    )
    assert "C" in test_labels
    assert "C" in train_labels


def test_import_to_regression_pipeline_csv_supports_group_and_stratified_modes(pipeline_context_csv):
    _assert_pipeline_regression_modes(pipeline_context_csv)


def test_import_to_regression_pipeline_excel_supports_group_and_stratified_modes(pipeline_context_excel):
    _assert_pipeline_regression_modes(pipeline_context_excel)
