import os
import sys
import tempfile
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from backend.data_db.database import Database


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_models.db"
        db = Database(db_path)
        yield db
        db.close()


def _seed_scope(db: Database, system: str = "A", dataset: str = "D") -> tuple[int, int]:
    sys_id = db._upsert_system(system)
    dataset_id = db._upsert_dataset(sys_id, dataset)
    return int(sys_id), int(dataset_id)


def _seed_feature(db: Database, system_id: int, name: str) -> int:
    return db.features_repo.insert_feature(
        db.con,
        system_id=int(system_id),
        name=name,
        source="raw",
        unit=None,
        type=None,
        notes=name,
        lag_seconds=0,
    )


def test_save_and_fetch_model(temp_db: Database):
    system_id, dataset_id = _seed_scope(temp_db)
    input_feature = _seed_feature(temp_db, system_id, "Input")
    target_feature = _seed_feature(temp_db, system_id, "Target")

    results = [
        {"stage": "cv", "metric_name": "rmse", "metric_value": 0.5, "fold": 1},
        {"stage": "test", "metric_name": "rmse", "metric_value": 0.45, "fold": None},
    ]

    model_id = temp_db.save_model_run(
        dataset_id=dataset_id,
        name="Test regression",
        model_type="regression",
        algorithm_key="linear_regression",
        selector_key="none",
        preprocessing={"scale": True},
        filters={"systems": ["A"], "datasets": ["D"]},
        hyperparameters={"alpha": 1.0},
        parameters={"cv_folds": 5},
        features=[(input_feature, "input"), (target_feature, "target")],
        results=results,
    )

    listed = temp_db.list_models("regression")
    assert not listed.empty
    assert model_id in listed["model_id"].values

    details = temp_db.fetch_model(model_id)
    assert details is not None
    assert details["model_type"] == "regression"
    assert int(details["dataset_id"]) == int(dataset_id)
    assert any(f["role"] == "target" for f in details["features"])
    assert len(details["results"]) == len(results)


def test_delete_model(temp_db: Database):
    system_id, dataset_id = _seed_scope(temp_db, system="S", dataset="DD")
    fid = _seed_feature(temp_db, system_id, "Only")
    model_id = temp_db.save_model_run(
        dataset_id=dataset_id,
        name="Transient",
        model_type="forecasting",
        algorithm_key="arima",
        selector_key=None,
        preprocessing={},
        filters={"systems": ["S"], "datasets": ["DD"]},
        hyperparameters={},
        parameters={},
        features=[(fid, "input")],
        results=[],
    )

    assert model_id in temp_db.list_models()["model_id"].values
    temp_db.delete_model(model_id)
    assert model_id not in temp_db.list_models()["model_id"].values
    assert temp_db.fetch_model(model_id) is None
