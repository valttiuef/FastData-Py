import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.models import ImportOptions
from frontend.models.hybrid_pandas_model import HybridPandasModel
from frontend.models.selection_settings import SelectionSettingsPayload
from frontend.models.settings_model import SettingsModel


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_model(tmp_path: Path, *, app_name: str) -> HybridPandasModel:
    settings = SettingsModel(
        organization="FastDataTests",
        application=app_name,
    )
    settings.set_database_path(tmp_path / f"{app_name}.duckdb")
    settings.set_selection_db_path(tmp_path / f"{app_name}.sqlite")
    return HybridPandasModel(settings)


def test_empty_selection_payload_hides_features_when_selections_enabled(tmp_path: Path) -> None:
    model = _build_model(tmp_path, app_name="FastDataSelectionEmpty")
    try:
        csv_path = tmp_path / "selection-empty.csv"
        _write_csv(
            csv_path,
            [
                {"timestamp": "2026-01-01 00:00:00", "Temperature": 10.0},
                {"timestamp": "2026-01-01 01:00:00", "Temperature": 11.0},
            ],
        )
        model.import_files(
            [str(csv_path)],
            ImportOptions(
                system_name="System A",
                dataset_name="Dataset A",
                use_duckdb_csv_import=True,
            ),
        )
        all_features = model.features_df_unconstrained()
        assert not all_features.empty

        model.apply_selection_payload(
            SelectionSettingsPayload(
                include_selections=True,
                include_filters=False,
                include_preprocessing=False,
                feature_ids=[],
            )
        )
        assert model.features_df().empty

        model.apply_selection_payload(
            SelectionSettingsPayload(
                include_selections=False,
                include_filters=False,
                include_preprocessing=False,
                feature_ids=[],
            )
        )
        assert not model.features_df().empty
    finally:
        model._close_database()
        model._close_selection_database()


def test_import_extends_active_selection_with_new_features(tmp_path: Path) -> None:
    model = _build_model(tmp_path, app_name="FastDataSelectionImportExtend")
    try:
        csv_a = tmp_path / "selection-import-a.csv"
        _write_csv(
            csv_a,
            [
                {"timestamp": "2026-01-01 00:00:00", "FeatureA": 1.0},
                {"timestamp": "2026-01-01 01:00:00", "FeatureA": 2.0},
            ],
        )
        model.import_files(
            [str(csv_a)],
            ImportOptions(
                system_name="System A",
                dataset_name="Dataset A",
                use_duckdb_csv_import=True,
            ),
        )
        before_features = model.features_df_unconstrained()
        before_ids = {int(value) for value in before_features["feature_id"].tolist()}
        assert before_ids

        base_payload = SelectionSettingsPayload(
            include_selections=True,
            include_filters=False,
            include_preprocessing=False,
            feature_ids=sorted(before_ids),
        )
        setting_id = model.save_selection_setting(
            name="Active Selection",
            payload=base_payload.to_dict(),
            notes="",
            activate=True,
        )
        model.apply_selection_payload(base_payload)

        csv_b = tmp_path / "selection-import-b.csv"
        _write_csv(
            csv_b,
            [
                {"timestamp": "2026-01-02 00:00:00", "FeatureB": 5.0},
                {"timestamp": "2026-01-02 01:00:00", "FeatureB": 6.0},
            ],
        )
        model.import_files(
            [str(csv_b)],
            ImportOptions(
                system_name="System A",
                dataset_name="Dataset A",
                use_duckdb_csv_import=True,
            ),
        )

        after_features = model.features_df_unconstrained()
        after_ids = {int(value) for value in after_features["feature_id"].tolist()}
        new_ids = after_ids.difference(before_ids)
        assert new_ids

        saved = model.selection_setting(int(setting_id))
        assert saved is not None
        saved_payload = SelectionSettingsPayload.from_dict(saved.get("payload"))
        assert new_ids.issubset(set(int(value) for value in (saved_payload.feature_ids or [])))
        assert new_ids.issubset(model.selected_feature_ids)
    finally:
        model._close_database()
        model._close_selection_database()


def test_import_extends_label_only_active_selection_with_new_features(tmp_path: Path) -> None:
    model = _build_model(tmp_path, app_name="FastDataSelectionImportExtendLabels")
    try:
        csv_a = tmp_path / "selection-import-label-a.csv"
        _write_csv(
            csv_a,
            [
                {"timestamp": "2026-01-01 00:00:00", "FeatureA": 1.0},
                {"timestamp": "2026-01-01 01:00:00", "FeatureA": 2.0},
            ],
        )
        model.import_files(
            [str(csv_a)],
            ImportOptions(
                system_name="System A",
                dataset_name="Dataset A",
                use_duckdb_csv_import=True,
            ),
        )

        before_features = model.features_df_unconstrained()
        before_ids = {int(value) for value in before_features["feature_id"].tolist()}
        assert len(before_ids) == 1
        existing_id = next(iter(before_ids))

        label_for_existing = next(
            (
                label
                for label, fid in (model._feature_label_map() or {}).items()
                if int(fid) == existing_id
            ),
            "",
        )
        assert label_for_existing

        payload = SelectionSettingsPayload(
            include_selections=True,
            include_filters=False,
            include_preprocessing=False,
            feature_ids=[],
            feature_labels=[label_for_existing],
        )
        setting_id = model.save_selection_setting(
            name="Active Selection Label Only",
            payload=payload.to_dict(),
            notes="",
            activate=True,
        )
        model.apply_selection_payload(payload)
        assert existing_id in model.selected_feature_ids

        csv_b = tmp_path / "selection-import-label-b.csv"
        _write_csv(
            csv_b,
            [
                {"timestamp": "2026-01-02 00:00:00", "FeatureB": 5.0},
                {"timestamp": "2026-01-02 01:00:00", "FeatureB": 6.0},
            ],
        )
        model.import_files(
            [str(csv_b)],
            ImportOptions(
                system_name="System A",
                dataset_name="Dataset A",
                use_duckdb_csv_import=True,
            ),
        )

        after_features = model.features_df_unconstrained()
        after_ids = {int(value) for value in after_features["feature_id"].tolist()}
        new_ids = after_ids.difference(before_ids)
        assert new_ids
        assert new_ids.issubset(model.selected_feature_ids)

        saved = model.selection_setting(int(setting_id))
        assert saved is not None
        saved_payload = SelectionSettingsPayload.from_dict(saved.get("payload"))
        assert saved_payload.feature_ids
        assert new_ids.issubset(set(int(value) for value in (saved_payload.feature_ids or [])))
    finally:
        model._close_database()
        model._close_selection_database()


def test_delete_active_selection_reverts_to_default_state(tmp_path: Path) -> None:
    model = _build_model(tmp_path, app_name="FastDataSelectionDeleteActive")
    try:
        csv_path = tmp_path / "selection-delete-active.csv"
        _write_csv(
            csv_path,
            [
                {"timestamp": "2026-01-01 00:00:00", "FeatureA": 1.0},
                {"timestamp": "2026-01-01 01:00:00", "FeatureA": 2.0},
            ],
        )
        model.import_files(
            [str(csv_path)],
            ImportOptions(
                system_name="System A",
                dataset_name="Dataset A",
                use_duckdb_csv_import=True,
            ),
        )
        assert not model.features_df().empty

        payload = SelectionSettingsPayload(
            include_selections=True,
            include_filters=True,
            include_preprocessing=True,
            feature_ids=[],
            filters={"systems": ["System A"]},
            preprocessing={"timestep": "1h"},
        )
        setting_id = model.save_selection_setting(
            name="To Delete",
            payload=payload.to_dict(),
            notes="",
            activate=True,
        )
        model.apply_selection_payload(payload)
        assert model.features_df().empty

        model.delete_selection_setting(int(setting_id))

        assert model.active_selection_setting() is None
        assert model.selected_feature_ids == set()
        assert model.selection_filters == {}
        assert model.selection_preprocessing == {}
        assert not model.features_df().empty
    finally:
        model._close_database()
        model._close_selection_database()
