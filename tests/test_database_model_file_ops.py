import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.data_db.database import Database
from frontend.models.database_model import DatabaseModel
from frontend.models.settings_model import SettingsModel


def _create_db(path: Path) -> None:
    db = Database(path)
    try:
        db.ensure_default_system_dataset()
    finally:
        db.close()


def test_save_database_as_switches_active_database_path(tmp_path: Path) -> None:
    source = tmp_path / "source.duckdb"
    copy_target = tmp_path / "copy.duckdb"
    _create_db(source)

    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataSaveAsSwitchesPath",
    )
    settings.set_database_path(source)
    model = DatabaseModel(settings)
    try:
        model.save_database_as(copy_target)
        assert copy_target.exists()
        assert model.path == copy_target
        assert settings.database_path == copy_target
    finally:
        model._close_database()


def test_use_database_does_not_switch_path_when_access_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.duckdb"
    locked = tmp_path / "locked.duckdb"
    _create_db(source)
    _create_db(locked)

    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataUseDbValidation",
    )
    settings.set_database_path(source)
    model = DatabaseModel(settings)
    try:
        def _raise_access_error(path: Path) -> None:
            raise PermissionError(f"Database file is in use: {path}")

        monkeypatch.setattr(model, "_validate_database_access", _raise_access_error)

        with pytest.raises(PermissionError):
            model.use_database(locked)

        assert model.path == source
        assert settings.database_path == source
    finally:
        model._close_database()
