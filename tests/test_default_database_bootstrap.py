import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from backend.data_db.database import Database


def test_new_database_bootstraps_default_system_and_dataset(tmp_path) -> None:
    db_path = tmp_path / "fresh.duckdb"

    db = Database(db_path)
    try:
        dataset_id = db.ensure_default_system_dataset()
        assert int(dataset_id) > 0
        systems = db.list_systems()
        datasets = db.list_datasets()
    finally:
        db.close()

    assert "DefaultSystem" in systems
    assert "DefaultDataset" in datasets
