# --- @ai START ---
# model: gpt-6
# tool: codex
# role: regression-tests
# reviewed: yes
# date: 2026-10-16
# --- @ai END ---
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.data_db.database import Database


@pytest.fixture
def assets(tmp_path):
    db = Database(tmp_path / "assets.duckdb")
    with db.write_transaction() as con:
        con.execute("INSERT INTO systems (id, name) VALUES (1, 'A'), (2, 'B')")
        con.execute("INSERT INTO datasets (id, system_id, name) VALUES (1, 1, 'Same'), (2, 1, 'Other'), (3, 2, 'Same')")
        con.execute("INSERT INTO imports (id, dataset_id, file_path, file_name, file_sha256, csv_table_name) VALUES (1,1,'a','a','a','raw_a'), (2,2,'b','b','b','raw_b'), (3,3,'c','c','c','raw_c')")
        for table in ("raw_a", "raw_b", "raw_c"):
            con.execute(f"CREATE TABLE {table} (v INTEGER)")
        con.execute("INSERT INTO features (id, system_id, name, name_normalized) VALUES (1,1,'exclusive','exclusive'), (2,1,'shared','shared'), (3,2,'unrelated','unrelated'), (4,1,'unused','unused')")
        con.execute("INSERT INTO feature_import_map VALUES (1,1,1,1), (2,1,1,1), (2,1,2,2), (3,2,3,3)")
        con.execute("INSERT INTO feature_dataset_map VALUES (1,1,1), (2,1,1), (2,1,2), (3,2,3)")
        con.execute("INSERT INTO csv_feature_columns VALUES (1,1,'exclusive'), (1,2,'shared'), (2,2,'shared'), (3,3,'unrelated')")
        con.execute("INSERT INTO feature_tags VALUES (1,1,'tag','tag'), (2,2,'keep','keep')")
        con.execute("INSERT INTO group_labels VALUES (1,'exclusive','kind1'), (2,'shared','kind2'), (3,'unrelated','kind3')")
        con.execute("INSERT INTO group_label_scopes VALUES (1,1,1,1), (2,1,1,1), (2,1,2,2), (3,2,3,3)")
        con.execute("INSERT INTO group_points VALUES ('2026-01-01','2026-01-02',1,1), ('2026-01-01','2026-01-02',1,2), ('2026-01-01','2026-01-02',2,2), ('2026-01-01','2026-01-02',3,3)")
        con.execute("INSERT INTO csv_group_columns VALUES (1,1,'exclusive','kind1'), (1,2,'shared','kind2'), (2,2,'shared','kind2')")
        con.execute("INSERT INTO group_value_aliases (kind, feature_id, raw_value_norm, label, label_norm) VALUES ('kind1',1,'1','exclusive','exclusive'), ('kind2',2,'1','shared','shared')")
        con.execute("INSERT INTO measurements VALUES (1,'2026-01-01',1,1,1), (1,'2026-01-01',2,2,1), (2,'2026-01-01',2,3,2), (3,'2026-01-01',3,4,3)")
        con.execute("INSERT INTO model_runs (id,dataset_id,name,model_type,algorithm_key) VALUES (1,1,'removed','regression','test'), (2,2,'kept','regression','test')")
        con.execute("INSERT INTO model_imports VALUES (1,1), (2,2)")
        con.execute("INSERT INTO model_features VALUES (1,1,'x'), (2,2,'x')")
        con.execute("INSERT INTO model_results (id,model_id,stage,metric_name) VALUES (1,1,'test','mse'), (2,2,'test','mse')")
    yield db
    db.close()


@pytest.mark.parametrize("kind,value,system", [("import", 1, None), ("dataset", "Same", "A"), ("system", "A", None)])
def test_cascade_and_shared_data(assets, kind, value, system):
    result = assets.delete_filter_item(kind, value, system=system)
    with assets.connection() as con:
        def ids(table):
            return [r[0] for r in con.execute(f"SELECT id FROM {table} ORDER BY id").fetchall()]
        assert ids("imports") == ([3] if kind == "system" else [2, 3])
        assert ids("features") == ([3] if kind == "system" else [2, 3, 4])
        assert ids("group_labels") == ([3] if kind == "system" else [2, 3])
        assert ids("model_runs") == ([] if kind == "system" else [2])
        assert ids("model_results") == ([] if kind == "system" else [2])
        assert con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'raw_a'").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'raw_c'").fetchone()[0] == 1
        for table in ("measurements", "csv_feature_columns", "csv_group_columns", "feature_import_map", "group_label_scopes", "model_imports"):
            assert con.execute(f"SELECT count(*) FROM {table} WHERE import_id = 1").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM feature_tags WHERE feature_id = 1").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM group_value_aliases WHERE feature_id = 1").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM group_points WHERE dataset_id = 1").fetchone()[0] == 0
    assert result["groups_deleted"] == (2 if kind == "system" else 1)


def test_failure_rolls_back_everything(assets):
    original = assets._delete_asset
    def fail_after_cleanup(con, kind, item_id):
        original(con, kind, item_id)
        raise RuntimeError("injected failure")
    assets._delete_asset = fail_after_cleanup
    with pytest.raises(RuntimeError, match="injected failure"):
        assets.delete_filter_item("import", 1)
    with assets.connection() as con:
        assert con.execute("SELECT count(*) FROM imports").fetchone()[0] == 3
        assert con.execute("SELECT count(*) FROM features").fetchone()[0] == 4
        assert con.execute("SELECT count(*) FROM group_labels").fetchone()[0] == 3
        assert con.execute("SELECT count(*) FROM raw_a").fetchone()[0] == 0


@pytest.mark.parametrize("kind,value", [("system", "A"), ("dataset", "Same"), ("import", 1)])
@pytest.mark.parametrize("confirm", [False, True])
def test_context_deletion_requires_confirmation(monkeypatch, kind, value, confirm):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from frontend.widgets.filters_widget import FiltersWidget
    app = QApplication.instance() or QApplication([])
    widget = FiltersWidget()
    calls = []
    monkeypatch.setattr(widget._filters_view_model, "remove_filter_item", lambda *args, **kwargs: calls.append((args, kwargs)))
    def answer(*args):
        assert args[-1] == QMessageBox.StandardButton.No
        assert "cannot be undone" in args[2]
        return QMessageBox.StandardButton.Yes if confirm else QMessageBox.StandardButton.No
    monkeypatch.setattr(QMessageBox, "question", answer)
    widget._on_item_context_action("remove_" + kind, {"value": value, "label": "clicked item"})
    assert len(calls) == int(confirm)
    if confirm:
        assert calls[0][0] == (kind, value)
    widget.close()


def test_model_resolves_duplicate_dataset_names(assets):
    from frontend.models.database_model import DatabaseModel
    class Signal:
        pass
    class Model:
        _path = None
        database_changed = features_list_changed = groups_changed = Signal()
        def _ensure_database(self):
            return assets
        def list_datasets_for_filters(self, **kwargs):
            return [("Same", "A"), ("Same", "B")]
        def _invalidate_features_cache(self):
            self.invalidated = True
        def _emit_in_main_thread(self, *args):
            pass
    model = Model()
    DatabaseModel.remove_filter_item(model, "dataset", "Same", label="Same (B)", systems=["A", "B"])
    with assets.connection() as con:
        assert con.execute("SELECT id FROM imports ORDER BY id").fetchall() == [(1,), (2,)]
    assert model.invalidated


def test_dataset_requires_exact_owner(assets):
    with pytest.raises(ValueError, match="not found or is ambiguous"):
        assets.delete_filter_item("dataset", "Same")
    with assets.connection() as con:
        assert con.execute("SELECT count(*) FROM imports").fetchone()[0] == 3


def test_import_deletion_preserves_other_import_in_same_dataset(tmp_path):
    from backend.models import ImportOptions
    db = Database(tmp_path / "imports.duckdb")
    try:
        paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
        for index, path in enumerate(paths):
            path.write_text(f"Time,Temp\n2026-01-0{index + 1},10\n", encoding="utf-8")
        options = ImportOptions(system_name="System", dataset_name="Dataset", csv_header_rows=1,
                                auto_detect_datetime=False, date_column="Time")
        first = db.import_file(paths[0], options)[0]
        second = db.import_file(paths[1], options)[0]
        db.delete_filter_item("import", first)
        remaining = db.query_raw()
        assert set(remaining["import_id"].astype(int)) == {second}
        assert all(path.exists() for path in paths)
        with db.connection() as con:
            assert con.execute("SELECT count(*) FROM features").fetchone()[0] == 1
            assert con.execute("SELECT count(*) FROM datasets").fetchone()[0] == 1
    finally:
        db.close()
