import os
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from backend.data_db.database import Database
from backend.models import ImportOptions
from frontend.utils.exporting import ExportPlan, execute_export_plan


def test_execute_export_plan_writes_split_csv_files(tmp_path: Path) -> None:
    frame_a = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    frame_b = pd.DataFrame({"A": [5], "B": [6]})
    destination = tmp_path / "exports.csv"

    plan = ExportPlan(
        kind="dataframes",
        selected_format="csv",
        destination=destination,
        datasets={"First": frame_a, "Second": frame_b},
    )
    outcome = execute_export_plan(plan)

    assert outcome.ok
    assert outcome.open_path == tmp_path
    created = sorted(tmp_path.glob("exports_*_*.csv"))
    assert len(created) == 2

    loaded_first = pd.read_csv(created[0])
    loaded_second = pd.read_csv(created[1])
    assert set(loaded_first.columns) == {"A", "B"}
    assert set(loaded_second.columns) == {"A", "B"}


def test_import_to_export_pipeline_roundtrip_keeps_selected_values(tmp_path: Path) -> None:
    db_path = tmp_path / "pipeline.duckdb"
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Time,Temp,Pressure",
                "2026-01-01 00:00:00,10.0,1.0",
                "2026-01-01 01:00:00,11.0,1.1",
                "2026-01-01 02:00:00,12.0,1.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    db = Database(db_path)
    try:
        options = ImportOptions(
            system_name="ExportSystem",
            dataset_name="ExportDataset",
            csv_header_rows=1,
            auto_detect_datetime=False,
            date_column="Time",
        )
        import_ids = db.import_file(csv_path, options)
        assert len(import_ids) == 1

        raw = db.query_raw(system="ExportSystem", dataset="ExportDataset")
        assert isinstance(raw, pd.DataFrame)
        assert not raw.empty

        value_col = "v" if "v" in raw.columns else "value"
        export_frame = raw.loc[:, ["t", value_col]].copy()
        export_frame = export_frame.rename(columns={"t": "Timestamp", value_col: "Value"})
        export_frame["Timestamp"] = pd.to_datetime(export_frame["Timestamp"], errors="coerce")
        export_frame["Value"] = pd.to_numeric(export_frame["Value"], errors="coerce")
        export_frame = export_frame.dropna(subset=["Timestamp", "Value"]).sort_values("Timestamp", kind="stable")
        export_frame = export_frame.reset_index(drop=True)
        assert not export_frame.empty

        destination = tmp_path / "pipeline_export.xlsx"
        plan = ExportPlan(
            kind="dataframes",
            selected_format="excel",
            destination=destination,
            datasets={"Imported data": export_frame},
        )
        outcome = execute_export_plan(plan)
        assert outcome.ok
        assert destination.exists()

        loaded = pd.read_excel(destination, sheet_name="Imported data")
        loaded["Value"] = pd.to_numeric(loaded["Value"], errors="coerce")
        loaded = loaded.dropna(subset=["Value"]).reset_index(drop=True)
        assert len(loaded) == len(export_frame)
        assert float(loaded["Value"].mean()) == pytest.approx(float(export_frame["Value"].mean()))
    finally:
        db.close()
