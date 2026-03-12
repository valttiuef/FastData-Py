import os
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from frontend.localization import tr
from frontend.widgets.export_dialog import ExportOption, ExportSelectionDialog


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_export_dialog_select_all_click_from_partial_selects_all(qapp) -> None:
    dialog = ExportSelectionDialog(
        title="Export",
        heading="Select",
        options=[
            ExportOption(key="a", label="A"),
            ExportOption(key="b", label="B"),
            ExportOption(key="c", label="C"),
        ],
        default_selected_keys=["a"],
    )
    assert dialog.selected_keys() == ["a"]
    assert dialog.select_all_checkbox.text() == tr("Select all")

    dialog.select_all_checkbox.click()

    assert set(dialog.selected_keys()) == {"a", "b", "c"}
    assert dialog.select_all_checkbox.text() == tr("Unselect all")


def test_export_dialog_select_all_becomes_unselect_all_when_all_selected(qapp) -> None:
    dialog = ExportSelectionDialog(
        title="Export",
        heading="Select",
        options=[
            ExportOption(key="a", label="A"),
            ExportOption(key="b", label="B"),
        ],
    )
    assert set(dialog.selected_keys()) == {"a", "b"}
    assert dialog.select_all_checkbox.text() == tr("Unselect all")

    dialog.select_all_checkbox.click()

    assert dialog.selected_keys() == []
    assert dialog.select_all_checkbox.text() == tr("Select all")
