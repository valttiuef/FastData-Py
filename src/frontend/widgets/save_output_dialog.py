
from __future__ import annotations
from typing import Mapping, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..localization import tr


class SaveOutputDialog(QDialog):
    def __init__(
        self,
        *,
        mode: str,
        model_label: str = "",
        defaults: Optional[dict[str, object]] = None,
        group_defaults: Optional[Mapping[int, str]] = None,
        title: Optional[str] = None,
        intro: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._mode = str(mode or "").strip().lower()
        self._defaults = defaults or {}
        self._group_defaults = dict(group_defaults or {})
        self._edits: dict[int, QLineEdit] = {}

        if title:
            self.setWindowTitle(title)
        elif self._mode == "predictions":
            label = str(model_label).strip() or tr("Model")
            self.setWindowTitle(tr("Save predictions: {model}").format(model=label))
        else:
            self.setWindowTitle(tr("Save timeline clusters"))

        layout = QVBoxLayout(self)
        if self._mode == "predictions":
            self._build_prediction_form(layout)
        else:
            self._build_group_form(layout, intro)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_prediction_form(self, layout: QVBoxLayout) -> None:
        form = QFormLayout()
        layout.addLayout(form)

        self.feature_name_edit = QLineEdit(str(self._defaults.get("name") or ""), self)
        self.feature_notes_edit = QLineEdit(str(self._defaults.get("notes") or ""), self)
        self.source_edit = QLineEdit(str(self._defaults.get("source") or ""), self)
        self.unit_edit = QLineEdit(str(self._defaults.get("unit") or ""), self)
        self.feature_type_edit = QLineEdit(str(self._defaults.get("type") or ""), self)
        self.tags_edit = QLineEdit(self)
        self.tags_edit.setPlaceholderText(tr("tag1, tag2"))
        self.tags_edit.setText(self._format_tags(self._defaults.get("tags")))

        form.addRow(tr("Name:"), self.feature_name_edit)
        form.addRow(tr("Notes:"), self.feature_notes_edit)
        form.addRow(tr("Source:"), self.source_edit)
        form.addRow(tr("Unit:"), self.unit_edit)
        form.addRow(tr("Type:"), self.feature_type_edit)
        form.addRow(tr("Tags:"), self.tags_edit)
        self.resize(420, 320)

    def _build_group_form(self, layout: QVBoxLayout, intro: Optional[str]) -> None:
        intro_label = QLabel(
            intro
            or tr("Set a database group label for each cluster, then save or cancel."),
            self,
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        rows_host = QWidget(self)
        rows_layout = QFormLayout(rows_host)
        rows_layout.setContentsMargins(8, 8, 8, 8)

        default_group_label = str(self._defaults.get("group_label") or "som_cluster")
        self.group_label_edit = QLineEdit(default_group_label, self)
        self.group_label_edit.setPlaceholderText(tr("Group label"))
        rows_layout.addRow(tr("Group label:"), self.group_label_edit)

        for cluster_id in sorted(self._group_defaults.keys()):
            name = str(self._group_defaults.get(cluster_id, "")).strip() or f"SOM_Cluster {cluster_id}"
            edit = QLineEdit(name, rows_host)
            edit.setPlaceholderText(tr("Group name"))
            rows_layout.addRow(tr("Cluster {id}:").format(id=cluster_id), edit)
            self._edits[int(cluster_id)] = edit

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_host)
        layout.addWidget(scroll, 1)
        self.resize(520, 420)

    @staticmethod
    def _format_tags(value: object) -> str:
        if isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value if str(item).strip()]
            return ", ".join(items)
        return str(value or "").strip()

    @staticmethod
    def _parse_tags(text: str) -> list[str]:
        return [item for item in (part.strip() for part in (text or "").split(",")) if item]

    def values(self) -> dict[str, object]:
        if self._mode == "predictions":
            return {
                "name": self.feature_name_edit.text().strip(),
                "notes": self.feature_notes_edit.text().strip(),
                "source": self.source_edit.text().strip(),
                "unit": self.unit_edit.text().strip(),
                "type": self.feature_type_edit.text().strip(),
                "tags": self._parse_tags(self.tags_edit.text()),
            }
        group_label = ""
        if hasattr(self, "group_label_edit"):
            group_label = (self.group_label_edit.text() or "").strip()
        group_names: dict[int, str] = {}
        for cluster_id, edit in self._edits.items():
            group_names[int(cluster_id)] = (edit.text() or "").strip()
        return {
            "group_label": group_label,
            "group_names": group_names,
        }


__all__ = ["SaveOutputDialog"]
