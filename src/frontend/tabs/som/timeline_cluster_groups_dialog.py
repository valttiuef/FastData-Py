
from __future__ import annotations
from typing import Mapping, Optional

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from ...localization import tr
from ...viewmodels.help_viewmodel import get_help_viewmodel
from ...widgets.help_widgets import InfoButton
from ...widgets.save_output_dialog import SaveOutputDialog


class TimelineClusterGroupsDialog(SaveOutputDialog):
    def __init__(
        self,
        cluster_defaults: Mapping[int, str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            mode="groups",
            defaults={"group_label": "som_cluster"},
            group_defaults=cluster_defaults,
            title=tr("Save timeline clusters"),
            intro=tr("Set a database group label for each cluster, then save or cancel."),
            parent=parent,
        )
        self._save_as_timeframes_checkbox = QCheckBox(
            tr("Save as timeframes"),
            self,
        )
        self._save_as_timeframes_checkbox.setChecked(True)
        help_row = QWidget(self)
        help_layout = QHBoxLayout(help_row)
        help_layout.setContentsMargins(0, 0, 0, 0)
        help_layout.setSpacing(6)
        help_layout.addWidget(self._save_as_timeframes_checkbox, 0)
        help_layout.addStretch(1)
        try:
            help_viewmodel = get_help_viewmodel()
            help_layout.addWidget(
                InfoButton("controls.som.timeline.save_as_timeframes", help_viewmodel, help_row),
                0,
            )
        except Exception:
            # Help system may not be initialised in tests or early startup.
            pass
        layout = self.layout()
        if layout is not None:
            layout.insertWidget(1, help_row)
        self.setModal(True)

    def group_names(self) -> dict[int, str]:
        values = self.values()
        return dict(values.get("group_names") or {})

    def group_label(self) -> str:
        return str(self.values().get("group_label") or "").strip()

    def save_as_timeframes(self) -> bool:
        return bool(self._save_as_timeframes_checkbox.isChecked())
