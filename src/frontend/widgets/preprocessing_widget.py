
from __future__ import annotations
from typing import Any, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (

    QComboBox,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QWidget,
    QSizePolicy,
)
from ..localization import tr

from .collapsible_section import CollapsibleSection
from .help_widgets import InfoButton
from ..viewmodels.help_viewmodel import HelpViewModel, get_help_viewmodel
from ..utils.time_steps import (
    MOVING_AVERAGE_OPTIONS,
    TIMESTEP_OPTIONS,
    label_to_seconds,
    seconds_to_label,
)

class PreprocessingWidget(CollapsibleSection):
    """Reusable widget that exposes the preprocessing controls used by the Data tab."""

    parameter_changed = Signal(str, object)
    parameters_changed = Signal(dict)

    def __init__(
        self,
        title: str = "Preprocessing",
        *,
        collapsed: bool = True,
        parent=None,
        help_viewmodel: Optional[HelpViewModel] = None,
    ):
        super().__init__(tr(title), collapsed=collapsed, parent=parent)
        self._help_viewmodel = help_viewmodel or self._resolve_help_viewmodel()

        grid = QGridLayout()
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        # Columns: 0=label, 1=combo/control, 2=edit/secondary control, 3=info button
        grid.setColumnStretch(0, 0)  # label fixed
        grid.setColumnStretch(1, 1)  # control 50%
        grid.setColumnStretch(2, 1)  # control 50%
        grid.setColumnStretch(3, 0)  # info fixed/small

        row = 0

        def add_pair_row(label_text: str, combo: QComboBox, edit: QLineEdit, info_key: str | None):
            nonlocal row

            lbl = self._make_label(label_text)
            lbl.setBuddy(combo)

            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            info = self._make_info(info_key) if info_key else None

            grid.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignRight)
            grid.addWidget(combo, row, 1)
            grid.addWidget(edit, row, 2)
            if info:
                grid.addWidget(info, row, 3, Qt.AlignmentFlag.AlignLeft)

            row += 1
            return lbl, info

        def add_single_row(label_text: str, control, info_key: str | None):
            """Single control takes full control area (columns 1+2)."""
            nonlocal row

            lbl = self._make_label(label_text)
            lbl.setBuddy(control)

            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            info = self._make_info(info_key) if info_key else None

            grid.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignRight)
            grid.addWidget(control, row, 1, 1, 2)  # span columns 1-2 (full width)
            if info:
                grid.addWidget(info, row, 3, Qt.AlignmentFlag.AlignLeft)

            row += 1
            return lbl, info

        # --- Timestep (combo + edit, 50/50)
        self.timestep_combo = QComboBox()
        for label, _seconds in TIMESTEP_OPTIONS:
            self.timestep_combo.addItem(tr(label), label)
        self._set_combo_by_data(self.timestep_combo, "auto")

        self.timestep_edit = QLineEdit()
        self.timestep_edit.setPlaceholderText(tr("auto"))

        self._lbl_timestep, self._timestep_info = add_pair_row(
            tr("Timestep:"), self.timestep_combo, self.timestep_edit,
            "controls.preprocessing.timestep"
        )

        # --- Moving average (combo + edit, 50/50)
        self.moving_average_combo = QComboBox()
        for label, _seconds in MOVING_AVERAGE_OPTIONS:
            self.moving_average_combo.addItem(tr(label), label)
        self._set_combo_by_data(self.moving_average_combo, "none")

        self.moving_average_edit = QLineEdit()
        self.moving_average_edit.setPlaceholderText(tr("auto"))

        self._lbl_ma, self._ma_info = add_pair_row(
            tr("Moving average:"), self.moving_average_combo, self.moving_average_edit,
            "controls.preprocessing.moving_average"
        )

        # --- Fill empty (single control spans full width)
        self.fill_combo = QComboBox()
        for label in ["none", "zero", "prev", "next"]:
            self.fill_combo.addItem(tr(label), label)
        self._set_combo_by_data(self.fill_combo, "prev")

        self._lbl_fill, self._fill_info = add_single_row(
            tr("Fill empty:"), self.fill_combo,
            "controls.preprocessing.fill"
        )

        # --- Aggregation (single control spans full width)
        self.agg_combo = QComboBox()
        for label in ["avg", "min", "max", "first", "last", "median"]:
            self.agg_combo.addItem(tr(label), label)
        self._set_combo_by_data(self.agg_combo, "avg")

        self._lbl_agg, self._agg_info = add_single_row(
            tr("Aggregation:"), self.agg_combo,
            "controls.preprocessing.aggregation"
        )

        self.bodyLayout().addLayout(grid)

        # --- Debounce + signals (keep your existing logic)
        self._debounce = QTimer(self)
        self._debounce.setInterval(250)
        self._debounce.setSingleShot(True)
        self._pending: tuple[str, object] | None = None
        self._debounce.timeout.connect(self._flush_debounce)

        self.timestep_combo.currentIndexChanged.connect(self._on_timestep_combo_changed)
        self.timestep_edit.textEdited.connect(lambda text: self._queue_emit("timestep", text))

        self.moving_average_combo.currentIndexChanged.connect(self._on_moving_average_combo_changed)
        self.moving_average_edit.textEdited.connect(lambda text: self._queue_emit("moving_average", text))

        self.fill_combo.currentIndexChanged.connect(
            lambda idx: self._emit_now("fill", self.fill_combo.itemData(idx))
        )
        self.agg_combo.currentIndexChanged.connect(
            lambda idx: self._emit_now("agg", self.agg_combo.itemData(idx))
        )

    # ------------------------------------------------------------------
    def _make_label(self, text: str) -> "QLabel":
        from PySide6.QtWidgets import QLabel

        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    def _make_info(self, help_key: str | None) -> QWidget | None:
        if help_key and self._help_viewmodel is not None:
            return InfoButton(help_key, self._help_viewmodel)
        return None

    def _resolve_help_viewmodel(self) -> Optional[HelpViewModel]:
        try:
            return get_help_viewmodel()
        except Exception:
            return None

    def _on_timestep_combo_changed(self) -> None:
        """Handle timestep combo box selection - populate edit field with seconds."""
        selected = str(self.timestep_combo.currentData() or "").strip().lower()

        if selected in ("auto", "none"):
            # Clear the edit field
            was = self.timestep_edit.blockSignals(True)
            self.timestep_edit.clear()
            self.timestep_edit.blockSignals(was)
        else:
            seconds = label_to_seconds(str(selected), TIMESTEP_OPTIONS)
            if seconds is not None:
                self.timestep_edit.setText(str(seconds))
        self._queue_emit("timestep", self.parameters().get("timestep"))

    def _on_moving_average_combo_changed(self) -> None:
        """Handle moving average combo box selection - populate edit field with seconds."""
        selected = str(self.moving_average_combo.currentData() or "").strip().lower()

        if selected in ("auto", "none"):
            was = self.moving_average_edit.blockSignals(True)
            self.moving_average_edit.clear()
            self.moving_average_edit.blockSignals(was)
        else:
            seconds = label_to_seconds(str(selected), MOVING_AVERAGE_OPTIONS)
            if seconds is not None:
                self.moving_average_edit.setText(str(seconds))
        self._queue_emit("moving_average", self.parameters().get("moving_average"))

    def _queue_emit(self, key: str, value: object) -> None:
        self._pending = (key, value)
        self._debounce.start()

    def _emit_now(self, key: str, value: object) -> None:
        self.parameter_changed.emit(key, value)
        self.parameters_changed.emit(self.parameters())

    def _flush_debounce(self) -> None:
        if not self._pending:
            return
        key, value = self._pending
        self._pending = None
        self._emit_now(key, value)

    # ------------------------------------------------------------------
    def parameters(self) -> dict[str, Any]:
        """Return the preprocessing parameters in the format HybridPandasModel expects."""

        timestep_mode = str(self.timestep_combo.currentData() or "auto").strip().lower()
        timestep_text = (self.timestep_edit.text() or "").strip()
        timestep: str | int | None = "auto"
        if timestep_text:
            timestep = int(timestep_text) if timestep_text.isdigit() else timestep_text
        elif timestep_mode in ("auto", "none"):
            timestep = timestep_mode

        fill = (self.fill_combo.currentData() or "none")
        fill = str(fill).strip().lower()

        ma_mode = str(self.moving_average_combo.currentData() or "none").strip().lower()
        ma_text = (self.moving_average_edit.text() or "").strip()
        moving_average: Any = None
        if ma_text:
            moving_average = int(ma_text) if ma_text.isdigit() else ma_text
        elif ma_mode == "auto":
            moving_average = "auto"
        elif ma_mode == "none":
            moving_average = None

        agg = (self.agg_combo.currentData() or "avg")
        agg = str(agg).strip().lower()

        return dict(
            timestep=timestep,
            fill=fill,
            moving_average=moving_average,
            agg=agg,
        )

    # ------------------------------------------------------------------
    def set_parameters(self, params: dict[str, Any]) -> None:
        params = params or {}

        # Set timestep mode/interval to matching value if found.
        timestep_val = params.get("timestep")
        if timestep_val is None or str(timestep_val).strip().lower() == "auto":
            was_combo = self.timestep_combo.blockSignals(True)
            self._set_combo_by_data(self.timestep_combo, "auto")
            self.timestep_combo.blockSignals(was_combo)
            was_edit = self.timestep_edit.blockSignals(True)
            self.timestep_edit.clear()
            self.timestep_edit.blockSignals(was_edit)
        elif str(timestep_val).strip().lower() == "none":
            was_combo = self.timestep_combo.blockSignals(True)
            self._set_combo_by_data(self.timestep_combo, "none")
            self.timestep_combo.blockSignals(was_combo)
            was_edit = self.timestep_edit.blockSignals(True)
            self.timestep_edit.clear()
            self.timestep_edit.blockSignals(was_edit)
        else:
            seconds = None
            if isinstance(timestep_val, int):
                seconds = timestep_val
            elif isinstance(timestep_val, str) and timestep_val.isdigit():
                seconds = int(timestep_val)
            
            if seconds is not None:
                label = seconds_to_label(seconds, TIMESTEP_OPTIONS)
                if label is not None:
                    was = self.timestep_combo.blockSignals(True)
                    self._set_combo_by_data(self.timestep_combo, label)
                    self.timestep_combo.blockSignals(was)
                    was = self.timestep_edit.blockSignals(True)
                    self.timestep_edit.setText(str(seconds))
                    self.timestep_edit.blockSignals(was)
                else:
                    # Set custom value in edit
                    was = self.timestep_edit.blockSignals(True)
                    self.timestep_edit.setText(str(seconds))
                    self.timestep_edit.blockSignals(was)
                    was = self.timestep_combo.blockSignals(True)
                    self._set_combo_by_data(self.timestep_combo, "auto")
                    self.timestep_combo.blockSignals(was)
            else:
                was = self.timestep_edit.blockSignals(True)
                self.timestep_edit.setText(str(timestep_val))
                self.timestep_edit.blockSignals(was)
                was = self.timestep_combo.blockSignals(True)
                self._set_combo_by_data(self.timestep_combo, "auto")
                self.timestep_combo.blockSignals(was)

        # Set moving average mode/interval to matching value if found.
        ma_val = params.get("moving_average")
        if ma_val is None or str(ma_val).strip().lower() in {"", "auto", "none"}:
            combo_value = "auto" if str(ma_val).strip().lower() == "auto" else "none"
            was_combo = self.moving_average_combo.blockSignals(True)
            self._set_combo_by_data(self.moving_average_combo, combo_value)
            self.moving_average_combo.blockSignals(was_combo)
            was_edit = self.moving_average_edit.blockSignals(True)
            self.moving_average_edit.clear()
            self.moving_average_edit.blockSignals(was_edit)
        else:
            seconds = None
            if isinstance(ma_val, int):
                seconds = ma_val
            elif isinstance(ma_val, str) and ma_val.isdigit():
                seconds = int(ma_val)
            
            if seconds is not None:
                label = seconds_to_label(seconds, MOVING_AVERAGE_OPTIONS)
                if label is not None:
                    was = self.moving_average_combo.blockSignals(True)
                    self._set_combo_by_data(self.moving_average_combo, label)
                    self.moving_average_combo.blockSignals(was)
                    was = self.moving_average_edit.blockSignals(True)
                    self.moving_average_edit.setText(str(seconds))
                    self.moving_average_edit.blockSignals(was)
                else:
                    # Set custom value in edit
                    was = self.moving_average_edit.blockSignals(True)
                    self.moving_average_edit.setText(str(seconds))
                    self.moving_average_edit.blockSignals(was)
                    was = self.moving_average_combo.blockSignals(True)
                    self._set_combo_by_data(self.moving_average_combo, "none")
                    self.moving_average_combo.blockSignals(was)
            else:
                was = self.moving_average_edit.blockSignals(True)
                self.moving_average_edit.setText(str(ma_val))
                self.moving_average_edit.blockSignals(was)
                was = self.moving_average_combo.blockSignals(True)
                self._set_combo_by_data(self.moving_average_combo, "auto")
                self.moving_average_combo.blockSignals(was)

        self._set_combo_value(self.fill_combo, params.get("fill"))
        self._set_combo_value(self.agg_combo, params.get("agg"))

        self.parameters_changed.emit(self.parameters())

    def _set_combo_value(self, combo: QComboBox, value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        target = text.lower()
        for idx in range(combo.count()):
            item_data = combo.itemData(idx)
            if item_data is None:
                continue
            if str(item_data).lower() == target:
                was = combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(was)
                return

    def _set_combo_by_data(self, combo: QComboBox, value: str) -> None:
        if value is None:
            return
        target = str(value)
        for idx in range(combo.count()):
            if combo.itemData(idx) == target:
                combo.setCurrentIndex(idx)
                return

