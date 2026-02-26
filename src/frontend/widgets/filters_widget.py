
from __future__ import annotations
from typing import Any, Iterable, Optional

import pandas as pd
from PySide6.QtCore import QDateTime, Qt, Signal, QObject
from PySide6.QtWidgets import (


    QDateTimeEdit,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QWidget,
)

import logging
logger = logging.getLogger(__name__)
from ..localization import tr

from .multi_check_combo import MultiCheckCombo

from .collapsible_section import CollapsibleSection

from ..models.database_model import DatabaseModel
from ..threading.runner import run_in_thread
from ..threading.utils import run_in_main_thread
from ..utils import toast_error, toast_success
from .help_widgets import InfoButton
from ..viewmodels.help_viewmodel import HelpViewModel, get_help_viewmodel



class FiltersWidgetViewModel(QObject):
    """Small helper that keeps filter options in sync with a database model."""

    systems_updated = Signal(list)
    datasets_updated = Signal(list)
    imports_updated = Signal(list)
    filters_refreshed = Signal()
    groups_updated = Signal(object)
    tags_updated = Signal(list)
    group_remove_finished = Signal(object)
    group_remove_failed = Signal(str)

    def __init__(
        self,
        model: Optional[DatabaseModel] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._model: Optional[DatabaseModel] = model
        self._remove_running = False
        if self._model is not None:
            self._model.database_changed.connect(self._on_database_changed)
            self._model.features_list_changed.connect(self._on_features_changed)
            self._model.groups_changed.connect(self._on_groups_changed)

    def refresh_filters(self) -> None:
        model = self._model

        def _load():
            systems: Iterable[str] = []
            datasets: Iterable[str] = []
            groups = pd.DataFrame(columns=["group_id", "kind", "label"])
            tags: Iterable[str] = []
            imports: Iterable[tuple[str, int]] = []
            if model is not None:
                try:
                    systems = model.list_systems()
                except Exception:
                    systems = []
                try:
                    datasets = model.list_datasets()
                except Exception:
                    datasets = []
                try:
                    groups = model.groups_df()
                except Exception:
                    groups = pd.DataFrame(columns=["group_id", "kind", "label"])
                try:
                    tags = model.list_feature_tags()
                except Exception:
                    tags = []
                try:
                    imports = model.list_imports()
                except Exception:
                    imports = []
            payload = {
                "systems": [(str(name), str(name)) for name in systems],
                "datasets": [(str(name), str(name)) for name in datasets],
                "groups": groups,
                "tags": [(str(tag), str(tag)) for tag in tags],
                "imports": [(str(label), int(import_id)) for label, import_id in imports],
            }
            return payload

        def _apply(payload) -> None:
            self.systems_updated.emit(payload["systems"])
            self.datasets_updated.emit(payload["datasets"])
            self.groups_updated.emit(payload["groups"])
            self.tags_updated.emit(payload["tags"])
            self.imports_updated.emit(payload["imports"])
            self.filters_refreshed.emit()

        run_in_thread(
            _load,
            on_result=lambda payload: run_in_main_thread(_apply, payload),
            owner=self,
            key="filters_load",
            cancel_previous=True,
        )

    def _on_database_changed(self, *_args) -> None:
        self.refresh_filters()

    def _on_features_changed(self) -> None:
        """Refresh filters when features change (tags may have been added/modified)."""
        self.refresh_filters()

    def _on_groups_changed(self) -> None:
        self._emit_groups()

    def _emit_systems(self) -> None:
        systems: Iterable[str] = []
        if self._model is not None:
            try:
                systems = self._model.list_systems()
            except Exception:
                systems = []
        items = [(str(name), str(name)) for name in systems]
        self.systems_updated.emit(items)

    def _emit_datasets(self) -> None:
        datasets: Iterable[str] = []
        if self._model is not None:
            try:
                datasets = self._model.list_datasets()
            except Exception:
                datasets = []
        items = [(str(name), str(name)) for name in datasets]
        self.datasets_updated.emit(items)

    def _emit_groups(self) -> None:
        df = pd.DataFrame(columns=["group_id", "kind", "label"])
        if self._model is not None:
            try:
                df = self._model.groups_df()
            except Exception:
                df = pd.DataFrame(columns=["group_id", "kind", "label"])
        self.groups_updated.emit(df)

    def _emit_tags(self) -> None:
        tags: Iterable[str] = []
        if self._model is not None:
            try:
                tags = self._model.list_feature_tags()
            except Exception:
                tags = []
        items = [(str(tag), str(tag)) for tag in tags]
        self.tags_updated.emit(items)

    def remove_group(self, group_id: int) -> None:
        if self._remove_running:
            return
        model = self._model
        if model is None:
            self.group_remove_failed.emit("Database is not available.")
            return
        self._remove_running = True
        run_in_thread(
            self._remove_group_sync,
            on_result=lambda payload: run_in_main_thread(self._on_group_remove_finished, payload),
            on_error=lambda message: run_in_main_thread(self._on_group_remove_failed, message),
            owner=self,
            key="filters_remove_group",
            group_id=int(group_id),
            model=model,
        )

    @staticmethod
    def _remove_group_sync(*, group_id: int, model: DatabaseModel) -> dict:
        return model.remove_group_by_id(int(group_id))

    def _on_group_remove_finished(self, payload: object) -> None:
        self._remove_running = False
        self.group_remove_finished.emit(payload if isinstance(payload, dict) else {})
        self.refresh_filters()

    def _on_group_remove_failed(self, message: str) -> None:
        self._remove_running = False
        self.group_remove_failed.emit(message)

    def refresh_imports(self, systems: Optional[list[str]], datasets: Optional[list[str]]) -> None:
        model = self._model

        def _load() -> list[tuple[str, int]]:
            if model is None:
                return []
            selected_systems = [str(s).strip() for s in (systems or []) if str(s).strip()]
            selected_datasets = [str(d).strip() for d in (datasets or []) if str(d).strip()]
            selected_system = selected_systems[0] if len(selected_systems) == 1 else None
            selected_dataset = selected_datasets[0] if len(selected_datasets) == 1 else None
            try:
                return list(
                    model.list_imports(
                        system=selected_system,
                        dataset=selected_dataset,
                        datasets=selected_datasets if selected_datasets else None,
                    )
                    or []
                )
            except Exception:
                return []

        run_in_thread(
            _load,
            on_result=lambda items: run_in_main_thread(self.imports_updated.emit, list(items)),
            owner=self,
            key="filters_imports_load",
            cancel_previous=True,
        )

class FiltersWidget(CollapsibleSection):
    """Reusable block that exposes the filtering controls used by multiple tabs."""

    date_range_changed = Signal()
    filters_changed = Signal()
    systems_changed = Signal()
    datasets_changed = Signal()
    imports_changed = Signal()
    filters_refreshed = Signal()
    months_changed = Signal()
    groups_changed = Signal()
    tags_changed = Signal()
    group_remove_finished = Signal(object)
    group_remove_failed = Signal(str)

    def __init__(
        self,
        title: str = "Filters",
        *,
        collapsed: bool = True,
        parent=None,
        model: Optional[DatabaseModel] = None,
        help_viewmodel: Optional[HelpViewModel] = None,
    ):
        super().__init__(tr(title), collapsed=collapsed, parent=parent)

        resolved_help = help_viewmodel
        if resolved_help is None:
            try:
                resolved_help = get_help_viewmodel()
            except Exception:
                resolved_help = None
        self._help_viewmodel = resolved_help

        grid = QGridLayout()
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # Combos for systems / datasets
        systems_label = self._make_label(tr("Systems:"))
        systems_info = self._make_info("controls.filters.systems_datasets")
        grid.addWidget(systems_label, 0, 0, Qt.AlignmentFlag.AlignRight)
        self.systems_combo = MultiCheckCombo(placeholder=tr("All systems"), summary_max=4)
        grid.addWidget(self.systems_combo, 0, 1)

        datasets_label = self._make_label(tr("Datasets:"))
        grid.addWidget(datasets_label, 0, 3, Qt.AlignmentFlag.AlignRight)
        self.datasets_combo = MultiCheckCombo(placeholder=tr("All datasets"), summary_max=4)
        grid.addWidget(self.datasets_combo, 0, 4)
        if systems_info:
            grid.addWidget(systems_info, 0, 5, Qt.AlignmentFlag.AlignLeft)

        # Date range
        from_label = self._make_label(tr("From:"))
        date_range_info = self._make_info("controls.filters.date_range")
        grid.addWidget(from_label, 1, 0, Qt.AlignmentFlag.AlignRight)
        self.dt_from = QDateTimeEdit()
        self.dt_from.setDisplayFormat("dd/MM/yyyy HH:mm")
        self.dt_from.setCalendarPopup(True)
        self.dt_from.setDateTime(QDateTime.currentDateTime().addYears(-10))
        grid.addWidget(self.dt_from, 1, 1)

        to_label = self._make_label(tr("To:"))
        grid.addWidget(to_label, 1, 3, Qt.AlignmentFlag.AlignRight)
        self.dt_to = QDateTimeEdit()
        self.dt_to.setDisplayFormat("dd/MM/yyyy HH:mm")
        self.dt_to.setCalendarPopup(True)
        self.dt_to.setDateTime(QDateTime.currentDateTime())
        grid.addWidget(self.dt_to, 1, 4)
        if date_range_info:
            grid.addWidget(date_range_info, 1, 5, Qt.AlignmentFlag.AlignLeft)

        # Months / groups row
        months_label = self._make_label(tr("Months:"))
        months_groups_info = self._make_info("controls.filters.months_groups")
        grid.addWidget(months_label, 2, 0, Qt.AlignmentFlag.AlignRight)
        self.months_combo = MultiCheckCombo(placeholder=tr("All months"), summary_max=4)
        grid.addWidget(self.months_combo, 2, 1)

        groups_label = self._make_label(tr("Groups:"))
        grid.addWidget(groups_label, 2, 3, Qt.AlignmentFlag.AlignRight)
        self.group_combo = MultiCheckCombo(placeholder=tr("All groups"), summary_max=4)
        self.group_combo.set_context_actions([("remove_group", tr("Remove group"))])
        grid.addWidget(self.group_combo, 2, 4)
        if months_groups_info:
            grid.addWidget(months_groups_info, 2, 5, Qt.AlignmentFlag.AlignLeft)

        imports_tags_info = self._make_info("controls.filters.imports_tags")

        imports_label = self._make_label(tr("Imports:"))
        grid.addWidget(imports_label, 3, 0, Qt.AlignmentFlag.AlignRight)
        self.imports_combo = MultiCheckCombo(placeholder=tr("All imports"), summary_max=2)
        grid.addWidget(self.imports_combo, 3, 1)

        tags_label = self._make_label(tr("Tags:"))
        grid.addWidget(tags_label, 3, 3, Qt.AlignmentFlag.AlignRight)
        self.tags_combo = MultiCheckCombo(placeholder=tr("All tags"), summary_max=4)
        grid.addWidget(self.tags_combo, 3, 4)
        if imports_tags_info:
            grid.addWidget(imports_tags_info, 3, 5, Qt.AlignmentFlag.AlignLeft)

        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 0)
        grid.setColumnStretch(3, 0)
        grid.setColumnStretch(4, 1)
        grid.setColumnStretch(5, 0)

        # Month entries
        months = [
            (tr("01 / Jan"), 1), (tr("02 / Feb"), 2), (tr("03 / Mar"), 3), (tr("04 / Apr"), 4),
            (tr("05 / May"), 5), (tr("06 / Jun"), 6), (tr("07 / Jul"), 7), (tr("08 / Aug"), 8),
            (tr("09 / Sep"), 9), (tr("10 / Oct"), 10), (tr("11 / Nov"), 11), (tr("12 / Dec"), 12),
        ]
        self.months_combo.set_items(months, check_all=True)

        # Signals
        self.dt_from.dateTimeChanged.connect(lambda _dt: self.date_range_changed.emit())
        self.dt_to.dateTimeChanged.connect(lambda _dt: self.date_range_changed.emit())
        self.systems_combo.selection_changed.connect(self.systems_changed.emit)
        self.datasets_combo.selection_changed.connect(self.datasets_changed.emit)
        self.imports_combo.selection_changed.connect(self.imports_changed.emit)
        self.months_combo.selection_changed.connect(self.months_changed.emit)
        self.group_combo.selection_changed.connect(self.groups_changed.emit)
        self.tags_combo.selection_changed.connect(self.tags_changed.emit)
        self.group_combo.context_action_triggered.connect(self._on_group_context_action)
        self.systems_combo.selection_changed.connect(lambda: self._refresh_imports_for_selection())
        self.datasets_combo.selection_changed.connect(lambda: self._refresh_imports_for_selection())

        for signal in (
            self.date_range_changed,
            self.systems_changed,
            self.datasets_changed,
            self.imports_changed,
            self.months_changed,
            self.groups_changed,
            self.tags_changed,
        ):
            signal.connect(self.filters_changed.emit)

        self.bodyLayout().addLayout(grid)

        self._filters_view_model = FiltersWidgetViewModel(model=model, parent=self)
        self._filters_view_model.systems_updated.connect(
            lambda items: self.set_systems(items, check_all=True)
        )
        self._filters_view_model.datasets_updated.connect(
            lambda items: self.set_datasets(items, check_all=True)
        )
        self._filters_view_model.imports_updated.connect(
            lambda items: self.set_imports(items, check_all=True)
        )
        self._filters_view_model.groups_updated.connect(self.set_groups)
        self._filters_view_model.tags_updated.connect(
            lambda items: self.set_tags(items, check_all=False)
        )
        self._filters_view_model.filters_refreshed.connect(self.filters_refreshed.emit)
        self._filters_view_model.group_remove_finished.connect(self._on_group_remove_finished)
        self._filters_view_model.group_remove_failed.connect(self._on_group_remove_failed)

    def _on_group_context_action(self, action_key: str, payload: object) -> None:
        if str(action_key or "").strip() != "remove_group":
            return
        data = payload if isinstance(payload, dict) else {}
        try:
            group_id = int(data.get("value"))
        except Exception:
            return
        group_label = str(data.get("label") or "").strip() or str(group_id)
        title = tr("Remove group")
        body = tr(
            "This will remove group \"{group}\" and its values from the database.\n\nDo you want to continue?"
        ).format(group=group_label)
        answer = QMessageBox.question(
            self,
            title,
            body,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._filters_view_model.remove_group(group_id)

    def _on_group_remove_finished(self, payload: object) -> None:
        data = payload if isinstance(payload, dict) else {}
        label_text = str(data.get("label") or "").strip()
        points_deleted = int(data.get("group_points_deleted") or 0)
        if label_text:
            message = tr("Removed group \"{group}\" ({points} values deleted).").format(
                group=label_text,
                points=points_deleted,
            )
        else:
            message = tr("Removed group ({points} values deleted).").format(points=points_deleted)
        try:
            toast_success(message, title=tr("Group removed"))
        except Exception:
            logger.warning("Exception in _on_group_remove_finished", exc_info=True)
        self.group_remove_finished.emit(data)

    def _on_group_remove_failed(self, message: str) -> None:
        text = str(message or "").strip() or tr("Failed to remove group.")
        try:
            toast_error(text, title=tr("Remove group"))
        except Exception:
            logger.warning("Exception in _on_group_remove_failed", exc_info=True)
        self.group_remove_failed.emit(text)

    # ------------------------------------------------------------------
    def set_systems(self, items: list[tuple[str, Any]], *, check_all: bool = True) -> None:
        self.systems_combo.set_items(items, check_all=check_all)

    def set_datasets(self, items: list[tuple[str, Any]], *, check_all: bool = True) -> None:
        self.datasets_combo.set_items(items, check_all=check_all)

    def set_imports(self, items: list[tuple[str, Any]], *, check_all: bool = True) -> None:
        self.imports_combo.set_items(items, check_all=check_all)

    def set_groups(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            self.group_combo.clear_items()
            return
        items = [(f"{row['kind']}: {row['label']}", int(row['group_id'])) for _, row in df.iterrows()]
        self.group_combo.set_items(items, check_all=False)

    def set_tags(self, items: list[tuple[str, Any]], *, check_all: bool = True) -> None:
        self.tags_combo.set_items(items, check_all=check_all)

    def refresh_filters(self) -> None:
        self._filters_view_model.refresh_filters()
        self._refresh_imports_for_selection()

    def _refresh_imports_for_selection(self) -> None:
        self._filters_view_model.refresh_imports(
            self.selected_systems(),
            self.selected_datasets(),
        )

    # ------------------------------------------------------------------
    def selected_systems(self) -> list[str]:
        return [str(v) for v in self.systems_combo.selected_values()]

    def selected_datasets(self) -> list[str]:
        return [str(v) for v in self.datasets_combo.selected_values()]

    def selected_import_ids(self) -> list[int]:
        return [int(v) for v in self.imports_combo.selected_values()]

    def selected_months(self) -> list[int]:
        return [int(v) for v in self.months_combo.selected_values()]

    def selected_group_ids(self) -> list[int]:
        return [int(v) for v in self.group_combo.selected_values()]

    def selected_tags(self) -> list[str]:
        return [str(v) for v in self.tags_combo.selected_values()]

    def filter_state(self) -> dict:
        start = self.start_timestamp()
        end = self.end_timestamp()
        return {
            "start": start.isoformat() if start is not None else None,
            "end": end.isoformat() if end is not None else None,
            "systems": self.selected_systems(),
            "Datasets": self.selected_datasets(),
            "datasets": self.selected_datasets(),
            "import_ids": self.selected_import_ids(),
            "months": self.selected_months(),
            "group_ids": self.selected_group_ids(),
            "tags": self.selected_tags(),
        }

    def apply_filter_state(self, state: dict | None) -> None:
        state = dict(state or {})
        def _set_qdt(widget: QDateTimeEdit, value: str | None):
            if not value:
                return
            try:
                ts = pd.Timestamp(value)
            except Exception:
                return
            qdt = QDateTime(
                ts.year, ts.month, ts.day,
                ts.hour, ts.minute, ts.second,
                int(ts.microsecond / 1000),
            )
            was = widget.blockSignals(True)
            widget.setDateTime(qdt)
            widget.blockSignals(was)

        _set_qdt(self.dt_from, state.get("start"))
        _set_qdt(self.dt_to, state.get("end"))

        systems = state.get("systems") or []
        self.systems_combo.set_selected_values(systems)
        datasets = state.get("datasets") or state.get("Datasets") or []
        self.datasets_combo.set_selected_values(datasets)
        import_ids = state.get("import_ids") or []
        self.imports_combo.set_selected_values(import_ids)
        months = state.get("months") or []
        self.months_combo.set_selected_values(months)
        groups = state.get("group_ids") or []
        self.group_combo.set_selected_values(groups)
        tags = state.get("tags") or []
        self.tags_combo.set_selected_values(tags)

    def get_settings(self) -> dict:
        return self.filter_state()

    def set_settings(self, settings: dict | None) -> None:
        widgets = [
            self,
            self.dt_from,
            self.dt_to,
            self.systems_combo,
            self.datasets_combo,
            self.imports_combo,
            self.months_combo,
            self.group_combo,
            self.tags_combo,
        ]
        previous_states = [(widget, widget.blockSignals(True)) for widget in widgets]
        try:
            self.apply_filter_state(settings or {})
        finally:
            for widget, was in reversed(previous_states):
                widget.blockSignals(was)

    # ------------------------------------------------------------------
    def start_timestamp(self) -> pd.Timestamp | None:
        return _parse_qdatetime(self.dt_from.dateTime())

    def end_timestamp(self) -> pd.Timestamp | None:
        return _parse_qdatetime(self.dt_to.dateTime())

    def _make_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def _make_info(self, help_key: str | None) -> QWidget | None:
        if help_key and self._help_viewmodel is not None:
            return InfoButton(help_key, self._help_viewmodel)
        return None


def _parse_qdatetime(qdt: QDateTime) -> pd.Timestamp | None:
    if not qdt or not qdt.isValid():
        return None
    date = qdt.date()
    time = qdt.time()
    try:
        return pd.Timestamp(
            year=date.year(),
            month=date.month(),
            day=date.day(),
            hour=time.hour(),
            minute=time.minute(),
            second=time.second(),
            microsecond=time.msec() * 1000,
        )
    except Exception:
        return None
