
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

NO_GROUP_FILTER_VALUE = -1
NO_TAG_FILTER_VALUE = "__NO_TAG__"



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

    def refresh_filters(
        self,
        *,
        systems: Optional[list[str]] = None,
        datasets: Optional[list[str]] = None,
        import_ids: Optional[list[int]] = None,
    ) -> None:
        model = self._model

        systems_scope = (
            [str(item).strip() for item in (systems or []) if str(item).strip()]
            if systems is not None
            else None
        )
        datasets_scope = (
            [str(item).strip() for item in (datasets or []) if str(item).strip()]
            if datasets is not None
            else None
        )
        import_ids_scope = (
            [int(item) for item in (import_ids or []) if item is not None]
            if import_ids is not None
            else None
        )

        def _load():
            systems_items: Iterable[str] = []
            datasets_items: list[tuple[str, str]] = []
            groups = pd.DataFrame(columns=["group_id", "kind", "label"])
            tags: Iterable[str] = []
            imports_items: list[tuple[str, int]] = []

            if model is not None:
                try:
                    systems_items = model.list_systems()
                except Exception:
                    systems_items = []

                try:
                    datasets_items = self.datasets_for_systems(systems_scope)
                except Exception:
                    datasets_items = []

                try:
                    groups = model.groups_df(
                        systems=systems_scope,
                        datasets=datasets_scope,
                        import_ids=import_ids_scope,
                    )
                except Exception:
                    groups = pd.DataFrame(columns=["group_id", "kind", "label"])

                try:
                    tags = model.list_feature_tags(
                        systems=systems_scope,
                        datasets=datasets_scope,
                        import_ids=import_ids_scope,
                    )
                except Exception:
                    tags = []

                try:
                    imports_items = self.imports_for_filters(
                        systems_scope,
                        datasets_scope,
                    )
                except Exception:
                    imports_items = []

            return {
                "systems": [(str(name), str(name)) for name in systems_items],
                "datasets": list(datasets_items),
                "groups": groups,
                "tags": [(str(tag), str(tag)) for tag in tags],
                "imports": list(imports_items),
            }

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
        self.datasets_updated.emit(self.datasets_for_systems(None))

    def _emit_groups(
        self,
        *,
        systems: Optional[list[str]] = None,
        datasets: Optional[list[str]] = None,
        import_ids: Optional[list[int]] = None,
    ) -> None:
        df = pd.DataFrame(columns=["group_id", "kind", "label"])
        if self._model is not None:
            try:
                df = self._model.groups_df(
                    systems=systems,
                    datasets=datasets,
                    import_ids=import_ids,
                )
            except Exception:
                df = pd.DataFrame(columns=["group_id", "kind", "label"])
        self.groups_updated.emit(df)

    def _emit_tags(
        self,
        *,
        systems: Optional[list[str]] = None,
        datasets: Optional[list[str]] = None,
        import_ids: Optional[list[int]] = None,
    ) -> None:
        tags: Iterable[str] = []
        if self._model is not None:
            try:
                tags = self._model.list_feature_tags(
                    systems=systems,
                    datasets=datasets,
                    import_ids=import_ids,
                )
            except Exception:
                tags = []
        items = [(str(tag), str(tag)) for tag in tags]
        self.tags_updated.emit(items)

    # @ai(gpt-5, codex, feature, 2026-03-11)
    def refresh_groups_and_tags_sync(
        self,
        *,
        systems: Optional[list[str]],
        datasets: Optional[list[str]],
        import_ids: Optional[list[int]],
        on_result=None,
    ) -> None:
        model = self._model
        groups = pd.DataFrame(columns=["group_id", "kind", "label"])
        tags: Iterable[str] = []
        if model is not None:
            try:
                groups = model.groups_df(
                    systems=systems,
                    datasets=datasets,
                    import_ids=import_ids,
                )
            except Exception:
                groups = pd.DataFrame(columns=["group_id", "kind", "label"])
            try:
                tags = model.list_feature_tags(
                    systems=systems,
                    datasets=datasets,
                    import_ids=import_ids,
                )
            except Exception:
                tags = []
        payload = {
            "groups": groups,
            "tags": [(str(tag), str(tag)) for tag in tags],
        }
        if callable(on_result):
            on_result(payload)
            return
        self.groups_updated.emit(payload["groups"])
        self.tags_updated.emit(payload["tags"])

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
        self.refresh_imports_sync(systems, datasets)

    def datasets_for_systems(
        self,
        systems: Optional[list[str]],
    ) -> list[tuple[str, str]]:
        model = self._model
        if model is None:
            return []
        if systems is not None:
            selected_systems = [str(s).strip() for s in systems if str(s).strip()]
            if not selected_systems:
                return []
        else:
            selected_systems = []
        dataset_entries: list[tuple[str, str]] = []
        list_for_filters = getattr(model, "list_datasets_for_filters", None)
        if callable(list_for_filters):
            try:
                raw_entries = list_for_filters(
                    systems=selected_systems if systems is not None else None
                )
            except Exception:
                raw_entries = []
            seen_pairs: set[tuple[str, str]] = set()
            for entry in raw_entries or []:
                if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                    continue
                dataset_name = str(entry[0] or "").strip()
                system_name = str(entry[1] or "").strip()
                if not dataset_name:
                    continue
                key = (dataset_name, system_name)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                dataset_entries.append(key)
        if not dataset_entries:
            try:
                items = model.list_datasets(systems=selected_systems if systems is not None else None)
            except Exception:
                items = []
            dataset_entries = [(str(name).strip(), "") for name in items if str(name).strip()]

        name_counts: dict[str, int] = {}
        for dataset_name, _system_name in dataset_entries:
            name_counts[dataset_name] = name_counts.get(dataset_name, 0) + 1

        out: list[tuple[str, str]] = []
        for dataset_name, system_name in dataset_entries:
            label = dataset_name
            if name_counts.get(dataset_name, 0) > 1 and system_name:
                label = f"{dataset_name} ({system_name})"
            out.append((label, dataset_name))
        return out

    def imports_for_filters(
        self,
        systems: Optional[list[str]],
        datasets: Optional[list[str]],
    ) -> list[tuple[str, int]]:
        model = self._model
        if model is None:
            return []
        if systems is not None:
            selected_systems = [str(s).strip() for s in systems if str(s).strip()]
            if not selected_systems:
                return []
        else:
            selected_systems = []
        if datasets is not None:
            selected_datasets = [str(d).strip() for d in datasets if str(d).strip()]
            if not selected_datasets:
                return []
        else:
            selected_datasets = []
        selected_system = selected_systems[0] if len(selected_systems) == 1 else None
        selected_dataset = selected_datasets[0] if len(selected_datasets) == 1 else None
        try:
            return list(
                model.list_imports(
                    system=selected_system,
                    dataset=selected_dataset,
                    datasets=selected_datasets if datasets is not None else None,
                    systems=selected_systems if systems is not None else None,
                )
                or []
            )
        except Exception:
            return []

    def refresh_datasets_sync(
        self,
        systems: Optional[list[str]],
        *,
        on_result=None,
    ) -> None:
        items = self.datasets_for_systems(systems)
        if callable(on_result):
            on_result(list(items))
            return
        self.datasets_updated.emit(list(items))

    def refresh_imports_sync(
        self,
        systems: Optional[list[str]],
        datasets: Optional[list[str]],
        *,
        on_result=None,
    ) -> None:
        items = self.imports_for_filters(systems, datasets)
        if callable(on_result):
            on_result(list(items))
            return
        self.imports_updated.emit(list(items))

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
        self._filters_batch_depth = 0
        self._filters_changed_pending = False
        self._pending_signal_names: set[str] = set()
        self._last_emitted_filter_signal_key: tuple | None = None
        self._dependency_refresh_token = 0
        self._dependency_refresh_active = False

        grid = QGridLayout()
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # Combos for systems / datasets
        systems_label = self._make_label(tr("Systems:"))
        systems_info = self._make_info("controls.filters.systems_datasets")
        grid.addWidget(systems_label, 0, 0, Qt.AlignmentFlag.AlignRight)
        self.systems_combo = MultiCheckCombo(placeholder=tr("All systems"), summary_max=4)
        self.systems_combo.set_empty_selection_means_all(False)
        self.systems_combo.set_preserve_missing_selected_values(True)
        grid.addWidget(self.systems_combo, 0, 1)

        datasets_label = self._make_label(tr("Datasets:"))
        grid.addWidget(datasets_label, 0, 3, Qt.AlignmentFlag.AlignRight)
        self.datasets_combo = MultiCheckCombo(placeholder=tr("All datasets"), summary_max=4)
        self.datasets_combo.set_empty_selection_means_all(False)
        self.datasets_combo.set_preserve_missing_selected_values(True)
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
        self.imports_combo.set_empty_selection_means_all(False)
        self.imports_combo.set_preserve_missing_selected_values(True)
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
        self.dt_from.dateTimeChanged.connect(lambda _dt: self._emit_filter_change("date_range_changed"))
        self.dt_to.dateTimeChanged.connect(lambda _dt: self._emit_filter_change("date_range_changed"))
        self.systems_combo.selection_changed.connect(self._on_systems_selection_changed)
        self.datasets_combo.selection_changed.connect(self._on_datasets_selection_changed)
        self.imports_combo.selection_changed.connect(self._on_imports_selection_changed)
        self.months_combo.selection_changed.connect(lambda: self._emit_filter_change("months_changed"))
        self.group_combo.selection_changed.connect(lambda: self._emit_filter_change("groups_changed"))
        self.tags_combo.selection_changed.connect(lambda: self._emit_filter_change("tags_changed"))
        self.group_combo.context_action_triggered.connect(self._on_group_context_action)

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
        if group_id <= 0:
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
        self._set_combo_items_preserving_selection(
            self.systems_combo,
            items,
            check_all=check_all,
        )

    def set_datasets(self, items: list[tuple[str, Any]], *, check_all: bool = True) -> None:
        self._set_combo_items_preserving_selection(
            self.datasets_combo,
            items,
            check_all=check_all,
            empty_placeholder="No datasets available",
            default_placeholder="All datasets",
        )

    def set_imports(self, items: list[tuple[str, Any]], *, check_all: bool = True) -> None:
        self._set_combo_items_preserving_selection(
            self.imports_combo,
            items,
            check_all=check_all,
            empty_placeholder="No imports available",
            default_placeholder="All imports",
        )

    def set_groups(self, df: pd.DataFrame) -> None:
        was = self.group_combo.blockSignals(True)
        try:
            items: list[tuple[str, Any]] = [(tr("No group"), NO_GROUP_FILTER_VALUE)]
            if df is not None and not df.empty:
                items.extend(
                    (f"{row['kind']}: {row['label']}", int(row["group_id"]))
                    for _, row in df.iterrows()
                )
            self.group_combo.set_items(items, check_all=False)
        finally:
            self.group_combo.blockSignals(was)

    def set_tags(self, items: list[tuple[str, Any]], *, check_all: bool = True) -> None:
        was = self.tags_combo.blockSignals(True)
        try:
            combined: list[tuple[str, Any]] = [(tr("No tag"), NO_TAG_FILTER_VALUE)]
            combined.extend(items or [])
            self.tags_combo.set_placeholder(
                "No tags available" if not (items or []) else "All tags"
            )
            self.tags_combo.set_items(combined, check_all=check_all)
        finally:
            self.tags_combo.blockSignals(was)

    def _all_combo_values(self, combo: MultiCheckCombo) -> list[Any]:
        values: list[Any] = []
        model = combo.model()
        if model is None:
            return values
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None:
                continue
            if item.data(MultiCheckCombo._ACTION_ROLE) == MultiCheckCombo._ACTION_TOGGLE_ALL:
                continue
            value = item.data(Qt.ItemDataRole.UserRole)
            if value is None:
                continue
            values.append(value)
        return values

    def _set_combo_placeholder_from_items(
        self,
        combo: MultiCheckCombo,
        items: Iterable[tuple[str, Any]] | None,
        *,
        empty_placeholder: str,
        default_placeholder: str,
    ) -> None:
        normalized_items = list(items or [])
        has_real_items = bool(normalized_items)
        combo.set_placeholder(empty_placeholder if not has_real_items else default_placeholder)

    @staticmethod
    def _is_full_visible_selection(
        selected_values: Iterable[Any] | None,
        visible_values: Iterable[Any] | None,
    ) -> bool:
        selected_set = set(selected_values or [])
        visible_set = set(visible_values or [])
        return bool(visible_set) and selected_set == visible_set

    # @ai(gpt-5, codex-cli, fix, 2026-03-12)
    def _set_combo_items_preserving_selection(
        self,
        combo: MultiCheckCombo,
        items: Iterable[tuple[str, Any]] | None,
        *,
        check_all: bool,
        empty_placeholder: str | None = None,
        default_placeholder: str | None = None,
    ) -> None:
        normalized_items = list(items or [])
        previous_visible_values = self._all_combo_values(combo)
        previous_selected_values = combo.selected_values()
        previous_remembered_values = combo.remembered_selected_values()
        had_full_selection = self._is_full_visible_selection(
            previous_selected_values,
            previous_visible_values,
        )

        was = combo.blockSignals(True)
        try:
            if empty_placeholder is not None and default_placeholder is not None:
                self._set_combo_placeholder_from_items(
                    combo,
                    normalized_items,
                    empty_placeholder=empty_placeholder,
                    default_placeholder=default_placeholder,
                )

            combo.set_items(normalized_items, check_all=check_all)
            if not check_all or not normalized_items:
                return

            values = [value for _label, value in normalized_items]
            if not (had_full_selection or not combo.selected_values()):
                return

            combo.set_selected_values(values)
            if combo.preserve_missing_selected_values():
                visible_set = set(values)
                extras = [value for value in previous_remembered_values if value not in visible_set]
                combo.set_remembered_selected_values(values + extras)
        finally:
            combo.blockSignals(was)

    def _select_values_or_all(
        self,
        combo: MultiCheckCombo,
        values: Iterable[Any] | None,
    ) -> None:
        if values is None:
            combo.set_selected_values(self._all_combo_values(combo))
            return
        normalized = list(values or [])
        if not normalized and combo.empty_selection_means_all():
            combo.set_selected_values(self._all_combo_values(combo))
            return
        combo.set_selected_values(normalized)

    def _select_int_values_or_all(
        self,
        combo: MultiCheckCombo,
        values: Iterable[Any] | None,
    ) -> None:
        if values is None:
            combo.set_selected_values(self._all_combo_values(combo))
            return
        normalized: list[int] = []
        for value in values or []:
            try:
                normalized.append(int(value))
            except Exception:
                continue
        if not normalized and combo.empty_selection_means_all():
            combo.set_selected_values(self._all_combo_values(combo))
            return
        combo.set_selected_values(normalized)

    def _state_values(self, state: dict[str, Any], key: str, legacy_key: str | None = None) -> Any:
        if key in state:
            return state.get(key)
        if legacy_key and legacy_key in state:
            return state.get(legacy_key)
        return None

    def refresh_filters(self) -> None:
        self._filters_view_model.refresh_filters()

    # ------------------------------------------------------------------
    def selected_systems(self) -> list[str]:
        return [str(v) for v in self.systems_combo.selected_values()]

    def selected_datasets(self) -> list[str]:
        return self._dataset_names_from_values(self.datasets_combo.selected_values())

    def selected_import_ids(self) -> list[int]:
        return [int(v) for v in self.imports_combo.selected_values()]

    # @ai(gpt-5, codex-cli, fix, 2026-03-12)
    def available_datasets(self) -> list[str]:
        return self._dataset_names_from_values(self._all_combo_values(self.datasets_combo))

    # @ai(gpt-5, codex-cli, fix, 2026-03-12)
    def available_import_ids(self) -> list[int]:
        out: list[int] = []
        for value in self._all_combo_values(self.imports_combo):
            try:
                out.append(int(value))
            except Exception:
                continue
        return out

    # @ai(gpt-5, codex, feature, 2026-03-10)
    def selected_datasets_for_data_scope(self) -> list[str]:
        return self._dataset_names_from_values(self.datasets_combo.remembered_selected_values())

    @staticmethod
    def _dataset_names_from_values(values: Iterable[Any] | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values or []:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    # @ai(gpt-5, codex, feature, 2026-03-10)
    def selected_import_ids_for_data_scope(self) -> list[int]:
        out: list[int] = []
        for value in self.imports_combo.remembered_selected_values():
            try:
                out.append(int(value))
            except Exception:
                continue
        return out

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
            "datasets": self.selected_datasets(),
            "import_ids": self.selected_import_ids(),
            "months": self.selected_months(),
            "group_ids": self.selected_group_ids(),
            "tags": self.selected_tags(),
        }

    def apply_filter_state(self, state: dict | None) -> None:
        self._apply_filter_state_with_dependencies(dict(state or {}))

    def _apply_filter_state_with_dependencies(self, state: dict[str, Any]) -> None:
        def _set_qdt(widget: QDateTimeEdit, value: str | None) -> None:
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
            try:
                widget.setDateTime(qdt)
            finally:
                widget.blockSignals(was)

        _set_qdt(self.dt_from, state.get("start"))
        _set_qdt(self.dt_to, state.get("end"))

        wanted_systems = [
            str(v).strip()
            for v in (self._state_values(state, "systems") or [])
            if str(v).strip()
        ]
        wanted_datasets = [
            str(v).strip()
            for v in (self._state_values(state, "datasets", legacy_key="Datasets") or [])
            if str(v).strip()
        ]

        wanted_import_ids: list[int] = []
        for value in (self._state_values(state, "import_ids") or []):
            try:
                wanted_import_ids.append(int(value))
            except Exception:
                continue

        wanted_months: list[int] = []
        for value in (state.get("months") or []):
            try:
                wanted_months.append(int(value))
            except Exception:
                continue

        wanted_group_ids: list[int] = []
        for value in (state.get("group_ids") or []):
            try:
                wanted_group_ids.append(int(value))
            except Exception:
                continue

        wanted_tags = [str(v) for v in (state.get("tags") or []) if str(v).strip()]

        # 1) systems
        self._select_values_or_all(self.systems_combo, wanted_systems)

        # 2) datasets for selected systems
        dataset_items = self._filters_view_model.datasets_for_systems(self.selected_systems())
        self.set_datasets(dataset_items, check_all=True)
        self._select_values_or_all(self.datasets_combo, wanted_datasets)

        # 3) imports for selected systems + datasets
        import_items = self._filters_view_model.imports_for_filters(
            self.selected_systems(),
            self.selected_datasets(),
        )
        self.set_imports(import_items, check_all=True)
        self._select_int_values_or_all(self.imports_combo, wanted_import_ids)

        # 4) groups + tags for full scope
        self._filters_view_model.refresh_groups_and_tags_sync(
            systems=self.selected_systems(),
            datasets=self.selected_datasets(),
            import_ids=self.selected_import_ids(),
        )

        # 5) restore months
        self.months_combo.set_selected_values(wanted_months)

        # 6) restore groups that still exist
        available_group_ids: set[int] = set()
        for value in self._all_combo_values(self.group_combo):
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed > 0:
                available_group_ids.add(parsed)
        self.group_combo.set_selected_values(
            [gid for gid in wanted_group_ids if gid in available_group_ids]
        )

        # 7) restore tags that still exist
        available_tags = {str(v) for v in self._all_combo_values(self.tags_combo) if str(v).strip()}
        self.tags_combo.set_selected_values(
            [tag for tag in wanted_tags if tag in available_tags]
        )

    def get_settings(self) -> dict:
        return self.filter_state()

    def set_settings(self, settings: dict | None) -> None:
        state = dict(settings or {})

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

        self._begin_filters_batch()
        try:
            self._apply_filter_state_with_dependencies(state)
        finally:
            for widget, was in reversed(previous_states):
                widget.blockSignals(was)
            self._end_filters_batch()

        self._emit_filter_change(
            "date_range_changed",
            "systems_changed",
            "datasets_changed",
            "imports_changed",
            "months_changed",
            "groups_changed",
            "tags_changed",
        )

    def _begin_filters_batch(self) -> None:
        self._filters_batch_depth += 1

    def _end_filters_batch(self) -> None:
        if self._filters_batch_depth <= 0:
            self._filters_batch_depth = 0
            return
        self._filters_batch_depth -= 1
        if self._filters_batch_depth == 0 and self._filters_changed_pending:
            self._filters_changed_pending = False
            pending = set(self._pending_signal_names)
            self._pending_signal_names.clear()
            self._emit_filter_signals_now(pending)

    def _emit_filter_change(self, *signal_names: str) -> None:
        valid_names = {name for name in signal_names if hasattr(self, name)}
        if self._filters_batch_depth > 0:
            self._filters_changed_pending = True
            self._pending_signal_names.update(valid_names)
            return
        self._emit_filter_signals_now(valid_names)

    def _emit_filter_signals_now(self, signal_names: set[str]) -> None:
        emit_key = (
            tuple(sorted(signal_names)),
            self.filter_state().get("start"),
            self.filter_state().get("end"),
            tuple(self.selected_systems()),
            tuple(self.selected_datasets()),
            tuple(self.selected_import_ids()),
            tuple(self.selected_months()),
            tuple(self.selected_group_ids()),
            tuple(self.selected_tags()),
        )
        if emit_key == self._last_emitted_filter_signal_key:
            return
        self._last_emitted_filter_signal_key = emit_key
        for name in (
            "date_range_changed",
            "systems_changed",
            "datasets_changed",
            "imports_changed",
            "months_changed",
            "groups_changed",
            "tags_changed",
        ):
            if name in signal_names:
                getattr(self, name).emit()
        self.filters_changed.emit()

    def _replace_combo_items_preserving_selection(
        self,
        combo: MultiCheckCombo,
        items: list[tuple[str, Any]],
        *,
        check_all: bool,
        previous_values: list[Any],
        empty_placeholder: str | None = None,
        default_placeholder: str | None = None,
    ) -> list[Any]:
        was = combo.blockSignals(True)
        try:
            if empty_placeholder is not None and default_placeholder is not None:
                self._set_combo_placeholder_from_items(
                    combo,
                    items,
                    empty_placeholder=empty_placeholder,
                    default_placeholder=default_placeholder,
                )
            if combo.preserve_missing_selected_values():
                remembered_values = combo.remembered_selected_values()
                combo.set_remembered_selected_values(remembered_values)
                combo.set_items(items, check_all=False)
            else:
                remembered_values = list(previous_values)
                previous_set = set(remembered_values)
                available_values = [value for _label, value in items]
                available_set = set(available_values)
                keep_values = [value for value in remembered_values if value in available_set]
                select_all = (
                    combo.empty_selection_means_all()
                    and bool(remembered_values)
                    and previous_set == available_set
                    and len(previous_set) == len(available_set)
                )
                combo.set_items(items, check_all=check_all)
                if select_all:
                    combo.set_selected_values(available_values)
                elif keep_values:
                    combo.set_selected_values(keep_values)
                elif check_all and combo.empty_selection_means_all():
                    combo.set_selected_values(available_values)
                else:
                    combo.set_selected_values([])
        finally:
            combo.blockSignals(was)
        return combo.selected_values()

    def _on_systems_selection_changed(self) -> None:
        self._dependency_refresh_token += 1
        token = self._dependency_refresh_token
        if not self._dependency_refresh_active:
            self._dependency_refresh_active = True
            self._begin_filters_batch()
        previous_datasets = self.selected_datasets()
        previous_imports = self.selected_import_ids()

        def _apply_datasets(items: list[tuple[str, Any]]) -> None:
            if token != self._dependency_refresh_token:
                self._dependency_refresh_active = False
                self._end_filters_batch()
                return
            self._replace_combo_items_preserving_selection(
                self.datasets_combo,
                items,
                check_all=True,
                previous_values=previous_datasets,
                empty_placeholder="No datasets available",
                default_placeholder="All datasets",
            )
            self._filters_view_model.refresh_imports_sync(
                self.selected_systems(),
                self.selected_datasets(),
                on_result=_apply_imports,
            )

        def _apply_imports(items: list[tuple[str, Any]]) -> None:
            if token != self._dependency_refresh_token:
                return
            self._replace_combo_items_preserving_selection(
                self.imports_combo,
                items,
                check_all=True,
                previous_values=previous_imports,
                empty_placeholder="No imports available",
                default_placeholder="All imports",
            )
            self._refresh_scoped_groups_and_tags(
                signal_name="systems_changed",
                token=token,
            )

        self._filters_view_model.refresh_datasets_sync(
            self.selected_systems(),
            on_result=_apply_datasets,
        )

    def _on_datasets_selection_changed(self) -> None:
        self._dependency_refresh_token += 1
        token = self._dependency_refresh_token
        if not self._dependency_refresh_active:
            self._dependency_refresh_active = True
            self._begin_filters_batch()
        previous_imports = self.selected_import_ids()

        def _apply_imports(items: list[tuple[str, Any]]) -> None:
            if token != self._dependency_refresh_token:
                self._dependency_refresh_active = False
                self._end_filters_batch()
                return
            self._replace_combo_items_preserving_selection(
                self.imports_combo,
                items,
                check_all=True,
                previous_values=previous_imports,
                empty_placeholder="No imports available",
                default_placeholder="All imports",
            )
            self._refresh_scoped_groups_and_tags(
                signal_name="datasets_changed",
                token=token,
            )

        self._filters_view_model.refresh_imports_sync(
            self.selected_systems(),
            self.selected_datasets(),
            on_result=_apply_imports,
        )

    def _on_imports_selection_changed(self) -> None:
        self._dependency_refresh_token += 1
        token = self._dependency_refresh_token
        if not self._dependency_refresh_active:
            self._dependency_refresh_active = True
            self._begin_filters_batch()
        self._refresh_scoped_groups_and_tags(
            signal_name="imports_changed",
            token=token,
        )

    def _refresh_scoped_groups_and_tags(self, *, signal_name: str, token: int) -> None:
        systems = self.selected_systems()
        datasets = self.selected_datasets()
        import_ids = self.selected_import_ids()

        def _apply(payload: dict[str, Any]) -> None:
            if token != self._dependency_refresh_token:
                self._dependency_refresh_active = False
                self._end_filters_batch()
                return
            groups_df = payload.get("groups") if isinstance(payload, dict) else None
            tags = payload.get("tags") if isinstance(payload, dict) else []
            if not isinstance(groups_df, pd.DataFrame):
                groups_df = pd.DataFrame(columns=["group_id", "kind", "label"])
            selected_group_ids = self.selected_group_ids()
            self.set_groups(groups_df)
            available_group_ids: set[int] = set()
            for value in self._all_combo_values(self.group_combo):
                try:
                    parsed = int(value)
                except Exception:
                    continue
                if parsed > 0:
                    available_group_ids.add(parsed)
            keep_group_ids = [gid for gid in selected_group_ids if gid in available_group_ids]
            self.group_combo.set_selected_values(keep_group_ids)
            selected_tags = self.selected_tags()
            self.set_tags(tags if isinstance(tags, list) else [])
            available_tags = {str(v) for v in self._all_combo_values(self.tags_combo) if str(v).strip()}
            keep_tags = [tag for tag in selected_tags if str(tag) in available_tags]
            self.tags_combo.set_selected_values(keep_tags)
            self._emit_filter_change(signal_name)
            self._dependency_refresh_active = False
            self._end_filters_batch()

        self._filters_view_model.refresh_groups_and_tags_sync(
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
            on_result=_apply,
        )

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
