
# main_window.pyw
import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (

    QApplication,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Use the centralized theme manager instead of calling styles directly
from ..style.theme_manager import theme_manager
from ..style.styles import detect_system_theme  # optional fallback

from backend.help_manager import get_help_manager
from core.paths import get_app_metadata_path, get_help_path, get_resource_path
from ..menu import init_menu_bar
from ..models.hybrid_pandas_model import HybridPandasModel
from ..viewmodels import get_log_view_model
from ..models.settings_model import SettingsModel
from ..tabs import add_tab_with_help
from ..tabs.tab_modules import get_runtime_tab_modules
from ..threading import run_in_main_thread, run_in_thread, stop_all_worker_threads
from ..utils import register_main_window, unregister_main_window
from .log_window import LogWindow
from .chat_window import ChatWindow
from .about_window import AboutWindow
from ..viewmodels.help_viewmodel import get_help_viewmodel
from ..widgets.toast import ToastManager

if TYPE_CHECKING:
    from ..tabs.tab_modules import TabModuleSpec

def _load_app_metadata() -> dict:
    """Load app metadata from the repository root."""
    meta_path = get_app_metadata_path()
    try:
        with meta_path.open(encoding="utf-8") as meta_file:
            data = json.load(meta_file)
    except FileNotFoundError:
        logging.getLogger(__name__).warning("appmeta.json not found at %s", meta_path)
        return {}
    except Exception as exc:  # pragma: no cover - defensive parsing fallback
        logging.getLogger(__name__).warning("Unable to read appmeta.json: %s", exc)
        return {}

    if not isinstance(data, dict):
        logging.getLogger(__name__).warning("Unexpected appmeta.json format: %r", data)
        return {}

    return data


@lru_cache(maxsize=1)
def _get_app_metadata() -> dict:
    """Cached accessor for app metadata."""
    return _load_app_metadata()


@dataclass(frozen=True)
class _TabSpec:
    key: str
    label: str
    help_key: str
    builder: Callable[[], QWidget]


class MainWindow(QMainWindow):
    _instance_created = False

    def __init__(self, *, status_callback: Callable[..., None] | None = None):
        if MainWindow._instance_created:
            raise RuntimeError("MainWindow has already been created")
        super().__init__()
        MainWindow._instance_created = True
        self._status_callback = status_callback
        metadata = _get_app_metadata()
        product_name = metadata.get("product_name") or metadata.get("app_name") or "FastData"
        self.setWindowTitle(product_name)

        self._report_startup_status(self.tr("Initializing settings..."), 35)
        # Models
        self.settings_model = SettingsModel()
        self._report_startup_status(self.tr("Initializing data model..."), 45)
        self.database_model = HybridPandasModel(self.settings_model)

        # Help
        self._report_startup_status(self.tr("Initializing help system..."), 55)
        self.help_viewmodel = None
        try:
            help_file = get_help_path()
            help_manager = get_help_manager(help_file)
            self.help_viewmodel = get_help_viewmodel(help_manager, parent=self)
        except Exception as exc:  # pragma: no cover - defensive guard for missing help file
            logging.getLogger(__name__).warning("Help system unavailable: %s", exc)

        # --- Central / Tabs ---------------------------------------------------
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)  # remove extra vertical jitter

        self._report_startup_status(self.tr("Initializing logging..."), 60)

        self._report_startup_status(self.tr("Building tabs..."), 65)
        # Tabs
        self._tab_modules = get_runtime_tab_modules()
        self._module_by_key = {module.key: module for module in self._tab_modules}

        self.tabs = QTabWidget(central)
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setIconSize(QSize(24, 24))

        self._placeholder_containers: dict[str, QWidget] = {}
        self._placeholder_labels: dict[str, QLabel] = {}
        self._placeholder_lookup: dict[QWidget, _TabSpec] = {}
        self._loaded_tabs: dict[str, QWidget] = {}
        self._pending_help_keys: set[str] = set()
        self._tab_specs = [
            _TabSpec(module.key, module.label, module.help_key, self._build_tab_factory(module))
            for module in self._tab_modules
            if module.lazy_load
        ]
        self._tab_spec_by_help_key = {spec.help_key: spec for spec in self._tab_specs}

        data_module = self._module_by_key.get("data")
        if data_module is not None:
            self._report_startup_status(self.tr("Loading {name} features...").format(name=data_module.label), 66)
            try:
                self.data_tab = data_module.builder(self)
                setattr(self, data_module.instance_attr, self.data_tab)
            except Exception:
                logging.getLogger(__name__).exception("Failed to load data tab")
                self._report_startup_status(
                    self.tr("ERROR: {name} failed to load (see logs)").format(name=data_module.label),
                    66,
                    error=True,
                )
                self.data_tab = self._build_error_tab(
                    self.tr("{name} failed to load.").format(name=data_module.label),
                    self.tr("Check the log for details and try restarting the app."),
                )
        else:
            self.data_tab = QWidget(self)

        if data_module is not None:
            add_tab_with_help(
                self.tabs,
                self.data_tab,
                data_module.label,
                data_module.help_key,
                self.help_viewmodel,
                tab_key=data_module.key,
            )

        for spec in self._tab_specs:
            placeholder, label = self._create_placeholder_tab(spec.label)
            self._placeholder_containers[spec.help_key] = placeholder
            self._placeholder_labels[spec.help_key] = label
            self._placeholder_lookup[placeholder] = spec
            self._pending_help_keys.add(spec.help_key)
            add_tab_with_help(
                self.tabs,
                placeholder,
                spec.label,
                spec.help_key,
                self.help_viewmodel,
                tab_key=spec.key,
            )

        self.tabs.setCurrentIndex(0)

        self._report_startup_status(self.tr("Preparing log view..."), 75)
        # Log
        self.log_view_model = get_log_view_model(parent=self)
        self.log_window = LogWindow(
            self.log_view_model,
            self.settings_model,
            parent=self,
        )

        # Standalone chat window (created on demand)
        self.chat_window: ChatWindow | None = None

        # Size policies that reduce "jumpiness"
        # Main tabs should greedily take space; log should not force expansion
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log_window.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        # Give the log a sane range so it doesn't pop to tiny heights/widths
        self.log_window.setMinimumWidth(300)

        # --- Splitter: Tabs | Log --------------------------------------------
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal, central)
        self._main_splitter.addWidget(self.tabs)
        self._main_splitter.addWidget(self.log_window)
        self._main_splitter.setOpaqueResize(False)  # draw after drag ends; feels steadier

        # Make only the log collapsible by drag; main area stays visible
        try:
            self._main_splitter.setCollapsible(0, False)
            self._main_splitter.setCollapsible(1, True)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

        # Track log visibility state; default to saved value
        self._log_visible = bool(self.settings_model.log_visible)
        self._updating_log_visibility = False

        # Update bookkeeping when user drags
        self._main_splitter.splitterMoved.connect(self._on_splitter_moved)

        layout.addWidget(self._main_splitter)
        self.setCentralWidget(central)

        # --- Status bar -------------------------------------------------------
        self._report_startup_status(self.tr("Preparing status bar..."), 80)
        sb = QStatusBar(self)

        self._progress = QProgressBar(self)
        self._progress.setMinimum(0)
        self._progress.setMaximum(100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        self._progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self._status_label = QLabel("", self)
        self._status_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self._status_label.setIndent(4)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        sb.addWidget(self._progress, 1)
        sb.addPermanentWidget(self._status_label)
        self.setStatusBar(sb)
        self._default_status = ""

        # Tab loading progress
        self._loading_total = len(self._tab_specs)
        self._loaded_count = 0
        self._update_tab_loading_status()

        if self._tab_specs:
            # Start background tab loading as soon as the event loop is free
            QTimer.singleShot(0, self._load_next_pending_tab)

        # DB events
        self.database_model.database_changed.connect(self._on_database_changed)
        self._on_database_changed(self.database_model.path)

        # Toast notifications (also forwarded by frontend.utils.toast_* helpers)
        self.toast_manager = ToastManager(anchor=self)

        # Register main window so other modules can use helpers
        try:
            register_main_window(self)
        except Exception:
            logger.warning("Exception in __init__", exc_info=True)

        # --- Menu bar / theme -------------------------------------------------
        self._report_startup_status(self.tr("Initializing menu..."), 85)
        init_menu_bar(self)
        for code, action in getattr(self, "language_actions", {}).items():
            action.setChecked(code == self.settings_model.language)

        self.set_log_visible(self.settings_model.log_visible)

        saved = self.settings_model.theme
        if saved not in ("dark", "light"):
            saved = detect_system_theme()
        self.current_theme = saved if saved in ("dark", "light") else "dark"
        self._report_startup_status(self.tr("Applying theme..."), 90)
        try:
            self.apply_theme(self.current_theme, apply_runtime=True)
        except Exception:
            logger.exception("Theme application failed at startup; falling back to light theme")
            self.current_theme = "light"
            try:
                self.apply_theme("light", apply_runtime=True)
            except Exception:
                logger.exception("Fallback theme application failed")
            self._report_startup_status(
                self.tr("Theme application failed; using safe defaults."),
                90,
                error=True,
            )

        # reflect checks after apply (if actions exist)
        al = getattr(self, "action_light", None)
        if al is not None:
            al.setChecked(self.current_theme == "light")
        ad = getattr(self, "action_dark", None)
        if ad is not None:
            ad.setChecked(self.current_theme == "dark")

        self.tabs.currentChanged.connect(self._maybe_load_selected_tab)

        self._report_startup_status(self.tr("Finalizing interface..."), 95)

    def _report_startup_status(self, text: str, percent: int, *, error: bool = False) -> None:
        if self._status_callback is None:
            return
        try:
            self._status_callback(text, percent, error)
        except TypeError:
            self._status_callback(text, percent)

    # --- Chat window ---------------------------------------------------------
    def show_chat_window(self) -> None:
        if self.chat_window is None:
            self.chat_window = ChatWindow(
                self.log_view_model,
                self.settings_model,
                help_viewmodel=self.help_viewmodel,
                parent=self,
            )
        self.chat_window.show()
        self.chat_window.raise_()
        self.chat_window.activateWindow()

    # --- Database menu actions -----------------------------------------------
    def _create_placeholder_tab(self, label: str) -> tuple[QWidget, QLabel]:
        placeholder = QWidget(self.tabs)
        layout = QVBoxLayout(placeholder)
        layout.setContentsMargins(24, 24, 24, 24)

        msg = QLabel(self.tr("Loading {name}…").format(name=label), placeholder)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setWordWrap(True)
        layout.addWidget(msg)

        return placeholder, msg

    def _build_error_tab(self, heading: str, message: str) -> QWidget:
        container = QWidget(self.tabs)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(24, 24, 24, 24)
        title = QLabel(heading, container)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        body = QLabel(message, container)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(body)
        return container

    def _maybe_load_selected_tab(self, index: int) -> None:
        widget = self.tabs.widget(index)
        spec = self._placeholder_lookup.get(widget)
        if spec is None:
            return
        self._load_tab(spec)

    def activate_tab(self, tab_key: str) -> None:
        """Switch to a tab using its module key or help key."""
        if not tab_key:
            return
        module = self._module_by_key.get(tab_key)
        help_key = module.help_key if module is not None else tab_key

        spec = self._tab_spec_by_help_key.get(help_key)
        if spec is not None:
            self._load_tab(spec)

        widget = self._placeholder_containers.get(help_key)
        if widget is None:
            widget = self._loaded_tabs.get(help_key)
        if widget is None and module is not None:
            widget = getattr(self, module.instance_attr, None)
        if widget is None:
            return
        index = self.tabs.indexOf(widget)
        if index >= 0:
            self.tabs.setCurrentIndex(index)

    def _load_next_pending_tab(self) -> None:
        if not self._pending_help_keys:
            return
        for spec in self._tab_specs:
            if spec.help_key in self._pending_help_keys:
                self._load_tab(spec)
                break

    def _load_tab(self, spec: _TabSpec, *, schedule_next: bool = True) -> None:
        if spec.help_key in self._loaded_tabs:
            return

        placeholder = self._placeholder_containers.get(spec.help_key)
        if placeholder is not None:
            placeholder.setEnabled(False)

        self._status_label.setText(self.tr("Loading {name}…").format(name=spec.label))
        self._progress.setVisible(True)

        try:
            widget = spec.builder()
            self._loaded_tabs[spec.help_key] = widget
            self._fill_placeholder(spec, widget)
            self._loaded_count += 1
            self._pending_help_keys.discard(spec.help_key)
        except Exception:
            logging.getLogger(__name__).exception("Failed to build tab %s", spec.help_key)
            self._pending_help_keys.discard(spec.help_key)
            self._loaded_count += 1
            self._loaded_tabs[spec.help_key] = placeholder or QWidget(self.tabs)
            error_text = self.tr("Failed to load {name}. Check the log for details.").format(name=spec.label)
            label = self._placeholder_labels.get(spec.help_key)
            if label is not None:
                label.setText(error_text)
            self._status_label.setText(error_text)
            self._report_startup_status(
                self.tr("ERROR: {name} failed to load (see logs)").format(name=spec.label),
                max(self._progress.value(), 0),
                error=True,
            )
        finally:
            if placeholder is not None:
                placeholder.setEnabled(True)
            self._update_tab_loading_status()
            if schedule_next:
                QTimer.singleShot(0, self._load_next_pending_tab)

    def _preload_lazy_tabs_for_startup(self, start_percent: int, end_percent: int) -> None:
        total = len(self._tab_specs)
        if total <= 0:
            return
        span = max(end_percent - start_percent, 0)
        for index, spec in enumerate(self._tab_specs, start=1):
            percent = start_percent + round(span * (index / total))
            self._report_startup_status(
                self.tr("Loading {name} features...").format(name=spec.label),
                percent,
            )
            self._load_tab(spec, schedule_next=False)

    def _fill_placeholder(self, spec: _TabSpec, widget: QWidget) -> None:
        container = self._placeholder_containers.get(spec.help_key)
        if container is None:
            return
        label = self._placeholder_labels.pop(spec.help_key, None)
        layout = container.layout()
        if layout is None:
            return
        if label is not None:
            layout.removeWidget(label)
            label.deleteLater()
        widget.setParent(container)
        layout.addWidget(widget)

    def _build_tab_factory(self, module: "TabModuleSpec") -> Callable[[], QWidget]:
        def _builder() -> QWidget:
            existing = getattr(self, module.instance_attr, None)
            if existing is not None:
                return existing
            widget = module.builder(self)
            setattr(self, module.instance_attr, widget)
            return widget

        return _builder

    def _ask_for_db_file(self, caption: str = "Select database file") -> str | None:
        caption = self.tr(caption)
        fn, _f = QFileDialog.getOpenFileName(
            self,
            caption,
            "",
            self.tr("Measurement data files (*.duckdb *.db *.duckdb3);;All Files (*)"),
        )
        return fn or None

    def _ask_for_selection_db_file(self, caption: str = "Select selection settings file") -> str | None:
        caption = self.tr(caption)
        fn, _f = QFileDialog.getOpenFileName(
            self,
            caption,
            "",
            self.tr("Database Files (*.db);;All Files (*)"),
        )
        return fn or None

    def _run_db_task(
        self,
        *,
        worker,
        start_message: str | None,
        success_message: str | None,
        error_title: str,
        error_prefix: str = "",
        start_title: str | None = None,
        success_title: str | None = None,
        start_kind: str = "info",
        on_success=None,
    ) -> None:
        if start_title is None:
            start_title = self.tr("Database")
        if success_title is None:
            success_title = self.tr("Done")
        if start_message:
            try:
                if start_kind == "warn":
                    self.toast_manager.warn(start_message, title=start_title)
                else:
                    self.toast_manager.info(start_message, title=start_title)
            except Exception:
                logger.warning("Exception in _run_db_task", exc_info=True)

        def _handle_success(result):
            if on_success:
                try:
                    on_success(result)
                except Exception as exc:
                    try:
                        self.toast_manager.error(str(exc), title=error_title)
                    except Exception:
                        logger.warning("Exception in _handle_success", exc_info=True)
                    return
            if success_message:
                try:
                    self.toast_manager.success(success_message, title=success_title)
                except Exception:
                    logger.warning("Exception in _handle_success", exc_info=True)

        def _handle_error(msg: str):
            message = f"{error_prefix}{msg}" if error_prefix else str(msg)
            try:
                self.toast_manager.error(message, title=error_title)
            except Exception:
                logger.warning("Exception in _handle_error", exc_info=True)

        run_in_thread(
            worker,
            on_result=lambda result: run_in_main_thread(_handle_success, result),
            on_error=lambda msg: run_in_main_thread(_handle_error, msg),
            owner=self,
            key="db_task",
        )

    def new_database(self):
        start = str(self.database_model.path)
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Create new database"),
            start,
            self.tr("Measurement data files (*.duckdb *.db);;All Files (*)"),
        )
        if not path:
            return
        target = Path(path)
        try:
            self._release_database_handles()
        except Exception:
            logger.warning("Exception in new_database", exc_info=True)

        def _worker():
            self.database_model.new_database(target)
            return target

        self._run_db_task(
            worker=_worker,
            start_message=self.tr("Creating database: {name}").format(name=target.name),
            success_message=self.tr("Created database: {name}").format(name=target.name),
            error_title=self.tr("Create database failed"),
            error_prefix=self.tr("Failed to create database: "),
            start_title=self.tr("Database"),
            success_title=self.tr("Database created"),
        )

    def open_database(self):
        sel = self._ask_for_db_file("Open database file")
        if not sel:
            return
        p = Path(sel)
        if not p.exists():
            message = self.tr("Selected database file does not exist: {path}").format(path=p)
            try:
                self.toast_manager.error(message, title=self.tr("Open database failed"))
            except Exception:
                logger.warning("Exception in open_database", exc_info=True)
            return
        try:
            self._release_database_handles()
        except Exception:
            logger.warning("Exception in open_database", exc_info=True)

        def _worker():
            self.database_model.use_database(p)
            return p

        self._run_db_task(
            worker=_worker,
            start_message=self.tr("Loading database: {name}").format(name=p.name),
            success_message=self.tr("Loaded database: {name}").format(name=p.name),
            error_title=self.tr("Open database failed"),
            error_prefix=self.tr("Failed to open database: "),
            start_title=self.tr("Database"),
            success_title=self.tr("Database loaded"),
        )

    def save_database_as(self):
        curp = Path(self.database_model.path)
        start = str(curp)
        dest, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save database as"),
            start,
            self.tr("Measurement data files (*.duckdb *.db);;All Files (*)"),
        )
        if not dest:
            return
        destp = Path(dest)
        try:
            self._release_database_handles()
        except Exception:
            logger.warning("Exception in save_database_as", exc_info=True)

        def _worker():
            self.database_model.save_database_as(destp)
            return destp

        self._run_db_task(
            worker=_worker,
            start_message=self.tr("Saving database: {name}").format(name=destp.name),
            success_message=self.tr("Saved database as: {name}").format(name=destp.name),
            error_title=self.tr("Save database failed"),
            error_prefix=self.tr("Failed to save database: "),
            start_title=self.tr("Database"),
            success_title=self.tr("Database saved"),
        )

    def reset_database(self):
        default = self.settings_model.default_database_path()
        message = self.tr("Resetting the database to the default local path.")
        try:
            self._release_database_handles()
        except Exception:
            logger.warning("Exception in reset_database", exc_info=True)

        def _worker():
            self.database_model.reset_database()
            return default

        self._run_db_task(
            worker=_worker,
            start_message=message,
            success_message=self.tr("Database reset to: {name}").format(name=Path(default).name),
            error_title=self.tr("Database reset failed"),
            error_prefix=self.tr("Failed to reset database: "),
            start_title=self.tr("Database reset"),
            success_title=self.tr("Database reset"),
            start_kind="warn",
        )

    def open_selection_settings_database(self):
        sel = self._ask_for_selection_db_file("Open selection settings database")
        if not sel:
            return
        p = Path(sel)
        if not p.exists():
            message = self.tr("Selected selection settings database file does not exist: {path}").format(path=p)
            try:
                self.toast_manager.error(message, title=self.tr("Open settings DB failed"))
            except Exception:
                logger.warning("Exception in open_selection_settings_database", exc_info=True)
            return

        def _worker():
            self.database_model.use_selection_database(p)
            return p

        self._run_db_task(
            worker=_worker,
            start_message=self.tr("Loading settings DB: {name}").format(name=p.name),
            success_message=self.tr("Loaded settings DB: {name}").format(name=p.name),
            error_title=self.tr("Open settings DB failed"),
            error_prefix=self.tr("Failed to open selection settings database: "),
            start_title=self.tr("Settings"),
            success_title=self.tr("Settings loaded"),
        )

    def save_selection_settings_database_as(self):
        curp = Path(self.database_model.selection_settings_path)
        start = str(curp)
        dest, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save selection settings as"),
            start,
            self.tr("Database Files (*.db);;All Files (*)"),
        )
        if not dest:
            return
        destp = Path(dest)

        def _worker():
            self.database_model.save_selection_database_as(destp)
            return destp

        self._run_db_task(
            worker=_worker,
            start_message=self.tr("Saving settings DB: {name}").format(name=destp.name),
            success_message=self.tr("Saved settings DB as: {name}").format(name=destp.name),
            error_title=self.tr("Save settings DB failed"),
            error_prefix=self.tr("Failed to save selection settings database: "),
            start_title=self.tr("Settings"),
            success_title=self.tr("Settings saved"),
        )

    def reset_selection_settings_database(self):
        answer = QMessageBox.question(
            self,
            self.tr("Reset settings database"),
            self.tr(
                "Are you sure you want to reset the selection settings database?\n\n"
                "This will restore the default local settings database path and may discard unsaved changes."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        default = self.settings_model.default_selection_db_path()
        message = self.tr("Resetting the selection settings database to the default local path.")

        def _worker():
            self.database_model.reset_selection_database()
            return default

        self._run_db_task(
            worker=_worker,
            start_message=message,
            success_message=self.tr("Settings reset to: {name}").format(name=Path(default).name),
            error_title=self.tr("Settings reset failed"),
            error_prefix=self.tr("Failed to reset selection settings database: "),
            start_title=self.tr("Settings reset"),
            success_title=self.tr("Settings reset"),
            start_kind="warn",
        )

    def refresh_databases(self) -> None:
        """Reload measurement, selection settings, and log databases."""
        try:
            self.toast_manager.info(self.tr("Refreshing databases..."), title=self.tr("Refreshing"))
        except Exception:
            logger.warning("Exception in refresh_databases", exc_info=True)
        try:
            self._release_database_handles()
        except Exception:
            logger.warning("Exception in refresh_databases", exc_info=True)
        try:
            self.database_model.refresh_selection_database()
        except Exception:
            logger.warning("Exception in refresh_databases", exc_info=True)
        try:
            self.database_model.refresh()
        except Exception:
            logger.warning("Exception in refresh_databases", exc_info=True)
        try:
            self.database_model.load_selection_state()
        except Exception:
            logger.warning("Exception in refresh_databases", exc_info=True)
        try:
            preferred = self.log_view_model.enabled_logger_names()
            self.log_view_model.reload_from_storage(preferred_loggers=preferred)
        except Exception:
            logger.warning("Exception in refresh_databases", exc_info=True)
        try:
            self.toast_manager.success(self.tr("Databases refreshed."), title=self.tr("Done"))
        except Exception:
            logger.warning("Exception in refresh_databases", exc_info=True)

    def _release_database_handles(self) -> None:
        for module in self._tab_modules:
            tab = getattr(self, module.instance_attr, None)
            if tab is None:
                continue
            closer = getattr(tab, "close_database", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.warning("Exception in _release_database_handles", exc_info=True)

    def _on_database_changed(self, path) -> None:
        try:
            name = Path(path).name if path else ""
        except Exception:
            name = ""
        self._default_status = self.tr("Database: {name}").format(name=name) if name else ""
        self.set_status_text(self._default_status)

    # --- Log splitter control (no hide(); sizes only) -------------------------
    def set_log_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if visible == self._log_visible and self.log_window.isHidden() == (not visible):
            self._update_log_action(visible)
            return

        self._updating_log_visibility = True
        try:
            self._log_visible = visible
            self.log_window.setVisible(visible)
            try:
                self._main_splitter.setCollapsible(1, not visible)
            except Exception:
                logger.warning("Exception in set_log_visible", exc_info=True)
            self._update_log_action(visible)
            try:
                self.settings_model.set_log_visible(visible)
            except Exception:
                logger.warning("Exception in set_log_visible", exc_info=True)
        finally:
            self._updating_log_visibility = False

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        if self._updating_log_visibility:
            return
        if not self.log_window.isVisible():
            return
        if self.log_window.width() <= 1:
            self.set_log_visible(False)

    def _update_log_action(self, visible: bool) -> None:
        action = getattr(self, "action_show_log", None)
        if not action:
            return
        if action.isChecked() == visible:
            return
        blocker = QSignalBlocker(action)
        action.setChecked(visible)

    # --- Status / Progress helpers -------------------------------------------
    def set_progress(self, percent: int):
        try:
            if percent is None:
                self._progress.setVisible(False)
                return
            pct = max(0, min(100, int(percent)))
            self._progress.setValue(pct)
            self._progress.setVisible(True)
        except Exception:
            logger.warning("Exception in set_progress", exc_info=True)

    def clear_progress(self):
        try:
            self._progress.setVisible(False)
            self._progress.setValue(0)
        except Exception:
            logger.warning("Exception in clear_progress", exc_info=True)

    def set_status_text(self, text: str):
        try:
            if text:
                message = str(text)
                self._status_label.setText(message)
                self.log_view_model.log_message(message, level=logging.INFO, origin="status")
            else:
                self._status_label.setText(self._default_status)
        except Exception:
            logger.warning("Exception in set_status_text", exc_info=True)

    def clear_status_text(self):
        try:
            self._status_label.setText(self._default_status)
        except Exception:
            logger.warning("Exception in clear_status_text", exc_info=True)

    def _update_tab_loading_status(self) -> None:
        try:
            if self._loading_total <= 0:
                self.clear_progress()
                self._status_label.setText(self._default_status)
                return

            loaded = max(0, min(self._loading_total, self._loaded_count))
            remaining = self._loading_total - loaded

            if remaining <= 0:
                self.clear_progress()
                ready_text = self._default_status or self.tr("All features ready")
                self._status_label.setText(ready_text)
                return

            if loaded == 0:
                self._progress.setVisible(True)
                self._progress.setValue(0)
                self._status_label.setText(self.tr("Loading features..."))
                return

            percent = int((loaded / self._loading_total) * 100)
            self._progress.setVisible(True)
            self._progress.setValue(percent)
            self._status_label.setText(self.tr("Loading feature ({loaded}/{total})…").format(loaded=loaded, total=self._loading_total))
        except Exception:
            logger.warning("Exception in _update_tab_loading_status", exc_info=True)

    # --- Theme ---------------------------------------------------------------
    def apply_theme(self, theme: str, *, apply_runtime: bool = False):
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("QApplication must be created before applying theme.")

        theme = theme if theme in ("dark", "light") else "dark"
        previous_theme = self.settings_model.theme
        if apply_runtime:
            tm = theme_manager()
            tm.set_theme(theme)

        self.current_theme = theme
        try:
            self.settings_model.set_theme(theme)
        except Exception:
            logger.warning("Exception in apply_theme", exc_info=True)

        al = getattr(self, "action_light", None)
        if al is not None:
            al.setChecked(theme == "light")
        ad = getattr(self, "action_dark", None)
        if ad is not None:
            ad.setChecked(theme == "dark")

        if not apply_runtime and theme != previous_theme:
            message = self.tr("Theme preference saved. Restart the application for the change to take effect.")
            self.set_status_text(message)
            try:
                self.toast_manager.info(message, title=self.tr("Restart required"))
            except Exception:
                logger.warning("Exception in apply_theme", exc_info=True)


    def available_languages(self) -> dict[str, str]:
        return {"en": self.tr("English"), "fi": self.tr("Finnish")}

    def change_language(self, language_code: str) -> None:
        previous = self.settings_model.language
        self.settings_model.set_language(language_code)
        for code, action in getattr(self, "language_actions", {}).items():
            action.setChecked(code == self.settings_model.language)
        if self.settings_model.language != previous:
            message = self.tr("Language preference saved. Restart the application for the change to take effect.")
            self.set_status_text(message)
            try:
                self.toast_manager.info(message, title=self.tr("Restart required"))
            except Exception:
                logger.warning("Exception in change_language", exc_info=True)

    def show_about_dialog(self):
        window = AboutWindow(self)
        window.exec()

    def open_help_docs(self) -> None:
        doc_path = get_resource_path("help/docs/help.html")
        if not doc_path.exists():
            try:
                self.toast_manager.error(self.tr("Help documentation not found."), title=self.tr("Help"))
            except Exception:
                logger.warning("Exception in open_help_docs", exc_info=True)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(doc_path)))

    # --- Close ----------------------------------------------------------------
    def closeEvent(self, event):
        try:
            self._release_database_handles()
        except Exception:
            logger.warning("Exception in closeEvent", exc_info=True)

        # Stop any running worker threads so the application can exit cleanly
        try:
            stop_all_worker_threads()
        except Exception:
            logger.warning("Exception in closeEvent", exc_info=True)

        # Shut down the log service before destroying Qt objects
        # This prevents "signal source has been deleted" errors when threads try to log after UI shutdown
        try:
            from backend.services.logging import get_log_service
            log_service = get_log_service()
            log_service.shutdown()
        except Exception:
            logger.warning("Exception in closeEvent", exc_info=True)

        MainWindow._instance_created = False
        try:
            unregister_main_window()
        except Exception:
            logger.warning("Exception in closeEvent", exc_info=True)

        # Call the base implementation so Qt does its normal cleanup
        super().closeEvent(event)

        # Explicitly quit the application when this window closes
        app = QApplication.instance()
        if app is not None:
            app.quit()
