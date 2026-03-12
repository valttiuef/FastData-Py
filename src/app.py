import sys
import os
import ctypes
import logging
import faulthandler
import atexit

from qt_compat import ensure_qt
ensure_qt()

# --- High-DPI policy & env clean-up (BEFORE QApplication) --------------------
# Avoid surprise scaling from environment overrides:
for k in ("QT_SCALE_FACTOR", "QT_SCREEN_SCALE_FACTORS"):
    os.environ.pop(k, None)

from PySide6.QtCore import Qt, QtMsgType, QEventLoop, QTimer, qInstallMessageHandler
from PySide6.QtGui import QGuiApplication

# Must be set before QApplication is created.
QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)
# -----------------------------------------------------------------------------

from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PySide6.QtGui import QPixmap, QColor, QIcon
from core.paths import get_icon_path, get_splash_image_path
from core.settings_manager import SettingsManager
from frontend.localization import init_localization_manager
from backend.services.logging import configure_logging, install_global_exception_hooks
from backend.services.logging.storage import crash_log_path

logger = logging.getLogger(__name__)
_FAULT_LOG_HANDLE = None
_FAULT_HANDLER_ENABLED_BY_APP = False
_FAULT_CLEANUP_REGISTERED = False


def _is_debugger_active() -> bool:
    if sys.gettrace() is not None:
        return True
    return any(name in sys.modules for name in ("debugpy", "pydevd"))


def _close_fault_logging() -> None:
    global _FAULT_LOG_HANDLE
    global _FAULT_HANDLER_ENABLED_BY_APP
    handle = _FAULT_LOG_HANDLE
    if handle is None:
        return
    try:
        if _FAULT_HANDLER_ENABLED_BY_APP and faulthandler.is_enabled():
            faulthandler.disable()
    except Exception:
        logger.debug("Failed to disable faulthandler during shutdown", exc_info=True)
    try:
        handle.flush()
        handle.close()
    except Exception:
        logger.debug("Failed to close crash log file handle", exc_info=True)
    finally:
        _FAULT_LOG_HANDLE = None
        _FAULT_HANDLER_ENABLED_BY_APP = False


def _enable_fault_logging() -> None:
    global _FAULT_LOG_HANDLE
    global _FAULT_HANDLER_ENABLED_BY_APP
    if _is_debugger_active():
        logger.info("Debugger detected; skipping faulthandler crash capture.")
        return
    handle = None
    try:
        handle = crash_log_path().open("a", encoding="utf-8")
        try:
            handle.write("=== faulthandler session start ===\n")
            handle.flush()
        except Exception:
            pass
        faulthandler.enable(file=handle, all_threads=True)
        _FAULT_LOG_HANDLE = handle
        _FAULT_HANDLER_ENABLED_BY_APP = True
    except Exception:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        logger.exception("Failed to enable faulthandler crash logging")


def _install_qt_message_bridge() -> None:
    qt_logger = logging.getLogger("qt")

    def _handler(mode, context, message):
        level = logging.INFO
        if mode in (QtMsgType.QtWarningMsg,):
            level = logging.WARNING
        elif mode in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
            level = logging.ERROR
        qt_logger.log(level, "Qt: %s", str(message or ""))

    qInstallMessageHandler(_handler)

class NonInteractiveSplash(QSplashScreen):
    def mousePressEvent(self, event) -> None:
        event.ignore()

    def mouseReleaseEvent(self, event) -> None:
        event.ignore()

    def mouseDoubleClickEvent(self, event) -> None:
        event.ignore()


def update_status(splash, app, text: str, *, percent: int | None = None, error: bool = False) -> None:
    if splash is None or app is None:
        return
    try:
        color = QColor("red") if error else QColor("white")
        message = text
        if percent is not None:
            message = f"{percent:>3}% - {text}"
        splash.showMessage(message, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, color)  # type: ignore
        app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    except Exception:
        logger.warning("Failed to update splash status", exc_info=True)


def _fatal_startup_error(app, splash, message: str, exc: Exception | None = None) -> None:
    detail = f"{message}"
    if exc is not None:
        detail = f"{message}\n\n{exc}"
    try:
        update_status(splash, app, f"ERROR: {message}", percent=0, error=True)
    except Exception:
        logger.warning("Failed to update splash status after startup error", exc_info=True)
    try:
        QMessageBox.critical(None, app.translate("startup", "Startup Error"), detail)
    except Exception:
        logger.warning("Failed to show startup error dialog", exc_info=True)
    try:
        if splash is not None:
            splash.close()
    except Exception:
        logger.warning("Failed to close splash after startup error", exc_info=True)
    try:
        if app is not None and app.overrideCursor() is not None:
            app.restoreOverrideCursor()
    except Exception:
        logger.warning("Failed to restore cursor after startup error", exc_info=True)


# @ai(gpt-5, codex, bugfix, 2026-03-12)
def main():
    configure_logging(name="app.startup")
    install_global_exception_hooks()
    global _FAULT_CLEANUP_REGISTERED
    if not _FAULT_CLEANUP_REGISTERED:
        atexit.register(_close_fault_logging)
        _FAULT_CLEANUP_REGISTERED = True
    _enable_fault_logging()
    _install_qt_message_bridge()

    # Windows taskbar icon
    if sys.platform.startswith("win"):
        app_id = "FastData"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)

    app = QApplication(sys.argv)
    try:
        app.aboutToQuit.connect(_close_fault_logging)
    except Exception:
        logger.debug("Failed to bind crash-log cleanup to aboutToQuit", exc_info=True)
    app.setStyle("Fusion")  # OK to keep

    settings_manager = SettingsManager("Visima", "FastData")
    localization = init_localization_manager(app)
    localization.apply_language(settings_manager.get_language())

    # Application/window icon
    icon_path = get_icon_path()
    app_icon = None
    if icon_path.exists():
        try:
            app_icon = QIcon(str(icon_path))
            app.setWindowIcon(app_icon)
        except Exception:
            app_icon = None

    # --- Splash screen ---
    splash = None
    try:
        splash_pix = QPixmap(str(get_splash_image_path()))
        splash = NonInteractiveSplash(splash_pix, Qt.WindowType.WindowStaysOnTopHint)  # type: ignore
        splash.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.SplashScreen)  # type: ignore
        splash.show()
        app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    except Exception as exc:
        _fatal_startup_error(app, splash, "Failed to create splash screen.", exc)
        sys.exit(1)

    # Import heavier modules only after the splash is visible to minimise perceived
    # startup latency.
    try:
        from frontend.style.theme_manager import init_theme_manager
        from frontend.style.styles import detect_system_theme
    except Exception as exc:
        _fatal_startup_error(app, splash, "Failed to load UI modules.", exc)
        sys.exit(1)

    def report_status(text: str, percent: int, error: bool = False) -> None:
        update_status(splash, app, text, percent=percent, error=error)

    update_status(splash, app, app.translate("startup", "Starting up..."), percent=0)

    # INIT THEME MANAGER *BEFORE* building UI
    report_status(app.translate("startup", "Loading theme..."), 10)
    try:
        theme = detect_system_theme()
        theme_mgr = init_theme_manager(app)
        theme_mgr.set_theme(theme)
        report_status(f"Theme '{theme}' applied", 20)
    except Exception as e:
        _fatal_startup_error(app, splash, "Theme load failed.", e)
        sys.exit(1)

    # --- Create main window ---
    report_status(app.translate("startup", "Creating main window..."), 30)
    app.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        from frontend.windows.main_window import MainWindow
        win = MainWindow(status_callback=report_status)
    except Exception as e:
        _fatal_startup_error(app, splash, "Main window failed to initialize.", e)
        sys.exit(1)
    if app_icon is not None:
        try:
            win.setWindowIcon(app_icon)
        except Exception:
            logger.warning("Failed to set main window icon", exc_info=True)

    # --- Finish splash ---
    try:
        update_status(splash, app, app.translate("startup", "Ready"), percent=100)
        if splash is not None:
            splash.finish(win)
        # Re-apply maximize after splash handoff; on some platforms the initial
        # state can be dropped during first show/polish.
        QTimer.singleShot(0, win.showMaximized)
        if app.overrideCursor() is not None:
            app.restoreOverrideCursor()
    except Exception as e:
        _fatal_startup_error(app, splash, "Failed during final UI setup.", e)
        sys.exit(1)

    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal uncaught exception in app entrypoint")
        raise
