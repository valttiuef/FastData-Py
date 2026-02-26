from PySide6.QtCore import QCoreApplication, QTimer

def run_in_main_thread(callback, *args, **kwargs):
    """Ensures the callback runs in the main Qt event loop (GUI thread)."""
    app = QCoreApplication.instance()
    if app is None:
        callback(*args, **kwargs)
        return
    QTimer.singleShot(0, app, lambda: callback(*args, **kwargs))
