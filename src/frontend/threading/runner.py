import atexit
import threading
import time
import weakref

from PySide6.QtCore import QCoreApplication, QThread, Signal
import logging

logger = logging.getLogger(__name__)

_active_threads: "set[GenericWorkerThread]" = set()
_owner_threads: "weakref.WeakKeyDictionary[object, dict[object | None, set[GenericWorkerThread]]]" = weakref.WeakKeyDictionary()
_owner_cleanup_hooks: "weakref.WeakKeyDictionary[object, bool]" = weakref.WeakKeyDictionary()
_cleanup_hook_registered = False


def _register_app_cleanup_hook() -> None:
    """Ensure worker threads are stopped when the Qt app is exiting."""

    global _cleanup_hook_registered
    if _cleanup_hook_registered:
        return

    app = QCoreApplication.instance()
    if app is None:
        return

    try:
        app.aboutToQuit.connect(lambda: stop_all_worker_threads(wait=False))
        _cleanup_hook_registered = True
    except Exception:
        # If we cannot install the hook (e.g. during early import), fail silently
        logger.warning("Exception in _register_app_cleanup_hook", exc_info=True)


class GenericWorkerThread(QThread):
    progress = Signal(int)
    result = Signal(object)
    finished = Signal(object)
    error = Signal(str)
    done = Signal()

    def __init__(
        self,
        target,
        *args,
        thread_priority: QThread.Priority | None = None,
        progress_min_interval_ms: int | None = 50,
        result_min_interval_ms: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.running = True
        self.stop_event = threading.Event()
        self._progress_min_interval = max(0, int(progress_min_interval_ms or 0)) / 1000.0
        self._result_min_interval = max(0, int(result_min_interval_ms or 0)) / 1000.0
        self._last_progress_emit = 0.0
        self._last_result_emit = 0.0
        self._thread_priority = thread_priority
        _active_threads.add(self)
        _register_app_cleanup_hook()

    def run(self):
        try:
            if self._thread_priority is not None:
                try:
                    self.setPriority(self._thread_priority)
                except Exception:
                    logger.warning("Exception in run", exc_info=True)

            def progress_callback(value):
                if self._progress_min_interval <= 0:
                    self.progress.emit(value)
                    return
                now = time.monotonic()
                if value in (0, 100) or (now - self._last_progress_emit) >= self._progress_min_interval:
                    self._last_progress_emit = now
                    self.progress.emit(value)

            def result_callback(entry):
                if self._result_min_interval <= 0:
                    self.result.emit(entry)
                    return
                now = time.monotonic()
                if (now - self._last_result_emit) >= self._result_min_interval:
                    self._last_result_emit = now
                    self.result.emit(entry)

            if 'progress_callback' in self.target.__code__.co_varnames:
                self.kwargs['progress_callback'] = progress_callback

            if 'stop_event' in self.target.__code__.co_varnames:
                self.kwargs['stop_event'] = self.stop_event

            if 'result_callback' in self.target.__code__.co_varnames:
                self.kwargs['result_callback'] = result_callback

            result = self.target(*self.args, **self.kwargs)

            if self.running and not self.stop_event.is_set():
                self.finished.emit(result)

        except Exception as e:
            logger.exception("Worker thread target failed")
            if self.running and not self.stop_event.is_set():
                self.error.emit(str(e))
        finally:
            _active_threads.discard(self)
            self.done.emit()

    def stop(self, wait=True):
        """Signal the function to stop and wait for thread to finish."""
        self.running = False
        self.stop_event.set()

        if wait:
            self.wait()

    def wait_until_finished(self, timeout=None):
        """Wait until the thread finishes. Optional timeout in seconds."""
        return self.wait(int(timeout * 1000)) if timeout is not None else self.wait()

def _register_owner_cleanup(owner: object) -> None:
    try:
        if owner in _owner_cleanup_hooks:
            return
    except TypeError:
        return
    try:
        destroyed = getattr(owner, "destroyed", None)
    except Exception:
        destroyed = None
    if destroyed is None:
        _owner_cleanup_hooks[owner] = True
        return
    try:
        owner_ref = weakref.ref(owner)
    except TypeError:
        return
    try:
        destroyed.connect(lambda *_: stop_owner_threads(owner_ref(), wait=False))
        try:
            _owner_cleanup_hooks[owner] = True
        except TypeError:
            logger.warning("Exception in _register_owner_cleanup", exc_info=True)
    except Exception:
        try:
            _owner_cleanup_hooks[owner] = True
        except TypeError:
            logger.warning("Exception in _register_owner_cleanup", exc_info=True)


def _track_thread(thread: GenericWorkerThread, owner: object | None, key: object | None) -> None:
    _active_threads.add(thread)
    if owner is None:
        return
    try:
        bucket = _owner_threads.setdefault(owner, {})
        bucket.setdefault(key, set()).add(thread)
        _register_owner_cleanup(owner)
    except TypeError:
        # Owner does not support weakrefs; skip owner tracking
        return


def _untrack_thread(thread: GenericWorkerThread, owner: object | None, key: object | None) -> None:
    _active_threads.discard(thread)
    if owner is None:
        return
    try:
        bucket = _owner_threads.get(owner)
    except Exception:
        bucket = None
    if not bucket:
        return
    try:
        threads = bucket.get(key)
    except Exception:
        threads = None
    if threads:
        threads.discard(thread)
        if not threads:
            bucket.pop(key, None)
    if not bucket:
        _owner_threads.pop(owner, None)


def stop_owner_threads(owner: object | None, *, key: object | None = None, wait: bool = False, timeout: float | None = 2.0) -> None:
    if owner is None:
        return
    try:
        bucket = _owner_threads.get(owner)
    except Exception:
        bucket = None
    if not bucket:
        return
    if key is None:
        threads = {t for group in bucket.values() for t in group}
    else:
        threads = set(bucket.get(key, set()))
    for thread in list(threads):
        try:
            thread.stop(wait=False)
            if not wait:
                continue
            wait_ms = int(timeout * 1000) if timeout is not None else -1
            finished_gracefully = thread.wait(wait_ms)
            if not finished_gracefully and thread.isRunning():
                try:
                    thread.requestInterruption()
                except Exception:
                    logger.warning("Exception in stop_owner_threads", exc_info=True)
                try:
                    thread.quit()
                except Exception:
                    logger.warning("Exception in stop_owner_threads", exc_info=True)
                thread.wait(wait_ms if wait_ms != -1 else 200)
            if thread.isRunning():
                try:
                    thread.terminate()
                    thread.wait(wait_ms if wait_ms != -1 else 200)
                except Exception:
                    logger.warning("Exception in stop_owner_threads", exc_info=True)
        except Exception:
            logger.warning("Exception in stop_owner_threads", exc_info=True)


def run_in_thread(
    func,
    on_result=None,
    on_progress=None,
    on_error=None,
    on_intermediate_result=None,
    *args,
    owner: object | None = None,
    key: object | None = None,
    cancel_previous: bool = False,
    thread_priority: QThread.Priority | None = QThread.Priority.LowPriority,
    progress_min_interval_ms: int | None = 50,
    result_min_interval_ms: int | None = None,
    **kwargs,
):
    if cancel_previous and owner is not None:
        stop_owner_threads(owner, key=key, wait=False)

    thread = GenericWorkerThread(
        func,
        *args,
        thread_priority=thread_priority,
        progress_min_interval_ms=progress_min_interval_ms,
        result_min_interval_ms=result_min_interval_ms,
        **kwargs,
    )
    if owner is not None:
        try:
            owner_ref = weakref.ref(owner)
        except TypeError:
            owner_ref = None
    else:
        owner_ref = None

    def _guard(callback):
        if callback is None:
            return None
        if owner_ref is None:
            return callback
        def _wrapped(*cb_args, **cb_kwargs):
            target = owner_ref()
            if target is None:
                return
            return callback(*cb_args, **cb_kwargs)
        return _wrapped

    if on_progress:
        thread.progress.connect(_guard(on_progress))
    if on_error:
        thread.error.connect(_guard(on_error))
    if on_intermediate_result:
        thread.result.connect(_guard(on_intermediate_result))
    if on_result:  # guard: only connect if provided
        thread.finished.connect(_guard(on_result))
    _track_thread(thread, owner, key)
    thread.done.connect(lambda: _untrack_thread(thread, owner, key))
    thread.start()
    return thread


def stop_all_worker_threads(wait: bool = True, timeout: float | None = 2.0) -> None:
    """Attempt to stop all active GenericWorkerThread instances.

    Args:
        wait: Whether to wait for threads to exit after signalling them.
        timeout: Optional timeout (in seconds) for waiting on each thread.
    """

    for thread in list(_active_threads):
        try:
            thread.stop(wait=False)
            if not wait:
                continue

            # QThread.wait expects milliseconds when timeout provided
            wait_ms = int(timeout * 1000) if timeout is not None else -1
            finished_gracefully = thread.wait(wait_ms)

            if not finished_gracefully and thread.isRunning():
                # Try to interrupt and quit the thread gracefully
                try:
                    thread.requestInterruption()
                except Exception:
                    logger.warning("Exception in stop_all_worker_threads", exc_info=True)
                try:
                    thread.quit()
                except Exception:
                    logger.warning("Exception in stop_all_worker_threads", exc_info=True)
                thread.wait(wait_ms if wait_ms != -1 else 200)

            if thread.isRunning():
                # As a last resort, force termination to avoid a hung process
                try:
                    thread.terminate()
                    thread.wait(wait_ms if wait_ms != -1 else 200)
                except Exception:
                    logger.warning("Exception in stop_all_worker_threads", exc_info=True)
        except Exception:
            # Best-effort shutdown; ignore failures to keep app exit robust
            logger.warning("Exception in stop_all_worker_threads", exc_info=True)


# Make sure worker threads can't keep the interpreter alive on abrupt exits
atexit.register(lambda: stop_all_worker_threads(wait=False))
