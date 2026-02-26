"""Helpers for working with PySide6 in environments without Qt libraries.

This module provides :func:`ensure_qt`, which attempts to import PySide6 and,
if it fails because the native Qt dependencies are missing (common in CI
containers), installs lightweight stub modules so that imports succeed during
smoke tests.  The real application still requires a functional PySide6
installation at runtime; the stubs merely offer placeholder classes used
purely for import-time attribute access.
"""

from __future__ import annotations

from importlib import import_module
import sys


def _has_working_qt() -> bool:
    """Return ``True`` if importing ``PySide6.QtWidgets`` succeeds."""

    try:
        import_module("PySide6.QtWidgets")
        return True
    except Exception:
        # Remove any partially-imported PySide6 modules so that stub
        # installation has a clean slate.
        for key in [k for k in sys.modules if k.startswith("PySide6")]:
            sys.modules.pop(key, None)
        return False


def ensure_qt() -> None:
    """Ensure that importing ``PySide6`` will succeed.

    When PySide6 is available this function does nothing.  If importing the
    package raises an exception (for instance ``ImportError`` caused by
    ``libGL`` being unavailable) we lazily install stub modules that satisfy
    attribute lookups used in the frontend code.  The stubs live under
    :mod:`frontend.qt_stub`.
    """

    if not _has_working_qt():
        from frontend.qt_stub import install_stubs

        install_stubs()

