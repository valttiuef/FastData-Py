from __future__ import annotations
"""Frontend package initialisation."""


from qt_compat import ensure_qt


# Ensure that importing frontend modules during smoke tests succeeds even when
# the PySide6 shared libraries are not available.  The real application still
# requires a functional PySide6 installation; this simply provides lightweight
# stand-ins for headless environments.
ensure_qt()

