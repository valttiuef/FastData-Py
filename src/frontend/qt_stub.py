from __future__ import annotations
"""Lightweight PySide6 stubs used for import-only smoke tests.

The real application depends on PySide6 at runtime.  When that library cannot
be imported (e.g. in a headless CI container without Qt), :func:`install_stubs`
injects simplified stand-ins into :mod:`sys.modules` so that our modules can be
imported without raising ``ImportError``.
"""


from types import ModuleType
from typing import Any, Callable
import sys


class _QtEnum(int):
    """A minimal integer-like enum used by the stubs."""

    def __new__(cls, value: int = 0) -> "_QtEnum":  # noqa: D401 - simple shim
        return int.__new__(cls, value)

    # Basic bitwise/boolean operations to keep UI code happy when combining
    # enum flags (e.g. ``Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter``)
    def __or__(self, other: Any) -> "_QtEnum":  # noqa: D401 - simple shim
        return _QtEnum(int(self) | int(other))

    __ror__ = __or__

    def __and__(self, other: Any) -> "_QtEnum":  # noqa: D401 - simple shim
        return _QtEnum(int(self) & int(other))

    def __invert__(self) -> "_QtEnum":  # noqa: D401 - simple shim
        return _QtEnum(~int(self))

    def __getattr__(self, _: str) -> "_QtEnum":  # allow chained attr access
        return _QtEnum()


class _Signal:
    """Bare-bones replacement for :class:`PySide6.QtCore.Signal`."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self._subscribers: list[Callable[..., Any]] = []

    def __get__(self, instance: Any, owner: type | None = None) -> "_Signal":
        return self

    def connect(self, slot: Callable[..., Any]) -> None:
        self._subscribers.append(slot)

    def emit(self, *args: Any, **kwargs: Any) -> None:
        for slot in list(self._subscribers):
            try:
                slot(*args, **kwargs)
            except Exception:
                # Swallow exceptions because these stubs are only used in tests
                pass


class _QtModule(ModuleType):
    """Module type that lazily fabricates stub classes on attribute access."""

    def __getattr__(self, name: str) -> Any:  # noqa: D401 - dynamic shim
        if name == "Signal":
            value = _Signal
        elif name == "Qt":
            value = _QtEnum()
        else:
            value = type(name, (), {"__module__": self.__name__})
        setattr(self, name, value)
        return value


def install_stubs() -> None:
    """Install stub PySide6 modules into :mod:`sys.modules`."""

    if "PySide6" in sys.modules:
        return

    package = ModuleType("PySide6")
    package.__path__ = []  # type: ignore[attr-defined]
    package.__file__ = "<qt-stub>"
    sys.modules["PySide6"] = package

    for submodule_name in ("QtCore", "QtGui", "QtWidgets", "QtCharts", "QtDataVisualization"):
        module = _QtModule(f"PySide6.{submodule_name}")
        module.__file__ = "<qt-stub>"
        module.__package__ = "PySide6"
        setattr(package, submodule_name, module)
        sys.modules[f"PySide6.{submodule_name}"] = module

