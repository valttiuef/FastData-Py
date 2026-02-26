
from __future__ import annotations
import sys
from dataclasses import dataclass, field

from PySide6.QtCore import QCoreApplication

from typing import Any, Callable, Sequence

TabBuilder = Callable[[Any], Any]


@dataclass(frozen=True)
class TabModuleSpec:
    key: str
    label: str
    help_key: str
    instance_attr: str
    builder: TabBuilder
    dev_enabled: bool
    release_enabled: bool
    version: str
    libraries: Sequence[str] = field(default_factory=tuple)
    known_issues: Sequence[str] = field(default_factory=tuple)
    pyinstaller_excludes: Sequence[str] = field(default_factory=tuple)
    lazy_load: bool = True

    def is_enabled(self, *, frozen: bool) -> bool:
        return self.release_enabled if frozen else self.dev_enabled


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def _build_data_tab(window: Any) -> Any:
    from .data.data_tab import DataTab

    return DataTab(window.database_model, parent=window)


def _build_selections_tab(window: Any) -> Any:
    from .selections.selection_tab import SelectionsTab

    return SelectionsTab(window.database_model, parent=window)


def _build_statistics_tab(window: Any) -> Any:
    from .statistics.statistics_tab import StatisticsTab

    return StatisticsTab(window.database_model, parent=window)


def _build_charts_tab(window: Any) -> Any:
    from .charts import ChartsTab

    return ChartsTab(window.database_model, parent=window)


def _build_som_tab(window: Any) -> Any:
    from .som.som_tab import SomTab

    return SomTab(window.database_model, parent=window)


def _build_regression_tab(window: Any) -> Any:
    from .regression.regression_tab import RegressionTab

    return RegressionTab(window.database_model, parent=window)


def _build_forecasting_tab(window: Any) -> Any:
    from .forecasting.forecasting_tab import ForecastingTab

    return ForecastingTab(window.database_model, parent=window)




def _localize_label(label: str) -> str:
    app = QCoreApplication.instance()
    if app is None:
        return label
    return app.translate("tabs", label)
TAB_MODULES: tuple[TabModuleSpec, ...] = (
    TabModuleSpec(
        key="data",
        label=_localize_label("Data"),
        help_key="tab.data",
        instance_attr="data_tab",
        builder=_build_data_tab,
        dev_enabled=True,
        release_enabled=True,
        version="0.1.0",
        libraries=("duckdb", "pandas", "numpy", "PySide6"),
        known_issues=("None reported.",),
        lazy_load=False,
    ),
    TabModuleSpec(
        key="selections",
        label=_localize_label("Selections"),
        help_key="tab.selections",
        instance_attr="selections_tab",
        builder=_build_selections_tab,
        dev_enabled=True,
        release_enabled=True,
        version="0.1.0",
        libraries=("sqlite3", "duckdb", "PySide6"),
        known_issues=("None reported.",),
        pyinstaller_excludes=("frontend.tabs.selections",),
    ),
    TabModuleSpec(
        key="statistics",
        label=_localize_label("Statistics"),
        help_key="tab.statistics",
        instance_attr="statistics_tab",
        builder=_build_statistics_tab,
        dev_enabled=True,
        release_enabled=True,
        version="0.1.0",
        libraries=("pandas", "numpy", "PySide6"),
        known_issues=("None reported.",),
        pyinstaller_excludes=("frontend.tabs.statistics", "backend.services.statistics_service"),
    ),
    TabModuleSpec(
        key="charts",
        label=_localize_label("Charts"),
        help_key="tab.charts",
        instance_attr="charts_tab",
        builder=_build_charts_tab,
        dev_enabled=True,
        release_enabled=True,
        version="0.1.0",
        libraries=("pandas", "PySide6"),
        known_issues=("None reported.",),
        pyinstaller_excludes=("frontend.tabs.charts",),
    ),
    TabModuleSpec(
        key="som",
        label=_localize_label("SOM"),
        help_key="tab.som",
        instance_attr="som_tab",
        builder=_build_som_tab,
        dev_enabled=True,
        release_enabled=True,
        version="0.1.0",
        libraries=("minisom", "numpy", "PySide6"),
        known_issues=("None reported.",),
        pyinstaller_excludes=("frontend.tabs.som", "backend.services.som_service"),
    ),
    TabModuleSpec(
        key="regression",
        label=_localize_label("Regression"),
        help_key="tab.regression",
        instance_attr="regression_tab",
        builder=_build_regression_tab,
        dev_enabled=True,
        release_enabled=True,
        version="0.1.0",
        libraries=("scikit-learn", "pandas", "numpy", "PySide6"),
        known_issues=("None reported.",),
        pyinstaller_excludes=("frontend.tabs.regression", "backend.services.regression_service"),
    ),
    TabModuleSpec(
        key="forecasting",
        label=_localize_label("Forecasting"),
        help_key="tab.forecasting",
        instance_attr="forecasting_tab",
        builder=_build_forecasting_tab,
        dev_enabled=False,
        release_enabled=False,
        version="0.1.0",
        libraries=("scikit-learn", "pandas", "numpy", "PySide6"),
        known_issues=("PyInstaller does not reckonize types when exe is created and predicted values gets pointed to wrong timestamps based on the forecasting horizon.",),
        pyinstaller_excludes=("frontend.tabs.forecasting", "backend.services.forecasting_service", "sktime"),
    ),
)


def get_tab_modules() -> tuple[TabModuleSpec, ...]:
    return TAB_MODULES


def get_runtime_tab_modules(*, frozen: bool | None = None) -> list[TabModuleSpec]:
    if frozen is None:
        frozen = is_frozen_app()
    return [module for module in TAB_MODULES if module.is_enabled(frozen=frozen)]


def get_release_tab_modules() -> list[TabModuleSpec]:
    return [module for module in TAB_MODULES if module.release_enabled]


def get_release_excludes() -> list[str]:
    excludes: list[str] = []
    for module in TAB_MODULES:
        if not module.release_enabled:
            excludes.extend(module.pyinstaller_excludes)
    return sorted(set(excludes))
