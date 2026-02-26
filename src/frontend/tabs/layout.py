from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QHBoxLayout, QTabBar, QTabWidget, QWidget

from ..widgets.help_widgets import InfoButton
from .tab_icons import resolve_tab_icon

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from ..viewmodels.help_viewmodel import HelpViewModel

def add_tab_with_help(
    tabs: QTabWidget,
    widget: QWidget,
    label: str,
    help_key: str,
    help_viewmodel: Optional["HelpViewModel"] = None,
    *,
    index: Optional[int] = None,
    tab_key: Optional[str] = None,
):
    """Insert a tab with a help button on the right-hand side.

    Exposes the same help button styling used by :func:`create_main_tabs` so
    callers can attach tabs incrementally (for lazy-loading, testing, etc.).
    """
    tab_bar = tabs.tabBar()
    if index is None:
        tab_index = tabs.addTab(widget, label)
    else:
        tab_index = tabs.insertTab(index, widget, label)

    tabs.setTabIcon(tab_index, resolve_tab_icon(tab_key=tab_key, help_key=help_key))

    if help_viewmodel is not None:
        container = QWidget(tab_bar)
        lay = QHBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        btn = InfoButton(help_key, help_viewmodel, parent=container)

        lay.addWidget(btn)

        tab_bar.setTabButton(tab_index, QTabBar.ButtonPosition.RightSide, container)

    return tab_index


def create_main_tabs(
    parent,
    data_tab_widget,
    selections_tab_widget=None,
    statistics_tab_widget=None,
    charts_tab_widget=None,
    som_tab_widget=None,
    regression_tab_widget=None,
    forecasting_tab_widget=None,
    help_viewmodel: Optional["HelpViewModel"] = None,
):
    """Create and return a QTabWidget populated with the app's main tabs.

    Parameters
    - parent: QWidget parent for the QTabWidget
    - data_tab_widget: a QWidget instance to use for the Data tab
    - som_tab_widget: optional QWidget with the SOM visualisation tab
    """
    tabs = QTabWidget(parent)
    tabs.setDocumentMode(True)
    tabs.setTabPosition(QTabWidget.TabPosition.North)
    tabs.setIconSize(QSize(20, 20))

    add_tab_with_help(tabs, data_tab_widget, "Data", "tab.data", help_viewmodel, tab_key="data")
    if selections_tab_widget is not None:
        add_tab_with_help(
            tabs,
            selections_tab_widget,
            "Selections",
            "tab.selections",
            help_viewmodel,
            tab_key="selections",
        )
    if charts_tab_widget is not None:
        add_tab_with_help(tabs, charts_tab_widget, "Charts", "tab.charts", help_viewmodel, tab_key="charts")
    if statistics_tab_widget is not None:
        add_tab_with_help(
            tabs,
            statistics_tab_widget,
            "Statistics",
            "tab.statistics",
            help_viewmodel,
            tab_key="statistics",
        )
    if som_tab_widget is not None:
        add_tab_with_help(tabs, som_tab_widget, "SOM", "tab.som", help_viewmodel, tab_key="som")
    if regression_tab_widget is not None:
        add_tab_with_help(
            tabs,
            regression_tab_widget,
            "Regression",
            "tab.regression",
            help_viewmodel,
            tab_key="regression",
        )
    if forecasting_tab_widget is not None:
        add_tab_with_help(
            tabs,
            forecasting_tab_widget,
            "Forecasting",
            "tab.forecasting",
            help_viewmodel,
            tab_key="forecasting",
        )

    tabs.setCurrentIndex(0)

    return tabs
