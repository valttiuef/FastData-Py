
from __future__ import annotations
import logging
from typing import Optional

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

import platform, subprocess, configparser, pathlib

from core.paths import get_resource_path

logger = logging.getLogger(__name__)

THEMES = {
    "dark": {
        "BG_1": "#2d2d2d",
        "BG_2": "#3a3a3a",
        "BG_3": "#4a4a4a",
        "TXT_1": "#dddddd",
        "BRD_1": "#3a3a3a",
        "BRD_2": "#4a4a4a",
        "IN_BG": "#1e1e1e",
        "BG_DIS": "#2a2a2a",
        "TXT_DIS": "#777777",
        "BRD_DIS": "#555555",
        "BRD_3": "#444444",
        "IN_BG_DIS": "#2a2a2a",
        "SEL_BG": "#505050",
    },
    "light": {
        "BG_1": "#f4f4f4",
        "BG_2": "#e6e6e6",
        "BG_3": "#ededed",
        "TXT_1": "#222222",
        "BRD_1": "#cfcfcf",
        "BRD_2": "#cfcfcf",
        "IN_BG": "#ffffff",
        "BG_DIS": "#f2f2f2",
        "TXT_DIS": "#9a9a9a",
        "BRD_DIS": "#d6d6d6",
        "BRD_3": "#dddddd",
        "IN_BG_DIS": "#f2f2f2",
        "SEL_BG": "#dcdcdc",
    },
}

THEME_STYLESHEETS = {
    "dark": "style/dark.qss",
    "light": "style/light.qss",
}

_QSS_TEMPLATE_CACHE: dict[str, str] = {}

def detect_system_theme() -> str:
    """
    Best-effort cross-platform detection.
    Returns "dark" or "light". Falls back to "light".
    """
    sysname = platform.system()

    if sysname == "Windows":
        # Windows 10/11: registry key AppsUseLightTheme (1=light, 0=dark)
        try:
            from PySide6.QtCore import QSettings
            s = QSettings(
                r"HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
                QSettings.NativeFormat,
            )
            v = s.value("AppsUseLightTheme")
            if v is not None:
                return "light" if int(v) == 1 else "dark"
        except Exception:
            logger.warning("Exception in detect_system_theme", exc_info=True)

    elif sysname == "Darwin":
        # macOS: "defaults read -g AppleInterfaceStyle" prints "Dark" in dark mode
        try:
            out = subprocess.check_output(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                stderr=subprocess.STDOUT
            ).decode().strip()
            return "dark" if out == "Dark" else "light"
        except Exception:
            # No key => light (AppleInterfaceStyle only exists in Dark)
            return "light"

    else:
        # Linux / *nix
        # GNOME 42+: color-scheme is 'default' or 'prefer-dark'
        try:
            out = subprocess.check_output(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"]
            ).decode().strip().strip("'")
            if "dark" in out.lower():
                return "dark"
        except Exception:
            logger.warning("Exception in detect_system_theme", exc_info=True)

        # Older GNOME/GTK: theme name often ends with '-dark'
        try:
            out = subprocess.check_output(
                ["gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"]
            ).decode().strip().strip("'")
            if out.lower().endswith("dark"):
                return "dark"
        except Exception:
            logger.warning("Exception in detect_system_theme", exc_info=True)

        # KDE: ~/.config/kdeglobals [General] ColorScheme often contains 'Dark'
        try:
            cfg = configparser.ConfigParser()
            cfg.read(pathlib.Path.home() / ".config" / "kdeglobals", encoding="utf-8")
            scheme = cfg.get("General", "ColorScheme", fallback="")
            if "dark" in scheme.lower():
                return "dark"
        except Exception:
            logger.warning("Exception in detect_system_theme", exc_info=True)

    return "light"


def _qcolor(value: str) -> QColor:
    color = QColor()
    color.setNamedColor(value)
    return color


def _build_palette(theme_colors: dict[str, str]) -> QPalette:
    palette = QPalette()

    active_inactive = (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive)

    def set_roles(groups: tuple[QPalette.ColorGroup, ...], role: QPalette.ColorRole, value: str) -> None:
        qcolor = _qcolor(value)
        for group in groups:
            palette.setColor(group, role, qcolor)

    set_roles(active_inactive, QPalette.Window, theme_colors["BG_1"])
    set_roles(active_inactive, QPalette.WindowText, theme_colors["TXT_1"])
    set_roles(active_inactive, QPalette.Base, theme_colors["IN_BG"])
    set_roles(active_inactive, QPalette.AlternateBase, theme_colors["BG_2"])
    set_roles(active_inactive, QPalette.ToolTipBase, theme_colors["BG_2"])
    set_roles(active_inactive, QPalette.ToolTipText, theme_colors["TXT_1"])
    set_roles(active_inactive, QPalette.Text, theme_colors["TXT_1"])
    set_roles(active_inactive, QPalette.Button, theme_colors["BG_2"])
    set_roles(active_inactive, QPalette.ButtonText, theme_colors["TXT_1"])
    set_roles(active_inactive, QPalette.Highlight, theme_colors["SEL_BG"])
    set_roles(active_inactive, QPalette.HighlightedText, theme_colors["TXT_1"])
    set_roles(active_inactive, QPalette.Link, theme_colors["SEL_BG"])
    set_roles(active_inactive, QPalette.PlaceholderText, theme_colors["TXT_DIS"])

    set_roles(active_inactive, QPalette.Mid, theme_colors["BRD_1"])
    set_roles(active_inactive, QPalette.Midlight, theme_colors["BRD_2"])
    set_roles(active_inactive, QPalette.Dark, theme_colors["BRD_2"])
    set_roles(active_inactive, QPalette.Shadow, theme_colors["BRD_3"])
    set_roles(active_inactive, QPalette.Light, theme_colors["BG_3"])

    palette.setColor(QPalette.Disabled, QPalette.Window, _qcolor(theme_colors["BG_1"]))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, _qcolor(theme_colors["TXT_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.Text, _qcolor(theme_colors["TXT_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, _qcolor(theme_colors["TXT_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.Base, _qcolor(theme_colors["IN_BG_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.Button, _qcolor(theme_colors["BG_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.AlternateBase, _qcolor(theme_colors["BG_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.ToolTipBase, _qcolor(theme_colors["BG_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.ToolTipText, _qcolor(theme_colors["TXT_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, _qcolor(theme_colors["BG_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, _qcolor(theme_colors["TXT_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.Link, _qcolor(theme_colors["BRD_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.PlaceholderText, _qcolor(theme_colors["TXT_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.Mid, _qcolor(theme_colors["BRD_3"]))
    palette.setColor(QPalette.Disabled, QPalette.Dark, _qcolor(theme_colors["BRD_DIS"]))
    palette.setColor(QPalette.Disabled, QPalette.Shadow, _qcolor(theme_colors["BRD_DIS"]))

    return palette


def apply_theme(app: QApplication, theme: str) -> None:
    if theme not in THEMES:
        raise ValueError(f"Unknown theme '{theme}'")
    palette = _build_palette(THEMES[theme])
    app.setPalette(palette)
    qss_rel_path = THEME_STYLESHEETS.get(theme)
    if not qss_rel_path:
        return
    try:
        p = get_resource_path(qss_rel_path)
        if p.exists():
            cache_key = str(p.resolve())
            raw = _QSS_TEMPLATE_CACHE.get(cache_key)
            if raw is None:
                raw = p.read_text(encoding="utf-8")
                _QSS_TEMPLATE_CACHE[cache_key] = raw
            app.setStyleSheet(raw)
    except Exception:
        # Best-effort: ignore stylesheet failures and keep palette
        logger.warning("Exception in apply_theme", exc_info=True)
