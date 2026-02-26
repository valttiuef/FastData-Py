"""
Centralized path resolution for the FastData application.

This module provides a single source of truth for resolving all application paths,
including resource files, databases, logs, help files, and configuration files.
It handles both development and distributed (PyInstaller) environments.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def _get_base_path() -> Path:
    """
    Get the base path for the application.
    
    In a PyInstaller bundle, this returns the temporary extraction directory (_MEIPASS).
    In development, this returns the repository root.
    
    Returns:
        The base path as a Path object.
    """
    # If running in PyInstaller bundle
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    
    # In development, go up from src/core to the repo root
    return Path(__file__).resolve().parents[2]


def _get_user_data_path() -> Path:
    """
    Get the user data directory for application settings and user databases.
    
    On Windows: %LOCALAPPDATA%/FastData
    On macOS: ~/Library/Application Support/FastData
    On Linux: ~/.local/share/FastData
    
    Returns:
        The user data path as a Path object.
    """
    if os.name == "nt":  # Windows
        base = Path(os.getenv("LOCALAPPDATA", Path.home()))
    elif sys.platform == "darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        base = Path.home() / ".local" / "share"
    
    return base / "FastData"


# ============================================================================
# PUBLIC API - Path Resolution Functions
# ============================================================================


def get_resource_path(relative_path: str) -> Path:
    """
    Get the absolute path to a resource file.
    
    Resources include icons, images, stylesheets, and help files.
    Works correctly both in development and in PyInstaller bundles.
    
    Args:
        relative_path: Path relative to the resources directory (e.g., "icons/fastdata_icon.ico")
    
    Returns:
        Absolute path to the resource as a Path object.
    
    Example:
        icon_path = get_resource_path("icons/fastdata_icon.ico")
        stylesheet_path = get_resource_path("style/dark.qss")
    """
    return _get_base_path() / "resources" / relative_path


def get_app_metadata_path() -> Path:
    """
    Get the path to appmeta.json file.
    
    In a PyInstaller bundle, will prefer appmeta_build.json if it exists.
    This file is generated at build time and contains build information.
    
    Returns:
        Absolute path to appmeta.json (or appmeta_build.json) as a Path object.
    """
    base = _get_base_path()
    
    # In PyInstaller bundles, prefer the build-time generated metadata
    if getattr(sys, "_MEIPASS", None):
        build_meta = base / "appmeta_build.json"
        if build_meta.exists():
            return build_meta
    
    # Fall back to source appmeta.json
    return base / "appmeta.json"


def get_help_path() -> Path:
    """
    Get the path to the help directory.
    
    Returns:
        Absolute path to the help directory as a Path object.
    """
    return get_resource_path("help")


def get_theme_stylesheet_path(theme: str) -> Path:
    """
    Get the path to a theme-specific stylesheet (QSS file).

    Returns:
        Absolute path to the theme stylesheet as a Path object.
    """
    return get_resource_path(f"style/{theme}.qss")


def get_splash_image_path() -> Path:
    """
    Get the path to the splash screen image.
    
    Returns:
        Absolute path to splash.png as a Path object.
    """
    return get_resource_path("images/splash.png")


def get_icon_path() -> Path:
    """
    Get the path to the application icon.
    
    Returns:
        Absolute path to fastdata_icon.ico as a Path object.
    """
    return get_resource_path("icons/fastdata_icon.ico")


def get_licenses_path() -> Path:
    """
    Get the path to the third-party licenses file.
    
    Returns:
        Absolute path to third_party_licenses.html as a Path object.
    """
    return get_resource_path("licenses/third_party_licenses.html")


# ============================================================================
# DATABASE PATHS
# ============================================================================


def get_default_database_path() -> Path:
    """
    Get the default path for the measurement database.
    
    Creates the directory if it doesn't exist.
    
    Returns:
        Absolute path to the default database file as a Path object.
    """
    db_folder = _get_user_data_path() / "databases"
    db_folder.mkdir(parents=True, exist_ok=True)
    return db_folder / "FastData.duckdb"


def get_default_selection_db_path() -> Path:
    """
    Get the default path for the selection settings database.
    
    Creates the directory if it doesn't exist.
    
    Returns:
        Absolute path to the default selection database file as a Path object.
    """
    db_folder = _get_user_data_path() / "selection_settings"
    db_folder.mkdir(parents=True, exist_ok=True)
    return db_folder / "selections.db"


def get_default_log_database_path() -> Path:
    """
    Get the default path for the application log database.
    
    Creates the directory if it doesn't exist.
    
    Returns:
        Absolute path to the default log database file as a Path object.
    """
    db_folder = _get_user_data_path() / "logs"
    db_folder.mkdir(parents=True, exist_ok=True)
    return db_folder / "application.db"


# ============================================================================
# DIRECTORY FUNCTIONS
# ============================================================================


def get_base_directory() -> Path:
    """
    Get the application base directory.
    
    In development, this is the repository root.
    In a PyInstaller bundle, this is the temporary extraction directory.
    
    Returns:
        The base directory path as a Path object.
    """
    return _get_base_path()


def get_user_data_directory() -> Path:
    """
    Get the user data directory for application data.
    
    Returns:
        The user data directory path as a Path object.
    """
    directory = _get_user_data_path()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_resource_exists(relative_path: str) -> Optional[Path]:
    """
    Check if a resource file exists and return its path, or None if not found.
    
    Args:
        relative_path: Path relative to the resources directory.
    
    Returns:
        Absolute path to the resource if it exists, None otherwise.
    """
    path = get_resource_path(relative_path)
    return path if path.exists() else None
