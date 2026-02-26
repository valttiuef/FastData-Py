"""
CENTRALIZED PATH RESOLUTION SYSTEM
===================================

This document describes the new centralized path resolution system in FastData.
All path references in the application now go through a single, centralized module
to ensure consistent behavior both in development and in distributed (PyInstaller) builds.

FILE: src/core/paths.py
=======================

The `paths.py` module is the single source of truth for all path resolution in the application.

KEY FEATURES:
- Works seamlessly in both development and PyInstaller bundle environments
- Handles Windows, macOS, and Linux path conventions automatically
- Centralizes all resource, database, and configuration path definitions
- Provides convenient helper functions for specific file types

USAGE:
======

Instead of scattered Path references like:
    Path(__file__).resolve().parents[3] / "resources" / "icons" / "fastdata_icon.ico"
    
Use the centralized API:
    from core.paths import get_icon_path
    icon_path = get_icon_path()

AVAILABLE FUNCTIONS:
====================

RESOURCE PATHS:
-   get_resource_path(relative_path) → Path
        Generic function for any resource file
        Example: get_resource_path("icons/fastdata_icon.ico")

-   get_app_metadata_path() → Path
        Path to appmeta.json

-   get_help_path() → Path
        Path to the help directory

-   get_theme_stylesheet_path(theme) → Path
        Path to a theme-specific QSS stylesheet

-   get_splash_image_path() → Path
        Path to the splash screen image

-   get_icon_path() → Path
        Path to the application icon

-   get_licenses_path() → Path
        Path to third-party licenses HTML file


DATABASE PATHS:
-   get_default_database_path() → Path
        Path to the default measurement database
        Location: %LOCALAPPDATA%/FastData/databases/FastData.duckdb (Windows)
                  ~/.local/share/FastData/databases/FastData.duckdb (Linux)
                  ~/Library/Application Support/FastData/databases/FastData.duckdb (macOS)

-   get_default_selection_db_path() → Path
        Path to the default selection settings database
        Location: %LOCALAPPDATA%/FastData/selection_settings/selections.db

-   get_default_log_database_path() → Path
        Path to the default log database
        Location: %LOCALAPPDATA%/FastData/logs/application.db


DIRECTORY FUNCTIONS:
-   get_base_directory() → Path
        Application base directory (repo root in dev, _MEIPASS in PyInstaller)

-   get_user_data_directory() → Path
        User data directory (creates directory if needed)

-   ensure_resource_exists(relative_path) → Path | None
        Check if a resource exists and return its path, or None

FILES UPDATED TO USE CENTRALIZED PATHS:
======================================

1. src/app.py
   - Uses get_icon_path(), get_splash_image_path(), get_theme_stylesheet_path()

2. src/frontend/windows/main_window.py
   - Uses get_app_metadata_path(), get_help_path(), get_theme_stylesheet_path()

3. src/frontend/windows/about_window.py
   - Uses get_app_metadata_path()

4. src/frontend/windows/licenses_window.py
   - Uses get_licenses_path()

5. src/core/settings_manager.py
   - Uses get_default_database_path(), get_default_selection_db_path(), get_default_log_database_path()


HOW IT WORKS:
=============

1. DEVELOPMENT MODE:
   - _get_base_path() returns the repository root (2 levels up from src/core/paths.py)
   - All resource paths are resolved relative to the repo root
   - All user data is stored in platform-specific user directories

2. PYINSTALLER BUNDLE MODE:
   - _get_base_path() detects sys._MEIPASS (set by PyInstaller)
   - Returns the temporary extraction directory
   - All resource paths are resolved relative to this directory
   - All user data is still stored in platform-specific user directories

This ensures that the application always finds its resources correctly, whether
running from source or from a distributed executable.


EXAMPLE: ADDING A NEW RESOURCE
==============================

If you need to reference a new resource file, follow these steps:

1. Add a function to src/core/paths.py:
    
    def get_my_new_file_path() -> Path:
        """Get the path to my new resource file."""
        return get_resource_path("mysubdir/myfile.ext")

2. Import and use it in your code:
    
    from core.paths import get_my_new_file_path
    
    file_path = get_my_new_file_path()
    content = file_path.read_text()

This approach ensures your code will work correctly in all deployment scenarios.
"""
