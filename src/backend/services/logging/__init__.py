"""Logging utilities available to both backend and frontend."""

from .log_service import LogEvent, LogService, get_log_service
from .logging_setup import configure_logging, install_global_exception_hooks
from .storage import (
    create_log_database,
    default_log_db_path,
    delete_log_database,
    fetch_all_log_records,
    fetch_log_records,
    get_log_database,
    load_log_database,
    save_log_record,
)

__all__ = [
    "LogEvent",
    "LogService",
    "configure_logging",
    "create_log_database",
    "default_log_db_path",
    "delete_log_database",
    "fetch_all_log_records",
    "fetch_log_records",
    "get_log_database",
    "get_log_service",
    "install_global_exception_hooks",
    "load_log_database",
    "save_log_record",
]
