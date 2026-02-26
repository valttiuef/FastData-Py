"""Window for displaying application information."""
import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (

    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ..localization import tr

from core.paths import get_app_metadata_path
from .licenses_window import LicensesWindow



def _load_app_metadata() -> dict:
    """Load app metadata from the repository root."""

    meta_path = get_app_metadata_path()
    try:
        with meta_path.open(encoding="utf-8") as meta_file:
            data = json.load(meta_file)
    except FileNotFoundError:
        logging.getLogger(__name__).warning("appmeta.json not found at %s", meta_path)
        return {}
    except Exception as exc:  # pragma: no cover - defensive parsing fallback
        logging.getLogger(__name__).warning("Unable to read appmeta.json: %s", exc)
        return {}

    if not isinstance(data, dict):
        logging.getLogger(__name__).warning("Unexpected appmeta.json format: %r", data)
        return {}

    return data


@lru_cache(maxsize=1)
def _get_app_metadata() -> dict:
    """Cached accessor for app metadata."""

    return _load_app_metadata()


class AboutWindow(QDialog):
    """Window with application information and quick access to licenses."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        metadata = _get_app_metadata()

        product_name = metadata.get("product_name") or metadata.get("app_name") or "FastData"
        description = metadata.get("description") or ""
        company_name = metadata.get("company_name") or ""
        version = metadata.get("version") or ""
        build = metadata.get("build") or ""
        license_type = metadata.get("license") or ""
        repo_url = metadata.get("repo_url") or ""
        issues_url = metadata.get("issues_url") or ""
        releases_url = metadata.get("releases_url") or ""

        self.setWindowTitle(tr("About {product_name}").format(product_name=product_name))
        self.setModal(True)
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # --- Title ---
        title_parts = [product_name]
        if description:
            title_parts.append(description)
        title_label = QLabel(" — ".join(title_parts), self)
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # --- Details ---
        detail_lines: list[str] = []
        if version:
            detail_lines.append(f"<b>{tr('Version')}:</b> {version}")
        if build:
            detail_lines.append(f"<b>{tr('Build')}:</b> {build}")
        if license_type:
            detail_lines.append(f"<b>{tr('License')}:</b> {license_type}")
        if company_name:
            detail_lines.append(f"<b>{tr('Developer')}:</b> {company_name}")
        detail_lines.append(f"<b>{tr('Framework')}:</b> PySide6 (Qt for Python)")

        details = QLabel("<br>".join(detail_lines), self)
        details.setTextFormat(Qt.TextFormat.RichText)
        details.setWordWrap(True)
        layout.addWidget(details)

        # --- Links ---
        links_lines: list[str] = []
        if repo_url:
            links_lines.append(f'<a href="{repo_url}">{tr("View Repository")}</a>')
        if issues_url:
            links_lines.append(f'<a href="{issues_url}">{tr("Report Issue")}</a>')
        if releases_url:
            links_lines.append(f'<a href="{releases_url}">{tr("View Releases")}</a>')

        if links_lines:
            links_label = QLabel(" | ".join(links_lines), self)
            links_label.setTextFormat(Qt.TextFormat.RichText)
            links_label.setOpenExternalLinks(True)
            links_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(links_label)

        # --- Copyright ---
        copyright_lines: list[str] = []
        if company_name:
            current_year = datetime.now().year
            copyright_lines.append(
                tr("© {year} {company}. All rights reserved.").format(year=current_year, company=company_name)
            )

        if copyright_lines:
            copyright_label = QLabel("<br>".join(copyright_lines), self)
            copyright_label.setTextFormat(Qt.TextFormat.RichText)
            copyright_label.setWordWrap(True)
            copyright_label.setStyleSheet("color: gray; font-size: 10px;")
            copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(copyright_label)

        # --- Buttons ---
        button_box = QDialogButtonBox(parent=self)
        button_box.setStandardButtons(QDialogButtonBox.StandardButton.Close)

        licenses_button = QPushButton(tr("Show Licenses…"), self)
        button_box.addButton(licenses_button, QDialogButtonBox.ButtonRole.ActionRole)
        licenses_button.clicked.connect(self._show_licenses)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)

    def _show_licenses(self) -> None:
        window = LicensesWindow(self)
        window.exec()
