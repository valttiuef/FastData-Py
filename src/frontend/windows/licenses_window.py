"""Window for displaying third-party licenses."""
from PySide6.QtWidgets import (

    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)
from ..localization import tr

from core.paths import get_licenses_path



class LicensesWindow(QDialog):
    """Window to display third-party licenses shipped with the app."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Third-Party Licenses"))
        self.setModal(True)
        self.resize(720, 540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._browser = QTextBrowser(self)
        self._browser.setOpenExternalLinks(True)
        self._browser.setReadOnly(True)
        layout.addWidget(self._browser, stretch=1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self._load_licenses()

    def _load_licenses(self) -> None:
        """Load the licenses HTML from the resources directory."""

        license_path = get_licenses_path()
        try:
            html = license_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._browser.setPlainText(tr("License file not found."))
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._browser.setPlainText(tr("Unable to load licenses: {error}").format(error=exc))
        else:
            self._browser.setHtml(html)
