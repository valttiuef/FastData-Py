from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
# frontend/windows/import_options_dialog.py

from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

import pandas as pd

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (

    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QSpinBox, QComboBox, QCheckBox,
    QDialogButtonBox, QWidget, QGroupBox, QPushButton,
    QToolButton, QSizePolicy, QStyle
)
from ..localization import tr

from backend.models import ImportOptions, HeaderRoles
from backend.importing import _get_encoding_candidates
from frontend.models.database_model import DatabaseModel
from ..widgets.fast_table import FastTable
from ..widgets.dataframe_table_model import DataFrameTableModel
from ..widgets.help_widgets import InfoButton
from ..viewmodels.help_viewmodel import HelpViewModel, get_help_viewmodel
from ..threading.runner import run_in_thread
from ..threading.utils import run_in_main_thread
from ..tabs.data.import_preview_logic import should_enable_duckdb_csv_import

import datetime
import numpy as np
import os

if TYPE_CHECKING:
    from ..tabs.data.viewmodel import DataViewModel

# ---- Small helpers -----------------------------------------------------------
import re
import warnings
from collections import Counter

_EPOCH_S_LO = 10**9      # ~2001
_EPOCH_S_HI = 2_000_000_000  # ~2033
_EPOCH_MS_LO = 10**12
_EPOCH_MS_HI = 2_000_000_000_000
_DUCKDB_CSV_AUTO_BYTES = 100 * 1024 * 1024

# quick regexes for common date orders
_PATTERNS = [
    # ISO-like: YYYY-MM-DD with optional time
    (re.compile(r"^\s*\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?)?\s*$"), "%Y-%m-%d %H:%M:%S"),
    # YYYY/MM/DD ...
    (re.compile(r"^\s*\d{4}/\d{1,2}/\d{1,2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\s*$"), "%Y/%m/%d %H:%M:%S"),
    # DD/MM/YYYY or MM/DD/YYYY with time (dot-separated): 03/12/2025 13.47 or 03/12/2025 13.47.00
    (re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}(?:[ T]\d{2}[:\.]?\d{2}(?:[:\.]?\d{2})?)?\s*$"), "%d/%m/%Y %H:%M:%S"),
    # DD.MM.YYYY with time (dot-separated or colon-separated)
    (re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}(?:[ T]\d{2}[:\.]?\d{2}(?:[:\.]?\d{2})?)?\s*$"), "%d.%m.%Y %H:%M:%S"),
    # DD-MM-YYYY with time
    (re.compile(r"^\s*\d{1,2}-\d{1,2}-\d{4}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\s*$"), "%d-%m-%Y %H:%M:%S"),
    # DD/MM/YYYY or MM/DD/YYYY (no time)
    (re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$"), "%d/%m/%Y"),
    # DD.MM.YYYY (no time)
    (re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$"), "%d.%m.%Y"),
]

_TIME_DOT_RE = re.compile(r"\b\d{1,2}\.\d{2}(?:\.\d{2})?\b")  # e.g. 9.00 or 9.00.30

def _normalize_incomplete_time(s: str) -> str:
    """
    Normalize incomplete dot-separated times by:
    1. Adding missing seconds to 2-part times (e.g., '13.47' -> '13.47.00')
    2. Converting dots to colons for time component (e.g., '13.47.00' -> '13:47:00')
    """
    # First, add missing seconds if needed (HH.MM pattern without seconds)
    s = re.sub(r'(\d{1,2})\.(\d{2})(?![\d.])', r'\1.\2.00', s)
    # Then convert dots to colons in the time component (HH.MM.SS -> HH:MM:SS)
    # Match pattern: digits.digits.digits at the end (time with dots)
    s = re.sub(r'(\d{1,2})\.(\d{2})\.(\d{2})', r'\1:\2:\3', s)
    return s

def _looks_epoch(x) -> Optional[str]:
    """Return 's' or 'ms' if value looks like unix epoch seconds or millis."""
    try:
        if pd.isna(x): return None
        xv = float(str(x).strip())
        if _EPOCH_S_LO <= xv <= _EPOCH_S_HI:
            return "s"
        if _EPOCH_MS_LO <= xv <= _EPOCH_MS_HI:
            return "ms"
    except Exception:
        logger.warning("Exception in _looks_epoch", exc_info=True)
    return None

def _infer_dayfirst_from_samples(samples: list[str]) -> bool:
    """Heuristic: for DD/MM/YYYY vs MM/DD/YYYY, if many have first two tokens > 12, it must be dayfirst."""
    dayfirst_votes = 0
    total_votes = 0
    for s in samples:
        if not isinstance(s, str): 
            s = str(s)
        m = re.match(r"^\s*(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})", s)
        if not m:
            continue
        a, b, y = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        # If a > 12 -> dayfirst; if b > 12 -> not dayfirst; else ambiguous
        if a > 12 and b <= 12:
            dayfirst_votes += 1; total_votes += 1
        elif b > 12 and a <= 12:
            total_votes += 1
    # default to True for EU if strong signal, else False
    return dayfirst_votes >= max(1, total_votes)

def _infer_common_format(samples: list[str], dayfirst_guess: bool) -> tuple[list[str], bool]:
    """
    Try to map common shapes to strftime tokens. Returns a small list of candidate formats.
    We intentionally keep it coarse: pandas can still parse with 'coerce' if it's slightly off.
    """
    hits = Counter()
    dot_time_seen = False
    for s in samples:
        if s is None or (isinstance(s, float) and np.isnan(s)): 
            continue
        s = str(s).strip()
        if not s:
            continue
        if _TIME_DOT_RE.search(s):
            dot_time_seen = True
        for rgx, fmt in _PATTERNS:
            if rgx.match(s):
                hits[fmt] += 1
                break
    # rank formats by support
    fmts = [f for f, _ in hits.most_common()]
    # tweak ambiguous dd/mm/yyyy vs mm/dd/yyyy
    if any("%d/%m/%Y" in f for f in fmts) and not dayfirst_guess:
        fmts = [f.replace("%d/%m/%Y", "%m/%d/%Y") for f in fmts]
    # Normalize to include a bare date too
    out = []
    for f in fmts[:3]:  # cap
        if " %H:%M:%S" in f:
            out.append(f)
            out.append(f.replace(" %H:%M:%S", ""))
        else:
            out.append(f)
    # add ISO fallback if nothing matched
    if not out:
        out = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
    return (out[:3], dot_time_seen)

def _guess_datetime_info_from_df(df: pd.DataFrame, sample_rows: int = 50) -> tuple[Optional[int], Optional[str], list[str], bool, bool]:
    """
    Return (best_col_index, best_col_name, candidate_formats, dayfirst, dot_time)
    based on the first `sample_rows` non-null values per column.
    
    Avoids false positives by:
    - Requiring a minimum threshold of successfully parsed datetimes (5+ values)
    - Preferring columns with a higher *percentage* of valid datetimes (70%+)
    - Rejecting columns with too few samples
    - Scoring by percentage success rather than absolute count
    - Accepting columns that are already datetime-like types
    """
    best_idx = None
    best_score = -1.0  # use percentage (0-100) instead of count
    best_dayfirst = False
    best_dot_time = False
    best_fmts: list[str] = []
    best_name: Optional[str] = None
    
    # Minimum requirements to consider a column as datetime
    MIN_DATETIME_VALUES = 5      # need at least this many valid datetimes
    MIN_SUCCESS_PERCENTAGE = 70  # at least 70% of sampled values must parse

    cols = list(df.columns)
    for ci, c in enumerate(cols):
        series = df[c].head(sample_rows).dropna()
        if series.empty:
            continue
        
        total = len(series)
        
        # Require minimum sample size
        if total < MIN_DATETIME_VALUES:
            continue

        # Check if column is already datetime-like type (pandas read it as datetime)
        if pd.api.types.is_datetime64_any_dtype(series):
            # Column is already datetime! Accept it with high confidence
            score = 100.0  # Perfect score for pre-parsed datetime columns
            if score > best_score:
                best_score = score
                best_idx = ci
                best_name = str(c)
                best_dayfirst = False  # Already parsed, direction doesn't matter
                best_dot_time = False
                best_fmts = ["mixed"]  # Use pandas' mixed format detector
            continue

        # Epoch fast-path
        epoch_votes = 0
        for v in series:
            kind = _looks_epoch(v)
            if kind:
                epoch_votes += 1
        
        if epoch_votes >= max(3, int(0.6 * total)):
            score = (epoch_votes / total) * 100  # percentage
            if score > best_score:
                best_score = score
                best_idx = ci
                best_name = str(c)
                best_dayfirst = False
                best_dot_time = False
                best_fmts = ["epoch_ms" if _looks_epoch(series.iloc[0]) == "ms" else "epoch_s"]
            continue

        # string-based inference
        samples = [str(x) for x in series.tolist()]
        
        # Normalize incomplete dot-separated times (e.g., "13.47" -> "13.47.00")
        normalized_samples = [_normalize_incomplete_time(s) for s in samples]
        
        # Reject columns that are purely numeric (like IDs) that only parse as years
        # Check if original samples contain date separators (/, -, ., space) or time patterns
        has_date_structure = False
        for s in samples:
            if re.search(r'[/\-\.\s]', s):  # Contains date/time separator characters
                has_date_structure = True
                break
        
        if not has_date_structure:
            # Purely numeric or simple strings without date structure - likely not a datetime column
            continue
        
        # try both dayfirst True/False quickly and take the better
        # Don't use format="mixed" because it doesn't handle dot-separated times well
        # Let dateutil parser handle the flexible parsing
        # Suppress warnings about format inference - this is intentional fast guessing
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Could not infer format, so each element will be parsed individually",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Parsing dates in .* format when dayfirst=.* was specified",
                category=UserWarning,
            )
            parsed_true = pd.to_datetime(pd.Series(normalized_samples), errors="coerce", dayfirst=True)
            parsed_false = pd.to_datetime(pd.Series(normalized_samples), errors="coerce", dayfirst=False)
        ok_true = parsed_true.notna().sum()
        ok_false = parsed_false.notna().sum()

        # Require minimum datetime matches
        ok_count = max(ok_true, ok_false)
        if ok_count < MIN_DATETIME_VALUES:
            continue
        
        # Calculate success percentage
        success_pct = (ok_count / total) * 100
        
        # Only consider if success rate meets threshold
        if success_pct < MIN_SUCCESS_PERCENTAGE:
            continue

        dayfirst_guess = ok_true >= ok_false
        # finer disambiguation for dd/mm vs mm/dd
        if ok_true == ok_false:
            dayfirst_guess = _infer_dayfirst_from_samples(samples)

        # Check for dot-time patterns in original samples before normalization
        # (e.g., "22/08/2024 17.28" has dot-separated time)
        dot_time = any(_TIME_DOT_RE.search(str(s)) for s in samples if isinstance(s, str))
        
        # Infer format from normalized samples (with colons instead of dots)
        # This ensures the detected format strings match the normalized data with colons
        # e.g., "22/08/2024 17.28" -> "22/08/2024 17:28:00" for proper format matching
        fmts, _ = _infer_common_format(normalized_samples, dayfirst_guess)
        score = success_pct  # use percentage, not count

        if score > best_score:
            best_score = score
            best_idx = ci
            best_name = str(c)
            best_dayfirst = bool(dayfirst_guess)
            best_dot_time = bool(dot_time)
            best_fmts = fmts

    return best_idx, best_name, best_fmts, best_dayfirst, best_dot_time


def _read_first_line(path: str, encoding_candidates: tuple[str, ...] = ("utf-8", "utf-8-sig", "cp1252")) -> str:
    """
    Read just the first *textual* line of a file using a few common encodings.
    Returns empty string on failure.
    """
    try:
        # Try pandas' encoding if it exists in options; here we just probe
        for enc in encoding_candidates:
            try:
                with open(path, "r", encoding=enc, errors="replace") as f:
                    return f.readline().rstrip("\n\r")
            except Exception:
                continue
    except Exception:
        logger.warning("Exception in _read_first_line", exc_info=True)
    return ""


def _read_csv_with_encoding_fallback(path: str, user_encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file, trying multiple encodings if necessary.
    
    Args:
        path: Path to the CSV file
        user_encoding: User-specified encoding (tried first if provided)
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame from the CSV file
    
    Raises:
        UnicodeDecodeError: If all encoding attempts fail
    """
    # Use shared encoding candidates helper
    encodings_to_try = _get_encoding_candidates(user_encoding)
    
    last_error: Optional[Exception] = None
    
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e
            continue
    
    # If all encodings failed, raise the last error
    if last_error:
        raise last_error
    
    # Fallback (shouldn't reach here normally)
    return pd.read_csv(path, **kwargs)


def _quick_guess_delimiter_from_line(line: str) -> Optional[str]:
    """
    Very fast delimiter guesser based on ONLY the first line.
    Strategy: count candidate occurrences and pick the max (ties resolved by a priority order).
    Handles tabs, commas, semicolons, pipes, colons.
    """
    if not line:
        return None

    candidates = ["\t", ",", ";", "|", ":"]
    counts = {c: line.count(c) for c in candidates}
    # If no candidate appears, return None
    if max(counts.values(), default=0) == 0:
        return None

    # Tie-breaker priority (common CSV first): tab, comma, semicolon, pipe, colon
    priority = {c: i for i, c in enumerate(candidates)}
    # Choose delimiter with highest count; if tie, lower priority index wins
    best = sorted(counts.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]
    return best


def _is_excel(path: str) -> bool:
    low = path.lower()
    return low.endswith(".xlsx") or low.endswith(".xls")


def _is_csv_like(path: str) -> bool:
    low = path.lower()
    # Treat .csv, .tsv, .txt as "CSV-like" for UI purposes
    return any(low.endswith(ext) for ext in (".csv", ".tsv", ".txt"))


class _OptionalRowField(QWidget):
    """
    Row index (0-based) or None:
      [□ None]  [ Header (row 0) | 1.. ]
    - None  => not used
    - 0     => use the column header row
    - >=1   => use the given data row (1-based visually)
    """
    def __init__(self, parent=None, minimum: int = 0, maximum: int = 99, value: Optional[int] = None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)

        self.spin = QSpinBox(self)
        self.spin.setRange(minimum, maximum)   # allow 0
        self.spin.setSpecialValueText("0")
        self.spin.setKeyboardTracking(False)   # apply only when valid

        lay.addWidget(self.spin, 1)

        self.setValue(value)

    def value(self) -> Optional[int]:
        return int(self.spin.value())

    def setValue(self, v: Optional[int]):
        if v is None:
            self.spin.setDisabled(True)
            self.spin.setValue(self.spin.minimum())  # show "Header (row 0)" but disabled
        else:
            self.spin.setDisabled(False)
            self.spin.setValue(int(v))


def _make_collapsible_group(title: str, inner: QWidget) -> QWidget:
    """
    Creates a simple collapsible section:
      [▸] Advanced
      (when open)
      [▾] Advanced
           <inner widget>
    """
    container = QWidget()
    v = QVBoxLayout(container)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)

    toggle = QToolButton()
    toggle.setText(title)
    toggle.setCheckable(True)
    toggle.setChecked(False)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.RightArrow)
    toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _on_toggled(checked: bool):
        inner.setVisible(checked)
        toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    toggle.toggled.connect(_on_toggled)

    # Start collapsed
    inner.setVisible(False)

    v.addWidget(toggle)
    v.addWidget(inner)
    return container


class ImportOptionsDialog(QDialog):
    """
    Simplified import dialog:
      - Shows only *basic* settings up front
      - Auto-guesses CSV delimiter from the FIRST LINE (fast)
      - Only shows CSV options for CSV-like files and Excel options for Excel files
      - Pushes everything else into an "Advanced" collapsible section
      - Keeps a lightweight header preview
    """
    def __init__(
        self,
        files: Sequence[str | Path],
        opts: Optional[ImportOptions] = None,
        *,
        database_model: DatabaseModel,
        data_view_model: "DataViewModel",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("Import Options"))
        self.files = [str(f) for f in files]
        self.opts = opts or ImportOptions()
        self._database_model = database_model
        self._data_view_model = data_view_model
        self._help_viewmodel = self._resolve_help_viewmodel()
        self._preview_request_id = 0
        self._choices_request_id = 0
        self._datasets_request_id = 0
        self._startup_tasks_scheduled = False
        self._default_system_name = self.opts.system_name or "DefaultSystem"
        self._default_dataset_name = self.opts.dataset_name or "DefaultDataset"

        root = QVBoxLayout(self)

        # Top: file info + quick preview
        top = QHBoxLayout()
        root.addLayout(top)

        first_name = Path(self.files[0]).name
        self.file_label = QLabel(
            tr("Files selected: {count}  |  First: {name}").format(count=len(self.files), name=first_name)
        )
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.file_label, 1)

        # Small icon-only refresh button (refresh preview using current settings)
        self._refresh_btn = QToolButton(self)
        self._refresh_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        try:
            icon = self.style().standardIcon(QStyle.SP_BrowserReload)
        except Exception:
            icon = None
        if icon:
            self._refresh_btn.setIcon(icon)
        self._refresh_btn.setToolTip(tr("Refresh preview (use current settings)"))
        # Clicking refresh should refresh preview but skip guessing
        self._refresh_btn.clicked.connect(lambda: self._request_preview_refresh(guess=False))
        top.addWidget(self._refresh_btn, 0)

        self.preview_table = FastTable(
            self,
            select="rows",
            single_selection=False,
            editable=False,
            sorting_enabled=False,
        )
        self._preview_model = DataFrameTableModel(pd.DataFrame(), include_index=False)
        self.preview_table.setModel(self._preview_model)
        self.preview_table.setMinimumHeight(140)
        root.addWidget(self.preview_table, 1)

        # ---------------------------------------------------------------------
        # BASIC SETTINGS (always visible, minimal)
        # ---------------------------------------------------------------------
        basic_box = QGroupBox(tr("Basic settings"), self)
        basic_form = QFormLayout(basic_box)

        # System: editable combobox populated from DB
        self.system_combo = QComboBox()
        self.system_combo.setEditable(True)

        # Dataset: editable combobox populated from DB; allow special token to use sheet name
        self.dataset_combo = QComboBox()
        self.dataset_combo.setEditable(True)
        self._SHEET_TOKEN = tr("<Use sheet name>")  # visible label for users

        # Populate immediate defaults first; DB choices are loaded asynchronously.
        self.system_combo.setCurrentText(self._default_system_name)
        self.dataset_combo.addItem(self._SHEET_TOKEN)
        if self._default_dataset_name == "__sheet__":
            self.dataset_combo.setCurrentText(self._SHEET_TOKEN)
        else:
            self.dataset_combo.setCurrentText(self._default_dataset_name)

        # When system selection changes, refresh datasets list
        def _on_system_changed(_text: str):
            self._request_datasets_refresh()

        self.system_combo.currentTextChanged.connect(_on_system_changed)

        basic_form.addRow(
            tr("System name:"),
            self._wrap_with_help(self.system_combo, "controls.import.system_name"),
        )
        basic_form.addRow(
            tr("Dataset name:"),
            self._wrap_with_help(self.dataset_combo, "controls.import.dataset_name"),
        )

        # Excel basic bits (only visible for Excel)
        self.excel_basic_container = QWidget()
        excel_basic_form = QFormLayout(self.excel_basic_container)
        excel_basic_form.setContentsMargins(0, 0, 0, 0)

        self.hdr_rows = QSpinBox()
        self.hdr_rows.setRange(0, 20)
        self.hdr_rows.setValue(self.opts.excel_header_rows if self.opts.excel_header_rows is not None else 0)
        excel_basic_form.addRow(
            tr("Header row amount:"),
            self._wrap_with_help(self.hdr_rows, "controls.import.excel_header_rows"),
        )

        # Attach basic containers
        basic_form.addRow(QLabel(tr("File-type basics")))
        basic_form.addRow(self.excel_basic_container)

        root.addWidget(basic_box, 0)

        # ---------------------------------------------------------------------
        # ADVANCED SETTINGS (collapsible)
        # ---------------------------------------------------------------------
        adv_inner = QWidget(self)
        adv_grid = QGridLayout(adv_inner)

        # ---- Header structure (Advanced)
        header_box = QGroupBox(tr("Header structure"), self)
        header_form = QFormLayout(header_box)

        hr = self.opts.header_roles or HeaderRoles()

        self.base_row = _OptionalRowField(value=hr.base_name_row)
        self.source_row = _OptionalRowField(value=hr.source_row)
        self.unit_row = _OptionalRowField(value=hr.unit_row)
        self.type_row_field = _OptionalRowField(value=hr.type_row)

        self.force_meta_cols_edit = QLineEdit(
            ", ".join(getattr(self.opts, "force_meta_columns", []) or [])
        )
        self.force_meta_cols_edit.setPlaceholderText(tr("e.g. Site, Dataset ID"))

        self.ignore_cols_edit = QLineEdit(
            ", ".join(getattr(self.opts, "ignore_column_prefixes", []) or [])
        )
        self.ignore_cols_edit.setPlaceholderText(tr("e.g. Time, AE_"))

        # Header split delimiter (configure before selecting which rows map to fields)
        self.header_split_delim = QLineEdit(hr.header_split_delim or "")
        self.header_split_delim.setPlaceholderText(tr("None"))

        # Excel header rows is editable in the basic settings (one place)
        header_form.addRow(
            tr("Header delimiter:"),
            self._wrap_with_help(self.header_split_delim, "controls.import.header_delimiter"),
        )
        header_form.addRow(
            tr("Base name:"),
            self._wrap_with_help(self.base_row, "controls.import.header_base_name"),
        )
        header_form.addRow(
            tr("Source:"),
            self._wrap_with_help(self.source_row, "controls.import.header_stream"),
        )
        header_form.addRow(
            tr("Unit:"),
            self._wrap_with_help(self.unit_row, "controls.import.header_unit"),
        )
        header_form.addRow(
            tr("Type:"),
            self._wrap_with_help(self.type_row_field, "controls.import.header_qualifier"),
        )
        header_form.addRow(
            tr("Force meta columns:"),
            self._wrap_with_help(self.force_meta_cols_edit, "controls.import.force_meta_columns"),
        )
        header_form.addRow(
            tr("Ignore column prefixes:"),
            self._wrap_with_help(self.ignore_cols_edit, "controls.import.ignore_column_prefixes"),
        )

        # ---- Timestamp (Advanced)
        ts_box = QGroupBox(tr("Timestamp"), self)
        ts_form = QFormLayout(ts_box)
        self.date_col_edit = QLineEdit(self.opts.date_column or "")
        self.dayfirst_cb = QCheckBox(tr("Assume day-first dates (03/04 = 3 April)"))
        self.dayfirst_cb.setChecked(bool(self.opts.assume_dayfirst))

        # dot time & explicit formats
        self.dot_time_cb = QCheckBox(tr("Treat '9.00' as '09:00' (dot → colon)"))
        self.dot_time_cb.setChecked(bool(getattr(self.opts, "dot_time_as_colon", True)))

        # join explicitly from list or empty list to satisfy static checkers
        self.explicit_formats_edit = QLineEdit(
            ", ".join(self.opts.datetime_formats or [])
        )
        self.explicit_formats_edit.setPlaceholderText(tr("auto"))
        self.date_col_edit.setPlaceholderText(tr("auto"))

        ts_form.addRow(
            tr("Date column:"),
            self._wrap_with_help(self.date_col_edit, "controls.import.date_column"),
        )
        ts_form.addRow(self._wrap_with_help(self.dayfirst_cb, "controls.import.assume_dayfirst"))
        ts_form.addRow(self._wrap_with_help(self.dot_time_cb, "controls.import.dot_time_as_colon"))
        ts_form.addRow(
            tr("Datetime formats:"),
            self._wrap_with_help(self.explicit_formats_edit, "controls.import.datetime_formats"),
        )

        # ---- CSV hints (Advanced)
        csv_box = QGroupBox(tr("CSV settings"), self)
        csv_form = QFormLayout(csv_box)
        self.csv_decimal = QLineEdit(self.opts.csv_decimal or "")
        self.csv_encoding = QLineEdit(self.opts.csv_encoding or "")
        self.csv_decimal.setPlaceholderText(tr("auto"))
        self.csv_encoding.setPlaceholderText(tr("auto"))

        # Guess delimiter up-front from first line (user can still override)
        self.csv_delim = QLineEdit(self.opts.csv_delimiter or "")
        self.csv_delim.setPlaceholderText(tr("auto"))

        csv_form.addRow(
            tr("Delimiter:"),
            self._wrap_with_help(self.csv_delim, "controls.import.csv_delimiter"),
        )

        csv_form.addRow(
            tr("Decimal:"),
            self._wrap_with_help(self.csv_decimal, "controls.import.csv_decimal"),
        )
        csv_form.addRow(
            tr("Encoding:"),
            self._wrap_with_help(self.csv_encoding, "controls.import.csv_encoding"),
        )

        self.duckdb_csv_cb = QCheckBox(tr("Use DuckDB CSV import"))
        auto_duckdb = False
        if _is_csv_like(self.files[0]):
            auto_duckdb = should_enable_duckdb_csv_import(self.files[0], _DUCKDB_CSV_AUTO_BYTES)
        opt_value = getattr(self.opts, "use_duckdb_csv_import", None)
        if opt_value is None:
            self.duckdb_csv_cb.setChecked(auto_duckdb)
        else:
            self.duckdb_csv_cb.setChecked(bool(opt_value))
        # Keep CSV controls visible but disable them for Excel imports.
        self.duckdb_csv_cb.setEnabled(_is_csv_like(self.files[0]))
        csv_form.addRow(
            self._wrap_with_help(self.duckdb_csv_cb, "controls.import.use_duckdb_csv_import"),
        )

        # Place groups in advanced grid
        adv_grid.addWidget(header_box, 0, 0)
        adv_grid.addWidget(ts_box,      0, 1)
        # Move CSV settings to right side under Timestamp as requested
        adv_grid.addWidget(csv_box,     1, 1)

        # Disable CSV settings when importing Excel (controls stay visible).
        csv_box.setEnabled(_is_csv_like(self.files[0]))

        # Wrap as collapsible
        adv_collapsible = _make_collapsible_group(tr("Advanced settings"), adv_inner)
        root.addWidget(adv_collapsible)

        # Bottom: buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self.resize(920, self.sizeHint().height())

        # Initial preview
        self._set_preview_loading()
        # Startup work is deferred to showEvent so dialog creation stays minimal.

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_startup_tasks()

    def _schedule_startup_tasks(self) -> None:
        if self._startup_tasks_scheduled:
            return
        self._startup_tasks_scheduled = True
        # Run lightweight DB choices first, then start preview slightly later to avoid
        # GIL contention during the dialog's first paint.
        QTimer.singleShot(0, self._request_initial_db_choices)
        QTimer.singleShot(180, self._request_preview_refresh)

    # ---- Public API ----------------------------------------------------------
    def build_options(self) -> ImportOptions:
        # Merge advanced header roles
        hr = HeaderRoles(
            base_name_row=self.base_row.value(),
            source_row=self.source_row.value(),
            unit_row=self.unit_row.value(),
            type_row=self.type_row_field.value(),
        )

        explicit_formats = [
            x.strip() for x in (self.explicit_formats_edit.text() or "").split(",") if x.strip()
        ] or None

        # Prefer the "basic" Excel header rows for quick edits; keep advanced in sync
        excel_header_rows = int(self.hdr_rows.value())

        # CSV delimiter: prefer user text; if blank, attempt quick auto-guess again
        csv_delim_text = self.csv_delim.text().strip()
        if not csv_delim_text and _is_csv_like(self.files[0]):
            # Guess again on build in case the file changed
            csv_delim_text = _quick_guess_delimiter_from_line(_read_first_line(self.files[0])) or None

        # Map dataset special token to internal sentinel
        dataset_text = self.dataset_combo.currentText().strip()
        if dataset_text == getattr(self, "_SHEET_TOKEN", tr("<Use sheet name>")):
            dataset_value = "__sheet__"
        else:
            dataset_value = dataset_text or "DefaultDataset"

        sys_text = self.system_combo.currentText().strip()
        # Header split options (put into HeaderRoles)
        header_split_delim = self.header_split_delim.text() or None
        # Build header_split_order automatically from the selected row mappings.
        # Collect (row_index, field_name) for fields that are enabled (not None).
        field_rows: list[tuple[int, str]] = []
        for name, widget in (('base', self.base_row), ('source', self.source_row), ('unit', self.unit_row), ('type', self.type_row_field)):
            try:
                v = widget.value()
            except Exception:
                v = None
            if v is not None:
                field_rows.append((int(v), name))

        # Sort by row index ascending; when several fields share the same row, the
        # canonical order (base, source, unit, type) is preserved by how we
        # iterated above.
        header_split_order = [fname for (_r, fname) in sorted(field_rows, key=lambda x: x[0])]
        hr.header_split_delim = header_split_delim
        hr.header_split_order = header_split_order or None

        forced_meta_cols = [
            x.strip() for x in (self.force_meta_cols_edit.text() or "").split(",") if x.strip()
        ] or None

        ignored_cols = [
            x.strip() for x in (self.ignore_cols_edit.text() or "").split(",") if x.strip()
        ] or None

        return replace(
            self.opts,
            system_name=sys_text or "DefaultSystem",
            dataset_name=dataset_value,
            excel_header_rows=excel_header_rows,
            csv_header_rows=excel_header_rows,
            header_roles=hr,
            date_column=self.date_col_edit.text().strip() or None,
            auto_detect_datetime= not self.date_col_edit.text().strip(),
            assume_dayfirst=bool(self.dayfirst_cb.isChecked()),
            # NEW:
            dot_time_as_colon=bool(self.dot_time_cb.isChecked()),
            datetime_formats=explicit_formats,
            csv_has_header=excel_header_rows > 0,
            csv_delimiter=(csv_delim_text or None),
            csv_decimal=(self.csv_decimal.text().strip() or None),
            csv_encoding=(self.csv_encoding.text().strip() or None),
            use_duckdb_csv_import=(
                bool(self.duckdb_csv_cb.isChecked()) if _is_csv_like(self.files[0]) else None
            ),
            force_meta_columns=forced_meta_cols,
            ignore_column_prefixes=ignored_cols,
        )

    def _make_info(self, help_key: str | None) -> QWidget | None:
        if help_key and self._help_viewmodel is not None:
            return InfoButton(help_key, self._help_viewmodel, parent=self)
        return None

    def _wrap_with_help(self, widget: QWidget, help_key: str | None) -> QWidget:
        info = self._make_info(help_key)
        if info is None:
            return widget
        container = QWidget(widget.parent())
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(widget, 1)
        layout.addWidget(info, 0, Qt.AlignmentFlag.AlignRight)
        return container

    def _resolve_help_viewmodel(self) -> Optional[HelpViewModel]:
        try:
            return get_help_viewmodel()
        except Exception:
            return None

    def _set_preview_loading(self) -> None:
        self._preview_model.set_dataframe(pd.DataFrame([{"Preview": tr("Loading...")}]))

    def _load_initial_db_choices(self, system_name: Optional[str], stop_event=None) -> dict[str, object]:
        db = self._database_model.create_connection()
        systems = db.list_systems()
        try:
            datasets = db.list_datasets(system_name or None)
        except Exception:
            datasets = db.list_datasets(None)
        return {
            "systems": list(systems or []),
            "datasets": list(datasets or []),
        }

    def _request_initial_db_choices(self) -> None:
        self._choices_request_id += 1
        request_id = self._choices_request_id
        run_in_thread(
            self._load_initial_db_choices,
            on_result=lambda payload, rid=request_id: run_in_main_thread(self._apply_initial_db_choices, payload, rid),
            on_error=lambda _msg, rid=request_id: run_in_main_thread(self._apply_initial_db_choices, {}, rid),
            owner=self,
            key="import_db_choices",
            cancel_previous=True,
            system_name=self.system_combo.currentText().strip() or None,
        )

    def _apply_initial_db_choices(self, payload: dict[str, object], request_id: int) -> None:
        if request_id != self._choices_request_id:
            return

        systems = [str(x) for x in (payload.get("systems") or []) if str(x).strip()]
        datasets = [str(x) for x in (payload.get("datasets") or []) if str(x).strip()]

        selected_system = self.system_combo.currentText().strip() or self._default_system_name
        selected_dataset = self.dataset_combo.currentText().strip()

        self.system_combo.blockSignals(True)
        self.system_combo.clear()
        self.system_combo.addItems(systems)
        self.system_combo.setCurrentText(selected_system)
        self.system_combo.blockSignals(False)

        self._apply_datasets_to_combo(datasets, selected_dataset)

    def _load_datasets_for_system(self, system_name: Optional[str], stop_event=None) -> list[str]:
        db = self._database_model.create_connection()
        return list(db.list_datasets(system_name or None) or [])

    def _request_datasets_refresh(self) -> None:
        self._datasets_request_id += 1
        request_id = self._datasets_request_id
        system_name = self.system_combo.currentText().strip() or None
        run_in_thread(
            self._load_datasets_for_system,
            on_result=lambda datasets, rid=request_id: run_in_main_thread(self._apply_datasets_result, datasets, rid),
            on_error=lambda _msg, rid=request_id: run_in_main_thread(self._apply_datasets_result, [], rid),
            owner=self,
            key="import_db_datasets",
            cancel_previous=True,
            system_name=system_name,
        )

    def _apply_datasets_result(self, datasets: list[str], request_id: int) -> None:
        if request_id != self._datasets_request_id:
            return
        selected_dataset = self.dataset_combo.currentText().strip()
        self._apply_datasets_to_combo(datasets, selected_dataset)

    def _apply_datasets_to_combo(self, datasets: list[str], selected_dataset: str) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItem(self._SHEET_TOKEN)
        self.dataset_combo.addItems([str(x) for x in datasets if str(x).strip()])
        if selected_dataset:
            self.dataset_combo.setCurrentText(selected_dataset)
        elif self._default_dataset_name == "__sheet__":
            self.dataset_combo.setCurrentText(self._SHEET_TOKEN)
        else:
            self.dataset_combo.setCurrentText(self._default_dataset_name)
        self.dataset_combo.blockSignals(False)

    def _request_preview_refresh(self, guess: bool = True) -> None:
        path = self.files[0]
        self._preview_request_id += 1
        request_id = self._preview_request_id
        self._set_preview_loading()

        run_in_thread(
            self._data_view_model.load_import_preview,
            on_result=lambda payload, rid=request_id: run_in_main_thread(self._apply_preview_payload, payload, rid),
            on_error=lambda message, rid=request_id: run_in_main_thread(self._apply_preview_error, message, rid),
            owner=self,
            key="import_preview",
            cancel_previous=True,
            file_path=path,
            csv_delimiter=self.csv_delim.text().strip() or None,
            csv_decimal=self.csv_decimal.text().strip() or None,
            csv_encoding=self.csv_encoding.text().strip() or None,
            base_header_index=self.base_row.value(),
            guess=guess,
            nrows=8,
            ncolumns=32,
        )

    def _apply_preview_payload(self, payload: dict[str, object], request_id: int) -> None:
        if request_id != self._preview_request_id:
            return
        frame = payload.get("display_df")
        if isinstance(frame, pd.DataFrame):
            self._preview_model.set_dataframe(frame)
            self.preview_table.resizeColumnsToContents()
        else:
            self._preview_model.set_dataframe(pd.DataFrame())

        error = payload.get("error")
        if error:
            self._preview_model.set_dataframe(pd.DataFrame([{"Preview error": str(error)}]))
            return

        header_rows = payload.get("header_rows")
        base_header_index = payload.get("base_header_index")
        date_column = payload.get("date_column")
        datetime_formats = payload.get("datetime_formats") or []
        assume_dayfirst = payload.get("assume_dayfirst")
        dot_time_as_colon = payload.get("dot_time_as_colon")
        delimiter_guess = payload.get("csv_delimiter_guess")

        try:
            if header_rows is not None:
                self.hdr_rows.setValue(int(header_rows))
        except Exception:
            logger.warning("Exception in _apply_preview_payload", exc_info=True)
        try:
            if base_header_index is not None:
                base_idx = int(base_header_index)
                self.base_row.setValue(base_idx)
                self.source_row.setValue(base_idx + 1)
                self.unit_row.setValue(base_idx + 2)
                self.type_row_field.setValue(base_idx + 3)
        except Exception:
            logger.warning("Exception in _apply_preview_payload", exc_info=True)
        try:
            if date_column:
                self.date_col_edit.setText(str(date_column))
            if not self.csv_delim.text().strip() and delimiter_guess:
                self.csv_delim.setText(str(delimiter_guess))
            if datetime_formats:
                self.explicit_formats_edit.setText(", ".join([str(fmt) for fmt in datetime_formats]))
            if assume_dayfirst is not None:
                self.dayfirst_cb.setChecked(bool(assume_dayfirst))
            if dot_time_as_colon:
                self.dot_time_cb.setChecked(True)
        except Exception:
            logger.warning("Exception in _apply_preview_payload", exc_info=True)

    def _apply_preview_error(self, message: str, request_id: int) -> None:
        if request_id != self._preview_request_id:
            return
        self._preview_model.set_dataframe(pd.DataFrame([{"Preview error": str(message)}]))

