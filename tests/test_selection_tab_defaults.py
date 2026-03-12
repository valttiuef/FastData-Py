import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from frontend.tabs.selections.selection_tab import _default_selection_payload


def test_default_selection_payload_enables_selections_only() -> None:
    payload = _default_selection_payload()

    assert payload.selections_enabled() is True
    assert payload.filters_enabled() is False
    assert payload.preprocessing_enabled() is False
