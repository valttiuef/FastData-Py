import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frontend.tabs.regression.regression_tab import RegressionTab


class _SidebarStub:
    def __init__(self) -> None:
        self.options = None

    def selected_target_payloads(self):
        return [{"feature_id": 2, "name": "Target"}]

    def available_feature_payloads(self):
        return [
            {"feature_id": 1, "name": "Input A"},
            {"feature_id": 2, "name": "Target"},
            {"feature_id": 3, "name": "Input B"},
        ]

    def payload_label(self, payload):
        return str(payload.get("name") or payload.get("label") or "")

    def set_stratify_options(self, options):
        self.options = list(options)


class _ViewModelStub:
    def available_group_kinds(self):
        return [("batch", "Batch")]


def test_regression_stratify_options_use_available_features_not_selected_inputs():
    tab = RegressionTab.__new__(RegressionTab)
    tab.sidebar = _SidebarStub()
    tab._view_model = _ViewModelStub()

    RegressionTab._update_stratify_options(tab)

    assert tab.sidebar.options is not None
    labels = [label for label, _payload in tab.sidebar.options]
    assert labels == ["Target", "Input A", "Input B", "Batch"]
