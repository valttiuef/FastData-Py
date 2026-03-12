import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


class _FakeDb:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def import_path(self, path, options=None):
        self.calls.append(("import_path", str(path)))


class _ImportHarness:
    def __init__(self) -> None:
        self.db = _FakeDb()
        self.progress_events: list[tuple[str, int, int, str]] = []
        self.notify_features_changed_calls = 0
        self.refresh_calls = 0
        self.reset_caches_calls = 0

        class _ProgressSignal:
            def __init__(self, outer):
                self._outer = outer

            def emit(self, phase, cur, tot, msg):
                self._outer.progress_events.append((str(phase), int(cur), int(tot), str(msg)))

        self.progress = _ProgressSignal(self)

    def notify_features_changed(self, *args, **kwargs):
        self.notify_features_changed_calls += 1

    def refresh(self):
        self.refresh_calls += 1

    def _reset_caches(self):
        self.reset_caches_calls += 1


def test_import_completion_notifies_feature_changes_without_database_refresh():
    from backend.models import ImportOptions
    from frontend.models.hybrid_pandas_model import HybridPandasModel

    harness = _ImportHarness()

    HybridPandasModel.import_files(
        harness,
        files=["a.csv", "b.csv"],
        options=ImportOptions(),
        progress_callback=None,
    )

    assert harness.notify_features_changed_calls == 1
    assert harness.reset_caches_calls == 1
    assert harness.refresh_calls == 0
    assert harness.db.calls == [("import_path", "a.csv"), ("import_path", "b.csv")]
