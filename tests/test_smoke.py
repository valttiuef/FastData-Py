# tests/test_smoke.py
import os
import sys
import traceback
from pathlib import Path
import importlib
import time

# Avoid GUI crashes in headless CI (Qt/PySide/PyQt etc.)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

IGNORE_MODULES = set()  # e.g. {"app"} if your entrypoint launches the UI on import
IGNORE_FILES = {"__init__.py", "test_smoke.py"}
IGNORE_DIR_NAMES = {"__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache", "build", "dist"}

def discover_module_names(src_dir: Path):
    for path in src_dir.rglob("*.py"):
        if path.name in IGNORE_FILES:
            continue
        if any(part in IGNORE_DIR_NAMES for part in path.parts):
            continue
        rel = path.relative_to(src_dir).with_suffix("")
        parts = list(rel.parts)
        if any(p.startswith("_") for p in parts):
            continue
        yield ".".join(parts)

def _import_all(src_dir: Path, verbose: bool = True):
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))  # make 'frontend', 'backend', etc. top-level
    failed = []
    start = time.perf_counter()
    for mod_name in sorted(set(discover_module_names(src_dir))):
        if mod_name in IGNORE_MODULES:
            if verbose:
                print(f"⏭️  Skipped: {mod_name}")
            continue
        try:
            importlib.import_module(mod_name)
            if verbose:
                print(f"✅ Imported: {mod_name}")
        except Exception:
            print(f"❌ Failed to import: {mod_name}")
            traceback.print_exc()
            failed.append(mod_name)
    dur = time.perf_counter() - start
    return failed, dur

def main():
    src_dir = (Path(__file__).parent.parent / "src").resolve()
    print(f"🚀 Smoke importing all modules under: {src_dir}\n")
    failed, dur = _import_all(src_dir, verbose=True)
    print("\n🏁 Smoke test finished.")
    if failed:
        print(f"\n⚠️  {len(failed)} module(s) failed to import:")
        for m in failed:
            print(f"  - {m}")
        print(f"\n⏱️ Duration: {dur:.2f}s")
        raise SystemExit(1)
    else:
        print("🎉 All modules imported successfully!")
        print(f"⏱️ Duration: {dur:.2f}s")
        raise SystemExit(0)

# ---------- Pytest entrypoint ----------
def test_smoke_imports():
    """Pytest-friendly wrapper."""
    src_dir = (Path(__file__).parent.parent / "src").resolve()
    print(f"\n🚀 Pytest smoke importing under: {src_dir}\n")
    failed, _ = _import_all(src_dir, verbose=True)
    assert not failed, f"{len(failed)} module(s) failed to import: {failed}"

# ---------- Script entrypoint ----------
if __name__ == "__main__":
    main()
