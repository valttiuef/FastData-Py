# main.spec  (single cross-platform spec with per-OS output folders)

import os, sys, platform, json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path

block_cipher = None
project_path = os.path.abspath(".")
src_path = os.path.join(project_path, "src")

# ✅ Make sure "src" is importable while the spec is executed (needed for collect_submodules)
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

from frontend.tabs.tab_modules import get_release_excludes, get_release_tab_modules

# -------- helpers --------
def run_git(args, cwd):
    """Return git command output or None (works even if git is missing)."""
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def normalize_semver3(v):
    """Convert '1.2.3.4' -> '1.2.3' for user-facing version strings."""
    parts = re.findall(r"\d+", str(v))
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return "0.0.0"

# -------- Load app metadata (base) --------
with open(os.path.join(project_path, "appmeta.json"), "r", encoding="utf-8") as f:
    META = json.load(f)

APP_NAME      = META.get("app_name", "FastData")
PRODUCT_NAME  = META.get("product_name", APP_NAME)
COMPANY_NAME  = META.get("company_name", "")
DESCRIPTION   = META.get("description", "")
APP_VERSION_RAW = META.get("version", "0.0.0")  # can be 1.0.0.0 (Windows style)
ICON_WIN_META = META.get("icon_windows")   # e.g. "resources/icons/fastdata_icon.ico"
ICON_LINUX    = META.get("icon_linux")
IDENTIFIER    = META.get("identifier", "")  # used at runtime if you want AppUserModelID

# User-facing version string (SemVer-ish)
APP_VERSION_3 = normalize_semver3(APP_VERSION_RAW)

# Git build stamp (computed at spec time, i.e. build time)
_commit   = run_git(["rev-parse", "--short", "HEAD"], project_path) or "nogit"
_describe = run_git(["describe", "--tags", "--always", "--dirty"], project_path) or _commit
_dirty    = _describe.endswith("-dirty")
_build_time_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

# This is what users should report (maps directly to a commit)
BUILD_STRING = f"{APP_VERSION_3}+g{_commit}" + ("-dirty" if _dirty else "")

# ----- Platform switches -----
IS_WIN = sys.platform.startswith("win")
ARCH = platform.machine().lower().replace("amd64", "x64").replace("x86_64", "x64")
PLAT = "win" if IS_WIN else "linux"

# ----- Per-platform venv + xgboost lib names -----
if IS_WIN:
    venv_site = os.path.join(project_path, ".venv-windows", "Lib", "site-packages")
    xgb_libname = "xgboost.dll"
    icon_file = ICON_WIN_META or os.path.join(project_path, "resources", "icons", "fastdata_icon.ico")
else:
    venv_site = os.path.join(
        project_path, ".venv-linux",
        "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages"
    )
    xgb_libname = "libxgboost.so"
    icon_file = None  # optional: ICON_LINUX

# (Optional but helpful) Ensure PyInstaller sees the intended venv packages first
# NOTE: This does NOT replace running PyInstaller inside the correct venv; it just reduces surprises.
if os.path.isdir(venv_site) and venv_site not in sys.path:
    sys.path.insert(0, venv_site)

# Build paths to xgboost artifacts
xgb_base = os.path.join(venv_site, "xgboost")
xgb_lib = os.path.join(xgb_base, "lib", xgb_libname)
xgb_version = os.path.join(xgb_base, "VERSION")

# ----- Output folder name inside dist/build -----
app_out_name = f"{APP_NAME}-{PLAT}-{ARCH or 'x64'}"

# ----- Files to bundle -----
binaries = [(xgb_lib, "xgboost/lib")] if os.path.exists(xgb_lib) else []

from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_submodules

# Start datas list
datas = []

# Bundle database SQL files (duckdb/postgres/common)
sql_dir = Path(project_path) / "src" / "backend" / "data_db" / "sql"
if sql_dir.exists():
    for p in sql_dir.rglob("*.sql"):
        rel = p.relative_to(sql_dir)
        dest_dir = os.path.join("backend", "data_db", "sql", rel.parent.as_posix())
        datas.append((str(p), dest_dir))

# Add duckdb metadata
datas += copy_metadata("duckdb")

release_modules = get_release_tab_modules()
release_libraries = {lib for module in release_modules for lib in module.libraries}
include_sktime = "sktime" in release_libraries

# --- optional legacy sktime support ---
# Only include sktime assets when a release module explicitly requires it.
if include_sktime:
    # This supports legacy sktime-based forecasting code paths kept as reference.
    # (sktime uses registries + optional deps + some dynamic patterns)
    datas += copy_metadata("sktime")
    datas += collect_data_files("sktime")

# Optional extra sktime non-Python package data for legacy/reference builds
# (collect_data_files already covers these; leaving it is fine if you like)
# datas += collect_data_files("sktime")

# Bundle resources folder
resource_dir = os.path.join(project_path, "resources")
if os.path.exists(resource_dir):
    for p in Path(resource_dir).rglob("*"):
        if p.is_file():
            rel = p.relative_to(resource_dir)
            dest_dir = os.path.join("resources", rel.parent.as_posix())
            datas.append((str(p), dest_dir))

# Bundle xgboost VERSION file if present
if os.path.exists(xgb_version):
    datas.append((xgb_version, "xgboost"))

# ---- Generate build-stamped metadata for runtime (keep repo clean) ----
_generated_dir = os.path.join(project_path, "build", "generated")
os.makedirs(_generated_dir, exist_ok=True)

_appmeta_build = os.path.join(_generated_dir, "appmeta_build.json")

META_BUILD = dict(META)
META_BUILD.update({
    # keep your base fields
    "version": APP_VERSION_3,             # user-facing
    "version_win": str(APP_VERSION_RAW),  # for Windows file properties if you want
    "build": BUILD_STRING,                # the “tell me this in bug reports” string
    "git_commit": _commit,
    "git_describe": _describe,
    "dirty": _dirty,
    "build_time_utc": _build_time_utc,
})

with open(_appmeta_build, "w", encoding="utf-8") as f:
    json.dump(META_BUILD, f, indent=2)

# Bundle into app root inside the frozen app
datas.append((_appmeta_build, "."))

# Also write build string into a simple file (optional)
_version_txt = os.path.join(project_path, "_build_VERSION.txt")
with open(_version_txt, "w", encoding="utf-8") as f:
    f.write(BUILD_STRING)
datas.append((_version_txt, "."))

# ✅ Fix for dynamic/lazy imports in backend.services (__getattr__ + import_module)
hiddenimports = []

# Always include the exact failing module explicitly:
hiddenimports += ["backend.services.clustering"]

# Try collecting all submodules too:
try:
    hiddenimports += collect_submodules("backend.services")
except Exception:
    pass

# --- sktime hidden imports for optional legacy support ---
# This can increase size/time, but keeps frozen builds stable when sktime is included.
if include_sktime:
    try:
        hiddenimports += collect_submodules("sktime")
    except Exception:
        # fallback: include common registry/typing areas that often trip frozen builds
        hiddenimports += [
            "sktime.datatypes",
            "sktime.datatypes._registry",
            "sktime.datatypes._check",
            "sktime.datatypes._hierarchical",
            "sktime.datatypes._hierarchical._check",
            "sktime.datatypes._hierarchical._convert",
            "sktime.split",
            "sktime.split.base",
            "sktime.split.base._base_splitter",
        ]

excludes = get_release_excludes()

# ---------- Optional Windows version resource ----------
version_info = None
if IS_WIN:
    from PyInstaller.utils.win32.versioninfo import (
        VSVersionInfo, FixedFileInfo, StringFileInfo, StringTable, StringStruct,
        VarFileInfo, VarStruct
    )

    def parse_ver_tuple(s, pad=4):
        nums = [int(x) for x in re.findall(r"\d+", str(s))[:pad]]
        nums += [0] * (pad - len(nums))
        return tuple(nums[:pad])

    FILEVERS = parse_ver_tuple(APP_VERSION_RAW)  # keep 4-part safe for Windows
    PRODVERS = FILEVERS

    version_info = VSVersionInfo(
        ffi=FixedFileInfo(
            filevers=FILEVERS,
            prodvers=PRODVERS,
            mask=0x3F,
            flags=0x0,
            OS=0x40004,
            fileType=0x1,
            subtype=0x0,
            date=(0, 0),
        ),
        kids=[
            StringFileInfo([StringTable(
                u'040904B0',
                [
                    StringStruct(u'CompanyName',      COMPANY_NAME),
                    StringStruct(u'FileDescription',  DESCRIPTION),
                    StringStruct(u'FileVersion',      str(APP_VERSION_RAW)),
                    StringStruct(u'InternalName',     APP_NAME),
                    StringStruct(u'OriginalFilename', f'{APP_NAME}_v{APP_VERSION_RAW}.exe'),
                    StringStruct(u'ProductName',      PRODUCT_NAME),
                    StringStruct(u'ProductVersion',   str(APP_VERSION_RAW)),
                    StringStruct(u'LegalCopyright',   u'© ' + COMPANY_NAME),
                    # Put commit build info somewhere visible in Properties → Details
                    StringStruct(u'Comments',         f"{DESCRIPTION} | {BUILD_STRING} | {_build_time_utc}"),
                ]
            )]),
            VarFileInfo([VarStruct(u'Translation', [0x0409, 1200])]),
        ]
    )

# ---------- PyInstaller build objects ----------
hookspath = [os.path.join(project_path, "hooks")] if include_sktime else []

a = Analysis(
    ["src/app.py"],
    # ✅ Include project root, src, and venv site-packages in search paths
    pathex=[project_path, src_path, venv_site],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=hookspath,
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=icon_file,
    version=version_info,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_out_name
)
