# AGENTS.md

## Purpose
This file gives coding agents a project-specific operating guide for FastData (`process-desktopapp`). Follow these rules when exploring, editing, testing, and packaging changes.

## Project Snapshot
- App type: local-first desktop app (PySide6/Qt)
- Language/runtime: Python `>=3.11` (see `pyproject.toml`)
- Entry point: `src/app.py`
- Main UI shell: `src/frontend/windows/main_window.py`
- Packaging: PyInstaller via `main.spec`, Windows installer via Inno Setup scripts
- Data layer: DuckDB + SQLite, all local files

## Repository Map
- `src/app.py`: startup flow (Qt init, splash, theme, main window)
- `src/frontend/`: GUI windows, tabs, widgets, charts, styles, localization, threading helpers
- `src/backend/`: services, DB access, repositories, importers, help manager, LLM providers
- `src/core/`: shared infrastructure (`paths.py`, settings/security/date utilities)
- `resources/`: styles, images/icons, help YAML/JSON, prompts, translations, license assets
- `tests/`: pytest and script-style verification for imports, DB/repositories, services, help
- `scripts/`: setup/build/release/doc generation scripts
- `.vscode/tasks.json`: canonical local tasks for setup/build/docs/licenses

## Non-Negotiable Conventions
- Use centralized path helpers from `src/core/paths.py`. Do not add new ad-hoc `Path(__file__)` resource lookups.
- Keep UI help content in sync: UI changes should update corresponding files in `resources/help/**`.
- Prefer extending existing tab/viewmodel/sidebar patterns over introducing new UI architecture.
- UI logic belongs in viewmodels. Run heavy work (DB writes, exports, saves) from viewmodels and fire threads via `src/frontend/threading/runner.py` + `src/frontend/threading/utils.py`; views should only dispatch signals and render state.
- For tab architecture, keep the control flow as: `sidebar/user action -> viewmodel starts work -> tab listens to viewmodel state/result signals and updates UI`. Avoid wiring start/request signals from sidebar through tab when the sidebar already has access to the viewmodel; only use tab-mediated starts when there is no practical alternative (for example, top-level window actions).
- File handling, long-running operations, and DB writes must run in background threads and should always surface user-facing toast messages for start/success/failure.
- Use `QMessageBox` only for confirmations (Yes/No style decisions before potentially destructive actions). Do not use info/warning message boxes for success/failure/status notifications; use toast messages instead.
- Views must confirm destructive operations (for example reset/clear/delete database/settings actions) before dispatching work.
- For tabs/sidebars that include `DataSelectorWidget`, treat it as the single source of truth for data selection. Prefer one direct data-fetch call from the selector (`fetch_base_dataframe` or `fetch_base_dataframe_for_features`) and let it apply active filters + preprocessing. Only read filter/preprocessing metadata separately when the tab explicitly needs that metadata for non-fetch logic (for example run context, labels, or summaries). Pass the resulting `DataFrame` into viewmodel methods (instead of passing selector fetch callbacks) so viewmodels can also accept data from other sources.
- Keep release tab behavior aligned with `src/frontend/tabs/tab_modules.py` (feature flags, lazy load, excludes).
- Respect local-first behavior. Do not introduce cloud-only dependencies for core workflows.
- Keep UI styles in resource stylesheets (`resources/style/*.qss`). Do not create inline widget styles in Python (for example `setStyleSheet(...)` with embedded CSS for feature UI); prefer dynamic properties/object names and QSS selectors, then toggle properties in code.

## Runtime Architecture Notes
- Startup (`src/app.py`) intentionally does light work before splash, then lazy-imports heavier UI modules.
- Main window uses tab module metadata from `src/frontend/tabs/tab_modules.py`.
- Data tab loads first; most other tabs are placeholder/lazy-loaded.
- Theme and localization are initialized early and used throughout.
- Forecasting tab is currently disabled in both dev/release flags in `tab_modules.py`.

## Database & Storage Model
- Measurement/process data DB: backend `data_db` (DuckDB-focused, SQL files under `src/backend/data_db/sql`)
- Settings/selections DB: backend `settings_db` (SQLite)
- Log DB: backend `logging_db` (SQLite)
- Default file locations are resolved via `core.paths` (platform-specific user data directories).

## Canonical Data Model (User Semantics)
- Treat `file/sheet` and `import` as equivalent in user-facing language. Internally, imports are DB tables created from ingested files/sheets.
- Imports are grouped into datasets. A dataset can contain multiple imports.
- Systems are higher-level containers that can contain multiple datasets.
- Systems own feature sets. Keep feature definitions/metadata consistent at system scope.
- Feature metadata should preserve `source`, `unit`, and `type` to support correct filtering, preprocessing, and modeling.
- Tags are primarily for feature organization/filtering (and optionally other entities where implemented). Tags are metadata and must not alter measured values.
- Preferred conceptual explanation order in docs/UI: `Files/Sheets (Imports) -> Datasets -> Systems -> Features -> Tags`.

## Setup, Run, Test
Use these first unless the user asks otherwise. On Windows, prefer the repo venv when running pytest.

- Windows setup: `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1`
- Linux/macOS setup: `./scripts/setup.sh`
- Run app from source: `python src/app.py`
- Run tests: `.venv-windows\Scripts\python -m pytest`
- Smoke-import coverage: `.venv-windows\Scripts\python -m pytest tests/test_smoke.py`
- Import parsing tests: `.venv-windows\Scripts\python -m pytest tests/test_imports.py`

## Build & Packaging
- PyInstaller build uses `main.spec` and dynamic release excludes from `tab_modules.py`.
- VS Code task: `Build EXE with Spec File`
- License artifact generation: `Generate Third-Party Licenses` -> `resources/licenses/third_party_licenses.html`
- Help docs generation: `python scripts/build_help_docs.py` (also wired in VS Code task)
- Windows installer build: `scripts/build-windows-installer.ps1`

When adding a dependency:
1. Update `pyproject.toml` and `requirements.txt` consistently.
2. Ensure packaging includes needed metadata/data files (check `main.spec`).
3. Regenerate third-party license artifact.

## Change Playbooks
### Adding a new tab/feature area
1. Implement tab/viewmodel/sidebar under `src/frontend/tabs/<feature>/`.
2. Register module in `src/frontend/tabs/tab_modules.py`.
3. Define release/dev enable flags and packaging excludes as needed.
4. Add icon/help key wiring and help docs in `resources/help/**`.
5. Add/extend backend services if needed under `src/backend/services/`.

### Adding a new resource file
1. Place it under `resources/`.
2. Add a path helper in `src/core/paths.py` when reusable.
3. Consume via helper function, not hardcoded relative paths.
4. Verify inclusion in packaged app (spec already walks `resources/**`).

### Updating help/documentation behavior
1. Edit YAML/JSON under `resources/help`.
2. Validate in-app rendering (help manager + viewmodel path).
3. Regenerate docs (`scripts/build_help_docs.py`) if user-facing docs are expected.

## Coding Style Expectations
- Keep type hints where existing modules already use them.
- Favor small, composable methods and clear naming (current code style).
- Add defensive guards around UI/file operations where failures should not crash startup.
- Preserve existing translation usage (`tr(...)` / `QCoreApplication.translate(...)`) in UI strings.
- Avoid broad refactors unless explicitly requested.

## Validation Before Handoff
Minimum checks after non-trivial edits:
1. Relevant targeted tests pass (`pytest <targeted-tests>`).
2. Smoke imports still pass when touched modules affect imports (`tests/test_smoke.py`).
3. App launches (`python src/app.py`) if UI/startup paths were changed.
4. If help/resources changed, verify path resolution and generated docs as applicable.
