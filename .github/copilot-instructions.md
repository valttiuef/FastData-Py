# Copilot Instructions for FastData

## Project Overview
FastData is a local-first Python desktop application for process data analysis. It uses PySide6 for the GUI, DuckDB and SQLite for data storage, and includes machine learning capabilities with scikit-learn and sktime.

**Key Characteristics:**
- Entry: `src/app.py`
- Main window: `src/frontend/windows/main_window.py`
- Data: DuckDB (data_db) + SQLite (settings_db, logging_db)
- Do not introduce new architectural patterns without explicit instruction
- Do not introduce mandatory cloud dependencies

## Technology Stack
- **Language:** Python >= 3.11
- **GUI Framework:** PySide6 (Qt for Python)
- **Databases:** DuckDB (analytical queries), SQLite (settings/logs)
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, sktime, MiniSom
- **LLM Integration:** OpenAI API

## Project Structure
```
├── .vscode/            # VS Code configuration (launch, tasks, settings)
├── resources/          # Application resources
│   ├── icons/          # Application icons
│   ├── images/         # Splash screen, logos
│   ├── licenses/       # Third-party license files
│   └── style/          # QSS stylesheets
├── scripts/            # Build and setup scripts
│   ├── setup.sh        # Linux/macOS setup script
│   ├── setup.ps1       # Windows setup script
│   └── build-windows-installer.ps1
├── src/
│   ├── app.py          # Application entry point
│   ├── qt_compat.py    # Qt compatibility layer
│   ├── backend/        # Data layer and services
│   │   ├── data_db/    # DuckDB database operations
│   │   ├── logging_db/ # Logging database operations
│   │   ├── settings_db/ # Settings database operations
│   │   ├── services/   # ML and analysis services
│   │   ├── models/     # Data models
│   │   └── importing/  # Data import utilities
│   ├── core/           # Core utilities
│   │   ├── datetime_utils.py
│   │   └── settings_manager.py
│   └── frontend/       # UI components
│       ├── windows/    # Main windows
│       ├── dialogs/    # Dialog windows
│       ├── tabs/       # Tab panels (each tab has its own viewmodel.py)
│       ├── widgets/    # Reusable widgets
│       ├── charts/     # Chart components
│       ├── models/     # Frontend data models
│       ├── threading/  # Background task utilities
│       ├── utils/      # UI utilities
│       └── style/      # Themes and styling
├── tests/              # Test files
├── main.spec           # PyInstaller build spec
├── pyproject.toml      # Project metadata and dependencies
└── requirements.txt    # Detailed dependencies list
```

## Development Setup
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -e .`
4. Run the application: `python src/app.py`
5. Prefer the repository virtual environment for validation commands when available
   - On Windows: `.venv-windows\Scripts\python.exe -m pytest ...`

## Code Style Guidelines
- Follow PEP 8 style guidelines
- Use type hints where practical
- Prefer descriptive variable and function names
- Keep functions focused and single-purpose
- Use docstrings for public functions and classes

## Testing
- Tests are located in the `tests/` directory
- Run tests with: `pytest tests/ -v`
- Use `QT_QPA_PLATFORM=offscreen` for headless testing
- Test fixtures should use temporary directories for database files
- After non-trivial edits, verify:
  1. Relevant tests pass
  2. Smoke imports pass
  3. App launches if startup/UI paths changed
  4. Resources resolve correctly if modified

## Architecture Guidelines

### Repository Structure
- `src/frontend/` → UI (tabs, windows, widgets, viewmodels, threading)
- `src/backend/` → services, repositories, DB access, importers, LLM providers
- `src/core/` → shared infrastructure (`paths.py`, settings, utilities)
- `resources/` → styles, help docs, prompts, translations, licenses
- `tests/` → pytest validation
- `scripts/` → setup/build tools

### Architectural Rules (Required)
- Use centralized path helpers from `src/core/paths.py`. Do not use ad-hoc `Path(__file__)`
- UI logic belongs in viewmodels. Views only dispatch signals and render state
- Heavy work (DB writes, exports, processing) must run in background threads using existing threading helpers
- Control flow pattern: `sidebar/user action → viewmodel starts work → tab listens → UI updates`
- Avoid routing start signals through tabs if the sidebar already has viewmodel access
- Use toast messages for status/success/failure
- Use `QMessageBox` only for confirmations
- Destructive actions must require confirmation before execution
- Keep styles in `resources/style/*.qss`. Do not embed inline widget styles
- Keep help content synchronized under `resources/help/**` when UI changes
- Keep release tab behavior aligned with `src/frontend/tabs/tab_modules.py`
- Respect local-first architecture. Do not introduce cloud-only dependencies
- When fixing issues, never suppress warnings/errors as a workaround; fix the root cause
- Global UI viewmodels (for example help/log) belong in `src/frontend/viewmodels/` and must expose shared getter functions
- Initialize shared/global viewmodels once in `src/frontend/windows/main_window.py`, then access them through getters instead of passing/storing ad-hoc per-widget instances
- Prefer pre-initialized controls/widgets when feasible (tables, charts, selectors) with stable default schemas/states so UI structure is visible before user actions
- For pre-initialized tables, keep initial column sets deterministic and preserve user-resized column widths across data refreshes when practical
- Prefer user-friendly UI naming wherever possible
- Table headers shown to users should use readable labels (Title Case / clear abbreviations), not internal snake_case keys; for example `bmu_x` → `BMU x`

### Backend Services
- Services in `backend/services/` handle ML and analysis operations
- Each service should be stateless when possible
- Use dependency injection for database connections

### Frontend Components
- UI components use the Model-View-ViewModel (MVVM) pattern
- ViewModels are co-located with their views (e.g., `tabs/data/viewmodel.py`)
- Keep UI logic separate from business logic
- Use Qt signals/slots for component communication

### Database Operations
- Use DuckDB for analytical queries and time-series data
- Use SQLite for application settings and logs
- Database operations should be in dedicated repository classes

## Async UX Pattern (Status/Progress/Toasts)

For user-triggered background jobs (imports, analysis, training, exports):

- Keep action button labels stable. Do not repurpose button text for progress
- Before starting work:
  - show an info toast (start message)
  - set status text describing the task
  - initialize progress to `0` when progress tracking is available
- During work:
  - keep status text for phase/context changes only (not numeric progress ticks)
  - do not drive status text from progress callbacks
  - update progress via callback/signals in the `0..100` range when possible
- Status text wording must be minimal and contextual:
  - include the task context in short form, using task-specific action words, for example "Finding correlations...", "Training SOM...", "Importing data...", "Forecasting finished."
  - do not include counts, percentages, stack traces, or long explanations in status text
  - for errors, include a short reason only when useful for immediate user context
  - put deeper details in toasts/logs instead
- On completion:
  - show success toast
  - set a clear finished status text
  - clear/hide progress indicator
- On warnings / no-result outcomes:
  - show warning toast
  - set status text describing the warning outcome
  - clear/hide progress indicator
- On errors:
  - show error toast
  - set status text with the failure reason
  - clear/hide progress indicator

## DataSelectorWidget Rule (Strict)

For tabs/sidebars that include `DataSelectorWidget`:

- Treat it as the single source of truth for data selection
- Always use async fetch methods in UI flows:
  - `fetch_base_dataframe_async`
  - or `fetch_base_dataframe_for_features_async`
- Do not use synchronous selector fetch methods on the UI thread, except in clearly non-UI/test-only code
- For dynamic refresh flows (for example feature checkbox changes, live chart/data updates):
  - use cancellable/superseding fetches (`cancel_previous=True` with stable owner/key)
  - ignore stale results when a newer request exists
- For button-triggered flows (for example SOM/Regression train):
  - show toast/status for "fetching data" and "data fetched → next task"
  - then pass the fetched `DataFrame` to the next viewmodel action
- For dynamic auto-updating flows, do not show fetch start/finish toasts by default
- Let the selector apply active filters and preprocessing
- Only read filter/preprocessing metadata separately when explicitly needed for non-fetch logic
- Pass the resulting `DataFrame` into viewmodel methods
- Do not pass selector fetch callbacks into viewmodels

## Common Patterns

### Adding New Features
1. Create/modify models in `backend/models/`
2. Implement service logic in `backend/services/`
3. Create a new tab folder with view and viewmodel (e.g., `frontend/tabs/mytab/`)
4. Build UI components in `frontend/widgets/` for reusable components

### Error Handling
- Use try/except blocks around database and file operations
- Log errors appropriately using the logging service
- Display user-friendly error messages in the UI

## Build and Distribution
- PyInstaller is used for creating executables
- Build spec is in `main.spec`
- VS Code tasks are available for building (see `.vscode/tasks.json`)

## Runtime Notes

- `src/app.py` performs light startup before lazy-importing heavy UI modules
- Main window uses metadata from `src/frontend/tabs/tab_modules.py`
- Data tab loads first; others may be lazy-loaded
- Theme and localization initialize early
- Prefer the repository virtual environment for validation commands when available
- On this repo, use `.venv-windows\Scripts\python.exe -m pytest ...` instead of assuming `pytest` is on `PATH`

## Data Model Semantics

Conceptual order:
Files/Sheets (Imports) → Datasets → Systems → Features → Tags

- Imports are DB tables created from ingested files/sheets
- Datasets group imports
- Systems group datasets and own feature sets
- Feature metadata must preserve `source`, `unit`, and `type`
- Tags are metadata only and must not alter measured values

## Dependency Rules

When adding dependencies:
1. Update `pyproject.toml`
2. Update `requirements.txt`
3. Ensure packaging works with `main.spec`
4. Regenerate third-party license artifact if required

Do not introduce mandatory cloud dependencies.

## AI Fingerprinting

For significant AI-generated changes, annotate with one of these forms:

**Single-line** (for new public functions/methods or significant single-function edits):
```
# @ai(model, tool, role, YYYY-MM-DD)
# @ai(gpt-4o, vscode, refactor, 2026-02-26)
```

**Block form** (for full-file changes or large multi-function architectural refactors):
```
# --- @ai START ---
# model: gpt-4o
# tool: vscode
# role: architectural-refactor
# reviewed: yes
# date: 2026-02-26
# --- @ai END ---
```

Do not annotate trivial edits. Markers must not affect runtime behavior.
