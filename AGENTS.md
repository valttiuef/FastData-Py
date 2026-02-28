# AGENTS.md

## Project Context

- Local-first desktop application (PySide6 / Qt)
- Python >= 3.11
- Entry: `src/app.py`
- Main window: `src/frontend/windows/main_window.py`
- Data: DuckDB (data_db) + SQLite (settings_db, logging_db)

## Repository Structure

- `src/frontend/` → UI (tabs, windows, widgets, viewmodels, threading)
- `src/backend/` → services, repositories, DB access, importers, LLM providers
- `src/core/` → shared infrastructure (`paths.py`, settings, utilities)
- `resources/` → styles, help docs, prompts, translations, licenses
- `tests/` → pytest validation
- `scripts/` → setup/build tools

Do not introduce new architectural patterns without explicit instruction.

## Architectural Rules

- Use centralized path helpers from `src/core/paths.py`. Do not use ad-hoc `Path(__file__)`.
- UI logic belongs in viewmodels. Views only dispatch signals and render state.
- Heavy work (DB writes, exports, processing) must run in background threads using existing threading helpers.
- Control flow pattern:  
  `sidebar/user action → viewmodel starts work → tab listens → UI updates`
- Avoid routing start signals through tabs if the sidebar already has viewmodel access.
- Use toast messages for status/success/failure.
- Use `QMessageBox` only for confirmations.
- Destructive actions must require confirmation before execution.
- Keep styles in `resources/style/*.qss`. Do not embed inline widget styles.
- Keep help content synchronized under `resources/help/**` when UI changes.
- Keep release tab behavior aligned with `src/frontend/tabs/tab_modules.py`.
- Respect local-first architecture. Do not introduce cloud-only dependencies.
- When fixing issues, never suppress warnings/errors as a workaround; fix the root cause.
- Global UI viewmodels (for example help/log) belong in `src/frontend/viewmodels/` and must expose shared getter functions.
- Initialize shared/global viewmodels once in `src/frontend/windows/main_window.py`, then access them through getters instead of passing/storing ad-hoc per-widget instances.
- Prefer pre-initialized controls/widgets when feasible (tables, charts, selectors) with stable default schemas/states so UI structure is visible before user actions.
- For pre-initialized tables, keep initial column sets deterministic and preserve user-resized column widths across data refreshes when practical.
- Prefer user-friendly UI naming wherever possible.
- Table headers shown to users should use readable labels (Title Case / clear abbreviations), not internal snake_case keys; for example `bmu_x` -> `BMU x`.

## Async UX Pattern (Status/Progress/Toasts)

For user-triggered background jobs (imports, analysis, training, exports):

- Keep action button labels stable. Do not repurpose button text for progress.
- Before starting work:
  - show an info toast (start message),
  - set status text describing the task,
  - initialize progress to `0` when progress tracking is available.
- During work:
  - keep status text for phase/context changes only (not numeric progress ticks),
  - do not drive status text from progress callbacks,
  - update progress via callback/signals in the `0..100` range when possible.
- Status text wording must be minimal and generic:
  - prefer short phrases like "Running...", "Finished.", "Failed."
  - do not include detailed explanations, counts, percentages, or error payloads in status text
  - put details in toasts/logs instead
- On completion:
  - show success toast,
  - set a clear finished status text,
  - clear/hide progress indicator.
- On warnings / no-result outcomes:
  - show warning toast,
  - set status text describing the warning outcome,
  - clear/hide progress indicator.
- On errors:
  - show error toast,
  - set status text with the failure reason,
  - clear/hide progress indicator.

## DataSelectorWidget Rule (Strict)

For tabs/sidebars that include `DataSelectorWidget`:

- Treat it as the single source of truth for data selection.
- Prefer one direct fetch call:
  - `fetch_base_dataframe`
  - or `fetch_base_dataframe_for_features`
- Let the selector apply active filters and preprocessing.
- Only read filter/preprocessing metadata separately when explicitly needed for non-fetch logic.
- Pass the resulting `DataFrame` into viewmodel methods.
- Do not pass selector fetch callbacks into viewmodels.

## Runtime Notes

- `src/app.py` performs light startup before lazy-importing heavy UI modules.
- Main window uses metadata from `src/frontend/tabs/tab_modules.py`.
- Data tab loads first; others may be lazy-loaded.
- Theme and localization initialize early.

## Data Model Semantics

Conceptual order:
Files/Sheets (Imports) → Datasets → Systems → Features → Tags

- Imports are DB tables created from ingested files/sheets.
- Datasets group imports.
- Systems group datasets and own feature sets.
- Feature metadata must preserve `source`, `unit`, and `type`.
- Tags are metadata only and must not alter measured values.

## Dependency Rules

When adding dependencies:
1. Update `pyproject.toml`
2. Update `requirements.txt`
3. Ensure packaging works with `main.spec`
4. Regenerate third-party license artifact if required

Do not introduce mandatory cloud dependencies.

## AI Fingerprinting

For significant AI-generated changes, annotate with one of these forms:

Single-line:

    # @ai(model, tool, role, YYYY-MM-DD)

Example:

    # @ai(gpt-4o, vscode, refactor, 2026-02-26)

Use single-line markers for:
- New public functions/methods (non-underscore names) when AI-authored.
- Significant single-function edits or refactors.

Block form:

    # --- @ai START ---
    # model: gpt-4o
    # tool: vscode
    # role: architectural-refactor
    # reviewed: yes
    # date: 2026-02-26
    # --- @ai END ---

Use block markers for:
- Full-file changes.
- Large multi-function or architectural refactors.

Do not annotate trivial edits.  
Markers must not affect runtime behavior.

## Validation After Changes

After non-trivial edits:
1. Relevant tests pass.
2. Smoke imports pass.
3. App launches if startup/UI paths changed.
4. Resources resolve correctly if modified.
