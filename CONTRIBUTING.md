## Contributing
- Keep feature documentation in `resources/help` in sync with UI changes.
- Update dependency/license notes above when adding major packages.
- Submit changes under the MIT License (see below).

## Getting started
1. **Set up Python 3.9+ and a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. **Install dependencies**
   ```bash
   pip install -e .
   ```
3. **Run the app**
   ```bash
   python src/app.py
   ```
   For VS Code users, you can also use the existing tasks: *Activate venv + Install* then *Run app* or press **F5**.

## Project structure
- `src/app.py` – Application entry point that sets up Qt, splash screen, theming, and the main window.
- `src/backend/` – Data and configuration layers: database access (`data_db`, `logging_db`, `settings_db`), import pipelines, domain models, and help manager.
- `src/frontend/` – Qt GUI: windows, tabs, dialogs, widgets, charts, and viewmodels that connect UI events to the backend.
- `src/core/` – Shared utilities (e.g., data helpers, feature engineering) used across the app.
- `resources/` – Icons, images, QSS themes, lang-files, prompt-files, licenses and the YAML/JSON help files consumed by the in-app documentation system.
- `examples/` – Self-contained demos such as the help system example showing how info buttons and popups are wired.
- `scripts/` – Developer scripts (packaging, linting, utilities) for local workflows.
- `tests/` – Pytest suites covering backend logic (help system, data handling, etc.).
- `requirements.txt` / `pyproject.toml` – Dependency lists for development and packaging.

## VSCode tasks

FastData includes a comprehensive set of VS Code tasks to streamline development, testing, and packaging workflows:

### Application Development
- **Activate venv + Install** – Sets up the Python virtual environment and installs all dependencies. Run this once after cloning the repository.
- **Run app** – Launches FastData directly from source. Useful for development and testing changes in real-time.

### Building & Packaging
- **Build EXE with Spec File** – Packages the application into a standalone Windows executable using PyInstaller and the `main.spec` configuration.
- **Generate Third-Party Licenses** – Generates `resources/licenses/third_party_licenses.html` containing all third-party license information for redistribution.
- **Generate Help Docs** – Runs `scripts/build_help_docs.py` to compile help content from YAML files into human-readable documentation.
- **Build Installer (Inno Setup)** – Creates the Windows installer executable using Inno Setup (requires Inno Setup to be installed).
- **Build App + Installer (Windows)** – Composite task that runs the complete build pipeline: generates licenses, builds the EXE, and creates the installer in sequence.

## Databases and models
- **Storage**: DuckDB and SQLite are used for local, file-based persistence of datasets and logs.
- **Modeling**: scikit-learn provide PCA, clustering, regression, and time-series pipelines; MiniSom powers SOM training/visualization.
- **LLM support**: OpenAI and Ollama connectors allow on-screen assistance and experiments with LLM-enhanced data exploration.

## Documentation system
Help content lives in `resources/help` as YAML/JSON. The backend help manager loads all files, the viewmodels expose them to the UI, and widgets like info buttons and popups render the content inline. See `examples/README_HELP_SYSTEM.md` for architecture and usage.

## Logging
User actions and data operations are logged automatically so sessions can be revisited later. Log stores are kept locally alongside datasets, making it easier to trace how results were produced.

## Localization
Localization is handled through localization.py which uses QTranslator and loads language files from resources/languages. Currently only Finnish language added as an example (not all words translated).