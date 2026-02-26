# FastData (Visima Process Analyzer)

<img src="resources/images/splash.png" width="150" alt="Splash screen">

FastData is a desktop process data analysis application developed by Valtteri Tiitta at the University of Eastern Finland (UEF) as part of the project Materials Solutions in the Green Transition – Visima. The project is co-financed by the European Union.

The application reimagines and extends an earlier MATLAB-based workflow in Python, combining a modern Qt-based graphical user interface with advanced data analysis, machine learning tooling, and integrated large-language-model (LLM) assistants. FastData is designed to support efficient exploration, processing, modeling, and interpretation of industrial process data.

FastData follows a local-first architecture: data storage, analysis, modeling, and visualization run entirely on the user’s machine. No cloud backend is required. Internet access is only needed if using OpenAI as an LLM provider; alternatively, local LLM backends such as Ollama can be used to keep all data processing fully offline.

The tool focuses on loading plant/process data, exploring it through visualizations, regression models, Self-Organizing Maps (SOM), and contextual help that is available directly in the interface. Some features are still under active development.

## Key capabilities
- **Process data management**: load and persist datasets in DuckDB/SQLite files and keep an audit trail of the work session through automatic logging.
- **Analysis and modeling**: run PCA, clustering, regression, and time-series tooling using scikit-learn; train SOM via MiniSom.
- **Interactive GUI**: PySide6 front end with tabs, dialogs, charts, and theming for a desktop-first experience.
- **LLM assist (experimental)**: optional OpenAI/Ollama integration for contextual questions and workflow guidance. Currently lightweight (no persistent history or deep project context tracking), it serves as a foundation for future model- and dataset-aware LLM integration.
- **Integrated help system**: structured YAML/JSON help files (`resources/help`) rendered directly inside the application so each widget can expose contextual documentation. The LLM assist can reference this help content for dynamic explanations. Documentation can also be generated from these YAML files using `scripts/build_help_docs.py` and the generated documentation can be opened through application.

## GUI

### Data
<img src="resources/screenshots/data.png" width="600" alt="Data screenshot">

### Charts
<img src="resources/screenshots/charts.png" width="600" alt="Charts screenshot">

### Statistics
<img src="resources/screenshots/statistics.png" width="600" alt="Statistics screenshot">

### SOM
<img src="resources/screenshots/som.png" width="600" alt="SOM screenshot">
<img src="resources/screenshots/som_features.png" width="600" alt="SOM features screenshot">
<img src="resources/screenshots/som_timeline.png" width="600" alt="SOM timeline screenshot">

### Regression
<img src="resources/screenshots/regression.png" width="600" alt="Regression screenshot">

### Chat / Help system
<img src="resources/screenshots/chat.png" width="600" alt="Chat/Help system screenshot">

## Dependencies and licenses

The application itself is released under the **MIT License** (see [LICENSE](LICENSE)).

This software depends on third-party libraries with the following licenses:

| Library | Purpose | License |
| --- | --- | --- |
| PySide6 / PySide6-Addons | Qt-based desktop GUI | LGPL-3.0 (Qt libraries dynamically linked) |
| duckdb | Local analytics database | MIT |
| SQLite (stdlib) | Embedded relational database | Public Domain |
| pandas, numpy | Data wrangling | BSD-3-Clause |
| scikit-learn | PCA, clustering, regression utilities | BSD-3-Clause |
| sktime | Legacy forecasting reference service (not used by active app flow) | BSD-3-Clause |
| MiniSom | Self-Organizing Maps | MIT |
| openai | OpenAI API integration | Apache-2.0 |
| ollama | Local LLM connector | MIT |
| openpyxl | Excel file reader | MIT |
| PyYAML | Help/documentation parsing | MIT |

A complete list of third-party licenses is available in
`resources/third_party_licenses.html`.

### Qt / PySide6 notice

This application uses PySide6 (Qt for Python) under the terms of the
GNU Lesser General Public License v3.0 (LGPL-3.0).

PySide6 and the Qt libraries are used **unmodified** and are
**dynamically linked**. Users may replace the Qt/PySide6 libraries with
compatible versions in accordance with the LGPL.

The full LGPL-3.0 license text is provided in `resources/licenses/LGPL-3.0.txt`.

Always review upstream licenses if you add or redistribute binaries.

## TODO / Known Issues

- **Forecasting (`scikit-learn`)**
  - Disabled until I figure out what is best way to implement this
  - Active forecasting now uses `scikit-learn` models and manual time-based splits.
  - The previous `sktime` implementation is preserved as a non-active reference in `src/backend/services/legacy_forecasting/forecasting_service_sktime.py`.

- **Translations**
  - Current translations are only examples and are incomplete.
  - Proper translation files should be created if multilingual support is required.

- **SQL schema**
  - The current database schema has grown organically and may be confusing in places.
  - A redesign and simplification of the schema should be considered.

- **LLM integration**
  - The current integration is basic and can be extended.
  - Possible improvements:
    - Add conversation history using saved logs
    - Improve message structure for clearer interactions
    - Provide better explanations of underlying models (e.g., decision trees, linear regression)

- **Code quality / refactoring**
  - A significant portion of the codebase was generated with LLM assistance.
  - While functional, it would benefit from systematic cleanup and refactoring.

- **Database structure**
  - Separate databases are currently used for logs, selections, and measurements.
  - Consolidating everything into a single database as the primary source of truth may simplify usage and data sharing.

- **Styling and language switching**
  - Changing theme or language currently requires restarting the application.
  - Dynamic switching would be preferable, but the current theme implementation needs performance improvements.

- **Cross-platform compatibility**
  - The application has only been tested on Windows.
  - Cross-platform testing (Linux, macOS) is needed to ensure compatibility.
  - OS-specific assumptions and dependencies should be reviewed.

- **Import freeze after mixed CSV/Excel sequence**
  - In some sessions, importing a large Excel workbook right after a DuckDB CSV import can appear to stall around sheet/chunk progress (commonly near 40%).
  - The import may continue only after additional UI interaction (for example clicking feature list/chart controls).
  - Workarounds:
    - Restart the app before importing the Excel workbook.
    - Import the Excel workbook into a fresh/opened database first, then import CSV files.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
