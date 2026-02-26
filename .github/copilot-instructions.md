# Copilot Instructions for FastData

## Project Overview
FastData is a Python desktop application for process data analysis. It uses PySide6 for the GUI, DuckDB and SQLite for data storage, and includes machine learning capabilities with scikit-learn and sktime.

## Technology Stack
- **Language:** Python 3.9+
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

## Architecture Guidelines

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
