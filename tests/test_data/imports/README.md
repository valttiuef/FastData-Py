# Import Test Data & Test Suite

This directory contains test files and a test suite for the FastData import functionality.

## Quick Start

Run all tests:
```bash
python tests/test_imports.py
```

Run specific test:
```bash
python tests/test_imports.py "CSV: Simple"
```

Run with verbose output:
```bash
python tests/test_imports.py -v
```

## Test Files

### CSV Files

| File | Description | Features |
|------|-------------|----------|
| `csv_simple_comma.csv` | Standard CSV with comma separator | ISO date format, 1 header row |
| `csv_semicolon_decimal_comma.csv` | European format | Semicolon separator, comma decimals (,) |
| `csv_eu_date_dot_time.csv` | European date with dot-separated time | DD/MM/YYYY HH.MM format |
| `csv_de_date_colon_time.csv` | German date with colon-separated time | DD.MM.YYYY HH:MM:SS format |
| `csv_epoch_seconds.csv` | Unix timestamp format | Epoch seconds as timestamp column |
| `csv_separate_date_time.csv` | Date and time in separate columns | First column is date detection target |
| `csv_with_meta.csv` | CSV with metadata columns | Meta columns alongside numeric data |

### Excel Files

| File | Description | Features |
|------|-------------|----------|
| `excel_simple.xlsx` | Basic Excel file | Single sheet, simple structure |
| `excel_multi_sheet.xlsx` | Multiple sheets | Tests sheet iteration |
| `excel_multi_header.xlsx` | Multiple header rows | 4-row header, common in industrial data |

## Test Coverage

The test suite validates:

- **DateTime Detection**: Various date formats (ISO, EU, DE), epoch seconds, dot-separated times
- **Delimiter Detection**: Comma, semicolon separators
- **Decimal Parsing**: Dot (.) and comma (,) decimal separators
- **Header Detection**: Single and multiple header row configurations
- **Metric Extraction**: Identifying metric columns and their metadata
- **Data Integrity**: Ensuring numeric values are correctly parsed
- **Multi-sheet Support**: Handling Excel files with multiple sheets

## Adding New Test Files

1. Create your test file in this directory
2. Add a test case to `tests/test_imports.py` with:
   - A descriptive name
   - File path
   - Expected validation checks
   - Optional custom `ImportOptions` if special handling is needed

Example:
```python
ImportTestCase(
    "CSV: My custom format",
    TEST_DATA_DIR / "my_test_file.csv",
    {
        "Has timestamp": (check_has_timestamp, "No timestamps"),
        "Has values": (check_has_values, "No values"),
    },
    import_options=ImportOptions(
        csv_delimiter=",",
        csv_decimal=".",
        # ... other options
    )
)
```

## Common Test Helpers

- `check_has_timestamp`: Validates timestamp column exists with values
- `check_has_values`: Validates numeric value column exists
- `check_has_base_names`: Validates metrics are identified
- `check_min_rows(n)`: Validates minimum row count
- `check_metrics_count(n)`: Validates expected metric count
- `check_header_rows(n)`: Validates header row detection

## Expected Test Results

Running on a healthy codebase should achieve:
- **10+ tests passing** out of 11
- Edge cases (epoch seconds, etc.) may require explicit configuration

## Notes

- Tests suppress Unicode warnings to keep output clean
- Each test is independent and can be run individually
- Verbose mode (`-v`) shows detailed check results and data inspection
