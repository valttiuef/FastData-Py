#!/usr/bin/env python3
"""
Test suite for CSV/Excel import detection and parsing.

Tests the import dialog's detection capabilities and the actual import process.
Can be run to verify import logic changes.

Usage:
    python tests/test_imports.py                    # Run all tests
    python tests/test_imports.py test_name          # Run specific test
    python tests/test_imports.py -v                 # Verbose output
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, List
import argparse
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.models import ImportOptions
from backend.importing.parsers import parse_file_to_tall
from backend.data_db.database import Database
from backend.data_db import database as db_module


class ImportTestCase:
    """Single import test case."""
    
    def __init__(self, name: str, file_path: Path, expected_checks: dict, 
                 import_options: Optional[ImportOptions] = None):
        """
        Args:
            name: Test case name
            file_path: Path to the test file
            expected_checks: Dict of check_name -> (check_func, error_message)
            import_options: Optional custom ImportOptions (uses defaults if None)
        """
        self.name = name
        self.file_path = file_path
        self.expected_checks = expected_checks
        self.import_options = import_options
        self.result = None
        self.error = None
        self.tall_df = None
        self.header_rows = None
    
    def run(self, verbose: bool = False) -> bool:
        """Run the test. Returns True if all checks pass."""
        if not self.file_path.exists():
            self.error = f"File not found: {self.file_path}"
            return False
        
        if verbose:
            print(f"\n  Testing: {self.name}")
            print(f"    File: {self.file_path.name}")
        
        try:
            # Import with provided options or defaults
            if self.import_options:
                options = self.import_options
            else:
                options = ImportOptions(
                    csv_header_rows=1,
                    excel_header_rows=4,
                    auto_detect_datetime=True,
                    assume_dayfirst=True,
                    dot_time_as_colon=True,
                )
            
            results = parse_file_to_tall(self.file_path, options)
            if not results:
                self.error = "No data imported"
                return False
            
            sheet_name, tall_df, header_rows = results[0]
            self.tall_df = tall_df
            self.header_rows = header_rows
            
            if verbose:
                print(f"    [PASS] Imported successfully")
                print(f"      Shape: {tall_df.shape[0]} rows x {tall_df.shape[1]} cols")
                print(f"      Header rows: {header_rows}")
                print(f"      Columns: {', '.join(tall_df.columns)}")
            
            # Run checks
            all_passed = True
            for check_name, (check_func, error_msg) in self.expected_checks.items():
                try:
                    result = check_func(tall_df, header_rows)
                    if result:
                        if verbose:
                            print(f"      [PASS] {check_name}")
                    else:
                        self.error = f"{check_name}: {error_msg}"
                        all_passed = False
                        if verbose:
                            print(f"      [FAIL] {check_name}: {error_msg}")
                except Exception as e:
                    self.error = f"{check_name}: {str(e)}"
                    all_passed = False
                    if verbose:
                        print(f"      [FAIL] {check_name}: {str(e)}")
            
            self.result = all_passed
            return all_passed
            
        except Exception as e:
            self.error = str(e)
            if verbose:
                print(f"    [FAIL] Import failed: {e}")
            return False


class DuckdbImportTestCase:
    """Import test case that exercises the DuckDB CSV import path."""

    def __init__(
        self,
        name: str,
        file_path: Path,
        expected_checks: dict,
        import_options: Optional[ImportOptions] = None,
    ):
        self.name = name
        self.file_path = file_path
        self.expected_checks = expected_checks
        self.import_options = import_options
        self.result = None
        self.error = None

    @staticmethod
    def _quote_ident(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def run(self, verbose: bool = False) -> bool:
        if not self.file_path.exists():
            self.error = f"File not found: {self.file_path}"
            return False

        if verbose:
            print(f"\n  Testing: {self.name}")
            print(f"    File: {self.file_path.name}")

        try:
            if self.import_options:
                options = self.import_options
            else:
                options = ImportOptions(
                    csv_header_rows=1,
                    excel_header_rows=4,
                    auto_detect_datetime=False,
                    date_column="Time",
                    use_duckdb_csv_import=True,
                )

            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "test_import.duckdb"
                db = Database(db_path)
                try:
                    supports_encoding = True
                    try:
                        db.con.execute(
                            "SELECT * FROM read_csv_auto(?, encoding='utf-8') LIMIT 0;",
                            [str(self.file_path)],
                        )
                    except Exception as e:
                        if "Invalid named parameter \"encoding\"" in str(e):
                            supports_encoding = False
                        else:
                            raise

                    original_get_enc = None
                    if not supports_encoding:
                        original_get_enc = db_module._get_encoding_candidates

                        def _no_encoding_candidates(_user_encoding):
                            return [None]

                        db_module._get_encoding_candidates = _no_encoding_candidates

                    try:
                        import_ids = db.import_file(self.file_path, options)
                    finally:
                        if original_get_enc is not None:
                            db_module._get_encoding_candidates = original_get_enc

                    if not import_ids:
                        self.error = "No imports created"
                        return False

                    row = db.con.execute(
                        "SELECT csv_table_name, csv_ts_column FROM imports WHERE id = ?;",
                        [int(import_ids[0])],
                    ).fetchone()
                    if not row:
                        self.error = "Missing import metadata"
                        return False

                    table_name, ts_col = row
                    if not table_name or not ts_col:
                        self.error = "Missing CSV table metadata"
                        return False

                    table_ref = self._quote_ident(table_name)
                    ts_ref = self._quote_ident(ts_col)

                    row_count = int(
                        db.con.execute(f"SELECT COUNT(*) FROM {table_ref};").fetchone()[0]
                    )
                    parsed_count = int(
                        db.con.execute(
                            f"SELECT COUNT(*) FROM {table_ref} WHERE {ts_ref} IS NOT NULL;"
                        ).fetchone()[0]
                    )

                    if verbose:
                        print(f"    [PASS] Imported with DuckDB")
                        print(f"      Rows: {row_count}")
                        print(f"      Parsed timestamps: {parsed_count}")

                    ctx = {
                        "db": db,
                        "row_count": row_count,
                        "parsed_count": parsed_count,
                        "table_name": table_name,
                        "ts_col": ts_col,
                    }

                    all_passed = True
                    for check_name, (check_func, error_msg) in self.expected_checks.items():
                        try:
                            result = check_func(ctx)
                            if result:
                                if verbose:
                                    print(f"      [PASS] {check_name}")
                            else:
                                self.error = f"{check_name}: {error_msg}"
                                all_passed = False
                                if verbose:
                                    print(f"      [FAIL] {check_name}: {error_msg}")
                        except Exception as e:
                            self.error = f"{check_name}: {str(e)}"
                            all_passed = False
                            if verbose:
                                print(f"      [FAIL] {check_name}: {str(e)}")

                    self.result = all_passed
                    return all_passed
                finally:
                    try:
                        db.close()
                    except Exception:
                        pass
        except Exception as e:
            self.error = str(e)
            if verbose:
                print(f"    [FAIL] Import failed: {e}")
            return False


# Helper checks
def check_has_timestamp(df: pd.DataFrame, header_rows: int) -> bool:
    """Check that dataframe has a valid timestamp column."""
    return "ts" in df.columns and df["ts"].notna().sum() > 0

def check_has_values(df: pd.DataFrame, header_rows: int) -> bool:
    """Check that dataframe has numeric values."""
    return "value" in df.columns and df["value"].notna().sum() > 0

def check_has_base_names(df: pd.DataFrame, header_rows: int) -> bool:
    """Check that metrics are identified (base_name column)."""
    return "base_name" in df.columns and df["base_name"].nunique() > 0

def check_min_rows(n: int):
    """Return a check function that validates minimum row count."""
    def check(df: pd.DataFrame, header_rows: int) -> bool:
        return len(df) >= n
    return check

def check_metrics_count(n: int):
    """Return a check function that validates metric count."""
    def check(df: pd.DataFrame, header_rows: int) -> bool:
        if "base_name" not in df.columns:
            return False
        return df["base_name"].nunique() == n
    return check

def check_header_rows(expected: int):
    """Return a check function that validates header row detection."""
    def check(df: pd.DataFrame, header_rows: int) -> bool:
        return header_rows == expected
    return check


def check_duckdb_parsed_all():
    """DuckDB: ensure all rows have parsed timestamps."""
    def check(ctx: dict) -> bool:
        return ctx["row_count"] > 0 and ctx["parsed_count"] == ctx["row_count"]
    return check


# Define test cases
TEST_DATA_DIR = Path(__file__).parent / "test_data" / "imports"

TEST_CASES = [
    # CSV with simple comma separator and ISO date format
    ImportTestCase(
        "CSV: Simple comma separator (ISO date)",
        TEST_DATA_DIR / "csv_simple_comma.csv",
        {
            "Has timestamp": (check_has_timestamp, "No valid timestamps found"),
            "Has values": (check_has_values, "No numeric values found"),
            "Has metrics": (check_has_base_names, "No metrics identified"),
            "Correct row count": (check_min_rows(3), "Expected at least 3 data rows"),
            "Correct metric count": (check_metrics_count(3), "Expected 3 metrics"),
            "Correct header rows": (check_header_rows(1), "Expected 1 header row"),
        }
    ),
    
    # CSV with semicolon separator and comma decimals
    ImportTestCase(
        "CSV: Semicolon separator + decimal comma",
        TEST_DATA_DIR / "csv_semicolon_decimal_comma.csv",
        {
            "Has timestamp": (check_has_timestamp, "No valid timestamps found"),
            "Has values": (check_has_values, "No numeric values found"),
            "Has metrics": (check_has_base_names, "No metrics identified"),
            "Decimal parsing": (
                lambda df, h: (df["value"].min() > 20 and df["value"].max() < 1100),
                "Decimal separator not parsed correctly"
            ),
        },
        import_options=ImportOptions(
            csv_header_rows=1,
            excel_header_rows=4,
            auto_detect_datetime=True,
            assume_dayfirst=True,
            dot_time_as_colon=True,
            csv_delimiter=";",
            csv_decimal=",",
        )
    ),
    
    # CSV with EU date format and dot-separated time
    ImportTestCase(
        "CSV: EU date (DD/MM/YYYY) + dot time (HH.MM)",
        TEST_DATA_DIR / "csv_eu_date_dot_time.csv",
        {
            "Has timestamp": (check_has_timestamp, "Dot-time format not recognized"),
            "Has values": (check_has_values, "No numeric values found"),
            "Correct metric count": (check_metrics_count(3), "Expected 3 metrics"),
        }
    ),
    
    # CSV with German date format (DD.MM.YYYY) and colon time
    ImportTestCase(
        "CSV: German date (DD.MM.YYYY) + colon time",
        TEST_DATA_DIR / "csv_de_date_colon_time.csv",
        {
            "Has timestamp": (check_has_timestamp, "German date format not recognized"),
            "Has values": (check_has_values, "No numeric values found"),
            "Correct row count": (check_min_rows(3), "Expected at least 3 data rows"),
        }
    ),
    
    # CSV with epoch seconds (Unix timestamp)
    ImportTestCase(
        "CSV: Epoch seconds timestamp",
        TEST_DATA_DIR / "csv_epoch_seconds.csv",
        {
            "Has timestamp": (check_has_timestamp, "Epoch seconds not recognized"),
            "Has values": (check_has_values, "No numeric values found"),
            "Correct metric count": (check_metrics_count(3), "Expected 3 metrics"),
        }
    ),

    # CSV with mixed timezone-aware and timezone-naive timestamps
    ImportTestCase(
        "CSV: Mixed timezones in timestamp column",
        TEST_DATA_DIR / "csv_mixed_timezones.csv",
        {
            "Has timestamp": (check_has_timestamp, "Mixed timezones should be parsed"),
            "Has values": (check_has_values, "No numeric values found"),
            "Correct row count": (check_min_rows(3), "Expected at least 3 data rows"),
            "Correct metric count": (check_metrics_count(1), "Expected 1 metric"),
        }
    ),

    # CSV with separate date and time columns
    ImportTestCase(
        "CSV: Separate date and time columns",
        TEST_DATA_DIR / "csv_separate_date_time.csv",
        {
            "Has timestamp": (check_has_timestamp, "Should parse first datetime column"),
            "Has values": (check_has_values, "No numeric values found"),
        }
    ),
    
    # CSV with metadata columns
    ImportTestCase(
        "CSV: With metadata columns",
        TEST_DATA_DIR / "csv_with_meta.csv",
        {
            "Has timestamp": (check_has_timestamp, "No valid timestamps found"),
            "Has values": (check_has_values, "No numeric values found"),
            "Has metrics": (check_has_base_names, "No metrics identified"),
            "Metadata preserved": (
                lambda df, h: ("Meta" in df.columns or df.shape[1] >= 4),
                "Metadata columns not preserved"
            ),
        }
    ),
    
    # CSV with multiple header rows
    ImportTestCase(
        "CSV: Multiple header rows",
        TEST_DATA_DIR / "csv_multi_header_rows.csv",
        {
            "Has timestamp": (check_has_timestamp, "No valid timestamps found"),
            "Has values": (check_has_values, "No numeric values found"),
            "Header detection": (
                lambda df, h: h >= 1,  # Should detect at least 1 header row
                "Expected to detect header rows"
            ),
        },
        import_options=ImportOptions(
            csv_header_rows=4,  # This CSV has multiple header rows
            excel_header_rows=4,
            auto_detect_datetime=True,
            assume_dayfirst=True,
            dot_time_as_colon=True,
        )
    ),
    
    # Excel - simple single sheet
    ImportTestCase(
        "Excel: Simple single sheet",
        TEST_DATA_DIR / "excel_simple.xlsx",
        {
            "Has timestamp": (check_has_timestamp, "No valid timestamps found"),
            "Has values": (check_has_values, "No numeric values found"),
            "Has metrics": (check_has_base_names, "No metrics identified"),
        }
    ),
    
    # Excel - multiple sheets
    ImportTestCase(
        "Excel: Multiple sheets",
        TEST_DATA_DIR / "excel_multi_sheet.xlsx",
        {
            "Has timestamp": (check_has_timestamp, "No valid timestamps found"),
            "Has values": (check_has_values, "No numeric values found"),
            "Data imported": (
                lambda df, h: len(df) > 0,
                "Expected data to be imported"
            ),
        }
    ),
    
    # Excel - with multiple header rows
    ImportTestCase(
        "Excel: Multiple header rows (4 rows)",
        TEST_DATA_DIR / "excel_multi_header.xlsx",
        {
            "Has timestamp": (check_has_timestamp, "No valid timestamps found"),
            "Has values": (check_has_values, "No numeric values found"),
            "Header detection": (check_header_rows(4), "Expected 4 header rows"),
            "Correct metric count": (check_metrics_count(3), "Expected 3 metrics"),
        }
    ),

    # DuckDB CSV import with two local timestamps in a single cell
    DuckdbImportTestCase(
        "DuckDB CSV: Double timestamp cell",
        TEST_DATA_DIR / "csv_duckdb_local_timestamp.csv",
        {
            "Parsed timestamps": (check_duckdb_parsed_all(), "Expected all timestamps to parse"),
        },
        import_options=ImportOptions(
            csv_header_rows=1,
            excel_header_rows=4,
            auto_detect_datetime=False,
            date_column="Time",
            datetime_formats=["%Y-%m-%d %H:%M:%S %z"],
            use_duckdb_csv_import=True,
        ),
    ),
]


def print_header():
    """Print test suite header."""
    print("\n" + "=" * 75)
    print("IMPORT TEST SUITE")
    print("Testing CSV/Excel detection and parsing functionality")
    print("=" * 75)


def print_summary(results: List[Tuple[str, bool, Optional[str]]]):
    """Print test results summary."""
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    print("\n" + "=" * 75)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 75)
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[FAILED] Some tests failed:")
        for name, passed, error in results:
            if not passed:
                print(f"  [FAIL] {name}")
                if error:
                    print(f"    Error: {error}")
    
    print()
    return passed == total


def run_tests(test_filter: Optional[str] = None, verbose: bool = False) -> bool:
    """Run test suite. Returns True if all tests pass."""
    print_header()
    
    # Filter tests if requested
    tests_to_run = TEST_CASES
    if test_filter:
        tests_to_run = [t for t in TEST_CASES if test_filter.lower() in t.name.lower()]
        if not tests_to_run:
            print(f"\n✗ No tests matching '{test_filter}'")
            return False
        print(f"\nRunning {len(tests_to_run)} matching test(s)...\n")
    else:
        print(f"\nRunning {len(TEST_CASES)} tests...\n")
    
    results = []
    for test_case in tests_to_run:
        passed = test_case.run(verbose=verbose)
        results.append((test_case.name, passed, test_case.error))
        
        if not verbose:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} {test_case.name}")
    
    return print_summary(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import test suite")
    parser.add_argument("test", nargs="?", help="Specific test to run (partial name match)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    success = run_tests(test_filter=args.test, verbose=args.verbose)
    sys.exit(0 if success else 1)
