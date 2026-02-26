
from __future__ import annotations
from pathlib import Path
from string import Template

SQL_DIR = Path(__file__).parent / "sql" / "duckdb"


def load_sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf-8")


def render_sql(tmpl: str, **kwargs) -> str:
    return Template(tmpl).substitute(**kwargs)
