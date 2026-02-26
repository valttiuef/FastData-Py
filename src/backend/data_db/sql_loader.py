# backend/sql_loader.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, Optional

_BLOCK_COMMENTS = re.compile(r"/\*.*?\*/", flags=re.S)

def _split_sql(script: str) -> Iterable[str]:
    # Remove block comments
    script = _BLOCK_COMMENTS.sub("", script)
    # Remove line comments
    lines = []
    for line in script.splitlines():
        s = line.strip()
        if s.startswith("--") or s == "":
            continue
        lines.append(line)
    cleaned = "\n".join(lines)

    # Split on semicolons into statements
    buf = []
    for ch in cleaned:
        buf.append(ch)
        if ch == ";":
            stmt = "".join(buf).strip()
            stmt = stmt[:-1].strip()  # drop trailing ';'
            if stmt:
                yield stmt
            buf = []
    tail = "".join(buf).strip()
    if tail:
        yield tail

def detect_engine_name(con) -> str:
    mod = con.__class__.__module__.lower()
    if "duckdb" in mod:
        return "duckdb"
    if "psycopg" in mod or "asyncpg" in mod or "pgdb" in mod or "psycopg2" in mod:
        return "postgres"
    return "postgres"

def load_and_execute_sql(con, root: Path, *, placeholders: Optional[dict] = None) -> None:
    engine = detect_engine_name(con)
    order = [
        root / "common" / "schema.sql",
        root / engine / "pre_create.sql",
        root / "common" / "views.sql",
        root / "common" / "indexes.sql",
        root / engine / "post_create.sql",
    ]
    for f in order:
        if not f.exists():
            continue
        sql = f.read_text(encoding="utf-8")
        if placeholders:
            for k, v in placeholders.items():
                sql = sql.replace("{{" + k + "}}", v)
        for stmt in _split_sql(sql):
            con.execute(stmt)
