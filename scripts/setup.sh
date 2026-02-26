#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv-linux"
PY="${PYTHON_BIN:-python3}"

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi

# ✅ Ensure pip exists in the venv (works even if missing initially)
"$VENV/bin/python" -m ensurepip --upgrade

# ✅ Upgrade pip to latest
"$VENV/bin/python" -m pip install --upgrade pip

# ✅ Install dependencies if requirements.txt exists
if [ -f requirements.txt ]; then
  "$VENV/bin/python" -m pip install -r requirements.txt
fi

echo "✅ Linux env ready at $VENV"
