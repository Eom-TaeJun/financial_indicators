#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m pytest -ra

if command -v ruff >/dev/null 2>&1; then
  ruff check .
elif python -m ruff --version >/dev/null 2>&1; then
  python -m ruff check .
else
  echo "ERROR: ruff is not installed. Install with: pip install -r requirements.txt" >&2
  exit 1
fi
