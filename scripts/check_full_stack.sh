#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

backend/.venv/bin/python -m ruff check backend/app backend/tests
backend/.venv/bin/python -m pytest backend/tests
backend/.venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path

venv_mineru = Path("backend/.venv/bin/mineru")
import shutil

checks = {
    "mineru package": importlib.util.find_spec("mineru") is not None,
    "PyMuPDF package": importlib.util.find_spec("fitz") is not None,
    "mineru command": venv_mineru.exists(),
    "LibreOffice command": shutil.which("libreoffice") is not None or shutil.which("soffice") is not None,
}
failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(f"{name}: {'ok' if ok else 'missing'}")
if failed:
    raise SystemExit(f"Missing full-stack dependency checks: {', '.join(failed)}")
PY

cd frontend
npm run lint
npm run build
cd ..

echo "Full-stack dependency check passed."
