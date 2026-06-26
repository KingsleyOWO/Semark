#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON:-python3}"

if [[ ! -d backend/.venv ]]; then
  "$python_bin" -m venv backend/.venv
fi

backend/.venv/bin/python -m pip install -U pip
backend/.venv/bin/python -m pip install -e 'backend[dev,mineru]'

if [[ ! -f backend/.env ]]; then
  cp backend/.env.example backend/.env
fi

cd frontend
npm ci
cd ..

echo "Full local install complete."
echo "Backend: cd backend && .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8585"
echo "Frontend: cd frontend && npm run dev"
echo "MinerU check: cd backend && .venv/bin/mineru --version"
echo "Set DOC_PARSER_VLM_MODEL and DOC_PARSER_VLM_BASE_URL in backend/.env for VLM enrichment."
