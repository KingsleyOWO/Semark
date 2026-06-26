#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/smoke_clone.sh [--repo <git-url>] [--docker] [--full-docker]

Examples:
  scripts/smoke_clone.sh
  scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git
  scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --docker
  scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --full-docker

Without --repo, the script tests the current working tree. With --repo, it clones
into a temporary directory and tests the clean clone.
USAGE
}

repo_url=""
run_docker="false"
run_full_docker="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      repo_url="${2:-}"
      shift 2
      ;;
    --docker)
      run_docker="true"
      shift
      ;;
    --full-docker)
      run_full_docker="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cleanup_dir=""
if [[ -n "$repo_url" ]]; then
  cleanup_dir="$(mktemp -d)"
  echo "Cloning $repo_url into $cleanup_dir/repo"
  git clone --depth 1 "$repo_url" "$cleanup_dir/repo"
  cd "$cleanup_dir/repo"
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi

if [[ ! -f backend/pyproject.toml || ! -f frontend/package.json ]]; then
  echo "This does not look like the doc-parser repository root." >&2
  exit 1
fi

echo "Checking backend"
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[dev,mineru]'
.venv/bin/mineru --version
.venv/bin/python -m ruff check app tests
.venv/bin/python -m pytest
cd ..

echo "Checking frontend"
cd frontend
npm ci
npm run lint
npm run build
cd ..

if [[ "$run_docker" == "true" ]]; then
  echo "Checking baseline Docker Compose"
  docker compose up --build -d
  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8585/api/health >/dev/null; then
      break
    fi
    sleep 2
  done
  curl -fsS http://127.0.0.1:8585/api/health
  curl -fsSI http://127.0.0.1:5070/ >/dev/null
  docker compose down
fi

if [[ "$run_full_docker" == "true" ]]; then
  echo "Checking full MinerU Docker Compose"
  docker compose -f docker-compose.full.yml up --build -d
  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8585/api/health >/dev/null; then
      break
    fi
    sleep 2
  done
  curl -fsS http://127.0.0.1:8585/api/health
  curl -fsSI http://127.0.0.1:5070/ >/dev/null
  docker compose -f docker-compose.full.yml exec -T backend mineru --version
  docker compose -f docker-compose.full.yml down
fi

if [[ -n "$cleanup_dir" ]]; then
  rm -rf "$cleanup_dir"
fi

echo "Smoke clone check passed."
