#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_demo_corpus.sh [options]

Options:
  --api <url>          Backend API base URL. Default: http://127.0.0.1:8585/api
  --ui <url>           UI base URL used in printed links. Default: http://127.0.0.1:5070
  --samples <dir>      Synthetic samples directory. Default: examples/samples
  --public-dir <dir>   Public downloaded corpus directory. Default: workspace/demo-corpus
  --include-public     Also upload files downloaded by scripts/fetch_demo_corpus.sh
  --profile <name>     fast or accurate. Default: fast
  --wait               Poll runs until they finish
  --timeout <seconds>  Poll timeout when --wait is used. Default: 900

Examples:
  scripts/run_demo_corpus.sh --profile fast
  scripts/fetch_demo_corpus.sh
  scripts/run_demo_corpus.sh --include-public --profile accurate --wait
USAGE
}

api_base="http://127.0.0.1:8585/api"
ui_base="http://127.0.0.1:5070"
samples_dir="examples/samples"
public_dir="workspace/demo-corpus"
include_public="false"
profile="fast"
wait_for_runs="false"
timeout_seconds=900

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api)
      api_base="${2:-}"
      shift 2
      ;;
    --ui)
      ui_base="${2:-}"
      shift 2
      ;;
    --samples)
      samples_dir="${2:-}"
      shift 2
      ;;
    --public-dir)
      public_dir="${2:-}"
      shift 2
      ;;
    --include-public)
      include_public="true"
      shift
      ;;
    --profile)
      profile="${2:-}"
      shift 2
      ;;
    --wait)
      wait_for_runs="true"
      shift
      ;;
    --timeout)
      timeout_seconds="${2:-}"
      shift 2
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

if [[ "$profile" != "fast" && "$profile" != "accurate" ]]; then
  echo "--profile must be fast or accurate" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

curl -fsS "$api_base/health" >/dev/null

files=()
while IFS= read -r file; do
  files+=("$file")
done < <(find "$samples_dir" -type f | sort)

if [[ "$include_public" == "true" ]]; then
  if [[ ! -d "$public_dir" ]]; then
    echo "Public corpus directory not found: $public_dir" >&2
    echo "Run scripts/fetch_demo_corpus.sh first, or omit --include-public." >&2
    exit 1
  fi
  while IFS= read -r file; do
    files+=("$file")
  done < <(find "$public_dir" -type f | grep -Ei '\.(pdf|html?|png|jpe?g|docx?|pptx?|xlsx?|odt|odp|ods)$' | sort || true)
fi

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No demo files found." >&2
  exit 1
fi

extract_json_field() {
  local field="$1"
  python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get(sys.argv[1], ""))' "$field"
}

json_list() {
  python3 - "$@" <<'JSON_LIST_PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
JSON_LIST_PY
}

run_status() {
  python3 -c 'import json,sys; print(json.load(sys.stdin).get("status", "unknown"))'
}

echo "Uploading ${#files[@]} demo files to $api_base"
doc_ids=()
for file in "${files[@]}"; do
  echo "- upload $file"
  response="$(curl -fsS -X POST -F "file=@${file}" "$api_base/ingest/upload")"
  doc_id="$(printf '%s' "$response" | extract_json_field doc_id)"
  if [[ -z "$doc_id" ]]; then
    echo "Could not parse doc_id from upload response for $file" >&2
    echo "$response" >&2
    exit 1
  fi
  doc_ids+=("$doc_id")
done

body="$(json_list "${doc_ids[@]}")"
echo "Creating ${#doc_ids[@]} $profile pipeline runs"
response="$(curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data "$body" \
  "$api_base/runs/batch-create?profile=$profile&use_cache=false")"

run_ids_file="$(mktemp)"
trap 'rm -f "$run_ids_file"' EXIT
RUN_RESPONSE="$response" python3 - <<'RUNS_PY' > "$run_ids_file"
import json
import os
import sys

data = json.loads(os.environ["RUN_RESPONSE"])
for item in data.get("created", []):
    print(item["run_id"])
if data.get("errors"):
    print(json.dumps(data["errors"], ensure_ascii=False), file=sys.stderr)
RUNS_PY

run_count="$(wc -l < "$run_ids_file" | tr -d ' ')"
echo "Created $run_count runs"
while IFS= read -r run_id; do
  [[ -n "$run_id" ]] || continue
  echo "- $run_id  $ui_base/viewer/$run_id"
done < "$run_ids_file"

if [[ "$wait_for_runs" == "true" ]]; then
  echo "Waiting for runs to finish, timeout ${timeout_seconds}s"
  deadline=$((SECONDS + timeout_seconds))
  while true; do
    pending=0
    failed=0
    while IFS= read -r run_id; do
      [[ -n "$run_id" ]] || continue
      status="$(curl -fsS "$api_base/runs/$run_id" | run_status)"
      printf '%s %s\n' "$status" "$run_id"
      case "$status" in
        pending|running) pending=$((pending + 1)) ;;
        failed|canceled) failed=$((failed + 1)) ;;
      esac
    done < "$run_ids_file"

    if [[ "$pending" -eq 0 ]]; then
      if [[ "$failed" -gt 0 ]]; then
        echo "$failed runs failed or were canceled" >&2
        exit 1
      fi
      echo "All demo runs finished successfully."
      break
    fi

    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "Timed out waiting for demo runs" >&2
      exit 1
    fi
    sleep 5
  done
fi

cat <<EOF

Demo run submitted.
Open the UI: $ui_base
Use Viewer links above to inspect Markdown, source maps, assets, quality, and semantic output.
EOF
