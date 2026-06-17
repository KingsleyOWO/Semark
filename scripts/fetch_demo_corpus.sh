#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/fetch_demo_corpus.sh [--out <directory>] [--skip-irs] [--skip-sec]

Downloads official public demo documents into a local workspace directory.
Downloaded files are intentionally not committed to the repository.

Environment:
  SEC_USER_AGENT  Required by SEC fair-access guidance for EDGAR requests.
                  Example: "Your Name your.email@example.com"
USAGE
}

out_dir="workspace/demo-corpus"
fetch_irs="true"
fetch_sec="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      out_dir="${2:-}"
      shift 2
      ;;
    --skip-irs)
      fetch_irs="false"
      shift
      ;;
    --skip-sec)
      fetch_sec="false"
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

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$out_dir"

if [[ "$fetch_irs" == "true" ]]; then
  echo "Downloading IRS Form W-9 PDF"
  curl -fL --retry 3 --retry-delay 2 \
    -o "$out_dir/irs_fw9.pdf" \
    "https://www.irs.gov/pub/irs-pdf/fw9.pdf"
fi

if [[ "$fetch_sec" == "true" ]]; then
  SEC_USER_AGENT="${SEC_USER_AGENT:-doc-parser-demo/0.1 open-source-demo@example.invalid}"
  OUT_DIR="$out_dir" SEC_USER_AGENT="$SEC_USER_AGENT" python3 - <<'SEC_PY'
import json
import os
import re
import urllib.request
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
user_agent = os.environ["SEC_USER_AGENT"]
headers = {
    "User-Agent": user_agent,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
submissions_url = "https://data.sec.gov/submissions/CIK0000320193.json"
request = urllib.request.Request(submissions_url, headers=headers)
print("Downloading SEC Apple submissions metadata")
with urllib.request.urlopen(request, timeout=30) as response:
    data = json.loads(response.read().decode("utf-8"))

recent = data.get("filings", {}).get("recent", {})
forms = recent.get("form", [])
accessions = recent.get("accessionNumber", [])
documents = recent.get("primaryDocument", [])
filing_dates = recent.get("filingDate", [])

selected = None
for form, accession, document, filing_date in zip(forms, accessions, documents, filing_dates):
    if form == "10-K" and document:
        selected = {
            "form": form,
            "accession": accession,
            "document": document,
            "filing_date": filing_date,
        }
        break

if not selected:
    raise SystemExit("No recent Apple 10-K filing found in SEC submissions data")

accession_no_dashes = selected["accession"].replace("-", "")
archive_url = (
    "https://www.sec.gov/Archives/edgar/data/320193/"
    f"{accession_no_dashes}/{selected['document']}"
)
archive_headers = dict(headers)
archive_headers["Host"] = "www.sec.gov"
request = urllib.request.Request(archive_url, headers=archive_headers)
print(f"Downloading SEC Apple {selected['filing_date']} 10-K HTML")
with urllib.request.urlopen(request, timeout=60) as response:
    content = response.read()

safe_date = re.sub(r"[^0-9-]", "", selected["filing_date"])
html_path = out_dir / f"sec_apple_10k_{safe_date}.html"
html_path.write_bytes(content)
meta_path = out_dir / "sec_apple_10k_metadata.json"
meta_path.write_text(json.dumps({**selected, "source_url": archive_url}, indent=2), encoding="utf-8")
print(f"Saved {html_path}")
SEC_PY
fi

cat <<EOF

Demo corpus ready: $out_dir

Suggested next step:
  scripts/run_demo_corpus.sh --include-public --profile fast
EOF
