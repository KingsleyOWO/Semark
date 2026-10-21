# Installation, Configuration and Deployment

**English** | [繁體中文](INSTALL.zh-TW.md) · [Back to README](../README.md)

## What It Handles

Supported inputs:

- PDF
- DOC/DOCX, PPT/PPTX, XLS/XLSX
- ODT/ODP/ODS
- HTML/HTM
- PNG/JPG/JPEG

DOCX files use native structure parsing first. If that fails, the pipeline automatically converts the file to PDF with LibreOffice and runs the existing PDF parser. The original Word file is preserved, and the converted PDF supplies page previews and text supplementation. Conversion warnings mark the quality report as `needs_review`; `parse_report.json` records the fallback reason and diagnostics. This fallback requires LibreOffice, which is included in the full Docker image.

Generated outputs include the main document plus automatically extracted child documents when the source contains independent retrieval units. For example, a long PDF can produce `main.md` for the body text and separate semantic Markdown files for individual forms, tables, flowcharts, figures, attachments, or other structured sections.

## What Gets Downloaded

- `git clone` downloads the source code, documentation, synthetic samples, and small curated demo snapshots only.
- `docker compose up --build` downloads operating-system packages, Python wheels, npm packages, MinerU dependencies, PyMuPDF, LibreOffice, CJK fonts, and the runtime dependencies needed by the full parser image.
- The first MinerU run may download parser/model cache files according to MinerU's own upstream behavior and license. Docker stores these in named volumes, not in Git.
- Semark does not bundle Ollama models, cloud model weights, API keys, private documents, generated outputs, or local caches. For local Ollama, pull the model yourself on the host with `ollama pull model-name`.
- `scripts/fetch_demo_corpus.sh` downloads optional public test files into `workspace/demo-corpus/`, which is ignored by Git.

## Runtime Modes

MinerU can run in three deployment modes:

- Simple mode: leave `SEMARK_MINERU_API_URL` empty. The `mineru` CLI auto-starts a temporary local API for each parse. This is easiest and most portable.
- Service mode: run a warm `mineru-api` locally and set `SEMARK_MINERU_API_URL`, for example `http://127.0.0.1:8601`. This avoids reloading parser resources every run.
- Remote mode: point `SEMARK_MINERU_API_URL` at a MinerU API/router on another machine, typically a GPU host.

If a configured MinerU API URL is unreachable, the app falls back to simple mode so parsing does not fail just because the warm service is down.

## Full Feature Quickstart

Use this path when you want the real product behavior: MinerU parsing plus optional VLM enrichment into structured semantic text.

```bash
./scripts/install_full_local.sh
```

This installs:

- Backend Python dependencies.
- PyMuPDF for PDF/image handling.
- MinerU pipeline dependencies via the `mineru` optional dependency group.
- Frontend npm dependencies.

Install LibreOffice separately when you want local HTML, DOC, PPT, ODT, or ODP conversion:

```bash
sudo apt install libreoffice
```

Then configure VLM in `backend/.env` if you want model-enriched forms, figures, diagrams, and semantic output. The enrichment model is used for form extraction and visual understanding. The reviewer model is used by the final quality gate for audit and controlled repair checks; leave it unset to reuse the enrichment model.

```env
SEMARK_VLM_BASE_URL=http://127.0.0.1:11434/v1
SEMARK_VLM_API_KEY=ollama
SEMARK_VLM_MODEL=your-vision-model

SEMARK_REVIEW_VLM_BASE_URL=http://127.0.0.1:11434/v1
SEMARK_REVIEW_VLM_API_KEY=ollama
SEMARK_REVIEW_VLM_MODEL=your-stronger-review-model
```

Start the services:

```bash
cd backend
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8585
```

```bash
cd frontend
npm run dev
```

Open `http://localhost:5070`, upload a document, and run the `accurate` profile for MinerU + VLM enrichment.

Check the full local environment:

```bash
./scripts/check_full_stack.sh
```

## Manual Local Quickstart

Backend:

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev,mineru]"
cp .env.example .env
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8585
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

For LAN access, bind the backend/frontend to `0.0.0.0` and open the chosen ports in your firewall.


## Docker Quickstart

The default Docker path is the full product path. `docker compose up --build` builds the backend with MinerU, PyMuPDF, LibreOffice, CJK fonts, and the document conversion/parsing dependencies needed for real document processing. The repository provides reproducible setup files; it does not commit or bundle model weights, API keys, private documents, generated outputs, or local cache files.

For Docker Desktop, or hosts where containers can reach a local Ollama service through `host.docker.internal`:

```bash
export SEMARK_VLM_BASE_URL=http://host.docker.internal:11434/v1
export SEMARK_VLM_API_KEY=ollama
export SEMARK_VLM_MODEL=your-vision-model
export SEMARK_REVIEW_VLM_BASE_URL=http://host.docker.internal:11434/v1
export SEMARK_REVIEW_VLM_API_KEY=ollama
export SEMARK_REVIEW_VLM_MODEL=your-stronger-review-model

docker compose up --build
```

Open:

- Frontend: `http://localhost:5070`
- Backend health: `http://localhost:8585/api/health`

On Linux/WSL, Docker bridge networking may not always reach a host Ollama service through `host.docker.internal`. If Settings -> VLM model probe times out, use the host-network compose file so containers can call Ollama at `127.0.0.1:11434`:

```bash
export SEMARK_VLM_MODEL=your-vision-model
export SEMARK_REVIEW_VLM_MODEL=your-stronger-review-model

docker compose -f docker-compose.full.host.yml up --build
```

If `5070` or `8585` is already in use, override both ports for the host-network compose file:

```bash
SEMARK_FRONTEND_PORT=35070 SEMARK_PORT=38585 \
  docker compose -f docker-compose.full.host.yml up --build
```

Accessing the frontend by IP address or `localhost` always works. If you want to reach it through a **named host** (for example an internal DNS name), allowlist it explicitly — arbitrary Host headers are rejected to prevent DNS-rebinding attacks against the unauthenticated backend:

```bash
SEMARK_FRONTEND_ALLOWED_HOSTS=semark.internal,docs.example.lan \
  docker compose up --build
```

For cloud or remote OpenAI-compatible providers, point both model endpoints at the provider instead of Ollama:

```bash
export SEMARK_VLM_BASE_URL=https://your-provider.example/v1
export SEMARK_VLM_API_KEY=your-api-key
export SEMARK_VLM_MODEL=your-vision-model
export SEMARK_REVIEW_VLM_BASE_URL=https://your-provider.example/v1
export SEMARK_REVIEW_VLM_API_KEY=your-api-key
export SEMARK_REVIEW_VLM_MODEL=your-stronger-review-model

docker compose up --build
```

The full image installs the backend with `.[mineru]`, includes PyMuPDF, LibreOffice for HTML/Office conversion, Chinese CJK fonts, MinerU pipeline extras, constrained PyTorch 2.6/2.7 plus torchvision, and the small compatibility dependency `six` for the MinerU pipeline backend. It provides the `mineru` CLI and stores the workspace plus MinerU/model caches in Docker volumes. First-time MinerU/model setup may download cache files according to MinerU's own behavior and license. The full image is intentionally larger because MinerU requires PyTorch at runtime.

`SEMARK_VLM_*` drives extraction/enrichment. `SEMARK_REVIEW_VLM_*` drives final audit/repair checks and can point to a stronger model; if omitted, it falls back to the enrichment model. The selected model must support image input when visual enrichment is enabled.

Legacy `DOC_PARSER_*` environment variables are still accepted for backward compatibility, but new examples and Docker files use `SEMARK_*`.

API-only development Docker is still available, but it intentionally omits the full MinerU/LibreOffice processing stack and is not the recommended path for end users:

```bash
docker compose -f docker-compose.api-only.yml up --build
```

The compose files keep the backend workspace in a named Docker volume and keep local path ingestion disabled by default. Review `THIRD_PARTY_LICENSES.md` before redistributing images or recommending model downloads.

> **Warning — your data lives in Docker volumes.** All uploaded documents, processing runs, and settings are stored in the named volumes `doc-parser-workspace` and `mineru-cache`. Running `docker compose down -v` (or `docker volume rm`) **permanently deletes all of them**. Use plain `docker compose down` for rebuilds/upgrades — it keeps the volumes. To back up before risky operations:
>
> ```bash
> docker run --rm -v doc-parser-workspace:/data -v "$(pwd)":/backup alpine \
>   tar czf /backup/semark-workspace-backup.tar.gz -C /data .
> ```


## Demo Samples

The repository includes synthetic samples under `examples/samples/` so demos do not depend on private files, customer data, or unclear redistribution rights:

- `synthetic_invoice.html`: English invoice metadata and line-item table.
- `synthetic_form.html`: English form-like fields and approval checklist.
- `synthetic_process_brief.html`: English process blocks and responsibility matrix.
- `synthetic_zh_purchase_request.html`: Traditional Chinese purchase request with fields and line items.
- `synthetic_zh_meeting_minutes.html`: Traditional Chinese meeting minutes with decisions and action items.

Run a local synthetic demo against an already running backend:

```bash
scripts/run_demo_corpus.sh --profile fast
```

Use `fast` for a no-VLM smoke test through the same API and parsing pipeline. Use `accurate` when you want configured VLM semantic enrichment as part of the demo:

```bash
scripts/run_demo_corpus.sh --profile accurate --wait
```

Optional public corpus downloads are intentionally kept outside Git in `workspace/demo-corpus/`:

```bash
SEC_USER_AGENT="Your Name your.email@example.com" scripts/fetch_demo_corpus.sh
scripts/run_demo_corpus.sh --include-public --profile fast
```

The public corpus script downloads:

- IRS Form W-9 PDF from `https://www.irs.gov/pub/irs-pdf/fw9.pdf`, useful for PDF forms and OCR checks.
- The latest Apple 10-K HTML found through the SEC EDGAR submissions API, useful for long-document and table-heavy parsing checks.

Downloaded public files are for local testing only and are not committed. Review each source's current terms before redistributing downloaded files.

Curated output snapshots are available under `examples/demos/`. These show the rendered source page beside generated semantic Markdown, chunks, and quality gate metadata so users can inspect the expected RAG-ready output without running a model.

## Optional Warm MinerU Service

From `backend/` after installing dependencies:

```bash
.venv/bin/mineru-api --host 127.0.0.1 --port 8601
```

Then set this in `backend/.env`:

```env
SEMARK_MINERU_API_URL=http://127.0.0.1:8601
```

Use Settings -> VLM Models -> MinerU Connection to verify the CLI version and whether the configured MinerU API URL is reachable.

## VLM Enrichment and Review

The app-level VLM is optional but recommended for complex forms, figures, diagrams, and tables. The adapter uses an OpenAI-compatible chat-completions interface and can be pointed at Ollama, OpenAI, vLLM, LMDeploy, or another compatible provider. Configure the endpoint reachable from the backend process:

```env
SEMARK_VLM_BASE_URL=http://127.0.0.1:11434/v1
SEMARK_VLM_API_KEY=ollama
SEMARK_VLM_MODEL=your-vision-model
```

For Ollama, use the `/v1` endpoint, for example `http://127.0.0.1:11434/v1`, and set `SEMARK_VLM_API_KEY=ollama`. For cloud OpenAI-compatible APIs, use the provider base URL, API key, and model name. The selected model must support the image input format used by the configured `image_mode` when visual enrichment is enabled.

Two model roles are supported:

- `SEMARK_VLM_*`: extraction/enrichment model used during the Enrich stage for forms, figures, diagrams, and optional table work.
- `SEMARK_REVIEW_VLM_*`: reviewer model used by the final quality gate to audit the generated semantic output and guide controlled repair checks. If unset, the reviewer falls back to `SEMARK_VLM_*`.

No default command sends documents to a cloud model; remote endpoints are opt-in through configuration.

### Collapsed Table Repair

Table repair is a separate switch from table enrichment. `vlm_enrich_tables` buys summaries for tables that parsed cleanly. `vlm_repair_collapsed_tables` — a `PipelineConfig` field, on by default, with no environment variable — re-reads only the tables whose cell boundaries the parser collapsed into one merged cell (26 of 128 tables in the reference corpus) and rebuilds the grid from the table image, so the cost is bounded by how badly the parse failed.

The original parse is kept alongside the repair as `table_body_mineru`, and every guard fails closed: a ragged grid, or a transcription holding less than 60% of the printed characters, is rejected and the original stands. A table with no crop image is not eligible. Because the image is treated as the authority, a table the model misreads is replaced by a wrong grid rather than a garbled one — those guards are what stands between the two.

### Semantic Output Language

The interface language and generated semantic document language are separate settings. In Settings -> Profiles -> Output Package, set Semantic output language to:

- `auto`: detect the output language from document content and enrichment text.
- `zh-TW`: force Traditional Chinese semantic section titles and summaries.
- `en`: force English semantic section titles and summaries.

Use `en` for English-only demo documents, `zh-TW` for Traditional Chinese corpora, and `auto` when the corpus is mixed. The selected value is recorded in each run manifest under `pipeline_config.package.semantic_output_language`.

## For maintainers: pre-release checks

### GitHub Release Dry Run

Before publishing, test from a clean clone rather than the working directory:

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git
```

To include the default full Docker check:

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --docker
```

To include the API-only development Docker check:

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --api-only-docker
```

The script installs full backend dependencies including MinerU, runs backend tests and lint, runs frontend lint and build, and optionally verifies Docker Compose health endpoints. The default Docker check also verifies that `mineru --version` works inside the backend container.

### Notes For Publishing

- Use `doc1/` as the GitHub repository root, not the outer workspace.
- Run `scripts/smoke_clone.sh` against the GitHub URL before announcing the repo.
- Treat `docker-compose.yml`, `docker-compose.full.host.yml`, and `scripts/install_full_local.sh` as the primary full-feature setup paths for MinerU + VLM demos.
- Treat `docker-compose.api-only.yml` as a lightweight development/API smoke path, not as the end-user product path.
- Do not commit `backend/.env`, `backend/workspace/`, generated outputs, or local model caches.
- Commit `backend/.env.example` as the portable configuration template.
- Review `THIRD_PARTY_LICENSES.md` for PyMuPDF, MinerU, and VLM model/provider obligations.
- Use synthetic/public sample documents only.
