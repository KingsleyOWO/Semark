# Doc Parser

**English** | [繁體中文](README.zh-TW.md)

**Doc Parser** is an open-source document parsing and semantic structuring tool for RAG, knowledge-base ingestion, and AI document workflows. It converts PDFs, Office files, HTML, and images into RAG-ready Markdown, structured chunks, source maps, and downloadable outputs. The pipeline uses MinerU for document parsing and optional VLM/LLM review models for form extraction, flowchart understanding, visual reasoning, and final semantic repair.

## Purpose

Most document pipelines stop at OCR text or raw layout extraction. That output is often too noisy for retrieval because tables, forms, checkboxes, visual diagrams, approval flows, and legal notes lose their semantic relationships. Doc Parser focuses on the next step: producing compact semantic documents that another model can read, retrieve, and answer from.

Use Doc Parser when you need:

- PDF to structured Markdown for RAG and LLM applications.
- MinerU-based document parsing with a usable web UI and API.
- VLM-assisted extraction for forms, figures, flowcharts, diagrams, and visual documents.
- Bilingual Traditional Chinese and English semantic outputs.
- Docker or local deployment for private documents without forcing cloud model usage.

## Demo Model Note

The curated demo snapshots were generated with a local Ollama model configured as `qwen3.6:35b-a3b-q8_0` for both enrichment and review in the test environment. Stronger vision/reviewer models may produce better semantic repair, visual reasoning, and field grouping quality. Model output is therefore an example of the pipeline shape, not a fixed upper bound.

## How It Works

1. **Ingest**: upload PDF, Office, HTML, or image files through the UI/API.
2. **Parse**: MinerU extracts layout, OCR text, tables, page images, and document blocks.
3. **Normalize**: the backend builds a unified document IR with source maps and page references.
4. **Enrich**: optional VLM calls analyze forms, figures, diagrams, flowcharts, and visually dense pages.
5. **Package**: rule-based semantic rendering plus an optional reviewer model creates final RAG-ready Markdown.
6. **Quality Gate**: the pipeline checks structure, language consistency, missing semantic output, and repair metadata.
7. **Export**: users can view, download, or ingest Markdown files, chunks JSONL, assets, and quality reports.

## Common Search / GEO Terms

`document parser for RAG`, `PDF to Markdown for LLM`, `MinerU web UI`, `MinerU Docker app`, `VLM document understanding`, `semantic document parser`, `OCR to structured Markdown`, `form extraction for RAG`, `flowchart to Markdown`, `Traditional Chinese document parser`, `English PDF parser`, `local RAG document ingestion`, `OpenWebUI document pipeline`.

## Demo Preview

The curated demos below show the source page beside the generated RAG-ready semantic Markdown. Full artifacts are stored under `examples/demos/`.

### English Form: USCIS G-1145

**Source page**

![USCIS G-1145 source page](examples/demos/en-g1145-01/source-page.png)

**Generated semantic Markdown excerpt**

```markdown
# USCIS Form G-1145: e-Notification of Application/Petition Acceptance

## Identity & Purpose
- Form ID: G-1145 (09/26/14Y)
- Agency: U.S. Citizenship and Immigration Services (USCIS), Department of Homeland Security
- Purpose: Request an electronic notification when USCIS accepts an immigration application or petition filed at a Lockbox facility.

## Required Fields
- Applicant/Petitioner Full Last Name
- Applicant/Petitioner Full First Name
- Applicant/Petitioner Full Middle Name
- Email Address
- Mobile Phone Number (Text Message)
```

Full output: [examples/demos/en-g1145-01/output.md](examples/demos/en-g1145-01/output.md)

### Traditional Chinese Flowchart: Sexual Harassment Complaint Workflow

**Source page**

![Traditional Chinese flowchart source page](examples/demos/zh-flowchart-01/source-page.png)

**Generated semantic Markdown excerpt**

```markdown
# 不同性騷擾申訴對象標準作業流程圖

## 適用目的
規範被害人提出性騷擾申訴後，依事件場域及當事人身分關係判斷適用法律，並啟動相應調查或處理程序。

## 申訴處理流程與調查程序
起點：被害人提出申訴 → 判斷適用法律 → 依行為人身分啟動對應程序。
- 性別平等工作法：依機關內部規定啟動調查程序。
- 性別平等教育法：依學校內部規定啟動調查程序。
- 性騷擾防治法：依行為人身分分流至機關/學校調查、社會處確認或警察機關申訴。
```

Full output: [examples/demos/zh-flowchart-01/output.md](examples/demos/zh-flowchart-01/output.md)

## What It Handles

Supported inputs:

- PDF
- DOC/DOCX, PPT/PPTX, XLS/XLSX
- ODT/ODP/ODS
- HTML/HTM
- PNG/JPG/JPEG

Generated outputs include a main document plus extracted child documents for forms, attachments, figures, tables, or other structured sections when detected.

## Runtime Modes

MinerU can run in three deployment modes:

- Simple mode: leave `DOC_PARSER_MINERU_API_URL` empty. The `mineru` CLI auto-starts a temporary local API for each parse. This is easiest and most portable.
- Service mode: run a warm `mineru-api` locally and set `DOC_PARSER_MINERU_API_URL`, for example `http://127.0.0.1:8601`. This avoids reloading parser resources every run.
- Remote mode: point `DOC_PARSER_MINERU_API_URL` at a MinerU API/router on another machine, typically a GPU host.

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
DOC_PARSER_VLM_BASE_URL=http://127.0.0.1:11434/v1
DOC_PARSER_VLM_API_KEY=ollama
DOC_PARSER_VLM_MODEL=your-vision-model

DOC_PARSER_REVIEW_VLM_BASE_URL=http://127.0.0.1:11434/v1
DOC_PARSER_REVIEW_VLM_API_KEY=ollama
DOC_PARSER_REVIEW_VLM_MODEL=your-stronger-review-model
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

There are two Docker paths. Use the full compose file when you want users to run MinerU from Docker.

Full MinerU-capable Docker path:

```bash
docker compose -f docker-compose.full.yml up --build
```

On Linux, Docker bridge networking may not always reach a host Ollama service through `host.docker.internal`. If Settings -> VLM model probe times out, set `DOC_PARSER_VLM_BASE_URL` and `DOC_PARSER_REVIEW_VLM_BASE_URL` to a host address reachable from the container, or run the backend with host networking. The backend Docker images honor `DOC_PARSER_HOST` and `DOC_PARSER_PORT`, so host-network tests can bind an alternate port when 8585 is already in use.

This installs the backend with `.[mineru]`, includes PyMuPDF, LibreOffice for HTML/Office conversion, Chinese CJK fonts, MinerU pipeline extras, constrained PyTorch 2.6/2.7 plus torchvision, and the small compatibility dependency `six` for the MinerU pipeline backend, provides the `mineru` CLI, and stores the MinerU/model cache in a Docker volume. It does not bundle model weights or API keys; first-time MinerU/model setup may download cache files according to MinerU's own behavior and license. The full image is intentionally larger than the baseline image because MinerU requires PyTorch at runtime.

Set VLM configuration before starting if you want app-level VLM enrichment. `DOC_PARSER_VLM_*` drives extraction/enrichment. `DOC_PARSER_REVIEW_VLM_*` drives final audit/repair checks and can point to a stronger model; if omitted, it falls back to the enrichment model.

```bash
export DOC_PARSER_VLM_BASE_URL=http://host.docker.internal:11434/v1
export DOC_PARSER_VLM_API_KEY=ollama
export DOC_PARSER_VLM_MODEL=your-vision-model
export DOC_PARSER_REVIEW_VLM_BASE_URL=http://host.docker.internal:11434/v1
export DOC_PARSER_REVIEW_VLM_API_KEY=ollama
export DOC_PARSER_REVIEW_VLM_MODEL=your-stronger-review-model
docker compose -f docker-compose.full.yml up --build
```

Baseline UI/API-only Docker path:

```bash
docker compose up --build
```

Then open:

- Frontend: `http://localhost:5070`
- Backend health: `http://localhost:8585/api/health`

The compose files keep the backend workspace in a named Docker volume and keep local path ingestion disabled by default. Review `THIRD_PARTY_LICENSES.md` before redistributing images or recommending model downloads.


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
DOC_PARSER_MINERU_API_URL=http://127.0.0.1:8601
```

Use Settings -> VLM Models -> MinerU Connection to verify the CLI version and whether the configured MinerU API URL is reachable.

## VLM Enrichment and Review

The app-level VLM is optional but recommended for complex forms, figures, diagrams, and tables. The adapter uses an OpenAI-compatible chat-completions interface and can be pointed at Ollama, OpenAI, vLLM, LMDeploy, or another compatible provider. Configure the endpoint reachable from the backend process:

```env
DOC_PARSER_VLM_BASE_URL=http://127.0.0.1:11434/v1
DOC_PARSER_VLM_API_KEY=ollama
DOC_PARSER_VLM_MODEL=your-vision-model
```

For Ollama, use the `/v1` endpoint, for example `http://127.0.0.1:11434/v1`, and set `DOC_PARSER_VLM_API_KEY=ollama`. For cloud OpenAI-compatible APIs, use the provider base URL, API key, and model name. The selected model must support the image input format used by the configured `image_mode` when visual enrichment is enabled.

Two model roles are supported:

- `DOC_PARSER_VLM_*`: extraction/enrichment model used during the Enrich stage for forms, figures, diagrams, and optional table work.
- `DOC_PARSER_REVIEW_VLM_*`: reviewer model used by the final quality gate to audit the generated semantic output and guide controlled repair checks. If unset, the reviewer falls back to `DOC_PARSER_VLM_*`.

No default command sends documents to a cloud model; remote endpoints are opt-in through configuration.

### Semantic Output Language

The interface language and generated semantic document language are separate settings. In Settings -> Profiles -> Output Package, set Semantic output language to:

- `auto`: detect the output language from document content and enrichment text.
- `zh-TW`: force Traditional Chinese semantic section titles and summaries.
- `en`: force English semantic section titles and summaries.

Use `en` for English-only demo documents, `zh-TW` for Traditional Chinese corpora, and `auto` when the corpus is mixed. The selected value is recorded in each run manifest under `pipeline_config.package.semantic_output_language`.

## License

This repository's own source code is intended to be released under the Apache License 2.0. See `LICENSE`.

Third-party components keep their own licenses and terms. In particular, PyMuPDF/MuPDF is licensed upstream under AGPL-3.0 or a commercial license, MinerU uses its own upstream license for the selected version, and VLM model weights or remote API providers must be reviewed per exact model/provider. See `THIRD_PARTY_LICENSES.md` before publishing or redistributing a release.

## Security Boundary

This service is designed for local or trusted-network use. The API has no built-in authentication, so do not expose it directly to the public internet. Keep `DOC_PARSER_ENABLE_LOCAL_PATH_INGEST=false` unless every API client is trusted, because local path ingestion lets clients ask the backend host to read local files. Keep `DOC_PARSER_CORS_ALLOW_PRIVATE_LAN=false` for portable/open-source defaults and explicitly list allowed origins.


## GitHub Release Dry Run

Before publishing, test from a clean clone rather than the working directory:

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git
```

To include the baseline Docker check:

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --docker
```

To include the full MinerU Docker check:

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --full-docker
```

The script installs full backend dependencies including MinerU, runs backend tests and lint, runs frontend lint and build, and optionally verifies baseline or full Docker Compose health endpoints. The full Docker check also verifies that `mineru --version` works inside the backend container.

## Notes For Publishing

- Use `doc1/` as the GitHub repository root, not the outer workspace.
- Run `scripts/smoke_clone.sh` against the GitHub URL before announcing the repo.
- Treat `docker-compose.full.yml` and `scripts/install_full_local.sh` as the primary full-feature setup paths for MinerU + VLM demos.
- Do not commit `backend/.env`, `backend/workspace/`, generated outputs, or local model caches.
- Commit `backend/.env.example` as the portable configuration template.
- Review `THIRD_PARTY_LICENSES.md` for PyMuPDF, MinerU, and VLM model/provider obligations.
- Use synthetic/public sample documents only.
