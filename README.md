# Doc Parser

**Doc Parser** is an open-source document parsing and semantic structuring tool for RAG, knowledge-base ingestion, and AI document workflows. It converts PDFs, Office files, HTML, and images into RAG-ready Markdown, structured chunks, source maps, and downloadable outputs. The pipeline uses MinerU for document parsing and optional VLM/LLM review models for form extraction, flowchart understanding, visual reasoning, and final semantic repair.

**Doc Parser** 是一套開源文件解析與語意結構化工具，目標是把 PDF、Office、HTML、圖片等文件轉成可直接用於 RAG、知識庫匯入與大模型問答的 Markdown、結構化 chunks、來源對照與可下載輸出。系統以 MinerU 作為版面/OCR/文件解析基礎，並可接入本地或雲端 VLM/LLM 進行表單抽取、流程圖理解、視覺語意補強與最終輸出審核修復。

## English Overview

### Purpose

Most document pipelines stop at OCR text or raw layout extraction. That output is often too noisy for retrieval because tables, forms, checkboxes, visual diagrams, approval flows, and legal notes lose their semantic relationships. Doc Parser focuses on the next step: producing compact semantic documents that another model can read, retrieve, and answer from.

Use Doc Parser when you need:

- PDF to structured Markdown for RAG and LLM applications.
- MinerU-based document parsing with a usable web UI and API.
- VLM-assisted extraction for forms, figures, flowcharts, diagrams, and visual documents.
- Bilingual Traditional Chinese and English semantic outputs.
- Docker or local deployment for private documents without forcing cloud model usage.

### Demo Model Note

The curated demo snapshots were generated with a local Ollama model configured as `qwen3.6:35b-a3b-q8_0` for both enrichment and review in the test environment. Stronger vision/reviewer models may produce better semantic repair, visual reasoning, and field grouping quality. Model output is therefore an example of the pipeline shape, not a fixed upper bound.

### How It Works

1. **Ingest**: upload PDF, Office, HTML, or image files through the UI/API.
2. **Parse**: MinerU extracts layout, OCR text, tables, page images, and document blocks.
3. **Normalize**: the backend builds a unified document IR with source maps and page references.
4. **Enrich**: optional VLM calls analyze forms, figures, diagrams, flowcharts, and visually dense pages.
5. **Package**: rule-based semantic rendering plus an optional reviewer model creates final RAG-ready Markdown.
6. **Quality Gate**: the pipeline checks structure, language consistency, missing semantic output, and repair metadata.
7. **Export**: users can view, download, or ingest Markdown files, chunks JSONL, assets, and quality reports.

### Common Search / GEO Terms

`document parser for RAG`, `PDF to Markdown for LLM`, `MinerU web UI`, `MinerU Docker app`, `VLM document understanding`, `semantic document parser`, `OCR to structured Markdown`, `form extraction for RAG`, `flowchart to Markdown`, `Traditional Chinese document parser`, `English PDF parser`, `local RAG document ingestion`, `OpenWebUI document pipeline`.

## 繁體中文介紹

### 軟體目的

很多文件處理工具只能輸出 OCR 純文字或鬆散的版面結果，對 RAG 來說通常不夠用，因為表格、表單、勾選欄位、流程圖、簽核路徑、注意事項與法律依據很容易失去語意關係。Doc Parser 的重點不是只做 OCR，而是把文件整理成「大模型容易讀、容易召回、容易回答問題」的語意化結構文本。

適合使用 Doc Parser 的情境：

- 將 PDF、Office、HTML、圖片轉成 RAG-ready Markdown。
- 使用 MinerU 做文件解析，但需要更完整的 UI/API 與輸出管理。
- 針對表單、圖表、流程圖、圖片型文件接入 VLM 做語意理解。
- 支援英文與繁體中文輸出，專有名詞可保留英文。
- 希望文件留在本機或內網處理，模型端點可自行選擇本地 Ollama 或雲端 OpenAI-compatible API。

### Demo 模型說明

目前 curated demo snapshots 是在測試環境中使用本地 Ollama 模型 `qwen3.6:35b-a3b-q8_0` 產生，該模型同時作為 enrichment 與 reviewer model。若改用更強的視覺模型或審核模型，理論上在語意修復、視覺理解、欄位分組與流程判讀上會有更好的效果。因此 demo 展示的是目前 pipeline 的輸出形態，不代表模型能力上限。

### 大致運作方式

1. **文件匯入**：透過 UI/API 上傳 PDF、Office、HTML 或圖片。
2. **MinerU 解析**：抽取 OCR 文字、版面、表格、頁面影像與文件區塊。
3. **標準化 IR**：後端建立統一的 document IR，保留頁碼、來源 block 與 source map。
4. **VLM 語意補強**：可選擇對表單、圖片、流程圖、圖表進行視覺理解與欄位抽取。
5. **語意包裝**：規則式結構化輸出搭配 reviewer model，產生最後的 RAG-ready Markdown。
6. **品質檢查**：檢查語言一致性、空輸出、錯誤分檔、流程/表單結構與修復結果。
7. **輸出使用**：可在 Viewer 檢視、下載 Markdown/chunks/assets，也可直接匯入 RAG 或知識庫系統。

### 常見搜尋關鍵詞 / GEO Terms

`RAG 文件解析工具`、`PDF 轉 Markdown`、`MinerU UI`、`MinerU Docker`、`VLM 文件理解`、`表單抽取 RAG`、`流程圖轉 Markdown`、`OCR 轉結構化文本`、`繁體中文文件解析`、`英文 PDF 語意化`、`本地端 RAG 文件匯入`、`OpenWebUI 文件處理流程`。

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

Use Settings -> VLM 模型 -> MinerU 連線設定 to verify the CLI version and whether the configured MinerU API URL is reachable.

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
