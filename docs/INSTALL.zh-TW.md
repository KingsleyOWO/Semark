# 安裝、設定與部署

[English](INSTALL.md) | **繁體中文** · [回到 README](../README.zh-TW.md)

## 支援輸入與輸出

支援輸入：

- PDF
- DOC/DOCX、PPT/PPTX、XLS/XLSX
- ODT/ODP/ODS
- HTML/HTM
- PNG/JPG/JPEG

DOCX 優先使用原生結構解析；解析失敗時，會自動透過 LibreOffice 轉成 PDF，再執行既有的 PDF 解析流程。原始 Word 會保留，轉出的 PDF 會用於頁面預覽與文字補充。若轉檔回報圖片或版面異常，品質報告會標示 `needs_review`，並在 `parse_report.json` 記錄備援原因與警告。此備援需要 LibreOffice（完整 Docker 映像已包含）。

產出內容包含主文文件，以及在偵測到獨立檢索單元時自動產生的子文件。例如一份長 PDF 可以輸出 `main.md` 作為主文，並另外產生表單、表格、流程圖、圖示、附件或其他結構化區塊的語意 Markdown 檔。

## 會下載哪些檔案

- `git clone` 只會下載原始碼、文件、synthetic samples 與小型 curated demo snapshots。
- `docker compose up --build` 會下載 OS packages、Python wheels、npm packages、MinerU dependencies、PyMuPDF、LibreOffice、CJK fonts，以及完整 parser image 所需的 runtime dependencies。
- 第一次執行 MinerU 時，可能會依 MinerU 上游行為與授權下載 parser/model cache。Docker 會把這些 cache 放在 named volumes，不會進 Git。
- Semark 不會內建 Ollama models、雲端模型權重、API keys、私人文件、generated outputs 或本機 cache。若使用本機 Ollama，請先在主機上執行 `ollama pull model-name`。
- `scripts/fetch_demo_corpus.sh` 會把可選的 public test files 下載到 `workspace/demo-corpus/`，該目錄已被 Git 忽略。

## Runtime Modes

MinerU 可用三種部署方式執行：

- Simple mode：不設定 `SEMARK_MINERU_API_URL`。`mineru` CLI 會在每次解析時自動啟動暫時的本地 API，最容易使用也最可攜。
- Service mode：在本機預先啟動常駐 `mineru-api`，並設定 `SEMARK_MINERU_API_URL`，例如 `http://127.0.0.1:8601`。這樣可以避免每次任務重新載入解析資源。
- Remote mode：將 `SEMARK_MINERU_API_URL` 指向另一台機器上的 MinerU API/router，通常是 GPU 主機。

如果設定的 MinerU API URL 無法連線，系統會 fallback 到 simple mode，避免 warm service 暫時不可用就讓解析直接失敗。

## Full Feature Quickstart

如果你要完整功能，也就是 MinerU 解析加上可選的 VLM 語意補強，使用這個安裝流程：

```bash
./scripts/install_full_local.sh
```

這會安裝：

- Backend Python dependencies。
- PyMuPDF，用於 PDF/image 處理。
- MinerU pipeline dependencies，透過 `mineru` optional dependency group 安裝。
- Frontend npm dependencies。

如果要在本機轉換 HTML、DOC、PPT、ODT 或 ODP，請另外安裝 LibreOffice：

```bash
sudo apt install libreoffice
```

若要啟用模型補強，請在 `backend/.env` 設定 VLM。Enrichment model 用於表單抽取與視覺理解；Reviewer model 用於最後 quality gate 的審核與受控修復。若 reviewer model 未設定，會沿用 enrichment model。

```env
SEMARK_VLM_BASE_URL=http://127.0.0.1:11434/v1
SEMARK_VLM_API_KEY=ollama
SEMARK_VLM_MODEL=your-vision-model

SEMARK_REVIEW_VLM_BASE_URL=http://127.0.0.1:11434/v1
SEMARK_REVIEW_VLM_API_KEY=ollama
SEMARK_REVIEW_VLM_MODEL=your-stronger-review-model
```

啟動服務：

```bash
cd backend
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8585
```

```bash
cd frontend
npm run dev
```

打開 `http://localhost:5070`，上傳文件，並選擇 `accurate` profile 執行 MinerU + VLM enrichment。

檢查完整本地環境：

```bash
./scripts/check_full_stack.sh
```

## Manual Local Quickstart

Backend：

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e ".[dev,mineru]"
cp .env.example .env
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8585
```

Frontend：

```bash
cd frontend
npm install
npm run dev
```

如果要從區網連線，將 backend/frontend 綁定到 `0.0.0.0`，並開啟對應防火牆連接埠。

## Docker Quickstart

預設 Docker 路徑就是完整功能。`docker compose up --build` 會用 full backend 建置，包含 MinerU、PyMuPDF、LibreOffice、中文 CJK fonts，以及實際文件處理需要的轉檔與解析依賴。Repository 提供的是可重現的設定檔與安裝流程；不會 commit 或內建 model weights、API keys、私人文件、generated outputs 或本機 cache 檔案。

如果使用 Docker Desktop，或主機環境中 container 可以透過 `host.docker.internal` 連到本機 Ollama：

```bash
export SEMARK_VLM_BASE_URL=http://host.docker.internal:11434/v1
export SEMARK_VLM_API_KEY=ollama
export SEMARK_VLM_MODEL=your-vision-model
export SEMARK_REVIEW_VLM_BASE_URL=http://host.docker.internal:11434/v1
export SEMARK_REVIEW_VLM_API_KEY=ollama
export SEMARK_REVIEW_VLM_MODEL=your-stronger-review-model

docker compose up --build
```

開啟：

- Frontend：`http://localhost:5070`
- Backend health：`http://localhost:8585/api/health`

在 Linux/WSL 上，Docker bridge networking 不一定能透過 `host.docker.internal` 連到主機上的 Ollama。如果 Settings -> VLM model probe timeout，請改用 host-network compose，讓容器直接用 `127.0.0.1:11434` 呼叫 Ollama：

```bash
export SEMARK_VLM_MODEL=your-vision-model
export SEMARK_REVIEW_VLM_MODEL=your-stronger-review-model

docker compose -f docker-compose.full.host.yml up --build
```

如果 `5070` 或 `8585` 已被占用，可以同時覆寫 host-network compose 的前後端 port：

```bash
SEMARK_FRONTEND_PORT=35070 SEMARK_PORT=38585 \
  docker compose -f docker-compose.full.host.yml up --build
```

用 IP 位址或 `localhost` 開啟前端一律可行。如果想透過**具名主機**（例如內部 DNS 名稱）存取，需要明確加入允許清單——其他 Host header 會被拒絕，避免 DNS rebinding 攻擊打到無認證的後端：

```bash
SEMARK_FRONTEND_ALLOWED_HOSTS=semark.internal,docs.example.lan \
  docker compose up --build
```

如果使用雲端或遠端 OpenAI-compatible provider，就把兩組模型 endpoint 指到對方服務，而不是 Ollama：

```bash
export SEMARK_VLM_BASE_URL=https://your-provider.example/v1
export SEMARK_VLM_API_KEY=your-api-key
export SEMARK_VLM_MODEL=your-vision-model
export SEMARK_REVIEW_VLM_BASE_URL=https://your-provider.example/v1
export SEMARK_REVIEW_VLM_API_KEY=your-api-key
export SEMARK_REVIEW_VLM_MODEL=your-stronger-review-model

docker compose up --build
```

Full image 會安裝 backend `.[mineru]`、PyMuPDF、LibreOffice、中文 CJK fonts、MinerU pipeline extras、受限制版本的 PyTorch 2.6/2.7 與 torchvision，以及 MinerU pipeline backend 需要的 `six`。它會提供 `mineru` CLI，並把 workspace 與 MinerU/model cache 放在 Docker volume。第一次 MinerU/model setup 可能依照 MinerU 上游行為與授權下載 cache 檔案。Full image 較大是正常的，因為 MinerU runtime 需要 PyTorch。

`SEMARK_VLM_*` 用於 extraction/enrichment；`SEMARK_REVIEW_VLM_*` 用於最後 audit/repair checks，也可以指向更強的模型。若省略 reviewer 設定，會 fallback 到 enrichment model。若啟用 visual enrichment，所選模型必須支援 image input。

舊版 `DOC_PARSER_*` 環境變數仍會被接受以維持相容性，但新的文件與 Docker 設定都以 `SEMARK_*` 為主。

API-only development Docker 仍然保留，但它刻意不包含完整 MinerU/LibreOffice processing stack，不建議作為一般使用者的主要路徑：

```bash
docker compose -f docker-compose.api-only.yml up --build
```

Compose files 會把 backend workspace 放在 named Docker volume，並預設關閉 local path ingestion。重新散布 image 或推薦模型下載前，請先閱讀 `THIRD_PARTY_LICENSES.md`。

> **警告——你的資料存放在 Docker volumes 裡。** 所有上傳的文件、處理紀錄與設定都存在 named volumes `doc-parser-workspace` 和 `mineru-cache`。執行 `docker compose down -v`（或 `docker volume rm`）會**永久刪除全部資料**。重建/升級請用不帶 `-v` 的 `docker compose down`——volume 會保留。進行有風險的操作前，建議先備份：
>
> ```bash
> docker run --rm -v doc-parser-workspace:/data -v "$(pwd)":/backup alpine \
>   tar czf /backup/semark-workspace-backup.tar.gz -C /data .
> ```

## Demo Samples

Repository 內含 `examples/samples/` synthetic samples，讓 demo 不依賴私人檔案、客戶資料或授權不明的文件：

- `synthetic_invoice.html`：英文 invoice metadata 與 line-item table。
- `synthetic_form.html`：英文表單欄位與 approval checklist。
- `synthetic_process_brief.html`：英文流程區塊與 responsibility matrix。
- `synthetic_zh_purchase_request.html`：繁體中文採購申請單，包含欄位與明細。
- `synthetic_zh_meeting_minutes.html`：繁體中文會議紀錄，包含決議與待辦事項。

對已啟動的 backend 執行本地 synthetic demo：

```bash
scripts/run_demo_corpus.sh --profile fast
```

`fast` 可用於不跑 VLM 的 smoke test，但仍會通過相同 API 與解析 pipeline。若要把已設定的 VLM semantic enrichment 納入 demo，使用 `accurate`：

```bash
scripts/run_demo_corpus.sh --profile accurate --wait
```

可選的 public corpus 下載刻意放在 Git 之外的 `workspace/demo-corpus/`：

```bash
SEC_USER_AGENT="Your Name your.email@example.com" scripts/fetch_demo_corpus.sh
scripts/run_demo_corpus.sh --include-public --profile fast
```

Public corpus script 會下載：

- IRS Form W-9 PDF：`https://www.irs.gov/pub/irs-pdf/fw9.pdf`，適合 PDF forms 與 OCR 檢查。
- SEC EDGAR submissions API 找到的最新 Apple 10-K HTML，適合長文件與表格密集文件測試。

下載的 public files 只用於本地測試，不會 commit。若要重新散布下載檔案，請自行確認來源目前條款。

Curated output snapshots 放在 `examples/demos/`。這些範例展示來源頁面、產出的 semantic Markdown、chunks 與 quality gate metadata，讓使用者不跑模型也能先檢查預期的 RAG-ready output。

## Optional Warm MinerU Service

在 `backend/` 安裝依賴後執行：

```bash
.venv/bin/mineru-api --host 127.0.0.1 --port 8601
```

然後在 `backend/.env` 設定：

```env
SEMARK_MINERU_API_URL=http://127.0.0.1:8601
```

到 Settings -> VLM 模型 -> MinerU 連線設定，可檢查 CLI version 與設定的 MinerU API URL 是否可連線。

## VLM Enrichment and Review

App-level VLM 是可選功能，但對複雜表單、圖表、流程圖、diagram 與表格建議啟用。Adapter 使用 OpenAI-compatible chat-completions interface，可接 Ollama、OpenAI、vLLM、LMDeploy 或其他相容 provider。請設定 backend process 能連到的 endpoint：

```env
SEMARK_VLM_BASE_URL=http://127.0.0.1:11434/v1
SEMARK_VLM_API_KEY=ollama
SEMARK_VLM_MODEL=your-vision-model
```

Ollama 請使用 `/v1` endpoint，例如 `http://127.0.0.1:11434/v1`，並設定 `SEMARK_VLM_API_KEY=ollama`。Cloud OpenAI-compatible API 則填入 provider base URL、API key 與 model name。若啟用 visual enrichment，所選模型必須支援設定的 `image_mode` 所使用的 image input format。

支援兩種模型角色：

- `SEMARK_VLM_*`：extraction/enrichment model，用於 Enrich stage 的表單、圖表、diagram 與可選表格處理。
- `SEMARK_REVIEW_VLM_*`：reviewer model，用於最後 quality gate 審核產出的 semantic output，並輔助受控 repair checks。若未設定，reviewer 會 fallback 到 `SEMARK_VLM_*`。

預設沒有任何指令會把文件送到雲端模型；remote endpoints 必須由使用者明確設定才會啟用。

### Collapsed Table Repair（表格崩欄修復）

表格修復與表格語意補強是兩個獨立的開關。`vlm_enrich_tables` 是為「解析正常的表格」加摘要；`vlm_repair_collapsed_tables`（`PipelineConfig` 欄位，預設開啟，沒有對應的環境變數）只處理「欄位邊界被解析器併成一格」的表格——參考語料中 128 張表裡的 26 張——並從表格圖片還原格線，所以成本與解析壞掉的程度成正比，不會擴散到整份文件。

原始解析結果會以 `table_body_mineru` 一併保留，而且所有防護都是 fail closed：格線數量不齊的結果、或轉錄字數不到印出字數 60% 的結果，一律拒收並沿用原始解析。沒有裁切圖的表格不列入處理範圍。由於這條路徑以圖片為準，模型讀錯的表格會被換成「錯的格線」而不是「糊掉的格線」——上述防護就是這兩者之間唯一的界線。

### Semantic Output Language

介面語言與產出的 semantic document language 是分開設定。在 Settings -> Profiles -> Output Package 中，Semantic output language 可設定為：

- `auto`：依文件內容與 enrichment text 自動判斷輸出語言。
- `zh-TW`：強制使用繁體中文 semantic section titles 與 summaries。
- `en`：強制使用英文 semantic section titles 與 summaries。

英文 demo 文件建議用 `en`，繁體中文 corpus 建議用 `zh-TW`，混合 corpus 可用 `auto`。此設定會記錄在每次 run manifest 的 `pipeline_config.package.semantic_output_language`。

## 維護者：發布前檢查

### GitHub Release Dry Run

發布前請用 clean clone 測試，而不是只測工作目錄：

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git
```

包含預設完整 Docker 檢查：

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --docker
```

包含 API-only development Docker 檢查：

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --api-only-docker
```

此 script 會安裝 full backend dependencies，包括 MinerU，執行 backend tests/lint、frontend lint/build，並可選擇驗證 Docker Compose health endpoints。預設 Docker 檢查也會確認 `mineru --version` 能在 backend container 內正常執行。

### Notes For Publishing

- GitHub repository root 應使用 `doc1/`，不是外層 workspace。
- 對 GitHub URL 執行 `scripts/smoke_clone.sh` 後再公開宣布 repo。
- `docker-compose.yml`、`docker-compose.full.host.yml` 與 `scripts/install_full_local.sh` 是 MinerU + VLM demo 的主要 full-feature setup path。
- `docker-compose.api-only.yml` 是 lightweight development/API smoke path，不是一般使用者的主要路徑。
- 不要 commit `backend/.env`、`backend/workspace/`、generated outputs 或 local model caches。
- `backend/.env.example` 是可攜的設定模板，應保留在 repository。
- 請確認 `THIRD_PARTY_LICENSES.md` 中 PyMuPDF、MinerU 與 VLM model/provider 的授權義務。
- 只使用 synthetic 或 public sample documents。
