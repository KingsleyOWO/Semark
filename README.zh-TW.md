# Doc Parser

[English](README.md) | **繁體中文**

**Doc Parser** 是一套開源文件解析與語意結構化工具，目標是把 PDF、Office、HTML、圖片等文件轉成可直接用於 RAG、知識庫匯入與大模型問答的 Markdown、結構化 chunks、來源對照與可下載輸出。當長文件中包含多個表單、表格、流程圖、圖示或附件時，系統可以自動拆成主文與獨立語意文件，方便後續資料整理、檢索與匯入知識庫。系統以 MinerU 作為版面、OCR 與文件解析基礎，並可接入本地或雲端 VLM/LLM 進行表單抽取、流程圖理解、視覺語意補強與最終輸出審核修復。

## 軟體目的

很多文件處理工具只能輸出 OCR 純文字或鬆散的版面結果，對 RAG 來說通常不夠用，因為表格、表單、勾選欄位、流程圖、簽核路徑、注意事項與法律依據很容易失去語意關係。Doc Parser 的重點不是只做 OCR，而是把文件整理成「大模型容易讀、容易召回、容易回答問題」的語意化結構文本。

適合使用 Doc Parser 的情境：

- 將 PDF、Office、HTML、圖片轉成 RAG-ready Markdown。
- 使用 MinerU 做文件解析，但需要更完整的 UI/API 與輸出管理。
- 針對表單、圖表、流程圖、圖片型文件接入 VLM 做語意理解。
- 自動分檔長文件中的表單、表格、流程圖、圖示或附件，形成主文與可獨立檢索的子文件。
- 支援英文與繁體中文輸出，專有名詞可保留英文。
- 希望文件留在本機或內網處理，模型端點可自行選擇本地 Ollama 或雲端 OpenAI-compatible API。

## Demo 模型說明

目前 curated demo snapshots 是在測試環境中使用本地 Ollama 模型 `qwen3.6:35b-a3b-q8_0` 產生，該模型同時作為 enrichment 與 reviewer model。若改用更強的 vision model 或 reviewer model，理論上在語意修復、視覺理解、欄位分組與流程判讀上會有更好的效果。因此 demo 展示的是目前 pipeline 的輸出形態，不代表模型能力上限。

## 大致運作方式

1. **文件匯入**：透過 UI/API 上傳 PDF、Office、HTML 或圖片。
2. **MinerU 解析**：抽取 OCR 文字、版面、表格、頁面影像與文件區塊。
3. **標準化 IR**：後端建立統一的 document IR，保留頁碼、來源 block 與 source map。
4. **VLM 語意補強**：可選擇對表單、圖片、流程圖、圖表進行視覺理解與欄位抽取。
5. **語意包裝與自動分檔**：規則式結構化輸出搭配 reviewer model，產生最後的 RAG-ready Markdown，並將獨立表單、表格、流程圖、圖示或附件拆成子文件。
6. **品質檢查**：檢查語言一致性、空輸出、錯誤分檔、流程/表單結構與修復結果。
7. **輸出使用**：可在 Viewer 檢視、下載主文、分檔語意文件、Markdown/chunks/assets，也可直接匯入 RAG 或知識庫系統。

## 常見搜尋關鍵詞 / GEO Terms

`RAG 文件解析工具`、`PDF 轉 Markdown`、`MinerU UI`、`MinerU Docker`、`VLM 文件理解`、`自動文件分檔`、`表單抽取 RAG`、`流程圖轉 Markdown`、`表格轉 Markdown`、`OCR 轉結構化文本`、`繁體中文文件解析`、`英文 PDF 語意化`、`本地端 RAG 文件匯入`、`OpenWebUI 文件處理流程`。

## Demo Preview

以下 curated demos 會把來源頁面與產出的 RAG-ready semantic Markdown 放在一起，完整檔案位於 `examples/demos/`。

### 英文表單：USCIS G-1145

**來源頁面**

![USCIS G-1145 source page](examples/demos/en-g1145-01/source-page.png)

**語意化 Markdown 摘要**

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

完整輸出：[examples/demos/en-g1145-01/output.md](examples/demos/en-g1145-01/output.md)

### 繁體中文流程圖：性騷擾申訴對象標準作業流程

**來源頁面**

![Traditional Chinese flowchart source page](examples/demos/zh-flowchart-01/source-page.png)

**語意化 Markdown 摘要**

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

完整輸出：[examples/demos/zh-flowchart-01/output.md](examples/demos/zh-flowchart-01/output.md)

## 支援輸入與輸出

支援輸入：

- PDF
- DOC/DOCX、PPT/PPTX、XLS/XLSX
- ODT/ODP/ODS
- HTML/HTM
- PNG/JPG/JPEG

產出內容包含主文文件，以及在偵測到獨立檢索單元時自動產生的子文件。例如一份長 PDF 可以輸出 `main.md` 作為主文，並另外產生表單、表格、流程圖、圖示、附件或其他結構化區塊的語意 Markdown 檔。

## Runtime Modes

MinerU 可用三種部署方式執行：

- Simple mode：不設定 `DOC_PARSER_MINERU_API_URL`。`mineru` CLI 會在每次解析時自動啟動暫時的本地 API，最容易使用也最可攜。
- Service mode：在本機預先啟動常駐 `mineru-api`，並設定 `DOC_PARSER_MINERU_API_URL`，例如 `http://127.0.0.1:8601`。這樣可以避免每次任務重新載入解析資源。
- Remote mode：將 `DOC_PARSER_MINERU_API_URL` 指向另一台機器上的 MinerU API/router，通常是 GPU 主機。

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
DOC_PARSER_VLM_BASE_URL=http://127.0.0.1:11434/v1
DOC_PARSER_VLM_API_KEY=ollama
DOC_PARSER_VLM_MODEL=your-vision-model

DOC_PARSER_REVIEW_VLM_BASE_URL=http://127.0.0.1:11434/v1
DOC_PARSER_REVIEW_VLM_API_KEY=ollama
DOC_PARSER_REVIEW_VLM_MODEL=your-stronger-review-model
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

Docker 有兩種啟動方式。如果希望使用者直接在 Docker 內跑 MinerU，請使用 full compose。

完整 MinerU-capable Docker：

```bash
docker compose -f docker-compose.full.yml up --build
```

在 Linux/WSL 上，Docker bridge networking 不一定能透過 `host.docker.internal` 連到主機上的 Ollama。如果 Settings -> VLM model probe timeout，請改用 host-network compose，讓容器直接用 `127.0.0.1:11434` 呼叫 Ollama：

```bash
export DOC_PARSER_VLM_MODEL=your-vision-model
export DOC_PARSER_REVIEW_VLM_MODEL=your-stronger-review-model
docker compose -f docker-compose.full.host.yml up --build
```

如果 `5070` 或 `8585` 已被占用，可以同時覆寫 host-network compose 的前後端 port：

```bash
DOC_PARSER_FRONTEND_PORT=35070 DOC_PARSER_PORT=38585 \
  docker compose -f docker-compose.full.host.yml up --build
```

Full image 會安裝 backend `.[mineru]`、PyMuPDF、LibreOffice、中文 CJK fonts、MinerU pipeline extras、受限制版本的 PyTorch 2.6/2.7 與 torchvision，以及 MinerU pipeline backend 需要的 `six`。它會提供 `mineru` CLI，並把 MinerU/model cache 放在 Docker volume。映像不包含 model weights 或 API keys；第一次 MinerU/model setup 可能依照 MinerU 上游行為與授權下載 cache 檔案。

如果要啟用 app-level VLM enrichment，請先設定模型端點。`DOC_PARSER_VLM_*` 用於 extraction/enrichment；`DOC_PARSER_REVIEW_VLM_*` 用於最後 audit/repair checks，也可以指向更強的模型。若省略 reviewer 設定，會 fallback 到 enrichment model。

```bash
export DOC_PARSER_VLM_BASE_URL=http://host.docker.internal:11434/v1
export DOC_PARSER_VLM_API_KEY=ollama
export DOC_PARSER_VLM_MODEL=your-vision-model
export DOC_PARSER_REVIEW_VLM_BASE_URL=http://host.docker.internal:11434/v1
export DOC_PARSER_REVIEW_VLM_API_KEY=ollama
export DOC_PARSER_REVIEW_VLM_MODEL=your-stronger-review-model
docker compose -f docker-compose.full.yml up --build
```

Baseline UI/API-only Docker：

```bash
docker compose up --build
```

開啟：

- Frontend：`http://localhost:5070`
- Backend health：`http://localhost:8585/api/health`

Compose files 會把 backend workspace 放在 named Docker volume，並預設關閉 local path ingestion。重新散布 image 或推薦模型下載前，請先閱讀 `THIRD_PARTY_LICENSES.md`。

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
DOC_PARSER_MINERU_API_URL=http://127.0.0.1:8601
```

到 Settings -> VLM 模型 -> MinerU 連線設定，可檢查 CLI version 與設定的 MinerU API URL 是否可連線。

## VLM Enrichment and Review

App-level VLM 是可選功能，但對複雜表單、圖表、流程圖、diagram 與表格建議啟用。Adapter 使用 OpenAI-compatible chat-completions interface，可接 Ollama、OpenAI、vLLM、LMDeploy 或其他相容 provider。請設定 backend process 能連到的 endpoint：

```env
DOC_PARSER_VLM_BASE_URL=http://127.0.0.1:11434/v1
DOC_PARSER_VLM_API_KEY=ollama
DOC_PARSER_VLM_MODEL=your-vision-model
```

Ollama 請使用 `/v1` endpoint，例如 `http://127.0.0.1:11434/v1`，並設定 `DOC_PARSER_VLM_API_KEY=ollama`。Cloud OpenAI-compatible API 則填入 provider base URL、API key 與 model name。若啟用 visual enrichment，所選模型必須支援設定的 `image_mode` 所使用的 image input format。

支援兩種模型角色：

- `DOC_PARSER_VLM_*`：extraction/enrichment model，用於 Enrich stage 的表單、圖表、diagram 與可選表格處理。
- `DOC_PARSER_REVIEW_VLM_*`：reviewer model，用於最後 quality gate 審核產出的 semantic output，並輔助受控 repair checks。若未設定，reviewer 會 fallback 到 `DOC_PARSER_VLM_*`。

預設沒有任何指令會把文件送到雲端模型；remote endpoints 必須由使用者明確設定才會啟用。

### Semantic Output Language

介面語言與產出的 semantic document language 是分開設定。在 Settings -> Profiles -> Output Package 中，Semantic output language 可設定為：

- `auto`：依文件內容與 enrichment text 自動判斷輸出語言。
- `zh-TW`：強制使用繁體中文 semantic section titles 與 summaries。
- `en`：強制使用英文 semantic section titles 與 summaries。

英文 demo 文件建議用 `en`，繁體中文 corpus 建議用 `zh-TW`，混合 corpus 可用 `auto`。此設定會記錄在每次 run manifest 的 `pipeline_config.package.semantic_output_language`。

## License

本 repository 自有原始碼預計以 Apache License 2.0 發布，詳見 `LICENSE`。

第三方元件維持各自授權與條款。特別是 PyMuPDF/MuPDF 上游為 AGPL-3.0 或 commercial license，MinerU 使用所選版本的上游 license，VLM model weights 或 remote API provider 也需要依實際模型/供應商確認。發布或重新散布前請閱讀 `THIRD_PARTY_LICENSES.md`。

## Security Boundary

本服務設計給本機或可信任內網使用。API 沒有內建 authentication，請勿直接暴露到 public internet。除非所有 API client 都可信任，請維持 `DOC_PARSER_ENABLE_LOCAL_PATH_INGEST=false`，因為 local path ingestion 會允許 client 要求 backend host 讀取本機檔案。開源預設建議維持 `DOC_PARSER_CORS_ALLOW_PRIVATE_LAN=false`，並明確列出允許的 origins。

## GitHub Release Dry Run

發布前請用 clean clone 測試，而不是只測工作目錄：

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git
```

包含 baseline Docker 檢查：

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --docker
```

包含 full MinerU Docker 檢查：

```bash
scripts/smoke_clone.sh --repo https://github.com/OWNER/REPO.git --full-docker
```

此 script 會安裝 full backend dependencies，包括 MinerU，執行 backend tests/lint、frontend lint/build，並可選擇驗證 baseline 或 full Docker Compose health endpoints。Full Docker 檢查也會確認 `mineru --version` 能在 backend container 內正常執行。

## Notes For Publishing

- GitHub repository root 應使用 `doc1/`，不是外層 workspace。
- 對 GitHub URL 執行 `scripts/smoke_clone.sh` 後再公開宣布 repo。
- `docker-compose.full.yml` 與 `scripts/install_full_local.sh` 是 MinerU + VLM demo 的主要 full-feature setup path。
- 不要 commit `backend/.env`、`backend/workspace/`、generated outputs 或 local model caches。
- `backend/.env.example` 是可攜的設定模板，應保留在 repository。
- 請確認 `THIRD_PARTY_LICENSES.md` 中 PyMuPDF、MinerU 與 VLM model/provider 的授權義務。
- 只使用 synthetic 或 public sample documents。
