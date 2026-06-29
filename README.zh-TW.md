# Semark

[English](README.md) | **繁體中文**

> 面向 RAG 的 Semantic Markdown 文件結構化工具：將 PDF、Office、HTML、圖片轉成 RAG-ready Markdown、chunks、source maps 與自動分檔輸出。

**Semark** 是一套開源 Semantic Markdown / 文件語意結構化工具，目標是把 PDF、Office、HTML、圖片等文件轉成可直接用於 RAG、知識庫匯入與大模型問答的 Markdown、結構化 chunks、來源對照與可下載輸出。當長文件中包含多個表單、表格、流程圖、圖示或附件時，系統可以自動拆成主文與獨立語意文件，方便後續資料整理、檢索與匯入知識庫。系統以 MinerU 作為版面、OCR 與文件解析 evidence，並可接入本地或雲端 VLM/LLM 進行表單抽取、流程圖理解、視覺語意補強、最終輸出審核修復與品質檢查。

## 軟體目的

很多文件處理工具只能輸出 OCR 純文字或鬆散的版面結果，對 RAG 來說通常不夠用，因為表格、表單、勾選欄位、流程圖、簽核路徑、注意事項與法律依據很容易失去語意關係。Semark 的重點不是只做 OCR，而是把文件整理成「大模型容易讀、容易召回、容易回答問題」的語意化結構文本。

適合使用 Semark 的情境：

- 將 PDF、Office、HTML、圖片轉成 RAG-ready Markdown。
- 使用 MinerU 做文件解析，但需要更完整的 UI/API 與輸出管理。
- 針對表單、圖表、流程圖、圖片型文件接入 VLM 做語意理解。
- 自動分檔長文件中的表單、表格、流程圖、圖示或附件，形成主文與可獨立檢索的子文件。
- 支援英文與繁體中文輸出，專有名詞可保留英文。
- 希望文件留在本機或內網處理，模型端點可自行選擇本地 Ollama 或雲端 OpenAI-compatible API。

## Demo 模型說明

目前 curated demo snapshots 是在測試環境中使用本地 Ollama 模型 `qwen3.6:35b-a3b-q8_0` 產生，該模型同時作為 enrichment 與 reviewer model。若改用更強的 vision model 或 reviewer model，理論上在語意修復、視覺理解、欄位分組與流程判讀上會有更好的效果。因此 demo 展示的是目前 pipeline 的輸出形態，不代表模型能力上限。

## 從這裡開始

- 第一次完整安裝：[從 GitHub 快速啟動](#從-github-快速啟動)。
- Docker 與 MinerU：[Docker Quickstart](#docker-quickstart)。
- 本機或雲端模型端點：[VLM Enrichment and Review](#vlm-enrichment-and-review)。
- 內建範例與可選 public corpus 下載：[Demo Samples](#demo-samples)。
- 不跑模型也能先看預期輸出：[Demo Preview](#demo-preview)。

## 大致運作方式

1. **文件匯入**：透過 UI/API 上傳 PDF、Office、HTML 或圖片。
2. **MinerU 解析**：抽取 OCR 文字、版面、表格、頁面影像與文件區塊。
3. **標準化 IR**：後端建立統一的 document IR，保留頁碼、來源 block 與 source map。
4. **VLM 語意補強**：可選擇對表單、圖片、流程圖、圖表進行視覺理解與欄位抽取。
5. **語意包裝與自動分檔**：規則式結構化輸出搭配 reviewer model，產生最後的 RAG-ready Markdown，並將獨立表單、表格、流程圖、圖示或附件拆成子文件。
6. **品質檢查**：檢查語言一致性、空輸出、錯誤分檔、流程/表單結構與修復結果。
7. **輸出使用**：可在 Viewer 檢視、下載主文、分檔語意文件、Markdown/chunks/assets，也可直接匯入 RAG 或知識庫系統。

## 常見搜尋關鍵詞 / GEO Terms

`Semark`、`Semark 文件解析`、`Semantic Markdown for RAG`、`RAG 文件解析工具`、`PDF 轉語意 Markdown`、`PDF 轉 Markdown`、`RAG-ready Markdown`、`LLM-ready Markdown`、`AI 文件解析`、`MinerU UI`、`MinerU Docker`、`VLM 文件理解`、`自動文件分檔`、`表單抽取 RAG`、`流程圖轉 Markdown`、`表格轉 Markdown`、`OCR 轉結構化文本`、`繁體中文文件解析`、`英文 PDF 語意化`、`本地端 RAG 文件匯入`、`OpenWebUI 文件處理流程`、`LlamaIndex 文件匯入`、`LangChain 文件匯入`。

建議 GitHub topics：`rag`、`pdf-to-markdown`、`semantic-markdown`、`document-ai`、`mineru`、`vlm`、`ocr`、`llm`、`knowledge-base`、`ollama`、`openai-compatible`、`traditional-chinese`。

## Demo Preview

以下 curated demos 會把來源頁面與產出的 RAG-ready semantic Markdown 放在一起，完整檔案位於 `examples/demos/`。

### 英文表單：USCIS G-1145

**來源頁面**

![USCIS G-1145 source page](examples/demos/en-g1145-01/source-page.png)

**完整語意化 Markdown**

```markdown
# USCIS Form G-1145: e-Notification of Application/Petition Acceptance

## Identity & Purpose
- **Form ID:** G-1145 (09/26/14Y)
- **Agency:** U.S. Citizenship and Immigration Services (USCIS), Department of Homeland Security
- **Purpose:** Request an electronic notification (e-Notification) via email and/or text message when USCIS accepts your immigration application or petition filed at a Lockbox facility. This service is provided as a convenience and does not grant any status or benefit.

## Instructions for Completion & Submission
- Complete this form and clip it to the first page of your application package.
- You will receive one e-Notification per form filed.
- **Delivery Timeline:** Notifications are sent within 24 hours after USCIS accepts the application.
- **Recipient Rules:**
  - Domestic customers: Email and/or text message.
  - Overseas customers: Email only.
- **Important Notes:** Undeliverable e-Notifications cannot be resent. The notification will display your receipt number and a link to check case status, but will not contain personal information. USCIS will also mail a physical receipt notice (I-797C) within 10 days of acceptance.

## Required Fields
Grouped by meaning for completion:
- **Applicant/Petitioner Identification:**
  - `Applicant/Petitioner Full Last Name`
  - `Applicant/Petitioner Full First Name`
  - `Applicant/Petitioner Full Middle Name`
- **Contact Information:**
  - `Email Address` (Required for all)
  - `Mobile Phone Number (Text Message)` (Required for domestic text notifications)

## Privacy Act Statement & Legal Disclosures
- **Authorities:** Collected pursuant to section 103(a) of the Immigration and Nationality Act (INA).
- **Purpose:** To request electronic notification upon USCIS acceptance of immigration forms.
- **Disclosure:** Provision is voluntary. Failure to provide information may prevent receipt of text/email notifications.
- **Routine Uses:** Information will be used/disclosed to DHS personnel and contractors per approved system of records notices [DHS/USCIS-007 & DHS/USCIS-001]. May also be shared for law enforcement or national security purposes.

## RAG Query Anchors
- Form G-1145 e-Notification purpose, Lockbox filing instructions, 24-hour notification timeline, domestic vs overseas delivery rules, I-797C receipt notice mailing timeframe, Privacy Act authorities and routine uses, field completion requirements.
```

### 繁體中文流程圖：性騷擾申訴對象標準作業流程

**來源頁面**

![Traditional Chinese flowchart source page](examples/demos/zh-flowchart-01/source-page.png)

**完整語意化 Markdown**

```markdown
# 不同性騷擾申訴對象標準作業流程圖

**版本資訊**：1130801製

## 語意摘要
本文件為「不同性騷擾申訴對象標準作業流程圖」之結構化語意描述。流程自被害人提出申訴開始，依據事件發生場域與當事人身分關係判斷適用之法規（《性別平等工作法》、《性別平等教育法》或《性騷擾防治法》），並依行為人身分啟動機關或學校內部調查程序；若無雇主或身分不明時，則移送警察或社會機關處理。後續將評估是否受理及是否續行調查。

## 流程邏輯與判斷節點

### 1. 申訴提出
- **起始動作**：被害人提出申訴

### 2. 適用法律判斷（依性騷擾事件發生之場域及當事人之身分關係）
- **情境一**：執行職務或求職時；或於非工作時間遭受所屬機關(學校)之同一人持續性性騷擾；或於非工作時間遭受不同機關學校具共同作業或業務往來關係之同一人持續性性騷擾。
  - **適用法規**：《性別平等工作法》
  - **後續處理**：若行為人是機關人員，依機關內部規定啟動調查程序。
- **情境二**：不論發生在上、下課期間或校內、外，事件之一方為學生，另一方為校長、教師、職員、工友或學生。
  - **適用法規**：《性別平等教育法》
  - **後續處理**：若行為人是學校人員，依學校內部規定啟動調查程序。
- **情境三**：非屬《性別平等工作法》及《性別平等教育法》適用對象時。
  - **適用法規**：《性騷擾防治法》
  - **後續處理**：向性騷擾事件發生地之警察機關提出申訴，或移送本府社會機關處理。

### 3. 受理與調查程序
- **判斷節點**：是否有不予受理之情形？
  - **是**：函復調查單位。
  - **否**：認應續行調查 → 續行調查。
```

## 支援輸入與輸出

支援輸入：

- PDF
- DOC/DOCX、PPT/PPTX、XLS/XLSX
- ODT/ODP/ODS
- HTML/HTM
- PNG/JPG/JPEG

產出內容包含主文文件，以及在偵測到獨立檢索單元時自動產生的子文件。例如一份長 PDF 可以輸出 `main.md` 作為主文，並另外產生表單、表格、流程圖、圖示、附件或其他結構化區塊的語意 Markdown 檔。

## 從 GitHub 快速啟動

最短的完整功能路徑是 Docker Compose：

```bash
git clone https://github.com/KingsleyOWO/Semark.git
cd Semark

# 可選：如果要接本機 Ollama vision model。
# Ollama 若尚未啟動，請在另一個 shell 啟動，並先拉取你要使用的模型。
ollama pull your-vision-model

export SEMARK_VLM_BASE_URL=http://host.docker.internal:11434/v1
export SEMARK_VLM_API_KEY=ollama
export SEMARK_VLM_MODEL=your-vision-model
export SEMARK_REVIEW_VLM_BASE_URL=http://host.docker.internal:11434/v1
export SEMARK_REVIEW_VLM_API_KEY=ollama
export SEMARK_REVIEW_VLM_MODEL=your-review-model

docker compose up --build
```

打開 `http://localhost:5070`。上傳 PDF、Office、HTML 或圖片，在文件清單勾選上傳的文件，然後使用 `accurate` 執行 MinerU 加上已設定的 VLM/LLM enrichment。`fast` 適合做不跑模型的輕量 smoke test。

在 Linux/WSL 上，如果 container 透過 `host.docker.internal` 連不到主機上的 Ollama，改用 host-network compose：

```bash
export SEMARK_VLM_MODEL=your-vision-model
export SEMARK_REVIEW_VLM_MODEL=your-review-model
docker compose -f docker-compose.full.host.yml up --build
```

任務成功後，進入 `Viewer` 檢查來源頁面、語意 Markdown、chunks、quality metadata 與分檔文件；進入 `文件管理` 可以批次下載或單獨下載產出的 Markdown/DOCX/TXT 文件。

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

本服務設計給本機或可信任內網使用。API 沒有內建 authentication，請勿直接暴露到 public internet。除非所有 API client 都可信任，請維持 `SEMARK_ENABLE_LOCAL_PATH_INGEST=false`，因為 local path ingestion 會允許 client 要求 backend host 讀取本機檔案。開源預設建議維持 `SEMARK_CORS_ALLOW_PRIVATE_LAN=false`，並明確列出允許的 origins。

## GitHub Release Dry Run

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

## Notes For Publishing

- GitHub repository root 應使用 `doc1/`，不是外層 workspace。
- 對 GitHub URL 執行 `scripts/smoke_clone.sh` 後再公開宣布 repo。
- `docker-compose.yml`、`docker-compose.full.host.yml` 與 `scripts/install_full_local.sh` 是 MinerU + VLM demo 的主要 full-feature setup path。
- `docker-compose.api-only.yml` 是 lightweight development/API smoke path，不是一般使用者的主要路徑。
- 不要 commit `backend/.env`、`backend/workspace/`、generated outputs 或 local model caches。
- `backend/.env.example` 是可攜的設定模板，應保留在 repository。
- 請確認 `THIRD_PARTY_LICENSES.md` 中 PyMuPDF、MinerU 與 VLM model/provider 的授權義務。
- 只使用 synthetic 或 public sample documents。
