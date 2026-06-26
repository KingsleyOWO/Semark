0. 目標與非目標
0.1 目標

將 PDF/Word/圖片/HTML 轉為：

Dataset View：高保真 dataset.md（可拿去產 dataset 或再加工）

RAG View：檢索友善 rag.md + assets/ + assets_index.jsonl（表單/資產可召回 + 可回傳）

支援大量文件、可中斷續跑、可重跑同檔比較參數效果（A/B Runs）。

有可視化 Debug：頁面渲染、block bbox 高亮、輸出差異比較、品質報告。

0.2 非目標

不做完整 RAG 系統（向量庫、Query orchestration 由下游接）

不做多人協作與多租戶

不做雲端高併發（但允許接遠端 VLM 推理服務）

1. 核心設計：doc_id 固定、run_id 多版本
1.1 身分與快取鍵

doc_id = sha256(file_bytes)：同一份檔案內容永遠同一 doc_id

run_id = ulid() 或 sha256(doc_id + config_hash + started_at)

config_hash = sha256(canonical_json(pipeline_config))

快取鍵（重要）

Parse cache key：(doc_id, mineru_engine_signature, parse_config_hash)

Enrich cache key：(doc_id, block_fingerprint, vlm_config_hash, prompt_version)

Package cache key：(doc_id, run_id, package_config_hash)（通常不需要共用）

這樣同檔要測參數：只要換 config → 自動 miss 重跑；要強制重跑：force=true 跳過 cache。

1.2 UI 操作（取代 “reset hash”）

New Run：用當前設定建立新 run（可選是否允許用 cache）

Re-run (use cache)：新 run、能用就用（快）

Force Re-run (ignore cache)：新 run、指定 stage 跳過 cache（慢但可測）

Invalidate cache：按 doc + stage 刪 cache（例如只清 Parse 或只清某些 block enrich）

2. 依賴與引擎選型
2.1 文件解析（PDF/Office/圖片）

MinerU 作為主解析器：

**安裝**：`pip install mineru`（非 magic-pdf）

**必要依賴**：
- PyTorch（CUDA）：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
- 文檔佈局：`pip install doclayout-yolo ultralytics`
- 文字處理：`pip install tokenizers transformers ftfy`
- 幾何運算：`pip install shapely pyclipper`
- 配置管理：`pip install omegaconf`

**環境變數**：`TORCH_FORCE_WEIGHTS_ONLY_LOAD=0`（PyTorch 2.9+ 必須）

CLI 參數（method/backend/lang/table/formula/page range/device/vram/model source 等）會完整暴露在本工具設定中。
opendatalab.github.io

content_list.json 為下游組裝/後處理最佳輸入：扁平、閱讀順序、含 bbox（0–1000 normalized）
opendatalab.github.io

輸出含 debug 檔 layout.pdf（閱讀順序/框）與 spans.pdf（pipeline 專用品質檢查）
opendatalab.github.io

注意：MinerU VLM backend 2.5+ 的 structured output 有重大變更，且與 pipeline backend 不相容，run manifest 必須記錄 backend/version。
opendatalab.github.io
+1

2.2 HTML 解析（可選）

兩級策略：

magic-html（便宜/CPU）：主體 HTML 抽取，可輸出純文字/Markdown，並支援公式抽取等特性。
GitHub

MinerU-HTML（Dripper，LLM-based）：分類 + state machine guided generation + fallback，並提供 FastAPI REST server；品質兜底。
GitHub
+1

2.3 VLM 補強（表單/圖/表）

Qwen3-VL（你可換其他 VLM，但 spec 以此為預設）：

Repo 標示 Apache-2.0，並提供 transformers 使用方式；且要求 transformers>=4.57.0。
GitHub
+1

3. 系統架構
3.1 元件

Core Orchestrator（後端）

Job queue / worker pool

Stage 狀態管理（Parse → Normalize → Enrich → Package → Index）

Adapter 層

MinerUAdapter：呼叫 mineru CLI 或 mineru-api（選其一）

HtmlAdapter：magic-html / dripper

VlmAdapter：本地 transformers 或 OpenAI-compatible endpoint（vLLM/LMDeploy）

Data Layer

SQLite：runs/jobs/cache index

File store：每個 doc/run 的產物、資產、manifest、IR、debug

Web UI（localhost）

Runs dashboard、Document viewer、Assets library、Diff、Doctor、Settings

4. 目錄與檔案規範（落地格式）

以 workspace/ 為根目錄：

workspace/
  store/
    docs/
      {doc_id}/
        source/
          original.{ext}
        cache/
          parse/{parse_cache_key}/   # 原生 MinerU 輸出落在這
            mineru_out/...
            mineru_meta.json
        runs/
          {run_id}/
            manifest.json
            document_ir.json
            source_map.json
            outputs/
              dataset.md
              rag.md
              assets_index.jsonl
              quality.json
              enrichments.jsonl
              chunks.jsonl              # 可選：供下游 ingestion
            assets/
              pages/                    # 每頁渲染圖（可選）
              figures/                  # 圖/圖表 crop
              forms/                    # 表單整頁 or 大區塊 crop
              tables/                   # 表格圖（可選）

5. 資料模型（Schema）
5.1 DocumentIR（document_ir.json）

以 MinerU content_list.json 為來源（扁平 blocks、閱讀順序、bbox 0–1000）。
opendatalab.github.io

Top-level
{
  "doc_id": "...",
  "run_id": "...",
  "source": { "path": "...", "ext": "pdf", "sha256": "...", "size_bytes": 123 },
  "engine": {
    "mineru": { "backend": "pipeline", "version": "2.x", "method": "auto", "lang": "chinese_cht" }
  },
  "pages": [
    { "page_idx": 0, "width_px": 2480, "height_px": 3508, "page_image_path": "assets/pages/p0000.png" }
  ],
  "blocks": [
    {
      "block_id": "b000123",
      "type": "text|table|image|equation|code|list",
      "page_idx": 0,
      "bbox_norm": [62,480,946,904],
      "reading_order": 12,
      "payload": { "text": "...", "text_level": 1 }
    }
  ]
}

block.type 對應 MinerU content_list types

text（含 text_level：0=正文，1/2/…=標題層級）
opendatalab.github.io

image（含 img_path、caption/footnote）
opendatalab.github.io

table

equation

若你使用 MinerU VLM backend：content_list 可能額外出現 code、list 等擴展類型（需在 schema 預留）。
opendatalab.github.io
+1

5.2 SourceMap（source_map.json）

用途：Viewer 點 MD ↔ 高亮 block、以及 chunk ↔ blocks 映射。

{
  "md_anchors": [
    { "anchor_id": "a001", "md_range": [120, 240], "block_ids": ["b000123","b000124"] }
  ],
  "chunks": [
    { "chunk_id": "c001", "view": "rag", "block_ids": ["b000200"], "attachments": ["asset://forms/f001.png"] }
  ]
}

5.3 Enrichments（enrichments.jsonl）

每筆對某 block 的 VLM 補強（只 patch、不改原文抽取）。

{"block_id":"b000200","kind":"form_guide","prompt_version":"v1",
 "model":"Qwen3-VL-...","decode":{"temperature":0.2,"top_p":0.8,"max_tokens":1024},
 "input":{"asset_path":"assets/forms/f001.png","context_blocks":["b..."]},
 "output":{"title":"...","triggers":["..."],"filling_guide":"...","field_schema":[...]},
 "quality":{"needs_review":false,"warnings":[]}}

5.4 Assets Index（assets_index.jsonl）— 表單可召回核心
{"type":"form_asset","asset_id":"f001","doc_id":"...","run_id":"...",
 "title":"...","triggers":["..."],"page_idx":0,
 "asset_path":"assets/forms/f001.png",
 "guide_ref":"enrichments.jsonl#b000200",
 "retrieval_text":"title + triggers + short_guide"}


下游 RAG ingestion：把 retrieval_text 當一個可檢索 doc chunk，命中就回傳 asset_path + guide_ref。

6. Pipeline 詳細規格
6.1 Stage A — Ingest

輸入：檔案/資料夾/HTML URL
輸出：doc_id、source/original.ext、DB 建立 doc record

Office 轉 PDF：使用本地 LibreOffice headless（doc/docx/ppt/pptx → pdf）

圖片：png/jpg 直接進 MinerU（若 MinerU 對圖片支援不足，則先包成單頁 pdf）

6.2 Stage B — Parse（MinerU/HTML）
B1: MinerU Parse

呼叫 mineru CLI（建議 subprocess）：

--path、--output 必填

可配置 --method [auto|txt|ocr]、--backend [...]、--lang、--start/--end、--formula/--table、--device、--vram、--source 
opendatalab.github.io

可配置環境變數：MINERU_PDF_RENDER_TIMEOUT、MINERU_TABLE_MERGE_ENABLE、MINERU_INTRA/INTER_OP_NUM_THREADS 等 
opendatalab.github.io

Parse 產物：保留 MinerU 原生輸出（md + structured + debug）

debug：layout.pdf（閱讀順序/框）、spans.pdf（pipeline 專用品質檢查）
opendatalab.github.io

structured：content_list.json（後處理主輸入）
opendatalab.github.io

你可以選擇跑 mineru-api（host/port 參數存在）但個人工具通常 CLI 就夠。
opendatalab.github.io

B2: HTML Parse

先 magic-html（輸出主體 HTML，可轉 Markdown/純文字）
GitHub

若品質不足 → Dripper（LLM-based，state machine guidance + fallback + FastAPI server）
GitHub

6.3 Stage C — Normalize（建 IR）

讀 MinerU content_list.json：

依 reading order 建 blocks

bbox 為 [x0,y0,x1,y1] 且 0–1000 normalized 
opendatalab.github.io

若產生每頁渲染圖：將 bbox_norm 轉 pixel bbox（viewer 用、crop 用）

6.4 Stage D — Enrich（VLM Patch）
D1: Gating（哪些 block 要跑 VLM）

預設只跑：

form：表單整頁或大區塊

figure：圖表/流程圖/截圖

（可選）table_summary：表格語意摘要（不重寫表格）

Gating heuristics（可配置）：

filename regex：申請單|表單|報支|請假|加班|進修|附件

page-level：文字占比低 + 大面積 table/image bbox

MinerU block：type=image 且 caption/footnote 空、或 table 欄位過於破碎

D2: VLM 輸入策略（解決你之前“固定切片”的疑慮）

不做固定切片；改用 MinerU bbox crop：

表單：優先整頁 assets/pages/pXXXX.png（或加 padding crop）

圖：用 image block bbox crop

可選：附加同頁縮圖作 context

D3: VLM 生成參數（可配置）

temperature/top_p/top_k/max_tokens/repetition_penalty（工具層自訂預設）

對 extraction 類任務建議低溫、偏 determinism（例如 temp 0.1–0.3）

Qwen3-VL 環境要求與使用方式以官方 repo 為準（transformers>=4.57.0；Apache-2.0）。
GitHub
+1

6.5 Stage E — Package（輸出 Dataset/RAG Views）
E1: dataset.md（保真）

使用 blocks 組裝：

text：保留原文字與標題層級（text_level → #/##…）
opendatalab.github.io

table：保留為 markdown table 或 html table（以 MinerU payload 為準）

image：僅引用資產（不強制加長描述）

不插入 VLM 長篇敘述（避免污染）

E2: rag.md（檢索友善 + 召回資產）

對 form/figure/table 插入：

semantic_caption/filling_guide (short) 摘要（可檢索）

asset://... 附件引用（viewer/下游可回傳）

同時輸出 assets_index.jsonl（必備）

6.6 Stage F — Chunk（可選，但建議內建）

輸出 chunks.jsonl（讓下游 ingestion 更省事）：

以 heading 為主、最大 token/字數限制、保留 chunk_id → block_ids → attachments

assets_index.jsonl 也可轉成 chunk（form_asset type）

7. Settings/Profiles（UI 必備）
7.1 內建 Profile

FAST

MinerU：pipeline、method=auto、table=true、formula=false

Enrich：只做表單偵測與資產輸出，不跑 VLM

BALANCED（預設）

MinerU：pipeline、auto、table=true、formula=true

Enrich：只跑 form + figure

ACCURATE

MinerU：必要時 method=ocr（或指定 lang）

Enrich：form + figure + table_summary

7.2 MinerU 參數映射（UI 全部可調）

method/backend/lang/url/start/end/formula/table/device/vram/source（全部出自 CLI help）
opendatalab.github.io

env：PDF render timeout、table merge、onnx threads（出自 CLI tools doc 的 env vars）
opendatalab.github.io

tools config：mineru.json 路徑（MINERU_TOOLS_CONFIG_JSON）
opendatalab.github.io

7.3 HTML 參數

extractor：magic-html / dripper

dripper endpoint（若你跑 FastAPI server）
GitHub

7.4 VLM 參數

model source：local transformers / remote openai-compatible endpoint

decoding：temperature/top_p/top_k/max_tokens/repetition_penalty

vision：crop_padding、include_page_thumbnail、form_mode=page|block

8. 後端 API Spec（FastAPI 建議）
8.1 Ingest/Run

POST /api/ingest {path|upload|url, options} → {doc_id}

POST /api/runs {doc_id, profile, config_overrides, use_cache=true, force_stages:[]} → {run_id}

POST /api/runs/{run_id}/cancel

POST /api/docs/{doc_id}/cache/invalidate {stages:["parse"|"enrich"], scope?}

8.2 Status/Artifacts

GET /api/runs（列表+filter）

GET /api/runs/{run_id}（stage 狀態、progress、timings、warnings）

GET /api/runs/{run_id}/document_ir

GET /api/runs/{run_id}/output?view=dataset|rag

GET /api/runs/{run_id}/assets_index

GET /api/assets/{doc_id}/{run_id}/{path}（靜態資產）

8.3 Viewer 支援

GET /api/runs/{run_id}/source_map

GET /api/runs/{run_id}/quality

9. 前端 UI Spec（頁面與元件）
9.1 Pages

Dashboard / Runs

新增資料夾/檔案、New Run、Re-run、Force Re-run、Invalidate cache

Runs table：doc_id、run_id、profile、stage、耗時、錯誤、warnings

Document Viewer

左：rag.md/dataset.md 切換渲染

右：page image + bbox overlay + block list（按 reading_order）

點 MD anchor → 高亮 blocks（依 source_map）

表單區：顯示表單圖 + filling guide + 下載/複製 asset path

Assets Library

以 assets_index.jsonl 為資料源

搜尋 triggers/title

點擊 → 顯示 asset + guide

Diff（Run A vs Run B）

同 doc_id 選兩個 run：

dataset.md diff / rag.md diff

assets_index diff（新增/刪除/修改）

quality 指標 diff

Doctor / Quality Report

Parse coverage：text/table/image count、discarded block 比例（若你接 MinerU VLM backend，content_list 可能包含 discarded blocks 的輸出差異需記錄）
opendatalab.github.io
+1

Debug 連結：layout.pdf/spans.pdf 快速打開 
opendatalab.github.io

Settings

MinerU / HTML / VLM / Profiles / Prompt versions

10. JobStore（SQLite）與狀態機
10.1 Tables（最低限度）

docs(doc_id, source_path, sha256, ext, created_at, meta_json)

runs(run_id, doc_id, config_json, config_hash, status, created_at, updated_at)

run_stages(run_id, stage, status, started_at, finished_at, error_json, stats_json)

cache_entries(cache_key, doc_id, stage, config_hash, path, created_at)

enrich_entries(doc_id, block_id, vlm_config_hash, prompt_version, output_json, created_at)

10.2 Stage 狀態

PENDING → RUNNING → SUCCEEDED/FAILED/CANCELED

支援 resume：

後端啟動時掃描 RUNNING 但 process 不存在 → 標記為 FAILED（可 retry）

retry 可指定從某 stage 開始

11. 品質、可追溯與可重現
11.1 manifest.json（每個 run 必出）

必含：

doc_id/run_id

解析引擎與版本（MinerU backend/version/method/lang/table/formula 等）
opendatalab.github.io
+1

VLM model/endpoint + decoding params

prompt_version

cache hit/miss 統計

timings（parse/enrich/package）

11.2 quality.json（每個 run 必出）

blocks 統計：text/table/image/equation counts（content_list types）
opendatalab.github.io

每頁 coverage（文字占比、圖占比）

enrich 覆蓋率（跑了哪些 kind、失敗率、needs_review 數）

12. 測試與驗收標準（MVP → v1）
12.1 MVP 驗收（最小可用）

可 ingest 資料夾，建立 doc list

可跑 MinerU parse（至少 pipeline backend），產出 dataset.md、rag.md、document_ir.json、assets_index.jsonl

Viewer 能：

顯示 rag.md + page image

點 rag.md 某段 → 高亮 bbox

Assets Library 可搜尋並開啟表單圖

同 doc 可多 run，Diff 能看到 rag.md/assets_index 變化

可中斷續跑（關 UI/重開仍可 resume 或 retry）

12.2 v1 強化

HTML：magic-html + dripper fallback（可選）
GitHub
+1

表單 VLM filling_guide + field_schema

chunk.jsonl 輸出（下游 ingestion 直接用）

13. 實作順序（建議你照這個拆 task）

Workspace/Store + SQLite JobStore

MinerUAdapter（CLI）+ Parse cache

Normalize → DocumentIR + page render + bbox overlay data

Package：dataset.md / rag.md（先不做 VLM）

Assets：forms/figures export + assets_index.jsonl（先 heuristic）

Web UI：Dashboard + Viewer + Assets library

Runs/Cache：New Run / Force Re-run / Invalidate cache + Diff

VlmAdapter + Enrichments.jsonl（最後接 Qwen3-VL）