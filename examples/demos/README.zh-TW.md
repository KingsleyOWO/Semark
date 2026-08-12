# Semark 精選輸出 Demo

[English](README.md) | **繁體中文**

這個目錄放的是以 `accurate` profile 成功執行後挑選出來的輸出快照。每個 demo 都把來源頁面的縮圖與產出的 RAG-ready Markdown 放在一起，讓讀者可以直接比對「視覺輸入」與「語意輸出」。兩個截圖 demo 另外附上只做解析／OCR 的原始輸出，因此不需要跑模型也看得出語意階段實際做了什麼。

## Demo 列表

- `zh-screenshot-guide-01/`：繁體中文截圖教學（國庫繳款書列印）。共兩頁、幾乎全由 UI 截圖構成的操作說明。示範選單列、紅框標示、資料輸入欄位與欄位提示如何變成可檢索的文字，而不是檢索不到的死圖連結。附 `raw-parse.md`，其中兩個畫面各自只是一行圖片連結。
- `en-screenshot-guide-01/`：英文截圖教學（VA Customer Engagement Portal）。兩頁文字與 UI 截圖混排的內容。示範一個會被純解析管線誤讀的頁面——OCR 錯字被升級成標題、操作步驟被當成章節標題——如何被重建回它真正的樣子：一份分步指南。附 `raw-parse.md`。
- `zh-flowchart-01/`：繁體中文流程圖 demo。示範一頁流程圖如何轉成精簡的語意 Markdown 與可供 RAG 匯入的 chunk JSONL。
- `en-g1145-01/`：英文表單 demo。示範一頁 USCIS 表單如何轉成語意 Markdown，包含分組後的用途、填寫說明、必填欄位、法律聲明與 RAG query anchors。

每個 demo 目錄下都有自己的 `README.md`，內含來源連結與更完整的說明。

## 檔案結構

每個 demo 目錄可能包含：

- `source-page.png`，多頁 demo 則為 `source-page-1.png` / `source-page-2.png`：用於視覺比對的來源頁面圖。
- `raw-parse.md`：解析／OCR 層自己的輸出，尚未經過任何語意階段。兩個截圖 demo 有附，作為語意輸出的對照基準。
- `output.md`：最終產出的語意 Markdown（主文）。
- `figure-example.md`：其中一份分檔的圖片語意文件，示範一張截圖如何成為可獨立檢索的語意檔案。
- `chunks.jsonl`：產生的 chunks，供檢索匯入使用。
- `quality_gate.json`：該次執行的品質閘門狀態、分數、issues 與修復 metadata。

這些快照是模型輔助產出的範例，不是對來源文件的法律或法遵解釋定本。
