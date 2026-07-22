# 繁體中文流程圖 Demo

這個 demo 是一張使用 `accurate` profile 處理的繁體中文流程圖。MinerU 先擷取頁面結構與 OCR evidence，再由設定的 VLM／reviewer model 將最終內容重整成 RAG-ready Semantic Markdown。

## 來源頁面

![來源頁面](source-page.png)

## 產出檔案

- [output.md](output.md)：最終 Semantic Markdown。
- [chunks.jsonl](chunks.jsonl)：由語意輸出建立的檢索 chunks。
- [quality_gate.json](quality_gate.json)：通過／失敗狀態、問題摘要與修復 metadata。

## 模型說明

這份 snapshot 在測試環境中使用本機 Ollama 模型 `qwen3.6:35b-a3b-q8_0`，同時作為 enrichment 與 reviewer model。改用更強的相容 vision 或 reviewer model，可能進一步改善視覺判讀與語意修復品質。

## 執行資訊

- Run ID：`01KW0VM74QCTCYD2Y0RJBWJZ3B`
- Document ID：`af6f53bd9cda7d1c`
- Profile：`accurate`
- 輸出語言：`zh-TW`
- Quality gate：`pass`
- Auto RAG ready：`true`

## 注意事項

來源文件本身就是一張流程圖，因此預期輸出是一份完整語意文件，而不是額外拆成多個子文件。只有當獨立圖片、表格或附件代表不同的可檢索單元時，才會另外分檔。
