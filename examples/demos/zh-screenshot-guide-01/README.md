# 繁體中文截圖教學 Demo：國庫繳款書列印

這個 demo 是一份共兩頁、主要由 UI 截圖構成的繁體中文操作說明，使用 `accurate` profile 處理。截圖中的選單列、紅框標示、資料輸入欄位與欄位提示，會被轉成語意描述及可檢索文字，而不是留在 RAG 無法讀取的圖片中。

## 來源頁面

![來源頁面 1](source-page-1.png)

![來源頁面 2](source-page-2.png)

## 產出檔案

- [raw-parse.md](raw-parse.md)：只做解析／OCR 的完整原始輸出；兩張截圖在結果中都只剩圖片連結。
- [output.md](output.md)：整合正文與圖片解釋的主文，也是 Viewer 顯示及下載「主文」時取得的內容。
- [figure-example.md](figure-example.md)：其中一份獨立圖片語意文件；將表單畫面整理成語意事實與場景描述。
- [chunks.jsonl](chunks.jsonl)：由語意輸出建立的檢索 chunks。
- [quality_gate.json](quality_gate.json)：品質檢查結果，分數為 1.0，沒有未解決問題。

## 建議比較重點

- 比較 [raw-parse.md](raw-parse.md) 與 [output.md](output.md)：原始解析中，整個資料輸入畫面只剩一條圖片連結；語意輸出則讓每個欄位提示都成為可檢索文字。
- 第 1 頁的紅框提示（「繳款書(01)條碼化作業」選項被紅框特別標示）會保留成檢索得到的文字。
- 表單截圖中的欄位提示，例如收入科目代號「12171002103」及機關代號「1710003」，會被轉錄一次，並整理成有根據的語意事實。
- 文件標題會從頁面內容還原為「列印國庫繳款書操作說明」，而不是使用通用章節標籤或檔名。

## 模型說明

這份 snapshot 在測試環境中使用本機 Ollama 模型 `qwen3.6:35b-a3b-q8_0`，同時作為 enrichment 與 reviewer model。改用更強的相容 vision 或 reviewer model，可能進一步改善視覺判讀與語意品質。

## 執行資訊

- Run ID：`01KXQEWEJTXCRG7SDNM2H5RN6C`
- Document ID：`028e33ae775f8034`
- Profile：`accurate`
- 輸出語言：`zh-TW`
- Quality gate：`pass`（1.0）

## 來源標示

來源頁面取自中華民國財政部國庫署公開的「國庫收支應用書表條碼化 Web 版」國庫繳款書列印操作說明：

- 系統：<https://veb.nta.gov.tw/>
- 機關資訊頁：<https://www.nta.gov.tw/singlehtml/241?cntId=8ab3b311519146c7bd4126f8a72fc260>

## 注意事項

這份教學說明的是公開政府系統的操作流程。畫面中的電話號碼與欄位範例值均為機關公開的非個人資訊。本輸出用於展示截圖型文件轉成 RAG-ready 語意內容後的形式，不取代官方操作說明。
