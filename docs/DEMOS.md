# Semark Demos

**English** | [繁體中文](DEMOS.zh-TW.md) · [Back to README](../README.md)

These demos were produced with v0.3 (2026-07), before page-by-page verification and the verified / not verified labels existed; they show the shape of the semantic Markdown.

## Demo Comparisons

Each demo pairs a real public source page with the generated RAG-ready semantic Markdown, and each one targets a concrete pain point. Full artifacts (output, figure documents, chunks, quality gate) are stored under `examples/demos/`.

| Pain point | What Semark delivers | Demo |
| --- | --- | --- |
| Screenshot-heavy layouts confuse parsers, and raw OCR noise leaks into the knowledge base | The reviewer model rebuilds a clean step-by-step guide; decorative icons are filtered out | [en screenshot guide](#english-screenshot-guide-va-portal-login) |
| Form OCR produces a flat field soup with no usable structure | Purpose, instructions, grouped fields, and legal notes in a fixed RAG template | [en form](#english-form-uscis-g-1145) |
| UI screenshots in manuals are invisible to retrieval — the menu names, highlighted buttons, and field hints are pixels, not text | Screenshots become semantic descriptions, menu/button paths, and grounded on-screen text | [zh screenshot guide](#traditional-chinese-screenshot-guide-treasury-payment-slips) |
| Flowcharts lose their decision logic the moment they become text | Conditions, branches, and responsible parties written out as structured Markdown | [zh flowchart](#traditional-chinese-flowchart-sexual-harassment-complaint-workflow) |

> **A note on personal information**: the screenshot demo sources above are public documents with no personal data. In real use, **Mask Private Info** (on by default) deterministically masks the personal content screenshots commonly leak (mailbox subjects and senders, domain account IDs, email account digits) before delivery — but masking is a pattern-based last line of defense. If personal information must not appear in your documents at all, the most reliable approach is to keep it out of the source screenshots in the first place (cover or blur before capturing). See [Screenshots and Personal Information](#screenshots-and-personal-information).

### English Screenshot Guide: VA Portal Login

**Pain point:** pages that mix prose with UI screenshots often get misread as fillable forms, and the resulting output is either template noise or raw OCR. Here the reviewer repair rebuilt the page into the guide it actually is.

**Source page** (page 1 of 2, the Authorized Use dialog)

![VA portal guide source page](../examples/demos/en-screenshot-guide-01/source-page-1.png)

**What a parse-only pipeline delivers** (raw output excerpt — [raw-parse.md](../examples/demos/en-screenshot-guide-01/raw-parse.md)):

```markdown
![](images/ad9812033c62175142ad60e21944a961cce1451dd0cc4e2c9b49f6885afd327b.jpg)

![](images/840bd0d482fed59b2cb8354121663026b1fe2c13f489952db74f61d881ce9538.jpg)

## 1. After reading the Authorized Use Only, click “I Agree”.

## AAuthorized Use Only

• Deny access to this system

![](images/50898e8b5a1d79fb899f60735826565bca2af476bcae51dcd3735ee9c387cb18.jpg)
```

The screenshots are dead image links (the portal panels' VA10091 / SAM.gov / contact facts simply do not exist as text), an OCR typo "AAuthorized" got promoted to a heading, and a mid-procedure step became a section title.

**What Semark delivers** (main document, excerpt — [full main document](../examples/demos/en-screenshot-guide-01/output.md))

```markdown
# Welcome to the VA Customer Engagement Portal (CEP)

After logging in through **ID.me**, the **Customer Engagement Portal** home page will appear. On the home page, locate the **Vendor Account Setup and Updates** "Login" button.

## Things you will need to get started
*   Your current banking information where you want your payments to go to.
*   If this is an update to a previous enrollment, you will need your previous banking information.

## Time to complete
Completing the webform should take **5 to 20 minutes**. Once you start your form, you cannot save your progress or information to go back to later.

## Instructions

### 1. Accept Terms of Use
After reading the **Authorized Use Only** notice, click "**I Agree**".

### 2. Navigate to Vendor Setup
Click the "**Login**" button under the **Vendor Account Setup and Updates** header.

> **Note:** Do not click on the "**Claim/Invoice Status**" Login button as it will direct you to the part of the Customer Engagement Portal for that activity.

## Vendor File Form (VA10091) Details
The Customer Engagement Portal Vendor File Form has 5 specific sections:
1.  Tax ID Information
2.  Payee Information
3.  Address/Bank Information
4.  Add Authorized Representative Contact/s
5.  Review
```

The decorative warning icon was filtered out, each meaningful screenshot became a figure document with grounded semantics, and the wrong-button warning survives as an explicit instruction a retriever can find.

### English Form: USCIS G-1145

**Pain point:** OCR of a government form yields an unstructured field soup; answering "what does this form do and how do I fill it in" needs purpose, rules, and grouped fields.

**Source page**

![USCIS G-1145 source page](../examples/demos/en-g1145-01/source-page.png)

**Generated semantic Markdown**

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

### Traditional Chinese Screenshot Guide: Treasury Payment Slips

**Pain point:** in screenshot-based manuals the facts that matter — which menu to click, which option is boxed in red, what to type into each field — live inside images, where OCR-only pipelines cannot retrieve them.

**Source page** (page 2 of 2, the data-entry screen)

![Treasury guide source page](../examples/demos/zh-screenshot-guide-01/source-page-2.png)

**What a parse-only pipeline delivers** — this is the complete raw output of the parsing/OCR layer for both pages ([raw-parse.md](../examples/demos/zh-screenshot-guide-01/raw-parse.md)):

```markdown
列印國庫繳款書操作說明  
![](images/3501ffcf4badafeb0891e3722c14444ba46ea14600b95ba58ebbf80e14e515ca.jpg)

版權所有©中華民國109年，財政部國庫署，建議解析度1024×768以上。客服電話:02-2395-7500#325,#32920160302-1517

【說明】

請使用 Google chrome 瀏覽器，連結至「國庫收支書表條碼化 web 版」首頁」（http://veb.nta.gov.tw/），點選畫面左上角「繳款書(01)條碼化作業」，進入資料登打畫面。

![](images/f93289d3a314f46910d41d95a3175e6bea439cd6b9d11ce84166e60d877ac1b4.jpg)

## 【說明】

1. 請依紅框說明文字輸入相關資訊後，點選右上角「列印」即可產製國庫繳款書（為利條碼讀取，請以雷射印表機列印）。

2. 備註欄案號，請填法院裁定遺管人之案號(如：台北地院108年度司繼字第0號)。
```

The entire data-entry screen — every field, every hint value, the red boxes — is that one dead image link. A retriever can answer nothing about the form from this.

**What Semark delivers** (main document, excerpt — [full main document](../examples/demos/zh-screenshot-guide-01/output.md))

```markdown
# 列印國庫繳款書操作說明

頂部導航欄包含四個條碼化作業選項：繳款書(01)、支出收回書(02)、收入退還書(03) 和 轉正通知書(05/04)。
「繳款書(01)條碼化作業」選項被紅框特別標示。

【說明】
請使用 Google chrome 瀏覽器，連結至「國庫收支書表條碼化 web 版」首頁」（http://veb.nta.gov.tw/），點選畫面左上角「繳款書(01)條碼化作業」，進入資料登打畫面。

頁面標題為 VEB001R 繳款書(01)條碼化作業。
收入科目代號需輸入「12171002103」。
收入科目名稱需輸入「其他雜項收入」。
收入機關代號與對帳機關代號均需輸入「1710003」。
填發機關、收入機關名稱及對帳機關名稱均需輸入「國庫署」。
繳款人欄位提示格式為「遺管人○○○繳納被繼承人○○○賸餘現金」。
備註欄位需輸入聯絡電話及案號。

## 【說明】
1. 請依紅框說明文字輸入相關資訊後，點選右上角「列印」即可產製國庫繳款書（為利條碼讀取，請以雷射印表機列印）。
2. 備註欄案號，請填法院裁定遺管人之案號(如：台北地院108年度司繼字第0號)。
```

Every red-box callout and field hint from the screenshots is now plain, retrievable text — ask a RAG system 「收入科目代號要填什麼」 and it can answer 「12171002103」 with this page as the source.

### Traditional Chinese Flowchart: Sexual Harassment Complaint Workflow

**Pain point:** a flowchart's decision logic (which law applies, who investigates, on what deadline) disappears when the image is dropped or OCR'd flat.

**Source page**

![Traditional Chinese flowchart source page](../examples/demos/zh-flowchart-01/source-page.png)

**Generated semantic Markdown**

```markdown
# 不同性騷擾申訴對象標準作業流程圖

版本日期：1130801製（民國113年8月1日）

適用目的：規範被害人提出性騷擾申訴後，依事件場域及當事人身分關係判斷適用法律，並啟動相應調查或處理程序之標準作業流程。

## 一、適用法律判斷基準

依性騷擾事件發生之場域及當事人之身分關係，判斷適用以下法律：

### 《性別平等工作法》適用情形

- 執行職務或求職時遭受性騷擾。
- 於非工作時間遭受所屬機關（學校）之同一人，為持續性性騷擾。
- 於非工作時間遭受不同機關學校，具共同作業或業務往來關係之同一人，為持續性性騷擾。

### 《性別平等教育法》適用情形

- 不論發生在上、下課期間或校內、外，事件之一方為學生，另一方為校長、教師、職員、工友或學生，即屬校園性別事件。

### 《性騷擾防治法》適用情形

- 非屬《性別平等工作法》及《性別平等教育法》適用對象時。

## 二、申訴處理流程與調查程序

起點：被害人提出申訴 → 判斷適用法律 → 依行為人身分啟動對應程序。

### （一）適用《性別平等工作法》或《性別平等教育法》

- 行為人是機關人員：依機關內部規定啟動調查程序。
- 行為人是學校人員：依學校內部規定啟動調查程序。

### （二）適用《性騷擾防治法》（依行為人身分分流）

行為人是機關人員／學校人員：

- 判斷是否有不予受理之情形？
- 是：移送本府社會處確認 → 認應續行調查 → 函復調查單位續行調查。
- 否：回歸依機關／學校內部規定啟動調查程序。

行為人無雇主、身分不明、不知道是誰或找不到：

- 向性騷擾事件發生地之警察機關提出申訴。

## 三、填寫與處理說明

- 本流程圖為標準作業指引，各機關（學校）應依內部規定配合執行調查程序。
- 涉及移送社會處確認或警察機關申訴之案件，應保留完整紀錄並函復相關調查單位。
```

## Demo Model Note

The curated demo snapshots were generated with a local Ollama model configured as `qwen3.6:35b-a3b-q8_0` for both enrichment and review in the test environment. Stronger vision/reviewer models may produce better semantic repair, visual reasoning, and field grouping quality. Model output is therefore an example of the pipeline shape, not a fixed upper bound.

## Screenshots and Personal Information

Semark treats screenshot-based documents (software manuals, internal how-to guides, onboarding walkthroughs) as first-class input:

- **Screenshot understanding**: UI screenshots are described semantically — what the screen shows, which menu items or buttons are highlighted, and the visible on-screen text — so the steps stay retrievable even though the source is an image. Captions are grounded against text that actually appears in the document, which suppresses hallucinated product or vendor names, and decorative icons or unreadable crops are filtered out of the RAG output and chunks.
- **Main text only**: the download menu can export just the main document — the same content the document view shows — when you do not want figure attachments in the deliverable.
- **Personal information masking**: real screenshots often leak content unrelated to the teaching material (mailbox subjects and senders, domain account IDs such as `CORP\x12345`, personal email account digits). **Mask Private Info** is enabled by default (toggle under Settings → Output Package); while on, Semark deterministically masks these patterns on every delivered surface (Markdown, split documents, chunks) while keeping instructional content such as account-format explanations intact.

The masking is a best-effort safety net for common patterns, not a guarantee. If personal or sensitive information must never appear in the output, blur or cover it in the source screenshots before processing — the pipeline cannot restore what was never uploaded, which makes that the only fully reliable option.
