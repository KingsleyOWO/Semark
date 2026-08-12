# Release Notes 2026-07-17 (v0.3.0)

This release focuses on screenshot-based how-to documents: understanding UI screenshots, keeping personal information out of the delivered output, and making downloads predictable.

## Highlights

### New demos: screenshots vs parse-only baseline

Two public screenshot how-to guides were added under `examples/demos/` — a Traditional Chinese treasury payment slip guide and pages from the VA Customer Engagement Portal vendor guide. Each demo ships the raw parse-only baseline (`raw-parse.md`, where entire screens are dead image links) next to the semantic output, and the README Demo Preview was restructured around the pain points each demo solves. Reviewing these demos also drove five export-quality fixes: guide-style title inference (操作說明/使用說明), bracket section labels (「【說明】」) no longer become document titles, repeated screenshot OCR lines are deduplicated in the main document, an applied semantic repair backfills the document title from the repaired heading, and the form-template gate respects the repair verdict instead of re-flagging reviewer-authored guides.

### Screenshot understanding

- Figure captions are grounded against text that actually appears in the document, suppressing hallucinated product or vendor names in captions and keywords.
- Screenshots are no longer mislabeled as flowcharts: menu-path evidence (`A > B > C`) still becomes structured content, but the stored image type stays truthful.
- Decorative icons and unreadable crops (no legible OCR text, self-declared blurry captions) are filtered out of the RAG weave and the chunk stream.
- OCR noise (page-number groups, cropped account fragments) is no longer promoted to headings, keeping chunk heading paths meaningful.
- Traditional Chinese output cleanup: OpenCC 了/瞭 round-trip damage is repaired on both the detection and conversion side, and mainland vocabulary is mapped to Taiwan usage (界面→介面, 圖標→圖示, 分辨率→解析度, …) with guards for terms like 界面活性劑.

### Personal information masking

- New deterministic privacy scrub for content transcribed from screenshots: mailbox subject/sender lines, domain account IDs (for example `CORP\x12345`), and personal numeric email account IDs (`d*****@…`) are masked on every delivered surface — Markdown, split documents, and chunks.
- Instructional content survives: account-format explanations and UI menu labels are explicitly preserved.
- Enabled by default; toggle under Settings → Output Package as **Mask Private Info** (遮蔽個人資訊).
- Masking is a best-effort safety net for common patterns. If personal or sensitive content must never appear in the output, blur or cover it in the source screenshots before processing.

### Downloads redesigned

- 主文 (main text) downloads now serve exactly the document the viewer renders — the previous separate render path could differ from the on-screen content and has been removed. Existing runs benefit immediately, no re-processing needed.
- The Documents page download flow is a per-action menu: choose content (main text only / all documents), choose format (MD/DOCX/TXT), download. No more global mode toggles that silently override the checked documents; the last choice is remembered.
- Batch downloads export each document once, from its newest run (`dedupe_by_doc` in the batch API, default off for compatibility), and the menu states the effective scope: runs selected → documents exported.
- Single-file cases download the file directly instead of a one-file ZIP; flat ZIP entries no longer overwrite each other when several runs share a source name.
- The run list loads completely (paged) instead of stopping at the first 100 runs.

### Quality gate alignment

- Gate verdicts now track content quality for how-to guides: table-of-contents dot leaders and quote-enclosed cropped UI labels no longer trigger the truncated-output warning, while genuinely truncated lines still do.
- Verdict cascade repair and table fidelity preservation in the packaging stage.

## Compatibility Notes

- `outputs/main_text.md` is no longer written for new runs; the download API resolves the main document from the split-document index. Files from older runs remain readable through the same endpoint (legacy fallback).
- The batch download API gained the optional `dedupe_by_doc` flag (default `false`), so existing API clients keep their exact-runs semantics.
- Backend test suite: 437 tests green.
