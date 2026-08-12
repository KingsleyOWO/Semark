# Semark Curated Output Demos

**English** | [繁體中文](README.zh-TW.md)

This directory contains curated output snapshots from successful `accurate`-profile runs.
Each demo keeps a small source-page image beside the generated RAG-ready Markdown so
readers can compare the visual input with the semantic output. The two screenshot demos
additionally ship the raw parse-only baseline, so the difference the semantic stage makes
is visible without running a model.

## Demos

- `zh-screenshot-guide-01/`: Traditional Chinese screenshot guide (Treasury payment slips).
  A two-page how-to built almost entirely from UI screenshots. Shows menu bars, red-box
  callouts, data-entry fields and field hints becoming retrievable text instead of dead
  image links. Ships `raw-parse.md`, where both screens are a single image link each.
- `en-screenshot-guide-01/`: English screenshot guide (VA Customer Engagement Portal).
  Two pages that mix prose with UI screenshots. Shows a page that a parse-only pipeline
  misreads — an OCR typo promoted to a heading, a mid-procedure step turned into a section
  title — rebuilt into the step-by-step guide it actually is. Ships `raw-parse.md`.
- `zh-flowchart-01/`: Traditional Chinese flowchart demo. Shows a one-page process
  diagram converted into concise semantic Markdown and chunk JSONL for RAG ingestion.
- `en-g1145-01/`: English form demo. Shows a one-page USCIS form converted into
  semantic Markdown with grouped purpose, instructions, required fields, disclosures,
  and RAG query anchors.

Each demo directory has its own `README.md` with the source link and a fuller walkthrough.

## Artifact Layout

Each demo directory may include:

- `source-page.png`, or `source-page-1.png` / `source-page-2.png` for multi-page demos:
  rendered source pages used for visual comparison.
- `raw-parse.md`: the parse/OCR layer's own output, before any semantic stage. Present in
  the two screenshot demos as the baseline the semantic output is compared against.
- `output.md`: final generated semantic Markdown (the main document).
- `figure-example.md`: one split figure document, showing how a screenshot is delivered as
  an independently retrievable semantic file.
- `chunks.jsonl`: generated chunks intended for retrieval ingestion.
- `quality_gate.json`: quality gate status, score, issues, and repair metadata for the run.

These snapshots are examples of model-assisted output, not golden legal or compliance
interpretations of the source documents.
