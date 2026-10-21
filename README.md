# Semark

**English** | [繁體中文](README.zh-TW.md)

> Turn PDFs, Word files, slides and images into RAG-ready Markdown, checked page by page against the source, with every part labelled verified or not.

Semark is a self-hosted document processing service. It extracts text, tables and pictures with MinerU and native Word parsing, then hands forms, flowcharts, charts and software screenshots to the vision model you choose (local Ollama or any OpenAI-compatible API). Finally it compares every page with the source and ships Markdown and chunks you can import into a knowledge base as they are.

![Semark turns screenshots into retrievable, grounded text](docs/assets/outreach/semark-vlm-screenshot-before-after.png)

## What makes it different

- **No source text goes missing.** Before delivery, every paragraph of the source is checked against the delivered documents and chunks, and the result goes into the quality report. Missing text is restored during review or clearly flagged; it never disappears silently.
- **It reads pictures.** Forms, flowcharts, charts and UI screenshots become retrievable text: fields, branch conditions, highlighted buttons and menu paths are kept. When the parser merges a table's columns, the grid is rebuilt from the table image.
- **Page-by-page verification, honest labels.** A model compares each source page with what was actually delivered, fixes what is wrong and checks again. Unreadable regions are marked as unreadable instead of guessed; pictures that could not be verified keep only the lettering read off them and are labelled "not verified" in the reading copy.
- **Your data stays with you.** Everything can run on your machine or LAN; no document leaves it unless you configure a remote endpoint.

## See it in action

Each demo puts a public source page next to Semark's output. The walkthroughs are in [docs/DEMOS.md](docs/DEMOS.md); the files are in `examples/demos/`.

| Pain point | What Semark does | Demo |
| --- | --- | --- |
| Screenshots in manuals are invisible to retrieval: menu names, highlights and field hints are just pixels | Screenshots become semantic descriptions, menu/button paths and on-screen text checked against the page | [Traditional Chinese screenshot guide](docs/DEMOS.md#traditional-chinese-screenshot-guide-treasury-payment-slips) |
| Flowcharts lose their decision logic once flattened to text | Conditions, branches and responsible parties as structured Markdown | [Traditional Chinese flowchart](docs/DEMOS.md#traditional-chinese-flowchart-sexual-harassment-complaint-workflow) |
| Screenshot-heavy pages get misclassified and OCR noise leaks into the knowledge base | Rebuilt as a clean step-by-step guide; decorative icons filtered out | [English screenshot guide](docs/DEMOS.md#english-screenshot-guide-va-portal-login) |
| Form OCR is a flat soup of fields | Purpose, rules, grouped fields and legal notes in a fixed layout | [English form](docs/DEMOS.md#english-form-uscis-g-1145) |

## Quickstart

You need Docker. For local models, install Ollama on the host and pull a model first:

```bash
git clone https://github.com/KingsleyOWO/Semark.git
cd Semark

export SEMARK_VLM_BASE_URL=http://host.docker.internal:11434/v1
export SEMARK_VLM_API_KEY=ollama
export SEMARK_VLM_MODEL=your-vision-model

docker compose up --build
```

Open `http://localhost:5070`, upload a document, select it and run `accurate`. When it finishes, compare source pages and output in `Viewer`, and download Markdown, DOCX or TXT from document management.

- On Linux/WSL, if the container cannot reach Ollama on the host, use `docker compose -f docker-compose.full.host.yml up --build`.
- For a cloud model, set `SEMARK_VLM_BASE_URL` and the API key to your provider's values. Verification can use a stronger model (`SEMARK_REVIEW_VLM_*`); without one it uses the same model.
- Local installs, a warm MinerU service, ports and host names, and the bundled samples are covered in [docs/INSTALL.md](docs/INSTALL.md).

> **Your data lives in Docker volumes.** `docker compose down -v` permanently deletes every uploaded document and result; use `docker compose down` without `-v` when upgrading.

## How it works

1. **Parse**: MinerU extracts text, layout and tables from PDFs and images; Word files are read from their native structure (converted to PDF only if that fails).
2. **Visual enrichment**: the vision model reads forms, flowcharts, charts and screenshots; tables with merged columns get their grid rebuilt from the image.
3. **Package**: a main document plus separately retrievable documents for forms, tables, flowcharts and pictures, with chunks and source maps.
4. **Verify page by page**: every page is listed, and a model compares the source page with what was delivered. When something is wrong it picks a tool (re-read a picture, edit a few words, restore missing content), then rebuilds the output and checks it again.
5. **Final validation**: source coverage, verification records and delivered content must agree before the result is published.

Details and settings: [backend/docs/agentic-processing.md](backend/docs/agentic-processing.md) (Traditional Chinese).

## What you get

- `documents/`: the package to import into RAG — `main.md` plus a document for each picture, table and form. Its text stays clean; verification status is kept as metadata.
- `chunks.jsonl`: ready-made chunks, each carrying its source block IDs.
- `delivery.md`: a reading copy with each picture's verification status in front of it.
- Quality and coverage reports: which pages are verified or pending, and whether all source text was delivered.

Labels you will see in the reading copy:

| Label | Meaning |
| --- | --- |
| `[Image content: automatic reading, verified]` | The model's reading of the picture was checked against the source |
| `[Image content: automatic reading, not verified]` | It could not be checked; only lettering read off the picture is kept |
| Unreadable-region marker | Part of the picture could not be read reliably and is left out instead of guessed |
| Incomplete-page marker | Part of that source page was not verified |

Traditional Chinese output uses the equivalent labels, such as 〔圖片內容：自動辨識，已核對〕.

## Processing time and limits

- **Verification takes time.** Each document has a default processing limit of 6 hours, drafting included; verification stops when it runs out. Example: a 97-page Word report full of pictures took about 2 hours to draft and 4.3 hours to verify on one DGX Spark with a local 27B vision model. To go faster, lower the limit or set `agentic.mode` to `off` when creating the run (see [the docs](backend/docs/agentic-processing.md)).
- **Not every page is guaranteed to pass.** In that report, 57 of 97 pages were fully verified; the rest were recorded as pending or not verified, never counted as passed — the same goes for pages the time limit did not reach.
- **Results depend on the model.** A wrong description of a picture (calling doughnuts cream puffs) may slip through; lettering and numbers on pictures face stricter checks.
- **OCR evidence is optional.** If PaddleOCR is available (`SEMARK_OCR_PYTHON_PATH` points at a Python with `paddleocr`), picture repairs use it as a second source of evidence; the default Docker image does not include it.
- **Screenshots can carry personal data.** Personal-information masking is on by default, but it is a last line of pattern matching; the reliable way is to blank personal content before uploading. See [docs/DEMOS.md](docs/DEMOS.md#screenshots-and-personal-information).

## Supported formats

PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, ODT/ODP/ODS, HTML, PNG/JPG. Output language can be `auto`, `zh-TW` or `en`; proper nouns can stay in English.

## Security

The service is meant for a local machine or a trusted network. The API has no authentication; do not expose it to the internet. Other settings are in [docs/INSTALL.md](docs/INSTALL.md).

## License

Semark's own source code is released under the Apache License 2.0; see `LICENSE`. Third-party components keep their own licenses: PyMuPDF/MuPDF is AGPL-3.0 or commercial, and MinerU, model weights and cloud providers have their own terms. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## More documentation

- [Installation, configuration and deployment](docs/INSTALL.md)
- [Demo comparisons](docs/DEMOS.md)
- [Page-by-page verification and repair](backend/docs/agentic-processing.md) (Traditional Chinese)
- [Release Notes 2026-08-12 (v0.3.1)](RELEASE_NOTES_20260812.md)
