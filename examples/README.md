# Demo Corpus

This directory contains synthetic public samples for exercising the document parser without publishing private or customer data.

## Included Synthetic Samples

- `samples/synthetic_invoice.html`: English invoice-like metadata and line-item table.
- `samples/synthetic_form.html`: English form-like fields and approval checklist.
- `samples/synthetic_process_brief.html`: English headings, step blocks, and responsibility table.
- `samples/synthetic_zh_purchase_request.html`: Traditional Chinese purchase request with structured fields, line items, and review notes.
- `samples/synthetic_zh_meeting_minutes.html`: Traditional Chinese meeting minutes with decisions and action items.

These files are intentionally simple HTML documents so they can be reviewed directly and converted by browser print-to-PDF if a PDF demo is needed.

## Local Synthetic Demo

Start the backend and frontend with local setup or Docker Compose, then run:

```bash
scripts/run_demo_corpus.sh --profile fast
```

This uploads all files in `examples/samples/` and creates pipeline runs through the backend API. Use `--wait` when you want the script to poll until runs finish:

```bash
scripts/run_demo_corpus.sh --profile fast --wait
```

Use `--profile accurate` when you want to demonstrate any configured VLM enrichment in addition to parsing.

## Optional Public Corpus

The repository does not commit third-party PDFs or filings. To download local-only public test files:

```bash
SEC_USER_AGENT="Your Name your.email@example.com" scripts/fetch_demo_corpus.sh
```

The script writes files into `workspace/demo-corpus/`, which is ignored by Git. It currently downloads:

- IRS Form W-9 PDF from the official IRS PDF URL.
- The latest Apple 10-K HTML resolved from the official SEC EDGAR submissions API.

Then include those files in a demo run:

```bash
scripts/run_demo_corpus.sh --include-public --profile fast --wait
```

## Recommended Demo Story

1. Run `fast` on the synthetic samples to prove clone/setup/upload/MinerU parsing/output works.
2. Open Viewer for the Traditional Chinese purchase request and show structured tables, Markdown output, source maps, and assets.
3. Configure VLM if available, rerun one form-like sample with `accurate`, and show semantic enrichment output.
4. Download the optional public corpus and run one long SEC filing to demonstrate long-document handling.
5. Keep external downloaded files out of commits and release archives.
