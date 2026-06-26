# Traditional Chinese Flowchart Demo

This demo shows a one-page Traditional Chinese flowchart processed with the accurate
profile. MinerU extracts the page structure and OCR evidence, then the configured
VLM/reviewer model rewrites the final output into RAG-ready semantic Markdown.

## Source Page

![Source page](source-page.png)

## Generated Output

- [output.md](output.md): final semantic Markdown.
- [chunks.jsonl](chunks.jsonl): retrieval chunks generated from the semantic output.
- [quality_gate.json](quality_gate.json): pass/fail status, issue summary, and repair metadata.

## Model Note

This snapshot was generated in the test environment with local Ollama model
`qwen3.6:35b-a3b-q8_0` as the enrichment/reviewer model. Stronger compatible
vision or reviewer models may improve visual reasoning and semantic repair quality.

## Run Metadata

- Run ID: `01KW0VM74QCTCYD2Y0RJBWJZ3B`
- Document ID: `af6f53bd9cda7d1c`
- Profile: `accurate`
- Output language: `zh-TW`
- Quality gate: `pass`
- Auto RAG ready: `true`

## Notes

The source document is a flowchart, so the expected output is a single semantic
document rather than separate child files. Independent figures, tables, or attachments
would be split only when they represent distinct retrievable units.
