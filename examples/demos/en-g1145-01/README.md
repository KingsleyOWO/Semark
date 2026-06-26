# English Form Demo: USCIS Form G-1145

This demo shows a one-page English government form processed with the accurate
profile. MinerU extracts OCR/layout evidence, then the configured VLM/reviewer
model rewrites the final output into RAG-ready semantic Markdown.

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

- Run ID: `01KW1AEKGKY236X2SBSYP9JJCG`
- Document ID: `f06c73937567793c`
- Profile: `accurate`
- Output language: `en`
- Quality gate: `pass`
- Auto RAG ready: `true`

## Source Attribution

The source page is rendered from USCIS Form G-1145, e-Notification of
Application/Petition Acceptance. See the official USCIS form page:
<https://www.uscis.gov/g-1145>

## Notes

The source document is a compact form, so the expected output is a single
semantic document rather than separate child files. The output is intended to
show the RAG-ready shape of generated Markdown, not to provide legal advice or
a substitute for the official form instructions.
