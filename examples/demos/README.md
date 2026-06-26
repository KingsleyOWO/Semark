# Curated Output Demos

This directory contains curated output snapshots from successful accurate-profile runs.
Each demo keeps a small source-page image beside the generated RAG-ready Markdown so
readers can compare the visual input with the semantic output.

## Demos

- `zh-flowchart-01/`: Traditional Chinese flowchart demo. Shows a one-page process
  diagram converted into concise semantic Markdown and chunk JSONL for RAG ingestion.

## Artifact Layout

Each demo directory may include:

- `source-page.png`: rendered source page used for visual comparison.
- `output.md`: final generated semantic Markdown.
- `chunks.jsonl`: generated chunks intended for retrieval ingestion.
- `quality_gate.json`: quality gate status and repair metadata for the run.

These snapshots are examples of model-assisted output, not golden legal or compliance
interpretations of the source documents.
