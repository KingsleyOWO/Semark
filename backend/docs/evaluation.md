# Evaluation Harness

This project now has a reproducible evaluation layer around the existing
pipeline artifacts. It does not require MinerU, Docling, PaddleOCR, olmOCR, or
Ollama to be installed before it can inspect existing runs.

## Run Baseline Metrics

From `backend/`:

```powershell
C:\Users\d35428\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m app.eval.runner --workspace workspace all
```

Outputs are written to:

```text
workspace/eval/reports/baseline_metrics.json
workspace/eval/reports/baseline_metrics.md
workspace/eval/reports/parser_candidate_matrix.json
workspace/eval/reports/parser_candidate_matrix.md
workspace/eval/reports/environment_inventory.json
workspace/eval/reports/retrieval_smoke.json
workspace/eval/reports/retrieval_smoke.md
```

The baseline report checks the existing `dataset.md`, `rag.md`,
`assets_index.jsonl`, `chunks.jsonl`, `quality.json`, `source_map.json`, and
`manifest.json` files. It records output sizes, block and asset counts,
enrichment counts, schema issues, source-map anchors, and retrieval asset tokens.

## Candidate Matrix

The parser matrix tracks the candidates from the upgrade plan:

- MinerU 3.x pipeline
- MinerU 3.x VLM / hybrid HTTP client
- Docling standard / VLM pipeline
- PaddleOCR PP-StructureV3 / PaddleOCR-VL
- olmOCR
- Marker as a reference-only baseline because of GPL and model license concerns

Availability is a probe, not an installation step. A parser is marked available
when the Python package or command appears in the active environment.

## Corpus

`benchmarks/corpus.public.json` contains two local backup-derived samples and
three public sample targets. Do not add private documents to this file. If a
private corpus is needed later, put it in a separate ignored manifest and record
only aggregate metrics in public reports.

`benchmarks/retrieval_queries.jsonl` contains starter retrieval checks. The next
step after the lexical smoke runner is to add an embedding-backed retrieval
runner with the production embedding stack, then score top-k chunk and asset
recall.

## Acceptance Gate

A parser/model candidate should not replace the current MinerU path unless it
improves at least one measured failure mode without breaking these contracts:

- `DocumentIR` keeps block ids, page ids, normalized bboxes, payloads, and source
  provenance.
- `assets_index.jsonl` keeps `retrieval_text`, `asset_path`, `block_id`, and
  evidence fields sufficient to return the original visual object.
- `source_map.json` keeps markdown-to-block traceability for the viewer.
- `manifest.json` records parser backend, version, VLM model, endpoint mode, and
  decoding parameters.
