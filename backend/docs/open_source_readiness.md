# Open-Source Readiness Checklist

Use this checklist before publishing the project.

## Repository Hygiene

- Remove private files from `workspace/store/docs` before publishing.
- Keep only public sample manifests and small synthetic fixtures in git.
- Add `.gitignore` rules for run outputs, model caches, local corpora, API keys,
  and generated benchmark reports.
- Replace local absolute paths in screenshots, manifests, and reports.

## Licensing

- Default parser path should use permissive or clearly documented licensing.
- Treat Marker as reference-only unless its GPL and model terms are acceptable
  for the target release.
- Keep a license matrix for MinerU, Docling, PaddleOCR, olmOCR, Qwen models,
  RAG-Anything, RAGFlow, and Chunkr.
- Document whether each model is suitable for personal, research, startup,
  commercial, or enterprise use.

## Product Positioning

Do not position the project as if multimodal document RAG does not exist.
RAG-Anything, RAGFlow, Chunkr, Docling, MinerU, PaddleOCR, and olmOCR all cover
parts of this space.

The sharper positioning is:

```text
Parser-agnostic document packaging for RAG: semantic text, visual asset recall,
source-map traceability, A/B run comparison, and benchmarkable outputs.
```

## Release Gate

- `cd backend && .venv/bin/python -m pytest` passes.
- `cd backend && .venv/bin/python -m ruff check app tests` passes.
- `cd frontend && npm run lint` has no errors.
- `cd frontend && npm run build` succeeds.
- API smoke checks pass for `/api/health`, upload, document listing, and cleanup.
- README includes quickstart, security boundary, architecture, sample output, and known limitations.
- Docker or setup docs describe CPU-only baseline and optional GPU/VLM paths.
- A small public demo corpus is available and reproducible.
- Root `LICENSE` is chosen and committed before GitHub publication.
- Dependency/model license matrix is reviewed for MinerU, PyMuPDF, VLM models, and optional parser baselines.
- No default command sends user documents to a cloud model.

