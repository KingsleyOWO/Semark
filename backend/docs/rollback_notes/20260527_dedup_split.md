# 2026-05-27 Deduplicated Split Document Output

## Restore Point

Snapshot before the deduplication attempt:

- `backend/.rollback_snapshots/20260527_dedup_split/package.py.before_dedup_split`

To restore the previous package-stage behavior manually:

```bash
cp backend/.rollback_snapshots/20260527_dedup_split/package.py.before_dedup_split backend/app/pipeline/stages/package.py
```

## Change Intent

When forms are exported as standalone documents, the main document should keep only a short relationship reference to those forms. It should not keep the original MinerU/OCR table blocks from the same form pages, because that duplicates content across `main.md` and `form_*.md` and can confuse generic RAG systems.

## Implementation Notes

- Added page exclusion support to the RAG markdown renderer.
- When structured output is a `form_collection`, page indices from form records are excluded from the main document render.
- Form page detection accepts both `page_idx` and `page_indices`; newer structured form records use `page_indices`.
- Form documents remain exported as standalone Markdown files with parent source metadata.

## Expected Output

- `outputs/documents/main.md`: source regulation/main body plus a `關聯表單與附件` section.
- `outputs/documents/form_*.md`: standalone form documents with semantic filling guidance and field information.
- Form table raw extraction should no longer appear in `main.md` when the form has been split out.

## Verified Run

- Run: `01KSM9V40MST1JYKRBJ9984NXZ`
- Output: `main.md` plus `form_0000.md` through `form_0003.md`.
- The main document keeps form references in `關聯表單與附件`, but no longer keeps raw form table blocks from pages 11, 12, 13, and 17.
- Pre-fix output backup for this run: `backend/workspace/store/docs/5a2a65cae02b001c/runs/01KSM9V40MST1JYKRBJ9984NXZ/outputs/.rollback_20260527_before_page_indices_fix/`
