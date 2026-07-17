# Backlog / Known Limitations

Findings from the 2026-07 architecture review that are acknowledged but not yet fixed.
Ordered roughly by impact. Contributions welcome.

## Operations

- **Disk retention & real deletion.** Deleting a run currently *hides* it (documents stay
  available); there is no artifact GC, no parse-cache pruning, and no disk-usage surfacing
  in the UI. Long-running installs grow by tens of MB per document. Planned: true run
  deletion (DB rows + run directory), parse-cache LRU per document, and a disk-stats endpoint.
- **Production frontend still runs the Vite dev server.** Move to a static production build
  behind a small reverse proxy (nginx) that also fronts `/api`.
- **Per-stage concurrency limits are declared but not enforced.** `enrich_gpu`/`enrich_http`
  in the task-queue pool config are currently informational; only the whole-pipeline cap is
  enforced. Planned: per-stage semaphores so GPU enrichment is serialized independently.
- **SQLite hardening.** Enable WAL + `busy_timeout`, and group multi-statement writes
  (run + stage-row creation) into explicit transactions.
- **Restart semantics for queued runs.** Queued-but-unstarted runs are canceled at startup
  (the queue is in-memory). A re-enqueue-on-boot policy could resume them instead.
- **Compose resource guidance.** No memory/CPU limits or GPU passthrough in the compose
  files; MinerU + a local LLM can exhaust host RAM. Document minimum RAM and add an
  opt-in `gpus: all` variant.
- **`outputs-summary` scans run directories per request.** Correct but O(runs); persist a
  "has documents" flag in the DB for indexed paging.

## API robustness

- Offload blocking work in async handlers (`md_to_docx`, ZIP assembly, upload writes,
  `copytree`/`rmtree`) to threads; stream large batch ZIPs instead of building them in RAM.
- Enforce an upload size limit and hash uploads in chunks.
- Guard `documents_index.json`/JSONL reads against corrupt files (return 404/422, not 500).
- Replace raw `str(e)` in batch-operation error payloads with generic messages (log details
  server-side).
- Cancel in-flight runs when their document is deleted; reconcile orphan directories at boot.

## Pipeline quality

- **`package.py` decomposition.** ~5.6k lines spanning nine responsibilities; the
  `source_md` / `delivered_md` aliasing in `run()` has already produced delivery bugs.
  Split into render / repair / export / privacy modules.
- Preserve pre-repair VLM audit evidence after a semantic-repair settle (currently the
  audit trail is replaced by the post-repair recheck).
- Emit a quality-gate warning when an enriched image block exports no asset (missing cached
  image file currently drops captions silently).
- Scope repeated-line dedup in split-main cleanup per heading section; exempt ordered-list
  markers.
- Make zh-TW conversion (s2twp, vocabulary mapping) and privacy scrubbing fence-aware so
  code blocks and quoted originals are untouched.
- Title inference: prefer the document's own H1 unless it scores as unreliable.

## Frontend polish

- Surface mutation failures (deletes/saves currently fail silently — add `onError` toasts)
  and guard double-clicks on per-row destructive buttons.
- Cross-page "select all" beyond 500 items hits the API limit and fails silently.
- Stop the 5s dashboard polling when no runs are active; batch per-row review-badge fetches.
- Settings: reset the VLM form's dirty state when switching roles/profiles (a save clicked
  mid-switch can write one role's form to the other role's config).
- Per-row profile selector on the dashboard displays a value the Run button ignores.
- Translate the handful of zh-TW strings remaining in the `en` locale table; localize
  `formatDate` by selected language.
- Viewer: anchor matching (click-to-source) fails for paragraphs containing formatting.
