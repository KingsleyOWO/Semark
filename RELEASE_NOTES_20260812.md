# Release Notes 2026-08-12 (v0.3.1)

**English** | [繁體中文](RELEASE_NOTES_20260812.zh-TW.md)

This is a delivery-correctness release. v0.3.0 shipped the screenshot pipeline; running a
corpus of research reports through it surfaced a class of defect the quality gate could not
see — output that is well-formed, passes every check, and is missing or garbling exactly the
text a retrieval query lands on. Every fix below was measured against that corpus rather than
against the few documents that first showed the symptom. (The corpus grew from 100 to 167
documents during the round, which is why the denominators below differ.)

Previous release: [Release Notes 2026-07-17 (v0.3.0)](RELEASE_NOTES_20260717.md).

## Highlights

### Body text, not the page it was printed on

- Page furniture (running heads, folios, volume lines) had a detector and no production
  caller: 1,826 lines were delivered as body paragraphs, inside 851 of 1,843 chunks. All three
  delivery surfaces now skip them, and short text that recurs in the same margin band on two
  or more pages is additionally tagged as furniture.
- The supplement pass — which re-reads each page and fills in what no block covers — was
  duplicating and reordering the prose it was meant to complete. Coverage is now directional,
  spans one page either side, ignores markdown escaping and whitespace, and falls back to bbox
  containment for text no comparison can match. Supplements are spliced into the parser's
  reading order per page instead of the whole document being re-sorted by vertical position;
  one supplement had been enough to interleave the columns of a two-column layout.
- Documents whose cover title is set vertically delivered a garbled `H1` — OCR of vertical
  type drops and swaps characters — while a clean copy of the same title sat unused in the IR
  as a horizontal running head. 49 of 167 documents were affected. The running head is now
  authoritative for characters but not for which segments a title has, so a cover kicker the
  running head omits is kept, and a series label it adds is refused.

### Tables

- **Multi-row headers.** A one-row-header assumption sent a table's sub-column names to a
  fallback (`欄位4: 22.40`, with nothing to say it is a share of imports) and shipped the
  sub-column row as the first data record. Header rows are now folded behind four guards, each
  with a live counter-example. Placeholder column names: 926 → 8.
- **Table titles.** Where the parser gave no caption (52 of 128 tables), the title came from a
  document-level string picked by substring match, truncated to 100 characters. The nearest
  caption printed above the table is now used; 46 of the 52 recover their printed name.
- **Numeric tables were being discarded as OCR noise.** Any purely numeric cell counted as
  "weak", so a wide statistics table was structurally guaranteed to read as garbage and have
  every data row dropped — 7 tables in 4 reports, whose absence then tripped the
  empty-output check, the VLM audit, and a reviewer pass of roughly 100k tokens. Well-formed
  figures are now read as data; lone digits and mash such as `4.64.74.7` still count as noise.
  11 of 11 tables recovered.
- **Attribution rows.** The 注／資料來源 line printed inside a table's own border looks exactly
  like a data row with only its first column filled, so it was emitted as a record whose
  heading is a sentence and whose one field states something the table never said. It is now
  recognised by its wording — narrowly, against the openings papers actually use — and
  rendered as a trailing line after the records. A row that does not announce itself stays
  data, because a row whose cell boundaries the parser lost is full-width too and the two are
  indistinguishable by shape.
- **Collapsed cell boundaries are now repaired.** New capability; see Compatibility Notes.
- **Table titles in `chunks.jsonl`.** The parser hands captions back as lists of strings; 76
  table titles shipped as the Python list repr, and `chunks.jsonl` is what retrieval reads.
  Captions are now flattened once, where the payload is built. The same shape had been making
  the figure caption-length gate fire on every properly captioned figure, so the model was
  re-captioning figures that already had a caption printed beside them.

### Traditional Chinese output

- The simplified-content detector converted with `s2t` and compared against the source, but
  that mapping treats 台/布/占/干/了 — every one of them standard zh-TW — as simplified. Ordinary
  Taiwanese prose therefore read as simplified, and was then rewritten by `s2twp`'s phrase
  table (公布→公佈, 關鍵零組件→關鍵零元件, 數據→資料, institution names 台→臺). Ambiguous variant
  pairs are folded before detection, conversion uses `s2tw`, and the author's own variant
  choice is restored positionally afterwards.
- Conversion never saw the parser's own text: its single call site was the VLM render path, so
  849 unambiguously simplified characters reached the output across 88 of 100 documents. It
  now runs as a normalize post-pass over every text-bearing payload field, whatever produced
  the block, with a kana guard applied line by line so a bibliography's Japanese entries are
  not "repaired".
- Two characters the detector could not see at all (凈, 説 — `s2t` maps them to themselves)
  passed byte-identical and never reached conversion: 51 and 25 occurrences.

### VLM robustness

- **A model the endpoint does not serve now reports unavailable.** The availability check
  returned true whenever the HTTP endpoint answered — including the branch that had just
  reported the configured model missing — so a model that was never pulled, or a typo in the
  name, did not stop the stage once with something a human could act on; it let every
  enrichment request go out and fail one at a time. An endpoint that enumerates no models
  stays permissive (some gateways return an empty list), and Ollama's implicit `:latest` tag
  is matched so a working bare-name configuration is not called broken.
- **Crops below the vision model's 32px patch factor are never sent.** Four crops of 32x31,
  34x31 and 37x31px caused 8 model-runner panics and 8 lost enrichments; the server reports
  this to the client as a generic resource-limit error that reads like an out-of-memory
  condition and is not one.

### API, operations, security

- **Named-host access is now opt-in.** The frontend dev server ran with all hosts allowed on
  `0.0.0.0`, so any website could DNS-rebind to it and reach the unauthenticated backend
  through the `/api` proxy. See `SEMARK_FRONTEND_ALLOWED_HOSTS` below; `localhost` and direct
  IP access are unchanged.
- **The privacy scrub now runs at the write boundary.** It had been applied at scattered call
  sites, leaving four delivered surfaces unmasked: `dataset.md`, the structured-repair
  main-document re-export, `chunks.jsonl` metadata, and the VLM-audit excerpts in
  `quality_gate.json` / `llm_vlm_outputs.md`.
- **Delivered writes are atomic.** No delivered artifact was written atomically, and split
  documents were deleted before their replacements were written, so a crash mid-export left
  deleted files still referenced by the old index. Stale split documents are now removed only
  after the new set lands.
- **Download lists past 500 runs.** With the default filter, the outputs-summary endpoint only
  ever inspected the newest 500 runs and reported the count within that window as the total,
  so installs past that point silently lost older documents from download lists. List queries
  also gained a stable tiebreaker — `created_at` has second resolution, so batch-created rows
  tied and could skip or duplicate rows across pages.
- **Enrich cache identity is parse-aware.** Keys folded in neither the parse configuration nor
  the parser version, so re-parsing under different settings could serve a caption belonging
  to a different figure, and switching the output language served captions in the old
  language. Previously cached rows re-enrich once.
- **Pipeline concurrency.** One module-level semaphore is now shared by the queue workers and
  the direct `background=false` path, which previously ran unbounded concurrent parse+VLM
  pipelines. The startup sweep also cancels PENDING runs orphaned by a restart (the queue is
  in-memory), and the task queue is stopped before the database disconnects so cancellation
  handlers can still write run status during shutdown.
- **Viewer.** The bbox overlay was vertically compressed on every page — on A4 portrait a
  page-bottom block drew at 71% height, so clicking a region selected the wrong block. Tables
  that the package stage deliberately keeps as raw HTML now render in the main-text pane
  instead of being silently dropped, through `rehype-raw` paired with `rehype-sanitize`
  (scripts, event handlers and inline styles stay stripped). Relative image paths resolve
  through the asset endpoint instead of 404ing against the SPA route.
- **The health endpoint reports the real application version** instead of a hardcoded `0.1.0`.

### Content that is not content

- A promotional insert — advertisements with prices, order lines and QR codes — occupies the
  right column of the last page in 32 documents while the left column still carries the
  reference list, so this had to be a column-level judgement: dropping the page would have
  deleted the references. The column boundary is derived from the page's own gutter rather
  than a hard-coded coordinate. It also keeps 105 covers and QR codes out of the model.
- A twenty-page report's bibliography was being scored with thresholds calibrated for a
  one-page form: `DOI:` and `https:` entries plus colon-terminated citation titles read as
  field labels, and 申請單 matched inside 申請單位 ("applicant unit"). 5 of 100 research reports
  were flagged as fillable forms, each raising an empty-structured-output issue and its
  downstream audit.
- Figure sections written by the model shipped as `##` headings at the same level as the
  article's own, naming no figure. They are now bold subsections under a single container
  heading. 311 spurious headings removed.

### Documentation and demos

- `examples/demos/README.md` gained the two screenshot demos added in v0.3.0 and the
  `raw-parse.md` / `figure-example.md` artifacts it never listed, and now has a Traditional
  Chinese counterpart.
- Both READMEs document collapsed-table repair, and three details that existed in only one
  language were synced.
- `BACKLOG.md` records the known limitations the architecture review surfaced and this round
  deliberately did not change.

## Compatibility Notes

- **New capability, on by default:** `vlm_repair_collapsed_tables` (a `PipelineConfig` field,
  no environment variable) re-reads tables whose cell boundaries the parser collapsed into a
  single merged cell — 26 of 128 tables in the reference corpus — and rebuilds the grid from
  the table image. It is independent of `vlm_enrich_tables`, which buys summaries for tables
  that parsed cleanly. The original parse is kept alongside as `table_body_mineru`; a ragged
  grid, or a transcription holding less than 60% of the printed characters, is rejected and
  the original stands. Set it to `false` to keep the previous behavior.
- **New variable:** `SEMARK_FRONTEND_ALLOWED_HOSTS` (comma-separated). Reaching the frontend
  through a named host now requires listing it. Access by `localhost` or by IP is unaffected.
- **Enrich cache keys changed.** Rows cached before this release re-enrich once.
- **Caption shape in the stored IR.** Documents parsed before this release keep list-shaped
  captions in their saved `document_ir.json`. A read-side guard covers them, so titles come
  out right either way; only a re-parse puts flat captions in the file.
- **The health endpoint's version string changes** from `0.1.0` to the real application
  version.
- `template_section_labels` added to `default.json` does not take effect in deployments that
  supply their own corpus ruleset, because corpus rules replace rather than merge.
- Backend test suite: 437 → 718 tests green.

## Verification and Caveats

- The pipeline fixes were verified by replaying all 167 stored document IRs offline, plus
  end-to-end runs on three documents. **A full live re-run of the whole corpus on the rebuilt
  image has not been done.**
- Where a threshold appears above — 0.60 coverage, 40-character header cells, the 60%
  short-read floor, the 106-character whole-table floor — it is a measurement of this corpus,
  not a principle. A corpus whose tables are annotated differently needs them re-measured.
- The supplement-coverage calibration is deliberately asymmetric: it prefers a repeated phrase
  over a line that exists in one copy and is gone. Two whole-table duplicates that fall below
  the length floor come back as a result of that choice.
- Because collapsed-table repair treats the image as the authority, a table the model misreads
  is replaced by a wrong grid rather than a garbled one. The fail-closed guards are what stands
  between those two outcomes.
- Known limitations that remain: image footnotes are still outside the supplement coverage
  text (the same bug class, not yet measured); drop caps still lose their first character at
  the parser level; and a Chinese paragraph citing Japanese inline keeps its simplified glyph
  (measured at one character corpus-wide — character-level splitting risks corrupting the
  Japanese).
