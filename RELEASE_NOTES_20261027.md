# Release Notes 2026-10-27 (v0.4.0)

**English** | [繁體中文](RELEASE_NOTES_20261027.zh-TW.md)

This release is about checking every page against the source before delivery. v0.3 could turn
tables, flowcharts and screenshots into semantic Markdown, but it could not say which parts to
trust: output could be well-formed and pass every quality check while misreading the lettering
on a picture or missing a paragraph of the source. In v0.4.0 every delivery is verified page by
page against the source; what can be fixed is fixed, and what cannot is labelled as such.

Previous release: [Release Notes 2026-08-12 (v0.3.1)](RELEASE_NOTES_20260812.md).

## Highlights

### Page-by-page verification against the source (new)

- **Every page is on the checklist**, including pages the parser produced nothing for. A model
  compares each source page image with the documents and chunks actually delivered, and records
  the page as verified, pending or not verified. An unclear source mapping, an oversized input,
  an invalid model answer or running out of time never counts as a pass.
- **Problems found are repaired automatically.** For one problem at a time, the model picks a
  tool: inspect the layout, read the original picture or the PDF's native text, re-read an
  enlarged region, make a precise edit to one picture, re-read a whole picture, restore missing
  content, or re-parse a single page. A tool's result is only a candidate: it must pass another
  check against the source before it is taken, and the documents and chunks are then rebuilt and
  checked again.
- **Published only after final validation.** Source coverage, verification records and the
  delivered content must agree before a result is published; an interrupted run resumes from its
  checkpoints.
- **One time limit per document: 6 hours**, parsing and drafting included. Work follows a plan
  within that time (at most three different methods per problem, all problems in turn) instead
  of retrying until time runs out. Adjust it with `config_overrides` when creating a run, or set
  `agentic.mode` to `off` to switch verification off.
- **Native Word pages need no model time**: pages with only text and tables are checked cell by
  cell for position and order against the text layer of the PDF converted from the Word file.

### Pictures: say when it cannot be read

- **The reading copy shows verification status.** `delivery.md` labels each picture verified or
  not verified; the `documents/` and chunks you import into RAG stay clean and carry the status
  as metadata (`source_verified`, `unverified_block_ids`).
- **Pictures that could not be verified keep only the lettering read off them**, not the
  model's own descriptions; unreadable regions are marked as unreadable instead of guessed.
- **Pictures are enlarged before reading.** Small pictures are scaled to about 3 megapixels
  before the vision model sees them; illegible characters are written as `[?]`.
- **More careful corrections.** A repair may not delete lettering already read; changing a word
  or number on a picture needs support from OCR or another reading; Cyrillic or Bopomofo letters
  that look like Latin ones are not treated as misspellings.
- **OCR is evidence, not the answer.** When PaddleOCR is available, re-reading a picture uses it
  as a second source of evidence, but an OCR line is delivered only when a model reads the same
  words (in testing, lines only OCR read were almost all misspelt).
- **A word or number is not read inside a longer one.** Re-reading a picture no longer drops
  "Active" as part of "Attractive", nor treats a count "×5" as already read in "15".
- Also fixed: pictures with transparent backgrounds, EMF/WMF vector pictures in Word, reviewing
  picture-dense pages in batches, and pictures delivered twice.

### Word documents

- Merged cells (rowspan/colspan), paragraphs and tables inside content controls (SDT), tables
  inside tables, and footnotes are kept.
- Text inserted with tracked changes is no longer lost: all revisions are accepted before
  parsing and the final text is delivered (previously whole inserted passages, and even one
  digit inserted into an amount, went missing).
- Long Word reports are no longer classified as a one-page form just because they mention a
  checklist or consent form, so a report's text is no longer replaced by a single picture.
- Word page numbers are mapped to pages of the converted PDF, so verification and repairs find
  the right source page.

### Delivered content

- Source tables in English documents are written in English (chunks used to read 「第 1 列：第 1 欄」, and
  main.md gained a Chinese-labelled copy of the table).
- When the delivery lacks part of the source, the "Additional Source Content" section at the end of
  main.md adds only the missing lines instead of repeating the whole block.
- Form guides no longer give a form sections it does not print (such as a default approval section).
- Short list items (a flowchart's role "- 被害人") are no longer deleted as OCR noise; a flowchart path
  is no longer taken as the document title.
- Latin form cues in filenames and page text must be whole words: "WebForm" in a filename or
  "authorized" in a notice no longer makes a user guide a form.

### PDF

- Headers and footers printed only once go back into the body; table rows lost at a page break
  in tables spanning pages are restored from the text layer.

### Batches and deployment

- A run can name another machine serving the same model with `vlm_base_url`, to split a batch
  across two machines.
- New optional settings `SEMARK_OCR_PYTHON_PATH` and `SEMARK_OCR_MODEL_CACHE` point to a Python
  with PaddleOCR installed and its model cache; the default Docker image does not include
  PaddleOCR.

### Documentation and demos

- The README was rewritten: it opens with what sets Semark apart, explains the output labels,
  and covers processing time and limits; installation and deployment details moved to
  `docs/INSTALL*.md` and the demos to `docs/DEMOS*.md`.
- The Treasury payment-slip screenshot guide demo was regenerated with v0.4.0 and ships the reading
  copy `delivery.md` with its verification labels; rerun with v0.4.0, the other three demos came
  out worse than before (see known limitations), so their v0.3 output stays, marked as such.
- `backend/docs/agentic-processing.md` records every decision in the verification flow, version
  by version, with how it was validated (Traditional Chinese).

## Compatibility notes

- **Processing takes much longer.** Verification is on by default for new runs; a long,
  picture-heavy document can take hours. Measured: a 97-page Word report full of pictures took
  about 6.5 hours on one DGX Spark with a local 27B vision model; 57 of its 97 pages were fully
  verified and the rest were recorded as pending or not verified. For the old speed, switch
  verification off (`agentic.mode: off`).
- **Existing runs are not re-run automatically**; old results show "no source verification
  record" in the UI.
- **Parse and draft caches are invalidated once** (the parser version and picture enlargement
  setting are part of the cache keys), so the first run of a document after upgrading parses and
  reads its pictures again.
- When running several jobs at once, make sure your model server can serve requests in
  parallel; local Ollama serving one model usually handles one request at a time.

## Known limitations

- A wrong description of a picture (calling doughnuts cream puffs) may slip through; lettering
  and numbers on pictures face stricter checks.
- A picture whose contents depend on each other, such as a table, and that has many unreadable
  regions is not delivered in part; the whole picture stays "not verified".
- Verification quality depends on the vision model; smaller models more often leave small
  print, handwriting and low-resolution screenshots unverified.
- A flowchart's branch structure can be misread and still pass verification: rerunning the
  flowchart demo put two branches under the wrong parent while the page was marked verified; the
  previous run of the same picture was right.
- With verification on, no whole-document rewrite runs, so forms and screenshot pages are laid out
  plainly rather than reorganised as in the v0.3 demos, and steps or OCR typos the parser makes into
  headings stay. A user guide that explains how to fill in a form may still be treated as a form.
- Filenames whose words run together (LeaveApplicationForm) no longer make a document a form by
  name alone; page content decides.
