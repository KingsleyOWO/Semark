# Read source-owned regions
Use read_regions for substantive table/text loss or unmapped visuals. For a mapped
missing picture use repair_image and preserve existing tables and neighboring text.
The program partitions the entire page, preferring complete PDF table rows and
matching native Word cell IDs. Tables remain full width, including merged cells.
Choose horizontal2 normally or horizontal4 for dense pages; grid4 is for non-table layouts.

Transcribe only the owned core and owned_rows. Preserve headings, exact wording,
row labels, numbers and units. Matched native_cells are source evidence; retain
their text and relationships. context_rows, crop overlap and the full-page image
provide context only. Do not repeat their content in another core.

If essential continuation/header context is missing, return status=needs_context,
complete=false and context_request=above/below/left/right/both/all. The program
can expand source crops within a fixed allowance; core ownership never changes.
For illegible source pixels return status=unreadable and context_request=none.
Never guess. For complete content use status=complete, complete=true and
context_request=none. Return table content as rows with supplied source_id, role and column/text cells, including blank cells. Never serialize tables as free Markdown in text. Put other source content in prose with page-coordinate bbox. observation is not delivered content.

After inspect_layout, up to three region_ids can provide source detail images.
They supplement full-page coverage and never certify correctness. Completed
readings are cached against source pixels, model and contract. The program binds every returned row and column to the source plan, rejects missing/duplicate/unowned rows and cells, and renders one table from the stored cell grid. Context headers are not returned again. Non-table regions retain prose. Format failures trigger a bounded split at complete source rows; ownership never changes.
All candidates still require source and actual delivery verification.
