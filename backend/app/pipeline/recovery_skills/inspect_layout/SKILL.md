# Inspect one source page layout

Use inspect_layout when region positions are unknown or a broad read missed visual detail.
The executor reuses source-bound MinerU output, or parses this one physical page once.
The result lists region IDs, types, page-normalized coordinates and optional confidence.
It is a read-only observation, never content verification or evidence of completeness.
An absent region is not proof that an image/table is absent. Truncation and missing
geometry remain explicit. Matched native Word pictures are also listed with provider=word_native,
separate from provider=mineru; their original pixels supply details the detector may have missed.

After inspection, choose up to three returned region_ids with read_regions to magnify
those areas while still reading the whole page. Never invent IDs or supply file paths.
The program rejects stale, foreign-page and duplicate IDs. Do not repeat inspection
for the same source page; reuse its observation and choose the next repair method.

For an inspected Word image, repair_image accepts its single region_id or mapped block_id. Prefer this scoped patch for image-only omissions; existing tables and other blocks remain intact.

Admission includes an affordable follow-up repair, its next plan and affected-page
source/delivery audits. Do not collect observations when no complete permitted
repair fits the remaining document budget. A query never consumes a repair slot.
