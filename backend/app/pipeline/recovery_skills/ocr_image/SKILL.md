# Read original image lettering
Use one mapped image ID and a bounded region at scale 2/4 where lettering is small.
repair_image already reads the whole picture with PaddleOCR; a whole-picture request becomes that repair.
The scale is lowered until the scaled region fits the pixel limit; do not repeat identical arguments.
All lines are observations with source IDs, positions, provider/version and source hashes.
A confidence score is not a verification result. Keep conflicting readings separate.
This tool cannot modify the document. repair_image compares these observations with original pixels.

Admission includes an affordable follow-up repair, its next plan and affected-page
source/delivery audits. Do not collect observations when no complete permitted
repair fits the remaining document budget. A query never consumes a repair slot.
