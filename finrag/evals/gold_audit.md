# Audit of the submitted 30-question gold set

## Result

The question design was strong and covered tables, narrative risk, transcripts, calculations and multi-document comparisons. It was not ready to use as evaluation ground truth without correction. The supplied records contained duplicated IDs, noncanonical filenames, combined source metadata and several factual or attribution errors.

## Corrections applied

- Replaced duplicate numeric IDs with unique IDs: `TSLA_001` through `TSLA_010`, `MSFT_001` through `MSFT_010`, and `JPM_001` through `JPM_010`.
- Normalized all sources to the project's filename contract.
- Recorded physical PDF page numbers, matching the application's `pdf_page` metadata rather than printed filing page labels.
- Split every cross-source reference into separately attributed contexts.
- Corrected Microsoft fiscal 2024 segment revenue to $77,728 million, $105,362 million and $62,032 million.
- Corrected Microsoft's fiscal 2024 provision for income taxes to $19,651 million.
- Corrected Microsoft's fiscal 2024 cash additions to property and equipment to $44,477 million and kept that full-year cash measure separate from Q4 capex including finance leases.
- Attributed Azure growth guidance to Amy Hood and corrected her year-over-year headcount statement to 3%.
- Reworded Tesla transcript records to remove unsupported claims about regulatory approval, transcript confirmation of 31.4 GWh and 40 GWh factory capacities, and a Cybercab 2026 volume-production commitment.
- Preserved JPMorgan Chase's label `total net revenue` and added a comparability warning when it is shown beside non-bank revenue.

## Important remaining check

Five local sources were directly checked in this workspace. `TSLA_2024_TRANSCRIPT.html` was not present here, so its corrected transcript claims were cross-checked against a public Q4 2024 call transcript. Before treating the set as final manager-approved ground truth, add the exact Tesla transcript used by the application and manually confirm `TSLA_006` through `TSLA_009` against that canonical file.

RAGAS scores should supplement this source audit, not replace it. The reference answer is itself part of the measurement instrument; an incorrect reference can make a correct chatbot answer look wrong.
