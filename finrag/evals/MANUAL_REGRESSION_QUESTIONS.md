# FinRAG manual regression questions

Use this checklist to verify the complete application after retrieval, planning,
prompt, validation, or orchestration changes. It targets the `phase1-v2` index:
MSFT and TSLA, fiscal years 2024 and 2025, with one 10-K and one Q4 transcript
for every company-year.

## Test setup

1. Select `phase1-v2`.
2. Start each numbered question in a fresh conversation unless a test explicitly
   says otherwise.
3. Clear all UI company, year, and document-type filters. The question itself
   should determine scope.
4. Enable multi-hop execution for section 8:

   ```dotenv
   FINRAG_MULTIHOP_ENABLED=true
   ```

5. For every answered question, verify that every citation opens evidence from
   the requested company, source year, and document type.
6. Record the decision, latency, cited source IDs, pass/fail result, and notes in
   the results table at the end of this file.

## Universal pass criteria

An answered test passes only when:

- The resolved company, source year, and document type match the question.
- Every material factual or numerical claim has a valid citation.
- No cited source falls outside the requested scope.
- Values preserve their row label, period, currency, scale, and sign.
- A comparison contains evidence for every requested company/year group.
- A calculation shows or clearly identifies its validated inputs and operation.
- Transcript guidance or opinion is not presented as an audited historical fact.
- Missing or ambiguous scope produces a safe decision instead of a guessed answer.

## 1. Simple 10-K summary

### M01 — Risk summary

> Summarize Microsoft's principal risk factors from its 2024 10-K in five concise bullets.

Expected:

- Decision: `answered`.
- Scope: MSFT / 2024 / 10-K.
- Evidence should come primarily from Item 1A.
- Each bullet should be traceable to cited filing evidence.

### M02 — AI infrastructure risks

> According to Tesla's 2025 10-K, summarize the AI-related resource requirements and supply-chain challenges described in Item 1A.

Expected:

- Decision: `answered`.
- Scope: TSLA / 2025 / 10-K.
- The response should distinguish actual disclosed dependencies from speculation.

### M03 — Business-segment summary

> According to Microsoft's 2024 10-K, summarize how the More Personal Computing segment is described in Note 19.

Expected:

- Decision: `answered`.
- Scope: MSFT / 2024 / 10-K.
- The answer should stay within the requested segment and cited note.

## 2. Transcript questions

### M04 — Margin commentary

> What did Tesla management say about automotive margins in its 2025 earnings transcript?

Expected:

- Decision: `answered`.
- Scope: TSLA / 2025 / transcript.
- Statements should be attributed only when speaker evidence supports attribution.

### M05 — Cloud and AI commentary

> What did Microsoft management say about cloud and AI demand in its 2025 earnings transcript?

Expected:

- Decision: `answered`.
- Scope: MSFT / 2025 / transcript.
- The answer should not silently import claims from the 10-K.

### M06 — Year-specific transcript comparison

> Compare Tesla management's automotive-margin commentary in its 2024 and 2025 earnings transcripts.

Expected:

- Decision: `answered`.
- Required evidence groups: TSLA / 2024 / transcript and TSLA / 2025 / transcript.
- Both years must be cited; the answer should identify changes without inventing causality.

## 3. Exact table values

### M07 — Microsoft revenue

> According to Microsoft's 2025 10-K, what was total revenue for the fiscal year ended June 30, 2025?

Expected:

- Decision: `answered`.
- Scope: MSFT / 2025 / 10-K.
- Expected value: USD 281,724 million.
- Confirm the answer uses the total-revenue row and 2025 column.

### M08 — Microsoft assets

> According to Microsoft's 2025 10-K, what were total assets as of June 30, 2025?

Expected:

- Decision: `answered`.
- Scope: MSFT / 2025 / 10-K.
- Expected value: USD 619,003 million.
- Confirm this is a balance-sheet date, not an annual flow period.

### M09 — Tesla automotive revenue

> According to Tesla's 2024 10-K, what were total automotive revenues for the year ended December 31, 2024?

Expected:

- Decision: `answered`.
- Scope: TSLA / 2024 / 10-K.
- Expected value: USD 77,070 million.
- The result must not substitute total company revenue or the 2023 comparison column.

## 4. Complete financial tables

### M10 — Microsoft balance sheet

> Give me the exact complete balance sheet table from Microsoft's 2025 10-K.

Expected:

- Decision: `answered`.
- Retrieval mode should be `exact_table` when a matching complete structure exists.
- Preserve all available rows, columns, units, and source order.
- The table should include total assets of 619,003 for 2025.

### M11 — Tesla income statement

> Show the complete consolidated statements of operations table from Tesla's 2025 10-K.

Expected:

- Decision: `answered`.
- Return the operations/income statement, not a similarly named comprehensive-income table.
- Preserve the 2025 and comparative columns and the stated unit scale.

### M12 — Tesla revenue table

> Show the complete revenue-by-source table from Tesla's 2024 10-K.

Expected:

- Decision: `answered` if that named table exists in the indexed structure.
- Return the specifically requested table rather than unrelated revenue excerpts.
- Do not truncate rows merely because ranked retrieval selected only part of the table.

## 5. Same-company, cross-year comparisons

### M13 — Microsoft revenue

> Compare Microsoft's total revenue in its 2024 and 2025 10-K filings in a table, and state the dollar increase.

Expected:

- Decision: `answered`.
- Required groups: MSFT / 2024 / 10-K and MSFT / 2025 / 10-K.
- Both filing years and values must be cited.
- Do not confuse a comparative 2024 column with the 2024 source filing.

### M14 — Tesla cash

> Compare Tesla's cash and cash equivalents at its 2024 and 2025 fiscal year ends according to the respective 10-K filings.

Expected:

- Decision: `answered`.
- Required groups: TSLA / 2024 / 10-K and TSLA / 2025 / 10-K.
- Preserve each balance-sheet date and unit scale.

### M15 — Microsoft current liabilities

> Compare Microsoft's total current liabilities as of June 30, 2024 and June 30, 2025 using the respective 10-K filings.

Expected:

- Decision: `answered`.
- Required groups: MSFT / 2024 / 10-K and MSFT / 2025 / 10-K.
- The answer should use the exact total-current-liabilities row for both years.

## 6. Cross-company comparisons

### M16 — Net income

> Compare Microsoft and Tesla net income for fiscal 2025 according to their 2025 10-K filings. State both values and the difference.

Expected:

- Decision: `answered`.
- Required groups: MSFT / 2025 / 10-K and TSLA / 2025 / 10-K.
- Both inputs and the calculated difference must use a consistent currency scale.

### M17 — Revenue

> Compare Microsoft and Tesla total revenue for fiscal 2024 according to their 2024 10-K filings. State both values and the difference.

Expected:

- Decision: `answered`.
- Required groups: MSFT / 2024 / 10-K and TSLA / 2024 / 10-K.
- Use total company revenue for both companies, not a segment value.

### M18 — Risk comparison

> Compare the principal AI and technology infrastructure risks disclosed by Microsoft and Tesla in their 2025 10-K filings.

Expected:

- Decision: `answered`.
- Required groups: MSFT / 2025 / 10-K and TSLA / 2025 / 10-K.
- Provide cited evidence for both companies and avoid claiming the risks are identical.

## 7. Percentage-change calculations

### M19 — Microsoft revenue change

> Using Microsoft's 2024 and 2025 10-K filings, calculate the percentage change in total revenue from fiscal 2024 to fiscal 2025. Show the formula and inputs.

Expected:

- Decision: `answered`.
- Use validated total-revenue values from the two source filings.
- Formula: `(2025 value - 2024 value) / 2024 value × 100`.
- Both input citations must be present.

### M20 — Tesla R&D change

> Using Tesla's 2024 and 2025 10-K filings, calculate the percentage change in research and development expense from 2024 to 2025. Show the formula and inputs.

Expected:

- Decision: `answered`.
- Use the same R&D metric and compatible units for both years.
- The calculator, rather than the answer model, should perform the arithmetic.

### M21 — Two-company calculation

> Calculate the percentage change in total revenue from 2024 to 2025 for both Microsoft and Tesla using their 10-K filings, and identify which company had the higher percentage change.

Expected:

- Decision: `answered`.
- Required groups: both companies × both years × 10-K.
- Four validated inputs and two calculations should be present.
- The conclusion must agree with the deterministic calculations.

## 8. Mixed 10-K and transcript multi-hop questions

### M22 — Revenue and explanations

> Compare Microsoft and Tesla revenue for 2024 and 2025 using their 10-Ks. Calculate each company's percentage change, identify which improved more, and explain the reasons management gave in the 2025 earnings transcripts.

Expected:

- Decision: `answered` when all requirements complete; otherwise a safe partial or insufficient-evidence response.
- Multi-hop route selected.
- Numeric evidence: both companies × 2024/2025 × 10-K.
- Narrative evidence: both companies × 2025 transcript.
- Historical changes must not be explained using unsupported future guidance.

### M23 — Operating income and drivers

> Compare Microsoft and Tesla operating income for 2024 and 2025 using their 10-Ks. Calculate each company's percentage change, identify which improved more, and explain the main drivers discussed in their 2025 earnings transcripts.

Expected:

- Multi-hop route selected.
- Exact operating-income facts should remain distinct from segment operating income.
- Narrative citations must cover both requested companies.

### M24 — R&D and management commentary

> Compare Microsoft and Tesla research and development expense for 2024 and 2025 using their 10-Ks. Calculate each company's percentage change, identify which grew R&D spending faster, and explain relevant management commentary from the 2025 earnings transcripts.

Expected:

- Multi-hop route selected.
- Four validated R&D inputs, two calculations, and transcript evidence for both companies.
- If commentary is unavailable for a company, report that limitation instead of inventing an explanation.

## 9. Missing-source handling

These expectations assume the active index is `phase1-v2`.

### M25 — Missing company

> Summarize JPMorgan's 2025 risk factors from its 10-K.

Expected:

- Decision: `data_unavailable` or the project's equivalent safe unavailable-source decision.
- No retrieval or generation over MSFT/TSLA evidence.

### M26 — Missing source year

> What was Microsoft's total revenue according to its 2023 10-K?

Expected:

- Decision: `data_unavailable` or `out_of_scope`, according to the established policy.
- The system must not answer using a 2023 comparative column inside a later filing.

### M27 — Unsupported document type

> Summarize Tesla's 2025 8-K filing.

Expected:

- Decision: `out_of_scope` or `data_unavailable`.
- The system must not silently substitute the 10-K or transcript.

## 10. Ambiguous company/year handling

### M28 — Missing company and year

> Summarize the principal risk factors.

Expected:

- Decision: `clarify`.
- Ask for at least company and source year; do not search all documents silently.

### M29 — Missing company

> Compare total revenue in 2024 and 2025.

Expected:

- Decision: `clarify`.
- Ask which company or companies should be compared.

### M30 — Missing company and year for transcript

> What did management say about automotive margins in the earnings call?

Expected:

- Decision: `clarify`.
- Ask for the company and year while recognizing transcript intent.

## Results log

| ID | Decision | Resolved scope correct? | Source coverage correct? | Answer/calculation correct? | Citations valid? | Latency | Pass/Fail | Notes |
|---|---|---|---|---|---|---:|---|---|
| M01 |  |  |  |  |  |  |  |  |
| M02 |  |  |  |  |  |  |  |  |
| M03 |  |  |  |  |  |  |  |  |
| M04 |  |  |  |  |  |  |  |  |
| M05 |  |  |  |  |  |  |  |  |
| M06 |  |  |  |  |  |  |  |  |
| M07 |  |  |  |  |  |  |  |  |
| M08 |  |  |  |  |  |  |  |  |
| M09 |  |  |  |  |  |  |  |  |
| M10 |  |  |  |  |  |  |  |  |
| M11 |  |  |  |  |  |  |  |  |
| M12 |  |  |  |  |  |  |  |  |
| M13 |  |  |  |  |  |  |  |  |
| M14 |  |  |  |  |  |  |  |  |
| M15 |  |  |  |  |  |  |  |  |
| M16 |  |  |  |  |  |  |  |  |
| M17 |  |  |  |  |  |  |  |  |
| M18 |  |  |  |  |  |  |  |  |
| M19 |  |  |  |  |  |  |  |  |
| M20 |  |  |  |  |  |  |  |  |
| M21 |  |  |  |  |  |  |  |  |
| M22 |  |  |  |  |  |  |  |  |
| M23 |  |  |  |  |  |  |  |  |
| M24 |  |  |  |  |  |  |  |  |
| M25 |  |  |  |  |  |  |  |  |
| M26 |  |  |  |  |  |  |  |  |
| M27 |  |  |  |  |  |  |  |  |
| M28 |  |  |  |  |  |  |  |  |
| M29 |  |  |  |  |  |  |  |  |
| M30 |  |  |  |  |  |  |  |  |

## Release gate

A refactor is acceptable only when:

- All missing-source and ambiguity tests make safe decisions.
- All exact-value tests return the correct row, period, and scale.
- Every comparison covers all required evidence groups.
- Complete-table tests preserve the complete matched source structure.
- Multi-hop tests either complete with validated evidence or fail safely with a
  precise missing-evidence explanation.
- Any regression is documented and accepted before merging.
