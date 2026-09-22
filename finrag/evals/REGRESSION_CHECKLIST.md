# Manual regression checklist

Run these questions after a change to retrieval, routing, compression, or answer
generation. Use the same index and settings when comparing two versions. Record
the answer and source links separately if a result fails; do not change a rule
only to make one question pass.

Date: __________  Git commit: __________  Index: __________

Reranking: on / off  Multi-hop: on / off

Mark a question `pass` only if its answer addresses the whole request, uses the
correct company/year/document, and cites supporting evidence. A supported
answer should not display raw HTML fragments or stop mid-table. An unavailable
source or live-data request should be declined clearly.

For the bundled `phase1-v2` index, Tesla's 2025 10-K statement reports revenue
of $94,827 million and net income of $3,855 million. The revenue comparison
should use MSFT $245,122 million (2024) and $281,724 million (2025), and TSLA
$97,690 million (2024) and $94,827 million (2025).

| # | Question | What to check | Result |
|---|---|---|---|
| 1 | Show the complete balance sheet table from Microsoft's 2025 10-K. | Full matching table, all rows and year columns, source citation. | ☐ Pass ☐ Fail |
| 2 | Show the complete cash flow statement from Tesla's 2024 10-K. | Primary cash-flow table, not a short summary table. | ☐ Pass ☐ Fail |
| 3 | Show the complete “Revenue by source” table from Tesla's 2025 10-K. | Named table and all its rows. | ☐ Pass ☐ Fail |
| 4 | List every Item heading in Microsoft's 2024 10-K. | Ordered Item headings without unrelated notes or omissions. | ☐ Pass ☐ Fail |
| 5 | What were Microsoft's total assets in its 2025 10-K? | Correct row, year, unit, and citation. | ☐ Pass ☐ Fail |
| 6 | Summarize Tesla's main competition risks in its 2024 10-K. | Grounded summary with filing citation. | ☐ Pass ☐ Fail |
| 7 | Which of those risks could affect vehicle sales? | Ask immediately after #6; follow-up uses the same scope. | ☐ Pass ☐ Fail |
| 8 | Compare Microsoft's 2024 and 2025 revenue using its 10-K filings. | Both years and their sources are clear. | ☐ Pass ☐ Fail |
| 9 | Compare Microsoft's and Tesla's 2025 revenue. Cite each filing. | Both companies covered; citations do not cross companies. | ☐ Pass ☐ Fail |
| 10 | Using MSFT and TSLA 2024 and 2025 10-Ks, calculate each company's revenue percentage change. Which grew more? | Both companies, correct inputs, arithmetic, units, and conclusion. | ☐ Pass ☐ Fail |
| 11 | Compare how Microsoft and Tesla discuss competition risk in their 2024 10-Ks. | Both companies covered with separate evidence. | ☐ Pass ☐ Fail |
| 12 | From Tesla's 2025 10-K, give revenue and net income, then summarize one risk factor. | $94,827 million revenue, $3,855 million net income, and a cited risk factor. | ☐ Pass ☐ Fail |
| 13 | What did Microsoft management say about Azure growth in its 2025 Q4 earnings-call transcript? | Uses the transcript rather than a 10-K. | ☐ Pass ☐ Fail |
| 14 | What is Microsoft's stock price right now? | Declines live-data request; does not invent a price. | ☐ Pass ☐ Fail |
| 15 | What does Apple's 2025 10-K say about revenue? | Reports that Apple is unavailable in the selected index. | ☐ Pass ☐ Fail |

Run all 15 with `FINRAG_MULTIHOP_ENABLED=false`. Then set it to `true`, restart
the app, and repeat #10 and #12. Compare correctness, citations, and response
time. Keep multi-hop enabled only if it helps those multi-task questions without
regressing the full set.

Failures and observations:

| # | Actual problem | Source or trace | General cause | Fix and rerun date |
|---|---|---|---|---|
| | | | | |
