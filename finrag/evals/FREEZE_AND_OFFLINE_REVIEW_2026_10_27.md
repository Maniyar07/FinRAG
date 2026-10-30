# Accepted-version freeze and offline regression review

Completed on 27 October 2026 with **zero provider API requests**. No production
code, local configuration, index version, Git HEAD or Git index was changed.
The additions for this task are one 132-line regression module and QA artifacts.

## Rollback point

The accepted working version was frozen before adding the broader regression
module. This is a workspace snapshot, including existing uncommitted and
untracked files; it is not a new Git commit or a claim that every captured change
was made during this task.

- Checkpoint: [accepted_20261027_805f01629e6d](results/checkpoints/accepted_20261027_805f01629e6d/RESTORE.md).
- Archive: `accepted_workspace.zip`, 240 files, approximately 13.78 MiB.
- Archive SHA-256: `7f7ab8054eec4dc79d0edcbfc8f84151fa3be65ffb0d88172867adbc4b3ec6ed`.
- Accepted core/profile revision: `fdbc06e79e961022c15ae7871894a298b17a2015fff3a8af7f4f5a5018d3d04e`.
- Preserved index: `phase1-v2-repair-20261024`.
- Previously accepted model profile: GPT-4.1 mini planning, GPT-4.1 evidence review.

The ZIP, manifest and captured Git patches are read-only. The manifest records
each file's checksum, dependency versions, Git state and hashes of the index
manifest and local configuration. Secrets and the virtual environment are not
in the archive; vector index files remain in their original versioned directory.
Read-only attributes protect against accidental edits, not privileged tampering.

The archive was extracted into `restore_rehearsal`; all 240 restored file hashes
matched. All 240 original workspace files still match too. The archive checksum,
Git HEAD, index manifest and local configuration hash were verified unchanged
after the tests.

For rollback, use the extracted code in a separate directory with the preserved
index and local configuration, following `RESTORE.md`. The snapshot uses the
existing Python environment for the rehearsal; dependency installation into a
new environment was not tested. Keep the current workspace intact.

## Broader regression results

| Check | Result |
| --- | --- |
| Full expanded offline suite | 705 tests and 70 subtests passed; no failures or skips |
| New cross-category matrix | 25 cases passed, included in the full-suite count |
| Restored-version smoke tests | 69 tests and 17 subtests passed |
| Restored saved-answer replay | Accepted business answer text and citations unchanged; broader transcript observations excluded and valid backlog/margin outlook retained |
| Real-index numeric graph checks | Five complete answers with citations; zero model and general-search calls |
| Real-index missing-period lookup | No row returned for FY2026 revenue requested from Tesla's 2025 filing |

The new [test module](../tests/test_frozen_offline_matrix.py) covers:

- Eight numeric cases: decimal, negative and zero values; missing facts; wrong
  reported period, company, document type or source year. A positive case uses
  a prior-year observation from a later filing.
- Four narrative cases: productive capital investment, a supported negative
  statement, an accounting-policy false positive and genuinely absent evidence.
- Seven comparison cases: a complete two-company, three-dimension comparison,
  then each of the six source details removed independently. Supported sections
  must survive; the missing dimension must remain explicitly unresolved even
  when the overall audit verdict claims completeness.
- Six transcript cases: valid business-specific outlook, broader business
  revenue, combined-segment revenue, shared product spending, historical-only
  evidence and genuinely absent outlook.

The real-index checks use known plans and actual stored filing tables. Expected
revenue values in million USD were Microsoft FY2025 281,724, Tesla FY2025 94,827,
Microsoft FY2024 245,122, Tesla FY2024 97,690, and Tesla FY2023 96,773 from its
2024 filing. Expectations come from the existing acceptance/regression records;
they were not added to application logic.

## Scope and evidence

Tests used the existing offline runner, which blocks external socket connections
and disables tracing. The index check and saved-answer replay block socket
connections entirely. The shared provider ledger remains at **388/400**.

These results verify deterministic contracts, controlled evidence-audit outcomes,
saved provider-response compatibility and exact indexed numeric paths. Narrative
audits are controlled fixtures, not fresh model judgments. This work does not
establish free-form query accuracy, browser latency, concurrency limits, or new
cost/latency percentage savings. The full suite emitted 45 warnings; passing
tests do not mean the project is warning-free or universally bug-free.

Artifacts:

- [Verification summary](results/frozen_regression_20261027.json).
- [Full-suite log](results/frozen_full_20261027.log) and [JUnit XML](results/frozen_full_20261027.junit.xml).
- [Restored smoke log](results/frozen_restore_smoke_20261027.log) and [replay log](results/frozen_restore_replay_20261027.log).
- [Real-index results](results/frozen_index_check_20261027.json) and [check script](results/frozen_index_check_20261027.py).

JUnit counts subtests as additional cases, so its 775-case total corresponds to
705 tests plus 70 subtests; these are not two different full-suite runs.
