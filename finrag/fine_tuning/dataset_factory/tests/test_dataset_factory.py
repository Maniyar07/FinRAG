from __future__ import annotations

import json
import csv
from collections import Counter
from pathlib import Path

import pytest

from fine_tuning.dataset_factory.models import (
    EvidenceBlock,
    GroundedAnswerTarget,
    GroundedClaimTarget,
    MasterExample,
    NumericFactRecord,
    ProvenanceRecord,
    QualityRecord,
    ScopeRecord,
)
from fine_tuning.dataset_factory.augmentation_candidates import (
    SAFETY_CATEGORIES,
    build_calculation_candidates,
    build_evidence_safety_candidates,
)
from fine_tuning.dataset_factory.review import apply_review_csv, write_review_csv
from fine_tuning.dataset_factory.split_export import (
    assign_splits,
    balance_dominant_query_types,
    cap_dataset_size,
    export_training_splits,
)
from fine_tuning.dataset_factory.table_candidates import mine_table_candidates
from fine_tuning.dataset_factory.teacher_candidates import (
    TeacherCandidate,
    TeacherClaim,
    _ensure_question_scope,
)
from fine_tuning.dataset_factory.teacher_synthesis import (
    SynthesisCandidate,
    SynthesisClaim,
    _ensure_question_scope as ensure_synthesis_question_scope,
)
from fine_tuning.dataset_factory import utils
from fine_tuning.dataset_factory.ai_review import write_ai_recommended_review_csv
from fine_tuning.dataset_factory.encoding_cleanup import normalize_supervision
from fine_tuning.dataset_factory.utils import sha256_text
from fine_tuning.dataset_factory.validators import validate_dataset, validate_example


TEST_WORK = Path(__file__).resolve().parents[2] / "work" / "unit-tests"


def work_dir(name: str) -> Path:
    path = TEST_WORK / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_example(
    *,
    example_id: str = "finrag_unit_00000001",
    group_id: str = "grp_0123456789abcdef",
    question: str = "What revenue did Microsoft report for fiscal 2025?",
    status: str = "pending",
    benchmark_overlap: bool = False,
) -> MasterExample:
    text = "Revenue | 2025 | $281,724 | (In millions)"
    return MasterExample(
        example_id=example_id,
        group_id=group_id,
        query_type="direct_extraction",
        difficulty="simple",
        usage="train_candidate",
        question=question,
        scope=ScopeRecord(tickers=["MSFT"], years=["2025"], doc_types=["10K"]),
        evidence=[
            EvidenceBlock(
                source_id="S1",
                parent_id="parent-1",
                ticker="MSFT",
                fiscal_year="2025",
                doc_type="10K",
                source="MSFT_2025_10K.pdf",
                section="Item 8",
                text=text,
                text_sha256=sha256_text(text),
            )
        ],
        numeric_facts=[
            NumericFactRecord(
                label="Revenue",
                value="281724",
                unit="USD millions",
                period="2025",
                source_ids=["S1"],
            )
        ],
        target=GroundedAnswerTarget(
            claims=[
                GroundedClaimTarget(
                    text="Microsoft reported revenue of USD 281,724 million in 2025.",
                    source_ids=["S1"],
                )
            ],
            source_ids=["S1"],
        ),
        provenance=ProvenanceRecord(
            generator="unit_test",
            source_record_ids=["parent-1"],
            canonical_fact_signature="a" * 64,
        ),
        quality=QualityRecord(
            automated_status="passed",
            review_status=status,
            benchmark_overlap=benchmark_overlap,
            reviewed_by="unit-test-reviewer" if status == "approved" else "",
            reviewed_at="2026-09-30T12:00:00+05:30" if status == "approved" else "",
        ),
    )


def test_valid_example_passes_financial_checks() -> None:
    errors, warnings = validate_example(make_example())
    assert errors == []
    assert warnings == []


def test_unsupported_number_is_rejected() -> None:
    example = make_example()
    target = example.target.model_copy(
        update={
            "claims": [
                GroundedClaimTarget(
                    text="Microsoft reported revenue of USD 999,999 million in 2025.",
                    source_ids=["S1"],
                )
            ]
        }
    )
    errors, _ = validate_example(example.model_copy(update={"target": target}))
    assert any(error.startswith("unsupported_target_numbers") for error in errors)


def test_exact_duplicate_question_is_dataset_failure() -> None:
    left = make_example()
    right = make_example(example_id="finrag_unit_00000002", group_id="grp_fedcba9876543210")
    validation = validate_dataset([left, right])
    assert not validation.passed
    assert any(error.startswith("duplicate_question") for error in validation.errors)


def test_split_assignment_is_group_stable() -> None:
    left = make_example(status="approved")
    right = make_example(
        example_id="finrag_unit_00000002",
        group_id=left.group_id,
        question="Show Microsoft's fiscal 2025 revenue as one grounded table row.",
        status="approved",
    )
    assigned = assign_splits([left, right])
    assert assigned[0].split == assigned[1].split


def test_review_cannot_approve_benchmark_overlap() -> None:
    example = make_example(benchmark_overlap=True)
    review_path = work_dir("review") / "review.csv"
    write_review_csv(review_path, [example])
    text = review_path.read_text(encoding="utf-8-sig")
    review_path.write_text(text.replace(",pending,", ",approved,"), encoding="utf-8-sig")
    with pytest.raises(ValueError, match="benchmark overlap"):
        apply_review_csv([example], review_path)


def test_review_requires_auditable_reviewer_metadata() -> None:
    example = make_example()
    review_path = work_dir("review-metadata") / "review.csv"
    write_review_csv(review_path, [example])
    text = review_path.read_text(encoding="utf-8-sig")
    review_path.write_text(text.replace(",pending,", ",approved,"), encoding="utf-8-sig")
    with pytest.raises(ValueError, match="reviewed_by is required"):
        apply_review_csv([example], review_path)


def test_export_blocks_small_dataset() -> None:
    example = assign_splits([make_example(status="approved")], train_ratio=0.99, validation_ratio=0.005)[0]
    if example.split != "train":
        example = example.model_copy(update={"split": "train"})
    with pytest.raises(ValueError, match="at least 500"):
        export_training_splits([example], work_dir("export") / "out")


def test_query_type_balancing_is_deterministic_and_auditable() -> None:
    rows = []
    for index in range(10):
        query_type = (
            "direct_extraction"
            if index < 5
            else "narrative" if index < 8 else "year_over_year"
        )
        rows.append(
            make_example(
                example_id=f"finrag_unit_balance_{index:02d}",
                group_id=f"grp_{index:016x}",
                question=f"What distinct fiscal 2025 fact number {index} did Microsoft report?",
                status="approved",
            ).model_copy(update={"query_type": query_type, "split": "train"})
        )
    policy_path = work_dir("balance") / "policy.json"
    policy_path.write_text(
        json.dumps({"maximum_single_query_type_fraction": 0.40}), encoding="utf-8"
    )
    selected, omitted = balance_dominant_query_types(rows, policy_path)
    repeated, repeated_omitted = balance_dominant_query_types(rows, policy_path)
    assert [row.example_id for row in selected] == [row.example_id for row in repeated]
    assert [row.example_id for row in omitted] == [
        row.example_id for row in repeated_omitted
    ]
    counts = {name: sum(row.query_type == name for row in selected) for name in {
        row.query_type for row in selected
    }}
    assert max(counts.values()) / len(selected) <= 0.40
    assert len(omitted) == 2


def test_dataset_cap_preserves_rare_query_types() -> None:
    rows = []
    for index in range(12):
        query_type = (
            "direct_extraction"
            if index < 8
            else "calculation" if index < 10 else "insufficient_evidence"
        )
        rows.append(
            make_example(
                example_id=f"finrag_unit_cap_{index:02d}",
                group_id=f"grp_{index:016x}",
                question=f"What distinct capped financial fact number {index} is requested?",
                status="approved",
            ).model_copy(update={"query_type": query_type, "split": "train"})
        )
    selected, omitted = cap_dataset_size(rows, 10)
    repeated, repeated_omitted = cap_dataset_size(rows, 10)
    assert len(selected) == 10
    assert {row.query_type for row in omitted} == {"direct_extraction"}
    assert [row.example_id for row in selected] == [row.example_id for row in repeated]
    assert [row.example_id for row in omitted] == [
        row.example_id for row in repeated_omitted
    ]


def test_table_miner_builds_multiple_query_types() -> None:
    parent_dir = work_dir("table") / "parents"
    parent_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "metadata": {
            "ticker": "MSFT",
            "fiscal_year": "2025",
            "doc_type": "10K",
            "source": "MSFT_2025_10K.pdf",
            "source_hash": "b" * 64,
            "parent_id": "parent-table",
            "section": "Item 8. Financial Statements",
        },
        "page_content": """
## Statements of Income

Revenue was as follows:

<table>
<thead><tr><th>(In millions)</th><th>2025</th><th>2024</th></tr></thead>
<tbody>
<tr><td>Revenue</td><td>$281,724</td><td>$245,122</td></tr>
<tr><td>Net income</td><td>$101,832</td><td>$88,136</td></tr>
</tbody>
</table>
""",
    }
    (parent_dir / "parent-table.json").write_text(json.dumps(record), encoding="utf-8")
    examples = mine_table_candidates(
        parent_dir,
        direct_limit=10,
        yoy_limit=10,
        targeted_table_limit=10,
        comparison_limit=0,
        complete_table_limit=10,
    )
    assert {example.query_type for example in examples} == {
        "direct_extraction", "year_over_year", "targeted_table", "complete_table"
    }
    assert validate_dataset(examples).passed


def test_calculation_augmentation_reuses_verified_arithmetic() -> None:
    parent_dir = work_dir("calculation-augmentation") / "parents"
    parent_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "metadata": {
            "ticker": "MSFT",
            "fiscal_year": "2025",
            "doc_type": "10K",
            "source": "MSFT_2025_10K.pdf",
            "source_hash": "c" * 64,
            "parent_id": "parent-calculation",
            "section": "Item 8",
        },
        "page_content": """
## Statements of Income

Revenue was as follows:

<table>
<thead><tr><th>(In millions)</th><th>2025</th><th>2024</th></tr></thead>
<tbody>
<tr><td>Revenue</td><td>$281,724</td><td>$245,122</td></tr>
<tr><td>Net income</td><td>$101,832</td><td>$88,136</td></tr>
</tbody>
</table>
""",
    }
    (parent_dir / "parent-calculation.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    mined = mine_table_candidates(
        parent_dir,
        direct_limit=5,
        yoy_limit=5,
        targeted_table_limit=0,
        comparison_limit=0,
        complete_table_limit=0,
    )
    base = next(row for row in mined if row.query_type == "year_over_year")
    base = base.model_copy(
        update={
            "quality": QualityRecord(
                automated_status="passed",
                review_status="approved",
                reviewed_by="unit-test-reviewer",
                reviewed_at="2026-10-02T17:30:00+05:30",
            ),
            "split": "train",
        }
    )
    additions = build_calculation_candidates([base], target_count=1)
    assert len(additions) == 1
    assert additions[0].query_type == "calculation"
    assert len(additions[0].calculations) == 1
    assert validate_dataset([base, additions[0]]).passed


def test_safety_augmentation_balances_reasons_and_splits() -> None:
    splits = ["train", "train", "train", "validation", "test"]
    sources = [
        make_example(
            example_id=f"finrag_unit_safety_{index:02d}",
            group_id=f"grp_{100 + index:016x}",
            question=(
                f"What distinct fiscal 2025 revenue fact variant {index} did "
                "Microsoft report?"
            ),
            status="approved",
        ).model_copy(update={"split": splits[index]})
        for index in range(5)
    ]
    additions = build_evidence_safety_candidates(sources, target_count=5)
    assert len(additions) == 5
    assert {row.query_type for row in additions} == {"insufficient_evidence"}
    assert {
        row.provenance.generator.split(":", 1)[1] for row in additions
    } == set(SAFETY_CATEGORIES)
    assert Counter(row.split for row in additions) == Counter(splits)
    assert validate_dataset([*sources, *additions]).passed


def test_insufficient_evidence_requires_canonical_uncited_response() -> None:
    source = make_example(status="approved").model_copy(update={"split": "train"})
    safety = build_evidence_safety_candidates(
        [
            source.model_copy(
                update={
                    "example_id": f"finrag_unit_safety_guard_{index:02d}",
                    "group_id": f"grp_{200 + index:016x}",
                    "question": f"What unique evidence guard input {index} is available?",
                    "split": split,
                }
            )
            for index, split in enumerate(
                ["train", "train", "train", "validation", "test"]
            )
        ],
        target_count=5,
    )[0]
    invalid_target = GroundedAnswerTarget(
        answer="The answer is unavailable.", claims=[], source_ids=[]
    )
    errors, _ = validate_example(
        safety.model_copy(update={"target": invalid_target})
    )
    assert "insufficient_evidence_target_must_use_canonical_response" in errors


def test_atomic_writer_retries_transient_windows_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    output = work_dir("atomic-writer") / "result.json"
    real_replace = utils.os.replace
    attempts = 0

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated transient lock")
        real_replace(source, destination)

    monkeypatch.setattr(utils.os, "replace", flaky_replace)
    utils.write_json(output, {"status": "ok"})
    assert attempts == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "ok"}


def test_teacher_question_scope_is_repaired_from_metadata() -> None:
    candidate = TeacherCandidate(
        query_type="narrative",
        question="What factor affected operating expenses?",
        claims=[TeacherClaim(text="A factor affected expenses.", evidence_quote="A factor affected expenses.")],
    )
    repaired = _ensure_question_scope(
        candidate,
        {"ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K"},
    )
    assert repaired.question.startswith("According to Microsoft's 2025 10-K,")


def test_synthesis_question_scope_is_repaired_from_metadata() -> None:
    candidate = SynthesisCandidate(
        question="How do the two sources describe demand?",
        claims=[
            SynthesisClaim(source_id="S1", text="Demand was discussed.", evidence_quote="Demand was discussed."),
            SynthesisClaim(source_id="S2", text="Demand improved.", evidence_quote="Demand improved."),
        ],
    )
    repaired = ensure_synthesis_question_scope(
        candidate,
        {"ticker": "TSLA", "fiscal_year": "2025"},
    )
    assert repaired.question.startswith(
        "According to Tesla's 2025 10-K and 2025 earnings-call transcript,"
    )


def test_ai_review_is_transparently_labeled_not_human() -> None:
    example = make_example()
    source = work_dir("ai-review") / "source.csv"
    output = work_dir("ai-review") / "recommended.csv"
    write_review_csv(source, [example])
    counts = write_ai_recommended_review_csv(
        source_csv=source,
        output_csv=output,
        examples=[example],
        teacher_reviews=[],
    )
    with output.open(newline="", encoding="utf-8-sig") as handle:
        row = next(csv.DictReader(handle))
    assert counts["approved"] == 1
    assert row["reviewed_by"] == "codex-deterministic-review-v1"
    assert "Human sign-off not recorded" in row["reviewer_notes"]


def test_mojibake_supervision_is_repaired_and_revalidated() -> None:
    broken = make_example(
        question="What did Microsoft report about managementâ€™s fiscal 2025 revenue?"
    )
    broken_target = broken.target.model_copy(
        update={
            "claims": [
                GroundedClaimTarget(
                    text="Microsoftâ€™s reported revenue was USD 281,724 million in 2025.",
                    source_ids=["S1"],
                )
            ]
        }
    )
    broken = broken.model_copy(update={"target": broken_target})
    errors, _ = validate_example(broken)
    assert "mojibake_in_supervised_question_or_target" in errors
    repaired, report = normalize_supervision([broken])
    repaired_errors, _ = validate_example(repaired[0])
    assert "mojibake_in_supervised_question_or_target" not in repaired_errors
    assert "management’s" in repaired[0].question
    assert report["changed_examples"] == 1
