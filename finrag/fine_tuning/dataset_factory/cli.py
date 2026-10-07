from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .ai_review import (
    generate_teacher_reviews,
    read_stored_reviews,
    write_ai_recommended_review_csv,
)
from .augmentation_candidates import (
    build_calculation_candidates,
    build_evidence_safety_candidates,
)
from .benchmark_guard import mark_benchmark_overlaps
from .encoding_cleanup import normalize_supervision
from .review import apply_review_csv, write_review_csv
from .seed_builder import build_approved_seed_examples
from .split_export import assign_splits, export_training_splits
from .table_candidates import mine_table_candidates
from .teacher_candidates import (
    excluded_gold_parent_ids,
    generate_narrative_candidates,
    load_narrative_pool,
)
from .teacher_synthesis import build_synthesis_pool, generate_synthesis_candidates
from .utils import read_jsonl, write_json, write_jsonl
from .validators import validate_dataset, with_validation_status


DATASET_FACTORY_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = DATASET_FACTORY_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_WORK = PACKAGE_ROOT / "work"
DEFAULT_MASTER = DEFAULT_WORK / "master_candidates.jsonl"


def _validation_payload(validation, examples) -> dict:
    return {
        "passed": validation.passed,
        "counts": {
            "examples": len(examples),
            "train_candidates": sum(
                example.usage == "train_candidate" for example in examples
            ),
            "automated_passed": sum(
                example.quality.automated_status == "passed" for example in examples
            ),
            "approved": sum(
                example.quality.review_status == "approved" for example in examples
            ),
            "approved_training": sum(
                example.usage == "train_candidate"
                and example.quality.review_status == "approved"
                and not example.quality.benchmark_overlap
                for example in examples
            ),
            "pending_training_review": sum(
                example.usage == "train_candidate"
                and example.quality.review_status == "pending"
                for example in examples
            ),
            "benchmark_clean_train_candidates": sum(
                example.usage == "train_candidate"
                and not example.quality.benchmark_overlap
                for example in examples
            ),
            "benchmark_overlap": sum(
                example.quality.benchmark_overlap for example in examples
            ),
        },
        "query_types": dict(sorted(Counter(example.query_type for example in examples).items())),
        "usage": dict(sorted(Counter(example.usage for example in examples).items())),
        "errors": list(validation.errors),
        "warnings": list(validation.warnings),
    }


def run_build(args: argparse.Namespace) -> int:
    seed_examples = build_approved_seed_examples(
        Path(args.drafts).resolve(),
        Path(args.approvals).resolve(),
        index_version=args.index_version,
    )
    candidates = mine_table_candidates(
        Path(args.parent_dir).resolve(),
        direct_limit=args.direct_limit,
        yoy_limit=args.yoy_limit,
        targeted_table_limit=args.targeted_table_limit,
        comparison_limit=args.comparison_limit,
        complete_table_limit=args.complete_table_limit,
    )
    examples = with_validation_status([*seed_examples, *candidates])
    examples = mark_benchmark_overlaps(examples, Path(args.benchmark).resolve())
    examples.sort(key=lambda example: example.example_id)
    validation = validate_dataset(examples)
    output = Path(args.output).resolve()
    write_jsonl(output, examples)
    report = _validation_payload(validation, examples)
    write_json(Path(args.report).resolve(), report)
    write_review_csv(Path(args.review_csv).resolve(), examples)
    print(
        f"Built {len(examples)} records: {len(seed_examples)} evaluation-reserved "
        f"prototypes and {len(candidates)} review-required training candidates."
    )
    print(f"Master dataset: {output}")
    print(f"Validation report: {Path(args.report).resolve()}")
    print(f"Review sheet: {Path(args.review_csv).resolve()}")
    if not validation.passed:
        print(f"Dataset-level validation failures: {len(validation.errors)}")
        return 1
    return 0


def run_validate(args: argparse.Namespace) -> int:
    path = Path(args.dataset).resolve()
    examples = read_jsonl(path)
    examples = with_validation_status(examples)
    if args.benchmark:
        examples = mark_benchmark_overlaps(examples, Path(args.benchmark).resolve())
    validation = validate_dataset(examples)
    output = Path(args.output).resolve() if args.output else path
    write_jsonl(output, examples)
    report_path = Path(args.report).resolve()
    write_json(report_path, _validation_payload(validation, examples))
    print(f"Validated {len(examples)} records; passed={validation.passed}")
    print(f"Report: {report_path}")
    return 0 if validation.passed else 1


def run_apply_review(args: argparse.Namespace) -> int:
    examples = read_jsonl(Path(args.dataset).resolve())
    reviewed = apply_review_csv(examples, Path(args.review_csv).resolve())
    validation = validate_dataset(reviewed)
    output = Path(args.output).resolve()
    write_jsonl(output, reviewed)
    print(
        f"Applied review statuses to {len(reviewed)} records; "
        f"approved={sum(row.quality.review_status == 'approved' for row in reviewed)}"
    )
    print(f"Reviewed master: {output}")
    return 0 if validation.passed else 1


def run_split(args: argparse.Namespace) -> int:
    examples = read_jsonl(Path(args.dataset).resolve())
    assigned = assign_splits(
        examples,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    validation = validate_dataset(assigned)
    output = Path(args.output).resolve()
    write_jsonl(output, assigned)
    counts = Counter(row.split or "unassigned" for row in assigned)
    print(f"Split assignments: {dict(sorted(counts.items()))}")
    print(f"Split master: {output}")
    return 0 if validation.passed else 1


def run_export(args: argparse.Namespace) -> int:
    examples = read_jsonl(Path(args.dataset).resolve())
    validation = validate_dataset(examples)
    if not validation.passed:
        raise ValueError(
            "Export blocked because dataset validation failed:\n- "
            + "\n- ".join(validation.errors[:25])
        )
    manifest = export_training_splits(
        examples,
        Path(args.output_dir).resolve(),
        minimum_train_examples=args.minimum_train_examples,
        include_metadata=args.include_metadata,
        policy_path=Path(args.policy).resolve() if args.policy else None,
        balance_query_types=args.balance_query_types,
        maximum_examples=args.maximum_examples,
    )
    print(f"Exported SFT dataset: {manifest['counts']}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    return 0


def run_prepare_final(args: argparse.Namespace) -> int:
    source = read_jsonl(Path(args.dataset).resolve())
    calculations = build_calculation_candidates(
        source, target_count=args.calculation_count
    )
    safety = build_evidence_safety_candidates(
        source, target_count=args.safety_count
    )
    additions = with_validation_status([*calculations, *safety])
    additions = mark_benchmark_overlaps(
        additions, Path(args.benchmark).resolve()
    )
    overlaps = [row.example_id for row in additions if row.quality.benchmark_overlap]
    if overlaps:
        raise ValueError(
            "Augmentation overlaps the locked benchmark: " + ", ".join(overlaps[:10])
        )
    existing_ids = {row.example_id for row in source}
    duplicate_ids = [row.example_id for row in additions if row.example_id in existing_ids]
    if duplicate_ids:
        raise ValueError(
            "Augmentation generated existing example IDs: " + ", ".join(duplicate_ids)
        )
    combined = with_validation_status([*source, *additions])
    combined = mark_benchmark_overlaps(
        combined, Path(args.benchmark).resolve()
    )
    assigned = assign_splits(
        combined,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    validation = validate_dataset(assigned)
    if not validation.passed:
        raise ValueError(
            "Final dataset validation failed:\n- "
            + "\n- ".join(validation.errors[:25])
        )

    master_output = Path(args.master_output).resolve()
    additions_output = Path(args.additions_output).resolve()
    additions_by_id = {row.example_id for row in additions}
    assigned_additions = [
        row for row in assigned if row.example_id in additions_by_id
    ]
    write_jsonl(master_output, assigned)
    write_jsonl(additions_output, assigned_additions)
    write_review_csv(Path(args.review_csv).resolve(), assigned_additions)
    manifest = export_training_splits(
        assigned,
        Path(args.output_dir).resolve(),
        minimum_train_examples=args.minimum_train_examples,
        include_metadata=args.include_metadata,
        policy_path=Path(args.policy).resolve(),
        balance_query_types=False,
        maximum_examples=args.maximum_examples,
    )
    report = {
        "passed": True,
        "source_examples": len(source),
        "master_examples": len(assigned),
        "added_examples": len(assigned_additions),
        "added_query_types": dict(
            sorted(Counter(row.query_type for row in assigned_additions).items())
        ),
        "added_splits": dict(
            sorted(Counter(row.split for row in assigned_additions).items())
        ),
        "safety_categories": dict(
            sorted(
                Counter(
                    row.provenance.generator.split(":", 1)[1]
                    for row in assigned_additions
                    if row.query_type == "insufficient_evidence"
                ).items()
            )
        ),
        "validation": _validation_payload(validation, assigned),
        "export_manifest": manifest,
    }
    write_json(Path(args.report).resolve(), report)
    print(
        f"Added {len(calculations)} calculation and {len(safety)} evidence-safety "
        "examples."
    )
    print(f"Final capped SFT dataset: {manifest['counts']}")
    print(f"Final SFT total: {sum(manifest['counts'].values())}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    return 0


def run_narrative_pool(args: argparse.Namespace) -> int:
    pool = load_narrative_pool(
        Path(args.parent_dir).resolve(),
        excluded_parent_ids=(
            excluded_gold_parent_ids(Path(args.drafts).resolve())
            if args.exclude_gold_parents
            else set()
        ),
        max_chars=args.max_chars,
        max_windows_per_parent=args.max_windows_per_parent,
    )
    counts = Counter(
        (
            str(row["metadata"]["ticker"]),
            str(row["metadata"]["fiscal_year"]),
            str(row["metadata"]["doc_type"]),
        )
        for row in pool
    )
    qualifier = "excluding gold-parent records" if args.exclude_gold_parents else "with fact-level benchmark guarding"
    print(f"Eligible narrative evidence windows ({qualifier}): {len(pool)}")
    for key, count in sorted(counts.items()):
        print(f"  {' | '.join(key)}: {count}")
    return 0


def run_generate_narrative(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    existing = read_jsonl(output) if args.resume and output.is_file() else []
    examples = generate_narrative_candidates(
        parent_dir=Path(args.parent_dir).resolve(),
        output=output,
        benchmark_path=Path(args.benchmark).resolve(),
        drafts_path=Path(args.drafts).resolve(),
        model=args.model,
        target_count=args.target_count,
        max_candidates_per_parent=args.max_candidates_per_parent,
        max_chars=args.max_chars,
        max_windows_per_parent=args.max_windows_per_parent,
        exclude_gold_parents=args.exclude_gold_parents,
        existing=existing,
        doc_type_filter=None if args.doc_type == "ALL" else args.doc_type,
        excluded_examples=(
            read_jsonl(Path(args.exclude_dataset).resolve())
            if args.exclude_dataset
            else []
        ),
    )
    examples = with_validation_status(examples)
    examples = mark_benchmark_overlaps(examples, Path(args.benchmark).resolve())
    validation = validate_dataset(examples)
    write_jsonl(output, examples)
    write_review_csv(Path(args.review_csv).resolve(), examples)
    write_json(Path(args.report).resolve(), _validation_payload(validation, examples))
    print(f"Narrative candidate output: {output}")
    return 0 if validation.passed else 1


def run_merge(args: argparse.Namespace) -> int:
    rows = [example for value in args.inputs for example in read_jsonl(Path(value).resolve())]
    ids = [row.example_id for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [value for value, count in Counter(ids).items() if count > 1]
        raise ValueError("Duplicate example IDs across inputs: " + ", ".join(duplicates[:20]))
    rows = with_validation_status(rows)
    rows = mark_benchmark_overlaps(rows, Path(args.benchmark).resolve())
    rows.sort(key=lambda row: row.example_id)
    validation = validate_dataset(rows)
    output = Path(args.output).resolve()
    write_jsonl(output, rows)
    write_review_csv(Path(args.review_csv).resolve(), rows)
    write_json(Path(args.report).resolve(), _validation_payload(validation, rows))
    print(f"Merged {len(rows)} records into {output}")
    return 0 if validation.passed else 1


def run_synthesis_pool(args: argparse.Namespace) -> int:
    pairs = build_synthesis_pool(
        Path(args.parent_dir).resolve(), max_chars=args.max_chars
    )
    counts = Counter(
        (
            str(filing["metadata"]["ticker"]),
            str(filing["metadata"]["fiscal_year"]),
        )
        for filing, _ in pairs
    )
    print(f"Eligible filing/transcript evidence pairs: {len(pairs)}")
    for key, count in sorted(counts.items()):
        print(f"  {' | '.join(key)}: {count}")
    return 0


def run_generate_synthesis(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    existing = read_jsonl(output) if args.resume and output.is_file() else []
    examples = generate_synthesis_candidates(
        parent_dir=Path(args.parent_dir).resolve(),
        output=output,
        benchmark_path=Path(args.benchmark).resolve(),
        model=args.model,
        target_count=args.target_count,
        max_candidates_per_pair=args.max_candidates_per_pair,
        max_chars=args.max_chars,
        existing=existing,
        excluded_examples=(
            read_jsonl(Path(args.exclude_dataset).resolve())
            if args.exclude_dataset
            else []
        ),
    )
    examples = mark_benchmark_overlaps(
        with_validation_status(examples), Path(args.benchmark).resolve()
    )
    validation = validate_dataset(examples)
    write_jsonl(output, examples)
    write_review_csv(Path(args.review_csv).resolve(), examples)
    write_json(Path(args.report).resolve(), _validation_payload(validation, examples))
    print(f"Synthesis candidate output: {output}")
    return 0 if validation.passed else 1


def run_ai_review(args: argparse.Namespace) -> int:
    examples = read_jsonl(Path(args.dataset).resolve())
    decisions_path = Path(args.decisions).resolve()
    if decisions_path.exists() and not args.resume:
        raise ValueError(
            f"AI review decisions already exist: {decisions_path}; use --resume"
        )
    existing = read_stored_reviews(decisions_path) if args.resume else []
    reviews = generate_teacher_reviews(
        examples,
        output=decisions_path,
        model=args.model,
        batch_size=args.batch_size,
        existing=existing,
    )
    counts = write_ai_recommended_review_csv(
        source_csv=Path(args.source_review_csv).resolve(),
        output_csv=Path(args.output_review_csv).resolve(),
        examples=examples,
        teacher_reviews=reviews,
        minimum_confidence=args.minimum_confidence,
    )
    print(f"AI-assisted review recommendations: {dict(sorted(counts.items()))}")
    print(f"Decision audit: {decisions_path}")
    print(f"Review CSV: {Path(args.output_review_csv).resolve()}")
    return 0


def run_normalize_supervision(args: argparse.Namespace) -> int:
    examples = read_jsonl(Path(args.dataset).resolve())
    normalized, normalization_report = normalize_supervision(examples)
    normalized = mark_benchmark_overlaps(
        with_validation_status(normalized), Path(args.benchmark).resolve()
    )
    validation = validate_dataset(normalized)
    output = Path(args.output).resolve()
    write_jsonl(output, normalized)
    payload = {
        **normalization_report,
        "validation": _validation_payload(validation, normalized),
    }
    write_json(Path(args.report).resolve(), payload)
    print(
        f"Normalized supervised text in {normalization_report['changed_examples']} "
        f"of {len(normalized)} records; passed={validation.passed}"
    )
    print(f"Normalized dataset: {output}")
    return 0 if validation.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and gate an evidence-grounded FinRAG LoRA dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build initial prototypes and table candidates.")
    build.add_argument(
        "--drafts",
        default=str(PROJECT_ROOT / "evals" / "gold_builder" / "work" / "drafts.json"),
    )
    build.add_argument(
        "--approvals",
        default=str(PROJECT_ROOT / "evals" / "gold_builder" / "work" / "review.csv"),
    )
    build.add_argument(
        "--parent-dir",
        default=str(PROJECT_ROOT / "Data" / "indexes" / "phase1-v2" / "parents"),
    )
    build.add_argument(
        "--benchmark", default=str(PROJECT_ROOT / "evals" / "ragas_gold_90.json")
    )
    build.add_argument("--index-version", default="phase1-v2")
    # Keep deterministic direct-extraction rows below the policy's 40% cap
    # once the recommended narrative and synthesis stages are added.
    build.add_argument("--direct-limit", type=int, default=350)
    build.add_argument("--yoy-limit", type=int, default=150)
    build.add_argument("--targeted-table-limit", type=int, default=50)
    build.add_argument("--comparison-limit", type=int, default=60)
    build.add_argument("--complete-table-limit", type=int, default=50)
    build.add_argument("--output", default=str(DEFAULT_MASTER))
    build.add_argument("--report", default=str(DEFAULT_WORK / "validation_report.json"))
    build.add_argument("--review-csv", default=str(DEFAULT_WORK / "review.csv"))
    build.set_defaults(handler=run_build)

    validate = subparsers.add_parser("validate", help="Revalidate a master JSONL file.")
    validate.add_argument("--dataset", default=str(DEFAULT_MASTER))
    validate.add_argument("--output")
    validate.add_argument(
        "--benchmark", default=str(PROJECT_ROOT / "evals" / "ragas_gold_90.json")
    )
    validate.add_argument("--report", default=str(DEFAULT_WORK / "validation_report.json"))
    validate.set_defaults(handler=run_validate)

    review = subparsers.add_parser(
        "apply-review", help="Apply manually edited review statuses to the master dataset."
    )
    review.add_argument("--dataset", default=str(DEFAULT_MASTER))
    review.add_argument("--review-csv", default=str(DEFAULT_WORK / "review.csv"))
    review.add_argument("--output", default=str(DEFAULT_WORK / "reviewed_master.jsonl"))
    review.set_defaults(handler=run_apply_review)

    split = subparsers.add_parser("split", help="Assign leakage-safe deterministic splits.")
    split.add_argument("--dataset", default=str(DEFAULT_WORK / "reviewed_master.jsonl"))
    split.add_argument("--output", default=str(DEFAULT_WORK / "split_master.jsonl"))
    split.add_argument("--train-ratio", type=float, default=0.84)
    split.add_argument("--validation-ratio", type=float, default=0.08)
    split.set_defaults(handler=run_split)

    export = subparsers.add_parser("export", help="Export only approved training records.")
    export.add_argument("--dataset", default=str(DEFAULT_WORK / "split_master.jsonl"))
    export.add_argument("--output-dir", default=str(DEFAULT_WORK / "sft"))
    export.add_argument("--minimum-train-examples", type=int, default=500)
    export.add_argument(
        "--maximum-examples",
        type=int,
        help="Deterministically cap the total exported train/validation/test rows.",
    )
    export.add_argument(
        "--policy",
        default=str(DATASET_FACTORY_ROOT / "configs" / "dataset_policy.json"),
    )
    export.add_argument("--include-metadata", action="store_true")
    export.add_argument(
        "--balance-query-types",
        action="store_true",
        help="Deterministically omit excess dominant-type rows from the train export only.",
    )
    export.set_defaults(handler=run_export)

    prepare_final = subparsers.add_parser(
        "prepare-final",
        help=(
            "Add deterministic calculation and evidence-safety examples, validate, "
            "split, and export a diversity-capped final SFT dataset."
        ),
    )
    prepare_final.add_argument(
        "--dataset", default=str(DEFAULT_WORK / "split_expanded_master.jsonl")
    )
    prepare_final.add_argument(
        "--benchmark", default=str(PROJECT_ROOT / "evals" / "ragas_gold_90.json")
    )
    prepare_final.add_argument("--calculation-count", type=int, default=25)
    prepare_final.add_argument("--safety-count", type=int, default=25)
    prepare_final.add_argument("--maximum-examples", type=int, default=800)
    prepare_final.add_argument("--minimum-train-examples", type=int, default=500)
    prepare_final.add_argument("--train-ratio", type=float, default=0.84)
    prepare_final.add_argument("--validation-ratio", type=float, default=0.08)
    prepare_final.add_argument(
        "--policy",
        default=str(DATASET_FACTORY_ROOT / "configs" / "dataset_policy.json"),
    )
    prepare_final.add_argument("--include-metadata", action="store_true")
    prepare_final.add_argument(
        "--master-output", default=str(DEFAULT_WORK / "final_800_master.jsonl")
    )
    prepare_final.add_argument(
        "--additions-output", default=str(DEFAULT_WORK / "final_800_additions.jsonl")
    )
    prepare_final.add_argument(
        "--review-csv", default=str(DEFAULT_WORK / "final_800_additions_review.csv")
    )
    prepare_final.add_argument(
        "--report", default=str(DEFAULT_WORK / "final_800_report.json")
    )
    prepare_final.add_argument(
        "--output-dir", default=str(DEFAULT_WORK / "sft_final_800")
    )
    prepare_final.set_defaults(handler=run_prepare_final)

    pool = subparsers.add_parser(
        "narrative-pool", help="Inspect eligible narrative parents without model calls."
    )
    pool.add_argument(
        "--parent-dir",
        default=str(PROJECT_ROOT / "Data" / "indexes" / "phase1-v2" / "parents"),
    )
    pool.add_argument(
        "--drafts",
        default=str(PROJECT_ROOT / "evals" / "gold_builder" / "work" / "drafts.json"),
    )
    pool.add_argument("--max-chars", type=int, default=12_000)
    pool.add_argument("--max-windows-per-parent", type=int, default=4)
    pool.add_argument("--exclude-gold-parents", action="store_true")
    pool.set_defaults(handler=run_narrative_pool)

    narrative = subparsers.add_parser(
        "generate-narrative",
        help="Generate separately gated narrative/transcript candidates with a teacher model.",
    )
    narrative.add_argument(
        "--parent-dir",
        default=str(PROJECT_ROOT / "Data" / "indexes" / "phase1-v2" / "parents"),
    )
    narrative.add_argument(
        "--drafts",
        default=str(PROJECT_ROOT / "evals" / "gold_builder" / "work" / "drafts.json"),
    )
    narrative.add_argument(
        "--benchmark", default=str(PROJECT_ROOT / "evals" / "ragas_gold_90.json")
    )
    narrative.add_argument("--model", default="gpt-4o-mini")
    narrative.add_argument("--target-count", type=int, default=250)
    narrative.add_argument("--max-candidates-per-parent", type=int, default=2)
    narrative.add_argument("--max-chars", type=int, default=12_000)
    narrative.add_argument("--max-windows-per-parent", type=int, default=4)
    narrative.add_argument(
        "--doc-type", choices=("ALL", "10K", "TRANSCRIPT"), default="ALL"
    )
    narrative.add_argument(
        "--exclude-dataset",
        help="Existing master JSONL whose IDs, groups, and questions must not be regenerated.",
    )
    narrative.add_argument("--exclude-gold-parents", action="store_true")
    narrative.add_argument("--resume", action="store_true")
    narrative.add_argument("--output", default=str(DEFAULT_WORK / "narrative_candidates.jsonl"))
    narrative.add_argument("--review-csv", default=str(DEFAULT_WORK / "narrative_review.csv"))
    narrative.add_argument("--report", default=str(DEFAULT_WORK / "narrative_report.json"))
    narrative.set_defaults(handler=run_generate_narrative)

    merge = subparsers.add_parser("merge", help="Merge independently built master JSONL files.")
    merge.add_argument("inputs", nargs="+")
    merge.add_argument(
        "--benchmark", default=str(PROJECT_ROOT / "evals" / "ragas_gold_90.json")
    )
    merge.add_argument("--output", default=str(DEFAULT_WORK / "combined_master.jsonl"))
    merge.add_argument("--review-csv", default=str(DEFAULT_WORK / "combined_review.csv"))
    merge.add_argument("--report", default=str(DEFAULT_WORK / "combined_report.json"))
    merge.set_defaults(handler=run_merge)

    synthesis_pool = subparsers.add_parser(
        "synthesis-pool", help="Inspect 10-K/transcript evidence pairs without model calls."
    )
    synthesis_pool.add_argument(
        "--parent-dir",
        default=str(PROJECT_ROOT / "Data" / "indexes" / "phase1-v2" / "parents"),
    )
    synthesis_pool.add_argument("--max-chars", type=int, default=8_000)
    synthesis_pool.set_defaults(handler=run_synthesis_pool)

    synthesis = subparsers.add_parser(
        "generate-synthesis",
        help="Generate separately gated 10-K/transcript synthesis candidates.",
    )
    synthesis.add_argument(
        "--parent-dir",
        default=str(PROJECT_ROOT / "Data" / "indexes" / "phase1-v2" / "parents"),
    )
    synthesis.add_argument(
        "--benchmark", default=str(PROJECT_ROOT / "evals" / "ragas_gold_90.json")
    )
    synthesis.add_argument("--model", default="gpt-4o-mini")
    synthesis.add_argument("--target-count", type=int, default=60)
    synthesis.add_argument("--max-candidates-per-pair", type=int, default=1)
    synthesis.add_argument("--max-chars", type=int, default=8_000)
    synthesis.add_argument(
        "--exclude-dataset",
        help="Existing master JSONL whose IDs, groups, and questions must not be regenerated.",
    )
    synthesis.add_argument("--resume", action="store_true")
    synthesis.add_argument("--output", default=str(DEFAULT_WORK / "synthesis_candidates.jsonl"))
    synthesis.add_argument("--review-csv", default=str(DEFAULT_WORK / "synthesis_review.csv"))
    synthesis.add_argument("--report", default=str(DEFAULT_WORK / "synthesis_report.json"))
    synthesis.set_defaults(handler=run_generate_synthesis)

    ai_review = subparsers.add_parser(
        "ai-review",
        help="Create transparent AI-assisted review recommendations for the combined dataset.",
    )
    ai_review.add_argument(
        "--dataset", default=str(DEFAULT_WORK / "combined_master.jsonl")
    )
    ai_review.add_argument(
        "--source-review-csv", default=str(DEFAULT_WORK / "combined_review.csv")
    )
    ai_review.add_argument(
        "--decisions", default=str(DEFAULT_WORK / "ai_review_decisions.jsonl")
    )
    ai_review.add_argument(
        "--output-review-csv", default=str(DEFAULT_WORK / "combined_review_ai.csv")
    )
    ai_review.add_argument("--model", default="gpt-4o-mini")
    ai_review.add_argument("--batch-size", type=int, default=2)
    ai_review.add_argument("--minimum-confidence", type=float, default=0.90)
    ai_review.add_argument("--resume", action="store_true")
    ai_review.set_defaults(handler=run_ai_review)

    normalize = subparsers.add_parser(
        "normalize-supervision",
        help="Repair mojibake in supervised questions and targets, then revalidate.",
    )
    normalize.add_argument(
        "--dataset", default=str(DEFAULT_WORK / "reviewed_ai_master.jsonl")
    )
    normalize.add_argument(
        "--benchmark", default=str(PROJECT_ROOT / "evals" / "ragas_gold_90.json")
    )
    normalize.add_argument(
        "--output", default=str(DEFAULT_WORK / "reviewed_ai_normalized_master.jsonl")
    )
    normalize.add_argument(
        "--report", default=str(DEFAULT_WORK / "encoding_normalization_report.json")
    )
    normalize.set_defaults(handler=run_normalize_supervision)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for field in (
        "direct_limit", "yoy_limit", "targeted_table_limit", "comparison_limit",
        "complete_table_limit", "calculation_count", "safety_count",
    ):
        if hasattr(args, field) and getattr(args, field) < 0:
            raise ValueError(f"--{field.replace('_', '-')} cannot be negative")
    return args.handler(args)


def entrypoint() -> int:
    """Console boundary: present expected gate failures without a traceback."""
    try:
        return main()
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
