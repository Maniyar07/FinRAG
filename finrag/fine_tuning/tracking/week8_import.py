"""Import completed FinRAG Week-8 results into an organized MLflow experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .base import ExperimentTracker


BASE_MODEL_FALLBACK = "Qwen/Qwen2.5-7B-Instruct"
EXPERIMENT_DEFAULT = "finrag-week8-lora"


@dataclass(frozen=True)
class MetricPoint:
    key: str
    value: float
    step: int


@dataclass(frozen=True)
class RunSpec:
    run_key: str
    run_name: str
    tags: Mapping[str, object]
    params: Mapping[str, object]
    metrics: Mapping[str, int | float]
    metric_history: tuple[MetricPoint, ...] = ()
    artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ImportPlan:
    import_id: str
    dataset_sha256: str
    base_model: str
    adapter_model: str
    run_specs: tuple[RunSpec, ...]
    source_files: tuple[Path, ...]
    lineage_warnings: tuple[str, ...] = ()


@dataclass
class ImportResult:
    run_key: str
    run_name: str
    action: str
    run_id: str | None
    artifact_uri: str | None = None


def load_json_object(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError:
        raise FileNotFoundError(f"Required tracking source is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _single_match(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        names = ", ".join(path.name for path in matches) or "none"
        raise ValueError(f"Expected exactly one {label} file matching {pattern!r}; found {names}")
    return matches[0]


def _require_schema(document: Mapping[str, object], expected: str, path: Path) -> None:
    actual = document.get("schema_version")
    if actual != expected:
        raise ValueError(f"{path.name}: expected schema {expected!r}, found {actual!r}")


def _sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_git_state(project_root: Path) -> dict[str, str]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current") or "detached"
        dirty = bool(run("status", "--porcelain"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "branch": "unavailable", "dirty": "unknown"}
    return {"commit": commit, "branch": branch, "dirty": str(dirty).lower()}


def _common_tags(
    *,
    import_id: str,
    dataset_sha256: str,
    git_state: Mapping[str, str],
    run_kind: str,
    run_key: str,
) -> dict[str, object]:
    return {
        "finrag.task": "T28",
        "finrag.project": "FinRAG",
        "finrag.source": "historical_import",
        "finrag.import_id": import_id,
        "finrag.run_group": f"week8-{dataset_sha256[:12]}",
        "finrag.run_key": run_key,
        "finrag.run_kind": run_kind,
        "finrag.dataset_sha256": dataset_sha256,
        "finrag.import_git_commit": git_state["commit"],
        "finrag.import_git_branch": git_state["branch"],
        "finrag.import_worktree_dirty": git_state["dirty"],
        "finrag.source_git_commit": "not_recorded_during_colab_run",
        "finrag.lineage_quality": "partial_historical",
    }


def _quality_metrics(
    model_summary: Mapping[str, object],
    preferences: Mapping[str, object],
    model: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name in (
        "factual_correct",
        "complete",
        "numeric_correct",
        "evidence_supported",
        "hallucination",
        "overall_acceptable",
    ):
        values = model_summary.get(metric_name)
        if not isinstance(values, Mapping):
            raise ValueError(f"Review summary is missing metric {metric_name!r}")
        for field_name in ("yes", "no", "na", "applicable", "yes_rate_percent"):
            value = values.get(field_name)
            if value is not None:
                metrics[f"review.{metric_name}.{field_name}"] = float(value)
    metrics["review.preference.model"] = float(preferences.get(model, 0))
    metrics["review.preference.tie"] = float(preferences.get("tie", 0))
    return metrics


def _serving_metrics(summary: Mapping[str, object]) -> dict[str, float]:
    numeric_fields = (
        "requests",
        "successful_requests",
        "errors",
        "wall_seconds",
        "latency_p50_ms",
        "latency_p95_ms",
        "requests_per_second",
        "successful_requests_per_second",
        "output_tokens_per_second",
        "source_id_checks_applicable",
    )
    metrics = {f"serving.{key}": float(summary[key]) for key in numeric_fields}
    format_counts = summary.get("format_counts")
    if not isinstance(format_counts, Mapping):
        raise ValueError("Serving summary is missing format_counts")
    requests = float(summary["requests"])
    applicable = float(summary["source_id_checks_applicable"])
    for key, value in format_counts.items():
        count = float(value)
        metrics[f"format.{key}.count"] = count
        denominator = applicable if str(key).startswith("source_ids_") else requests
        metrics[f"format.{key}.rate"] = count / denominator if denominator else 0.0
    return metrics


def build_import_plan(
    *,
    reports_dir: Path,
    project_root: Path,
    concurrency_1_path: Path | None = None,
    concurrency_2_path: Path | None = None,
) -> ImportPlan:
    reports_dir = reports_dir.resolve()
    training_path = reports_dir / "training_run_manifest.json"
    review_path = reports_dir / "human_review_summary_ai_assisted.json"
    review_report_path = reports_dir / "T26_AI_ASSISTED_REVIEW_REPORT.md"
    c1_path = (
        concurrency_1_path
        or _single_match(
            reports_dir,
            "t27_concurrency1_*.summary.json",
            "concurrency-one summary",
        )
    ).resolve()
    c2_path = (
        concurrency_2_path
        or _single_match(
            reports_dir,
            "t27_concurrency2_*.summary.json",
            "concurrency-two summary",
        )
    ).resolve()
    source_files = (training_path, review_path, review_report_path, c1_path, c2_path)
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required tracking source is missing: {path}")

    training = load_json_object(training_path)
    review = load_json_object(review_path)
    c1 = load_json_object(c1_path)
    c2 = load_json_object(c2_path)
    _require_schema(training, "finrag-training-run-v1", training_path)
    _require_schema(review, "finrag-ai-assisted-review-v1", review_path)
    _require_schema(c1, "finrag-serving-benchmark-v2", c1_path)
    _require_schema(c2, "finrag-serving-benchmark-v2", c2_path)
    if c1.get("concurrency") != 1 or c2.get("concurrency") != 2:
        raise ValueError("Serving summaries must represent concurrency one and two respectively")

    dataset = training.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("Training manifest is missing dataset metadata")
    dataset_sha256 = str(dataset.get("test_sha256", ""))
    hashes = {
        dataset_sha256,
        str(review.get("dataset_sha256", "")),
        str(c1.get("dataset_sha256", "")),
        str(c2.get("dataset_sha256", "")),
    }
    if "" in hashes or len(hashes) != 1:
        raise ValueError(f"Dataset lineage mismatch across Week-8 artifacts: {sorted(hashes)}")

    base_model = str(training.get("base_model_requested") or BASE_MODEL_FALLBACK)
    review_models = review.get("models")
    c1_models = c1.get("models")
    c2_models = c2.get("models")
    if not all(isinstance(value, Mapping) for value in (review_models, c1_models, c2_models)):
        raise ValueError("Quality and serving summaries must contain model mappings")
    model_keys = set(review_models)
    if model_keys != set(c1_models) or model_keys != set(c2_models):
        raise ValueError("Model names differ between quality and serving summaries")
    if base_model not in model_keys or len(model_keys) != 2:
        raise ValueError(f"Expected base model {base_model!r} plus exactly one adapter model")
    adapter_model = next(model for model in model_keys if model != base_model)

    import_id = _sha256_files(source_files)
    git_state = read_git_state(project_root.resolve())
    run_specs: list[RunSpec] = []

    training_config = training.get("training")
    lora_config = training.get("lora")
    results = training.get("results")
    adapter = training.get("adapter")
    if not all(isinstance(value, Mapping) for value in (training_config, lora_config, results, adapter)):
        raise ValueError("Training manifest is missing training, LoRA, results, or adapter metadata")
    training_params: dict[str, object] = {
        "method": training["method"],
        "base_model.requested": base_model,
        "base_model.training_checkpoint": training["adapter_config_base"],
        "hardware": training["hardware"],
        "dataset.sha256": dataset_sha256,
        **{f"dataset.{key}": value for key, value in dataset.items()},
        **{f"training.{key}": value for key, value in training_config.items()},
        **{f"lora.{key}": value for key, value in lora_config.items()},
        "adapter.sha256": adapter["sha256"],
        "adapter.size_bytes": adapter["size_bytes"],
    }
    training_key = "training.qlora"
    run_specs.append(
        RunSpec(
            run_key=training_key,
            run_name="01-training-qlora",
            tags={
                **_common_tags(
                    import_id=import_id,
                    dataset_sha256=dataset_sha256,
                    git_state=git_state,
                    run_kind="training",
                    run_key=training_key,
                ),
                "finrag.model_role": "adapter_training",
            },
            params=training_params,
            metrics={
                "training.runtime_seconds": float(results["training_runtime_seconds"]),
                "training.final_loss": float(results["final_training_loss"]),
                "adapter.size_bytes": float(adapter["size_bytes"]),
            },
            metric_history=tuple(
                MetricPoint("validation.loss", float(loss), epoch)
                for epoch, loss in enumerate(results["validation_loss_by_epoch"], start=1)
            ),
            artifacts=(training_path,),
        )
    )

    preferences = review.get("preferences")
    if not isinstance(preferences, Mapping):
        raise ValueError("Review summary is missing preferences")
    for order, (model, role) in enumerate(
        ((base_model, "base"), (adapter_model, "adapter")),
        start=2,
    ):
        run_key = f"quality.{role}"
        model_summary = review_models[model]
        if not isinstance(model_summary, Mapping):
            raise ValueError(f"Review summary for {model!r} is invalid")
        run_specs.append(
            RunSpec(
                run_key=run_key,
                run_name=f"{order:02d}-quality-{role}",
                tags={
                    **_common_tags(
                        import_id=import_id,
                        dataset_sha256=dataset_sha256,
                        git_state=git_state,
                        run_kind="quality_evaluation",
                        run_key=run_key,
                    ),
                    "finrag.model_role": role,
                    "finrag.model_name": model,
                    "finrag.review_type": review["reviewer_type"],
                },
                params={
                    "model.name": model,
                    "model.role": role,
                    "evaluation.reviewed_examples": review["reviewed_examples"],
                    "evaluation.reviewer_type": review["reviewer_type"],
                    "evaluation.reviewer_label": review["reviewer_label"],
                    "evaluation.dataset_sha256": dataset_sha256,
                    "evaluation.prediction_sha256": review["prediction_sha256"],
                },
                metrics=_quality_metrics(model_summary, preferences, model),
                artifacts=(review_path, review_report_path),
            )
        )

    order = 4
    for serving_document, summary_path in ((c1, c1_path), (c2, c2_path)):
        concurrency = int(serving_document["concurrency"])
        model_map = serving_document["models"]
        for model, role in ((base_model, "base"), (adapter_model, "adapter")):
            run_key = f"serving.c{concurrency}.{role}"
            model_summary = model_map[model]
            run_specs.append(
                RunSpec(
                    run_key=run_key,
                    run_name=f"{order:02d}-serving-c{concurrency}-{role}",
                    tags={
                        **_common_tags(
                            import_id=import_id,
                            dataset_sha256=dataset_sha256,
                            git_state=git_state,
                            run_kind="serving_benchmark",
                            run_key=run_key,
                        ),
                        "finrag.model_role": role,
                        "finrag.model_name": model,
                        "finrag.concurrency": concurrency,
                    },
                    params={
                        "model.name": model,
                        "model.role": role,
                        "serving.concurrency": concurrency,
                        "serving.server_max_num_seqs": serving_document["server_max_num_seqs"],
                        "serving.load_mode": serving_document["load_mode"],
                        "serving.max_tokens": serving_document["max_tokens"],
                        "evaluation.dataset_sha256": dataset_sha256,
                    },
                    metrics=_serving_metrics(model_summary),
                    artifacts=(summary_path,),
                )
            )
            order += 1

    return ImportPlan(
        import_id=import_id,
        dataset_sha256=dataset_sha256,
        base_model=base_model,
        adapter_model=adapter_model,
        run_specs=tuple(run_specs),
        source_files=source_files,
        lineage_warnings=(
            "The original Colab run did not record a Git commit; import-time Git state is logged separately.",
        ),
    )


def execute_import(
    *,
    plan: ImportPlan,
    tracker: ExperimentTracker,
    force: bool = False,
) -> list[ImportResult]:
    results: list[ImportResult] = []
    for spec in plan.run_specs:
        identity_tags = {
            "finrag.import_id": plan.import_id,
            "finrag.run_key": spec.run_key,
        }
        existing_run_id = tracker.find_run_id(identity_tags)
        if existing_run_id and not force:
            results.append(
                ImportResult(spec.run_key, spec.run_name, "skipped_existing", existing_run_id)
            )
            continue
        with tracker.start_run(spec.run_name, tags=spec.tags) as run:
            run.log_params(spec.params)
            run.log_metrics(spec.metrics)
            for point in spec.metric_history:
                run.log_metrics({point.key: point.value}, step=point.step)
            for artifact in spec.artifacts:
                run.log_artifact(artifact, artifact_path="verified_summaries")
            results.append(
                ImportResult(
                    spec.run_key,
                    spec.run_name,
                    "created",
                    run.info.run_id,
                    run.info.artifact_uri,
                )
            )
    return results


def _sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _receipt_payload(
    *,
    plan: ImportPlan,
    results: Sequence[ImportResult],
    tracking_uri: str,
    experiment_name: str,
    experiment_id: str | None,
    dry_run: bool,
) -> dict:
    return {
        "schema_version": "finrag-mlflow-import-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "import_id": plan.import_id,
        "dataset_sha256": plan.dataset_sha256,
        "base_model": plan.base_model,
        "adapter_model": plan.adapter_model,
        "lineage_warnings": list(plan.lineage_warnings),
        "source_files": [str(path) for path in plan.source_files],
        "runs": [result.__dict__ for result in results],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser(project_root: Path | None = None) -> argparse.ArgumentParser:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    state_dir = root / "fine_tuning" / "work" / "mlflow"
    parser = argparse.ArgumentParser(
        description="Import verified Week-8 FinRAG training/evaluation summaries into MLflow."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=root / "fine_tuning" / "reports" / "t26_t27",
    )
    parser.add_argument("--concurrency-1-summary", type=Path)
    parser.add_argument("--concurrency-2-summary", type=Path)
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI") or _sqlite_uri(state_dir / "mlflow.db"),
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", EXPERIMENT_DEFAULT),
    )
    parser.add_argument("--artifact-root", type=Path, default=state_dir / "artifacts")
    parser.add_argument(
        "--receipt",
        type=Path,
        default=state_dir / "week8_import_receipt.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create another copy even if the import already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the seven-run plan without importing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    args = build_parser(project_root).parse_args(argv)
    plan = build_import_plan(
        reports_dir=args.reports_dir,
        project_root=project_root,
        concurrency_1_path=args.concurrency_1_summary,
        concurrency_2_path=args.concurrency_2_summary,
    )

    if args.dry_run:
        dry_results = [
            ImportResult(spec.run_key, spec.run_name, "planned", None)
            for spec in plan.run_specs
        ]
        receipt = _receipt_payload(
            plan=plan,
            results=dry_results,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            experiment_id=None,
            dry_run=True,
        )
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 0

    args.artifact_root.mkdir(parents=True, exist_ok=True)
    from .mlflow_tracker import MlflowTracker

    tracker = MlflowTracker(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        artifact_location=args.artifact_root.resolve().as_uri(),
    )
    import_results = execute_import(plan=plan, tracker=tracker, force=args.force)
    receipt = _receipt_payload(
        plan=plan,
        results=import_results,
        tracking_uri=tracker.tracking_uri,
        experiment_name=args.experiment_name,
        experiment_id=tracker.experiment_id,
        dry_run=False,
    )
    _write_json_atomic(args.receipt, receipt)
    created = sum(result.action == "created" for result in import_results)
    skipped = sum(result.action == "skipped_existing" for result in import_results)
    print(f"PASS: MLflow import complete (created={created}, skipped={skipped})")
    print(f"Experiment: {args.experiment_name} (ID {tracker.experiment_id})")
    print(f"Tracking URI: {tracker.tracking_uri}")
    print(f"Receipt: {args.receipt.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
