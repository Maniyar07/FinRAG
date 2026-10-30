from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from fine_tuning.tracking.base import RunInfo
from fine_tuning.tracking.mlflow_tracker import MlflowRunSession
from fine_tuning.tracking import week8_import
from fine_tuning.tracking.week8_import import build_import_plan, execute_import


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "fine_tuning" / "reports" / "t26_t27"


class FakeSession:
    def __init__(self, run_name: str, run_id: str) -> None:
        self.info = RunInfo(
            run_id,
            run_name,
            f"file:///artifacts/{run_id}",
            "fake://tracking",
        )
        self.params = {}
        self.metrics = []
        self.tags = {}
        self.artifacts = []

    def log_params(self, values):
        self.params.update(values)

    def log_metrics(self, values, *, step=None):
        self.metrics.append((dict(values), step))

    def set_tags(self, values):
        self.tags.update(values)

    def log_artifact(self, path, *, artifact_path=None):
        self.artifacts.append((Path(path), artifact_path))


class FakeTracker:
    tracking_uri = "fake://tracking"
    experiment_id = "1"

    def __init__(self) -> None:
        self.sessions = []
        self.existing = {}

    def find_run_id(self, required_tags):
        return self.existing.get(tuple(sorted(required_tags.items())))

    @contextmanager
    def start_run(self, run_name, *, tags=None):
        session = FakeSession(run_name, f"run-{len(self.sessions) + 1}")
        session.tags.update(tags or {})
        self.sessions.append(session)
        yield session


def test_week8_plan_is_complete_and_consistent():
    plan = build_import_plan(reports_dir=REPORTS_DIR, project_root=PROJECT_ROOT)

    assert len(plan.run_specs) == 7
    assert len({spec.run_key for spec in plan.run_specs}) == 7
    assert plan.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert plan.adapter_model == "finrag-lora"
    assert plan.dataset_sha256 == (
        "ce5ee27758605e8d7cf37583011b42eccaa9af3a6c365c56e4b6b7866cb43f07"
    )

    adapter_quality = next(
        spec for spec in plan.run_specs if spec.run_key == "quality.adapter"
    )
    assert adapter_quality.metrics["review.overall_acceptable.yes_rate_percent"] == 82.5
    adapter_c2 = next(
        spec for spec in plan.run_specs if spec.run_key == "serving.c2.adapter"
    )
    assert adapter_c2.metrics["format.json_valid.rate"] == 1.0
    assert adapter_c2.metrics["serving.requests_per_second"] == 0.297


def test_import_is_idempotent_and_logs_only_summary_artifacts():
    plan = build_import_plan(reports_dir=REPORTS_DIR, project_root=PROJECT_ROOT)
    tracker = FakeTracker()
    existing_spec = plan.run_specs[0]
    tracker.existing[
        tuple(
            sorted(
                {
                    "finrag.import_id": plan.import_id,
                    "finrag.run_key": existing_spec.run_key,
                }.items()
            )
        )
    ] = "existing-training"

    results = execute_import(plan=plan, tracker=tracker)

    assert results[0].action == "skipped_existing"
    assert len(tracker.sessions) == 6
    assert all(
        artifact.suffix.lower() in {".json", ".md"}
        for session in tracker.sessions
        for artifact, _ in session.artifacts
    )
    assert not any(
        artifact.suffix.lower() in {".csv", ".jsonl"}
        for session in tracker.sessions
        for artifact, _ in session.artifacts
    )


def test_dataset_hash_mismatch_fails_before_tracking(monkeypatch):
    original_loader = week8_import.load_json_object

    def mismatched_loader(path):
        document = original_loader(path)
        if path.name == "human_review_summary_ai_assisted.json":
            document = dict(document)
            document["dataset_sha256"] = "different"
        return document

    monkeypatch.setattr(week8_import, "load_json_object", mismatched_loader)

    with pytest.raises(ValueError, match="Dataset lineage mismatch"):
        build_import_plan(reports_dir=REPORTS_DIR, project_root=PROJECT_ROOT)


class FakeMlflowClient:
    def __init__(self) -> None:
        self.artifacts = []

    def log_artifact(self, run_id, path, artifact_path=None):
        self.artifacts.append((run_id, path, artifact_path))


def test_tracker_blocks_raw_row_level_artifacts():
    raw = REPORTS_DIR / "blind_review_completed_ai_assisted.csv"
    session = MlflowRunSession(
        client=FakeMlflowClient(),
        info=RunInfo("run-1", "test", "file:///artifacts", "fake://tracking"),
        max_artifact_bytes=1_000,
    )

    with pytest.raises(ValueError, match="Refusing artifact"):
        session.log_artifact(raw)
