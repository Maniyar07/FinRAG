"""Safe MLflow implementation of the FinRAG experiment-tracking contract."""

from __future__ import annotations

import importlib
import json
import math
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator, Mapping

from .base import MetricValue, ParameterValue, RunInfo


SAFE_ARTIFACT_SUFFIXES = frozenset({".json", ".md", ".txt"})
BLOCKED_ARTIFACT_SUFFIXES = frozenset({".csv", ".jsonl", ".parquet"})


def _load_mlflow() -> ModuleType:
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow is not installed. Install the isolated tracking dependencies with "
            "`python -m pip install -r fine_tuning/tracking/requirements.txt`."
        ) from exc


def _stringify(value: object, *, limit: int = 5_900) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif value is None:
        rendered = "null"
    elif isinstance(value, (dict, list, tuple, set)):
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        rendered = str(value)
    if len(rendered) > limit:
        raise ValueError(f"Tracking value exceeds the safe {limit}-character limit")
    return rendered


def _validate_key(key: object) -> str:
    rendered = str(key).strip()
    if not rendered:
        raise ValueError("Tracking keys cannot be empty")
    if len(rendered) > 250:
        raise ValueError(f"Tracking key is too long: {rendered[:40]!r}")
    return rendered


class MlflowRunSession:
    """Run-bound operations implemented through ``MlflowClient``."""

    def __init__(
        self,
        *,
        client: object,
        info: RunInfo,
        max_artifact_bytes: int,
    ) -> None:
        self._client = client
        self._info = info
        self._max_artifact_bytes = max_artifact_bytes

    @property
    def info(self) -> RunInfo:
        return self._info

    def log_params(self, values: Mapping[str, object]) -> None:
        for key, value in values.items():
            self._client.log_param(
                self._info.run_id,
                _validate_key(key),
                _stringify(value),
            )

    def log_metrics(
        self,
        values: Mapping[str, MetricValue],
        *,
        step: int | None = None,
    ) -> None:
        metric_step = 0 if step is None else int(step)
        if metric_step < 0:
            raise ValueError("Metric step cannot be negative")
        for key, value in values.items():
            metric_key = _validate_key(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Metric {metric_key!r} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Metric {metric_key!r} must be finite")
            self._client.log_metric(
                self._info.run_id,
                metric_key,
                number,
                step=metric_step,
            )

    def set_tags(self, values: Mapping[str, object]) -> None:
        for key, value in values.items():
            self._client.set_tag(
                self._info.run_id,
                _validate_key(key),
                _stringify(value),
            )

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        source = path.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Tracking artifact does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix in BLOCKED_ARTIFACT_SUFFIXES or suffix not in SAFE_ARTIFACT_SUFFIXES:
            raise ValueError(
                f"Refusing artifact {source.name!r}. Only summary JSON, Markdown, and text "
                "files are allowed; raw row-level datasets/reviews are excluded."
            )
        size = source.stat().st_size
        if size > self._max_artifact_bytes:
            raise ValueError(
                f"Artifact {source.name!r} is {size:,} bytes; maximum is "
                f"{self._max_artifact_bytes:,} bytes"
            )
        self._client.log_artifact(
            self._info.run_id,
            str(source),
            artifact_path=artifact_path,
        )


class MlflowTracker:
    """Local or remote MLflow tracker with bounded, explicit artifact logging."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        artifact_location: str | None = None,
        max_artifact_bytes: int = 2_000_000,
    ) -> None:
        if not tracking_uri.strip():
            raise ValueError("tracking_uri is required")
        if not experiment_name.strip():
            raise ValueError("experiment_name is required")
        if max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive")

        self._mlflow = _load_mlflow()
        self._mlflow.set_tracking_uri(tracking_uri)
        self._client = self._mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = self._client.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
                tags={"finrag.managed_by": "fine_tuning.tracking"},
            )
        else:
            experiment_id = experiment.experiment_id

        self._tracking_uri = tracking_uri
        self._experiment_id = str(experiment_id)
        self._max_artifact_bytes = max_artifact_bytes

    @property
    def tracking_uri(self) -> str:
        return self._tracking_uri

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    def find_run_id(self, required_tags: Mapping[str, str]) -> str | None:
        """Return the newest non-deleted run with every requested exact tag."""

        if not required_tags:
            raise ValueError("At least one required tag is needed for run lookup")
        runs = self._client.search_runs(
            experiment_ids=[self._experiment_id],
            run_view_type=1,
            max_results=1_000,
            order_by=["attributes.start_time DESC"],
        )
        expected = {str(key): str(value) for key, value in required_tags.items()}
        for run in runs:
            if run.info.status == "FINISHED" and all(
                run.data.tags.get(key) == value for key, value in expected.items()
            ):
                return str(run.info.run_id)
        return None

    @contextmanager
    def start_run(
        self,
        run_name: str,
        *,
        tags: Mapping[str, ParameterValue | object] | None = None,
    ) -> Iterator[MlflowRunSession]:
        run_tags = {
            _validate_key(key): _stringify(value)
            for key, value in (tags or {}).items()
        }
        run_tags["mlflow.runName"] = run_name
        run = self._client.create_run(self._experiment_id, tags=run_tags)
        info = RunInfo(
            run_id=str(run.info.run_id),
            run_name=run_name,
            artifact_uri=str(run.info.artifact_uri),
            tracking_uri=self._tracking_uri,
        )
        session = MlflowRunSession(
            client=self._client,
            info=info,
            max_artifact_bytes=self._max_artifact_bytes,
        )
        try:
            yield session
        except BaseException as exc:
            try:
                session.set_tags(
                    {
                        "finrag.run_status": "failed",
                        "finrag.failure_type": type(exc).__name__,
                    }
                )
                self._client.set_terminated(info.run_id, status="FAILED")
            except Exception:
                # Preserve the original training/tracking failure if the backend
                # itself becomes unreachable while recording failure metadata.
                pass
            raise
        else:
            session.set_tags({"finrag.run_status": "finished"})
            self._client.set_terminated(info.run_id, status="FINISHED")
