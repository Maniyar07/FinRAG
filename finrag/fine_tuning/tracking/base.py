"""Backend-neutral experiment-tracking contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Mapping, Protocol, TypeAlias


ParameterValue: TypeAlias = str | int | float | bool
MetricValue: TypeAlias = int | float


@dataclass(frozen=True)
class RunInfo:
    """Stable information returned for a tracked run."""

    run_id: str
    run_name: str
    artifact_uri: str
    tracking_uri: str


class RunSession(Protocol):
    """Operations available while one tracking run is active."""

    @property
    def info(self) -> RunInfo: ...

    def log_params(self, values: Mapping[str, ParameterValue | object]) -> None: ...

    def log_metrics(
        self,
        values: Mapping[str, MetricValue],
        *,
        step: int | None = None,
    ) -> None: ...

    def set_tags(self, values: Mapping[str, ParameterValue | object]) -> None: ...

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None: ...


class ExperimentTracker(Protocol):
    """Minimal interface so training code is not coupled directly to MLflow."""

    @property
    def tracking_uri(self) -> str: ...

    @property
    def experiment_id(self) -> str: ...

    def find_run_id(self, required_tags: Mapping[str, str]) -> str | None: ...

    def start_run(
        self,
        run_name: str,
        *,
        tags: Mapping[str, ParameterValue | object] | None = None,
    ) -> ContextManager[RunSession]: ...
