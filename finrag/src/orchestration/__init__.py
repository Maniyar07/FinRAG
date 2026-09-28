"""Small contracts for bounded multi-step financial research."""

from src.orchestration.executor import (
    ExecutedCalculation,
    MultiHopExecutionResult,
    MultiHopExecutor,
)
from src.orchestration.models import (
    CalculationReference,
    EvidenceGroup,
    EvidenceRequirement,
    EvidenceType,
    FactReference,
    MultiHopPlan,
    PlannedCalculation,
    RequirementResult,
)
from src.orchestration.planner import (
    MultiHopPlanner,
    PlanPayload,
    should_use_multihop,
    validate_plan_scope,
)

__all__ = [
    "CalculationReference",
    "ExecutedCalculation",
    "EvidenceGroup",
    "EvidenceRequirement",
    "EvidenceType",
    "FactReference",
    "MultiHopPlan",
    "MultiHopExecutionResult",
    "MultiHopExecutor",
    "MultiHopPlanner",
    "PlanPayload",
    "PlannedCalculation",
    "RequirementResult",
    "should_use_multihop",
    "validate_plan_scope",
]
