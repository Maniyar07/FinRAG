"""Request-local tools and verified state for the LangChain research agent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.financial.calculator import CalculationOperation, CalculationRequest
from src.financial.models import ValidatedFinancialFact
from src.orchestration.calculation_runner import (
    ExecutedCalculation,
    calculation_as_fact,
)
from src.orchestration.evidence_merger import EvidenceMerger
from src.orchestration.executor import MultiHopExecutionResult, MultiHopExecutor
from src.orchestration.fact_selection import (
    _ordered_calculation_facts,
    _select_fact,
)
from src.orchestration.models import (
    CalculationReference,
    DOCUMENT_TYPES,
    EvidenceGroup,
    EvidenceRequirement,
    EvidenceType,
    MultiHopPlan,
    REQUIREMENT_ID_RE,
)
from src.orchestration.plan_payload import CALCULATION_INTENT_RE
from src.schemas import Scope
from src.tools.financial_calculator import FinancialCalculatorTool


class AgentInputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearchToolInput(AgentInputModel):
    requirement_id: str
    query: str = Field(min_length=3, max_length=1_000)
    evidence_type: EvidenceType
    document_type: str
    groups: tuple[EvidenceGroup, ...] = Field(min_length=1, max_length=18)
    requires_complete_coverage: bool = True

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str) -> str:
        if not REQUIREMENT_ID_RE.fullmatch(value):
            raise ValueError("requirement_id must be a lowercase identifier")
        return value

    @field_validator("document_type", mode="before")
    @classmethod
    def normalize_document_type(cls, value: object) -> str:
        document_type = str(value).strip().upper().replace("-", "")
        if document_type not in DOCUMENT_TYPES:
            raise ValueError("document_type must be 10K or TRANSCRIPT")
        return document_type

    @model_validator(mode="after")
    def validate_unique_groups(self) -> "ResearchToolInput":
        keys = [group.key for group in self.groups]
        if len(keys) != len(set(keys)):
            raise ValueError("groups must not contain duplicates")
        return self

    def to_requirement(self) -> EvidenceRequirement:
        return EvidenceRequirement(
            requirement_id=self.requirement_id,
            question=self.query,
            evidence_type=self.evidence_type,
            document_type=self.document_type,
            groups=self.groups,
            requires_complete_coverage=self.requires_complete_coverage,
        )


class CalculatorToolInput(AgentInputModel):
    calculation_id: str
    label: str = Field(min_length=3, max_length=300)
    operation: CalculationOperation
    input_fact_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Ordered validated fact IDs or previously completed calculation IDs."
        ),
    )
    periods: Decimal | None = Field(default=None, gt=0)

    @field_validator("calculation_id")
    @classmethod
    def validate_calculation_id(cls, value: str) -> str:
        if not REQUIREMENT_ID_RE.fullmatch(value):
            raise ValueError("calculation_id must be a lowercase identifier")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> "CalculatorToolInput":
        CalculationRequest(
            operation=self.operation,
            input_fact_ids=self.input_fact_ids,
            periods=self.periods,
        )
        return self


class CompleteResearchInput(AgentInputModel):
    """The terminal tool intentionally accepts no model-supplied data."""


def _allowed_document_types(scope: Scope) -> frozenset[str]:
    if scope.required_doc_types:
        return frozenset(scope.required_doc_types)
    if scope.doc_type:
        return frozenset((scope.doc_type,))
    return DOCUMENT_TYPES


def _canonical_fact_id(fact: ValidatedFinancialFact, source: dict) -> str:
    identity = "|".join(
        (
            str(source.get("source_hash") or ""),
            str(source.get("parent_id") or ""),
            fact.ticker,
            fact.metric.casefold(),
            fact.period.casefold(),
            fact.raw_value,
            (fact.row_label or "").casefold(),
            (fact.column_label or "").casefold(),
        )
    )
    return "F" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _safe_error(error: Exception) -> str:
    return " ".join(str(error).split())[:300] or type(error).__name__


@dataclass
class AgentResearchSession:
    """Request-local state shared only by this agent's two approved tools."""

    question: str
    permitted_scope: Scope
    executor: MultiHopExecutor
    calculator: FinancialCalculatorTool
    max_actions: int
    expected_plan: MultiHopPlan | None = None
    merger: EvidenceMerger = field(default_factory=EvidenceMerger)
    facts: dict[str, ValidatedFinancialFact] = field(default_factory=dict)
    calculations: list[ExecutedCalculation] = field(default_factory=list)
    requirements: list[EvidenceRequirement] = field(default_factory=list)
    requirement_traces: list[dict] = field(default_factory=list)
    calculation_traces: list[dict] = field(default_factory=list)
    completed_searches: dict[str, dict] = field(default_factory=dict)
    completed_calculations: dict[str, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    action_count: int = 0

    def _reserve_action(self, tool: str, arguments: dict) -> dict:
        if self.action_count >= self.max_actions:
            raise ToolException(
                f"Agent action limit ({self.max_actions}) reached; stop and report "
                "the available evidence."
            )
        self.action_count += 1
        event = {
            "step": self.action_count,
            "action": {"tool": tool, "arguments": arguments},
            "observation": {"status": "running"},
        }
        self.events.append(event)
        return event

    def _validate_requirement_scope(self, requirement: EvidenceRequirement) -> None:
        requested = {group.key for group in requirement.groups}
        permitted = set(self.permitted_scope.groups)
        if not requested.issubset(permitted):
            raise ValueError("search groups exceed the user-approved scope")
        if requirement.document_type not in _allowed_document_types(
            self.permitted_scope
        ):
            raise ValueError("document type exceeds the user-approved scope")

    def _expected_requirement(self, payload: ResearchToolInput) -> EvidenceRequirement:
        if self.expected_plan is None:
            return payload.to_requirement()
        expected = next(
            (
                item for item in self.expected_plan.requirements
                if item.requirement_id == payload.requirement_id
            ),
            None,
        )
        if expected is None:
            raise ValueError("requirement_id is not present in the typed checklist")
        if (
            payload.evidence_type != expected.evidence_type
            or payload.document_type != expected.document_type
            or tuple(group.key for group in payload.groups)
            != tuple(group.key for group in expected.groups)
            or payload.requires_complete_coverage
            != expected.requires_complete_coverage
        ):
            raise ValueError("search arguments do not match the typed checklist")
        return expected

    def search(self, payload: ResearchToolInput) -> str:
        cached = self.completed_searches.get(payload.requirement_id)
        if cached is not None:
            return json.dumps(
                {**cached, "already_completed": True},
                ensure_ascii=True,
                sort_keys=True,
            )
        event = self._reserve_action(
            "search_financial_documents",
            {
                "requirement_id": payload.requirement_id,
                "evidence_type": payload.evidence_type.value,
                "document_type": payload.document_type,
                "groups": [list(group.key) for group in payload.groups],
            },
        )
        try:
            if any(
                item.requirement_id == payload.requirement_id
                for item in self.requirements
            ):
                raise ValueError("requirement_id has already been used")
            requirement = self._expected_requirement(payload)
            self._validate_requirement_scope(requirement)
            plan = MultiHopPlan(
                original_question=self.question,
                requirements=(requirement,),
                calculations=(),
            )
            execution = self.executor.execute(
                plan,
                permitted_scope=self.permitted_scope,
            )

            original_sources = list(execution.bundle.sources)
            merged_sources = self.merger.add(execution.bundle)
            source_ids = {
                str(original.get("id")): str(merged.get("id"))
                for original, merged in zip(original_sources, merged_sources)
            }
            merged_by_id = {
                str(source.get("id")): source for source in self.merger.sources
            }
            role = {
                "requirement_id": requirement.requirement_id,
                "evidence_type": requirement.evidence_type.value,
                "document_type": requirement.document_type,
            }
            for merged_id in source_ids.values():
                source = merged_by_id.get(merged_id)
                if source is None:
                    continue
                roles = list(source.get("evidence_requirements") or [])
                if role not in roles:
                    roles.append(role)
                    source["evidence_requirements"] = roles

            facts: list[ValidatedFinancialFact] = []
            for incoming in execution.facts:
                merged_source_id = source_ids.get(incoming.source_id)
                source = merged_by_id.get(str(merged_source_id))
                if source is None:
                    continue
                fact = incoming.model_copy(
                    update={
                        "source_id": merged_source_id,
                        "fact_id": _canonical_fact_id(incoming, source),
                    }
                )
                self.facts[fact.fact_id] = fact
                facts.append(fact)

            raw_trace = next(
                iter((execution.trace or {}).get("requirements", ())), {}
            )
            trace = {
                **raw_trace,
                "requirement_id": requirement.requirement_id,
                "source_ids": list(dict.fromkeys(source_ids.values())),
                "fact_ids": [fact.fact_id for fact in facts],
                "facts": [
                    {
                        "fact_id": fact.fact_id,
                        "ticker": fact.ticker,
                        "metric": fact.metric,
                        "period": fact.period,
                        "raw_value": fact.raw_value,
                        "normalized_unit": fact.normalized_unit,
                        "currency": fact.currency,
                        "source_id": fact.source_id,
                        "row_label": fact.row_label,
                        "column_label": fact.column_label,
                    }
                    for fact in facts
                ],
            }
            self.requirements.append(requirement)
            self.requirement_traces.append(trace)
            missing_groups = list(raw_trace.get("missing_groups", ()))
            status = (
                "insufficient_evidence"
                if missing_groups or not trace["source_ids"]
                or (requirement.evidence_type == EvidenceType.NUMERIC and not facts)
                else "completed"
            )
            observation = {
                "status": status,
                "requirement_id": requirement.requirement_id,
                "retrieval_mode": raw_trace.get("retrieval_mode", "hybrid"),
                "candidate_count": int(raw_trace.get("candidate_count", 0) or 0),
                "source_ids": trace["source_ids"],
                "fact_ids": trace["fact_ids"],
                "facts": trace["facts"],
                "missing_groups": missing_groups,
                "issues": list(execution.issues),
            }
        except Exception as error:
            observation = {
                "status": "rejected",
                "requirement_id": payload.requirement_id,
                "error": _safe_error(error),
            }
        event["observation"] = observation
        if observation["status"] == "completed":
            self.completed_searches[payload.requirement_id] = observation
        return json.dumps(observation, ensure_ascii=True, sort_keys=True)

    def calculate(self, payload: CalculatorToolInput) -> str:
        cached = self.completed_calculations.get(payload.calculation_id)
        if cached is not None:
            return json.dumps(
                {**cached, "already_completed": True},
                ensure_ascii=True,
                sort_keys=True,
            )
        event = self._reserve_action(
            "financial_calculator",
            {
                "calculation_id": payload.calculation_id,
                "operation": payload.operation.value,
                "input_fact_ids": list(payload.input_fact_ids),
            },
        )
        try:
            if any(
                item.calculation_id == payload.calculation_id
                for item in self.calculations
            ):
                raise ValueError("calculation_id has already been used")
            expected = None
            if self.expected_plan is not None:
                expected = next(
                    (
                        item for item in self.expected_plan.calculations
                        if item.calculation_id == payload.calculation_id
                    ),
                    None,
                )
                if expected is None:
                    raise ValueError(
                        "calculation_id is not present in the typed checklist"
                    )
                if payload.operation != expected.operation:
                    raise ValueError(
                        "calculation operation does not match the typed checklist"
                    )

            available_facts = dict(self.facts)
            upstream_sources: list[str] = []
            input_ids = payload.input_fact_ids
            if expected is not None:
                source_map = {
                    str(source.get("id")): source
                    # Use every source already collected. bundle() deliberately
                    # narrows evidence to completed calculations, which would
                    # hide sources needed by the next calculation in a staged
                    # or cross-company plan.
                    for source in self.merger.sources
                }
                facts_by_requirement = {
                    str(trace.get("requirement_id")): tuple(
                        self.facts[fact_id]
                        for fact_id in trace.get("fact_ids", ())
                        if fact_id in self.facts
                    )
                    for trace in self.requirement_traces
                }
                selected = []
                for reference in expected.inputs:
                    if isinstance(reference, CalculationReference):
                        prior = next(
                            (
                                item for item in self.calculations
                                if item.calculation_id == reference.calculation_id
                            ),
                            None,
                        )
                        if prior is None:
                            raise ValueError("referenced calculation is unavailable")
                        fact = calculation_as_fact(prior)
                        available_facts[fact.fact_id] = fact
                        upstream_sources.extend(prior.result.source_ids)
                    else:
                        fact = _select_fact(
                            reference, facts_by_requirement, source_map
                        )
                        upstream_sources.append(fact.source_id)
                    selected.append(fact)
                selected = list(
                    _ordered_calculation_facts(expected, tuple(selected))
                )
                input_ids = tuple(fact.fact_id for fact in selected)

            for input_id in input_ids:
                if input_id in available_facts:
                    if available_facts[input_id].source_id not in upstream_sources:
                        upstream_sources.append(available_facts[input_id].source_id)
                    continue
                prior = next(
                    (
                        item for item in self.calculations
                        if item.calculation_id == input_id
                    ),
                    None,
                )
                if prior is None:
                    raise ValueError(f"Unknown fact or calculation ID: {input_id}")
                available_facts[input_id] = calculation_as_fact(prior)
                upstream_sources.extend(prior.result.source_ids)
            request = CalculationRequest(
                operation=expected.operation if expected is not None else payload.operation,
                input_fact_ids=input_ids,
                periods=expected.periods if expected is not None else payload.periods,
            )
            result = self.calculator.execute(
                request,
                available_facts=available_facts,
            )
            result = result.model_copy(
                update={
                    "source_ids": tuple(dict.fromkeys(upstream_sources)),
                }
            )
            completed = ExecutedCalculation(
                calculation_id=payload.calculation_id,
                label=expected.label if expected is not None else payload.label,
                result=result,
            )
            self.calculations.append(completed)
            observation = {
                "status": "completed",
                "calculation_id": payload.calculation_id,
                "operation": result.operation.value,
                "input_fact_ids": list(result.input_fact_ids),
                "source_ids": list(result.source_ids),
                "result": str(result.result),
                "result_unit": result.result_unit,
            }
            self.calculation_traces.append(observation)
        except Exception as error:
            observation = {
                "status": "rejected",
                "calculation_id": payload.calculation_id,
                "error": _safe_error(error),
            }
            self.calculation_traces.append(observation)
        event["observation"] = observation
        if observation["status"] == "completed":
            self.completed_calculations[payload.calculation_id] = observation
        return json.dumps(observation, ensure_ascii=True, sort_keys=True)

    def complete(self) -> str:
        event = self._reserve_action("complete_research", {})
        issues = self.final_issues()
        observation = {
            "status": "completed" if not issues else "insufficient_evidence",
            "issues": list(issues),
            "source_ids": [source["id"] for source in self.bundle().sources],
            "calculation_ids": [item.calculation_id for item in self.calculations],
        }
        event["observation"] = observation
        return json.dumps(observation, ensure_ascii=True, sort_keys=True)

    def complete_pending_calculations(self) -> int:
        """Deterministically finish typed calculations after all searches succeed."""
        if self.expected_plan is None:
            return 0
        completed_requirements = set(self.completed_searches)
        if any(
            item.requirement_id not in completed_requirements
            for item in self.expected_plan.requirements
        ):
            return 0
        completed_ids = {item.calculation_id for item in self.calculations}
        recovered = 0
        for calculation in self.expected_plan.calculations:
            if calculation.calculation_id in completed_ids:
                continue
            if self.action_count >= self.max_actions:
                break
            placeholders = tuple(
                (
                    value.calculation_id
                    if isinstance(value, CalculationReference)
                    else f"planned_input_{index}"
                )
                for index, value in enumerate(calculation.inputs, start=1)
            )
            observation = json.loads(
                self.calculate(
                    CalculatorToolInput(
                        calculation_id=calculation.calculation_id,
                        label=calculation.label,
                        operation=calculation.operation,
                        input_fact_ids=placeholders,
                        periods=calculation.periods,
                    )
                )
            )
            if observation.get("status") != "completed":
                break
            completed_ids.add(calculation.calculation_id)
            recovered += 1
        return recovered

    def search_from_kwargs(self, **kwargs: Any) -> str:
        return self.search(ResearchToolInput.model_validate(kwargs))

    def calculate_from_kwargs(self, **kwargs: Any) -> str:
        return self.calculate(CalculatorToolInput.model_validate(kwargs))

    def complete_from_kwargs(self, **kwargs: Any) -> str:
        CompleteResearchInput.model_validate(kwargs)
        return self.complete()

    def tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                name="search_financial_documents",
                description=(
                    "Search one focused in-scope 10-K or transcript evidence need. "
                    "Numeric searches return validated fact IDs for the calculator."
                ),
                func=self.search_from_kwargs,
                args_schema=ResearchToolInput,
                handle_tool_error=True,
            ),
            StructuredTool.from_function(
                name="financial_calculator",
                description=(
                    "Perform deterministic financial arithmetic using only validated "
                    "fact IDs returned by search_financial_documents."
                ),
                func=self.calculate_from_kwargs,
                args_schema=CalculatorToolInput,
                handle_tool_error=True,
            ),
            StructuredTool.from_function(
                name="complete_research",
                description=(
                    "Finish the research loop after every requested part has been "
                    "attempted. The application independently checks completeness."
                ),
                func=self.complete_from_kwargs,
                args_schema=CompleteResearchInput,
                handle_tool_error=True,
                return_direct=True,
            ),
        ]

    def bundle(self):
        return self.merger.build(
            scope=self.permitted_scope,
            facts=tuple(self.facts.values()),
            calculations=tuple(self.calculations),
        )

    def final_issues(self) -> tuple[str, ...]:
        bundle = self.bundle()
        issues: list[str] = []
        if not bundle.sources or not bundle.context:
            issues.append("no_verified_evidence")
        covered = {
            (
                str(source.get("ticker") or ""),
                str(source.get("fiscal_year") or ""),
                str(source.get("doc_type") or "").upper().replace("-", ""),
            )
            for source in bundle.sources
        }
        expected_groups = (
            (
                (group.ticker, group.fiscal_year, requirement.document_type)
                for requirement in self.expected_plan.requirements
                for group in requirement.groups
            )
            if self.expected_plan is not None
            else iter(self.permitted_scope.retrieval_groups)
        )
        missing = []
        for ticker, year, document_type in expected_groups:
            group_is_covered = any(
                actual_ticker == ticker
                and actual_year == year
                and (document_type is None or actual_type == document_type)
                for actual_ticker, actual_year, actual_type in covered
            )
            if not group_is_covered:
                missing.append((ticker, year, document_type))
        if missing:
            issues.append(
                "missing_evidence_groups:"
                + ",".join(f"{ticker}-{year}-{doc_type or 'ANY'}" for ticker, year, doc_type in missing)
            )
        if CALCULATION_INTENT_RE.search(self.question) and not self.calculations:
            issues.append("requested_calculation_not_completed")
        if self.expected_plan is not None:
            completed_requirements = {
                item.requirement_id for item in self.requirements
            }
            completed_calculations = {
                item.calculation_id for item in self.calculations
            }
            for item in self.expected_plan.requirements:
                if item.requirement_id not in completed_requirements:
                    issues.append(
                        f"checklist_requirement_not_completed:{item.requirement_id}"
                    )
            for item in self.expected_plan.calculations:
                if item.calculation_id not in completed_calculations:
                    issues.append(
                        f"checklist_calculation_not_completed:{item.calculation_id}"
                    )
        return tuple(issues)

    def execution(self) -> MultiHopExecutionResult:
        issues = self.final_issues()
        return MultiHopExecutionResult(
            bundle=self.bundle(),
            requirement_results=(),
            facts=tuple(self.facts.values()),
            calculations=tuple(self.calculations),
            issues=issues,
            trace={
                "requirements": list(self.requirement_traces),
                "calculations": list(self.calculation_traces),
                "merged_source_ids": [
                    source["id"] for source in self.bundle().sources
                ],
                "candidate_count": self.merger.candidate_count,
            },
        )
