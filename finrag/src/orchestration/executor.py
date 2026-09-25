"""Direct, bounded execution of multi-hop financial research plans."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from src.financial.fact_pipeline import FinancialFactPipeline
from src.financial.models import ValidatedFinancialFact
from src.orchestration.calculation_runner import (
    ExecutedCalculation,
    run_calculations,
)
from src.orchestration.fact_selection import (
    _fact_matches_metric,
    _fact_matches_period,
    _focused_numeric_sources,
    _numeric_search_instruction,
    _select_fact,
)
from src.orchestration.models import (
    EvidenceGroup,
    EvidenceRequirement,
    EvidenceType,
    FactReference,
    MultiHopPlan,
    RequirementResult,
)
from src.orchestration.evidence_merger import (
    EvidenceMerger as _EvidenceMerger,
    normalize_doc_type as _normalize_doc_type,
)
from src.schemas import RetrievalBundle, Scope
from src.retrieval.structured_lookup import StructuredDocumentLookup, VerifiedTableRow
from src.tools.document_search import (
    DocumentSearchPurpose,
    DocumentSearchRequest,
    DocumentSearchResult,
    DocumentSearchTool,
)
from src.tools.financial_calculator import FinancialCalculatorTool


@dataclass(frozen=True)
class MultiHopExecutionResult:
    """Final evidence plus compact diagnostics; not a mutable workflow state."""

    bundle: RetrievalBundle
    requirement_results: tuple[RequirementResult, ...]
    facts: tuple[ValidatedFinancialFact, ...]
    calculations: tuple[ExecutedCalculation, ...]
    issues: tuple[str, ...]
    trace: dict

    @property
    def complete(self) -> bool:
        return not self.issues and bool(self.bundle.sources and self.bundle.context)


def _purpose(requirement: EvidenceRequirement) -> DocumentSearchPurpose:
    question = requirement.question.casefold()
    if "risk" in question:
        return DocumentSearchPurpose.RISK
    if any(term in question for term in ("guidance", "outlook", "expect")):
        return DocumentSearchPurpose.GUIDANCE
    if any(term in question for term in ("accounting policy", "recognition policy")):
        return DocumentSearchPurpose.ACCOUNTING_POLICY
    if requirement.evidence_type == EvidenceType.NARRATIVE and (
        requirement.document_type == "TRANSCRIPT"
        or any(term in question for term in ("why", "explain", "driver", "reason"))
    ):
        return DocumentSearchPurpose.MANAGEMENT_EXPLANATION
    if len(requirement.groups) > 1:
        return DocumentSearchPurpose.COMPARISON
    return DocumentSearchPurpose.GENERAL


def _missing_groups(
    requirement: EvidenceRequirement,
    sources: list[dict],
) -> tuple[EvidenceGroup, ...]:
    covered = {
        (
            str(source.get("ticker") or "").upper(),
            str(source.get("fiscal_year") or ""),
            _normalize_doc_type(source.get("doc_type")),
        )
        for source in sources
    }
    return tuple(
        group
        for group in requirement.groups
        if (group.ticker, group.fiscal_year, requirement.document_type) not in covered
    )


class MultiHopExecutor:
    """Run planned searches directly; no graph engine or generic tool registry."""

    def __init__(
        self,
        *,
        document_search: DocumentSearchTool,
        fact_pipeline: FinancialFactPipeline | None = None,
        calculator: FinancialCalculatorTool | None = None,
        statement_lookup: StructuredDocumentLookup | None = None,
    ) -> None:
        self.document_search = document_search
        self.fact_pipeline = fact_pipeline or FinancialFactPipeline()
        self.calculator = calculator or FinancialCalculatorTool()
        self.statement_lookup = statement_lookup

    def _complete_statement_rows(
        self,
        requirement: EvidenceRequirement,
        references: tuple[FactReference, ...],
    ) -> tuple[VerifiedTableRow, ...]:
        """Return exact rows only when every planned numeric input is verified."""
        if (
            self.statement_lookup is None
            or requirement.evidence_type != EvidenceType.NUMERIC
            or requirement.document_type != "10K"
            or not references
            or any(not reference.metric_hint for reference in references)
        ):
            return ()

        unique_references = tuple({
            (
                reference.ticker,
                reference.fiscal_year,
                reference.period,
                str(reference.metric_hint),
            ): reference
            for reference in references
        }.values())
        if {group.key for group in requirement.groups} != {
            (reference.ticker, reference.fiscal_year)
            for reference in unique_references
        }:
            return ()

        selected: list[VerifiedTableRow] = []
        seen: set[tuple[str, str, str]] = set()
        for reference in unique_references:
            period_years = re.findall(r"\b(?:19|20)\d{2}\b", reference.period or "")
            if period_years and period_years != [reference.fiscal_year]:
                return ()
            row_scope = Scope(
                tickers=(reference.ticker,),
                years=(reference.fiscal_year,),
                doc_type="10K",
                requested_groups=((reference.ticker, reference.fiscal_year),),
            )
            rows = self.statement_lookup.statement_rows(
                str(reference.metric_hint), row_scope
            )
            if len(rows) != 1:
                return ()
            row = rows[0]
            source = {**row.source, "id": "S1"}
            verified = self.fact_pipeline.recover_exact_table_row(
                metric_hint=str(reference.metric_hint),
                period_year=reference.fiscal_year,
                sources=[source],
                permitted_scope=row_scope,
            )
            if len(verified.valid_facts) != 1:
                return ()
            key = (
                str(row.source.get("parent_id") or ""),
                row.label.casefold(),
                row.year,
            )
            if key not in seen:
                selected.append(row)
                seen.add(key)
        return tuple(selected)

    def execute(
        self,
        plan: MultiHopPlan,
        *,
        permitted_scope: Scope,
        query_expansions: tuple[str, ...] = (),
    ) -> MultiHopExecutionResult:
        merger = _EvidenceMerger()
        issues: list[str] = []
        requirement_results: list[RequirementResult] = []
        all_facts: dict[str, ValidatedFinancialFact] = {}
        facts_by_requirement: dict[str, tuple[ValidatedFinancialFact, ...]] = {}
        requirement_traces: list[dict] = []
        metric_hints_by_requirement: dict[str, tuple[str, ...]] = {}
        references_by_requirement: dict[str, tuple[FactReference, ...]] = {}
        for calculation in plan.calculations:
            for reference in calculation.inputs:
                references = list(
                    references_by_requirement.get(reference.requirement_id, ())
                )
                references.append(reference)
                references_by_requirement[reference.requirement_id] = tuple(references)
                if not reference.metric_hint:
                    continue
                current = list(
                    metric_hints_by_requirement.get(reference.requirement_id, ())
                )
                if reference.metric_hint not in current:
                    current.append(reference.metric_hint)
                metric_hints_by_requirement[reference.requirement_id] = tuple(current)

        for requirement in plan.requirements:
            scope = requirement.to_scope()
            planned_references = references_by_requirement.get(
                requirement.requirement_id, ()
            )
            structured_rows = self._complete_statement_rows(
                requirement, planned_references
            )
            search_query = requirement.question
            if requirement.evidence_type == EvidenceType.NUMERIC:
                search_query = (
                    f"{search_query} {_numeric_search_instruction(requirement.question)}"
                )
            elif "liquidity" in search_query.casefold():
                search_query += " financing debt covenants funding investment portfolio credit market risk"
            elif "tax reconciliation" in search_query.casefold():
                search_query += " income taxes statutory rate tax credits valuation allowance"
            if structured_rows:
                search = DocumentSearchResult(
                    search_query,
                    _purpose(requirement),
                    RetrievalBundle(
                        context="\n\n".join(
                            str(row.source.get("evidence_text") or "")
                            for row in structured_rows
                        ),
                        sources=[row.source for row in structured_rows],
                        scope=scope,
                        candidate_count=0,
                        covered_groups=scope.groups,
                    ),
                )
            else:
                search = self.document_search.execute(
                    DocumentSearchRequest(
                        query=search_query,
                        scope=scope,
                        purpose=_purpose(requirement),
                        query_expansions=query_expansions,
                    ),
                    permitted_scope=permitted_scope,
                )
            sources = merger.add(search.bundle)
            source_ids_for_requirement = {
                str(source.get("id")) for source in sources
            }
            for merged_source in merger.sources:
                if str(merged_source.get("id")) not in source_ids_for_requirement:
                    continue
                roles = list(merged_source.get("evidence_requirements") or [])
                role = {
                    "requirement_id": requirement.requirement_id,
                    "evidence_type": requirement.evidence_type.value,
                    "document_type": requirement.document_type,
                }
                if role not in roles:
                    roles.append(role)
                merged_source["evidence_requirements"] = roles
            missing = _missing_groups(requirement, sources)
            if not sources:
                issues.append(f"{requirement.requirement_id}:no_evidence")

            facts: tuple[ValidatedFinancialFact, ...] = ()
            rejected_count = 0
            rejected_facts: list[dict[str, str]] = []
            recovered_groups: list[list[str]] = []
            if requirement.evidence_type == EvidenceType.NUMERIC and sources:
                if structured_rows:
                    valid_map: dict[str, ValidatedFinancialFact] = {}
                    rejections = []
                else:
                    validation = self.fact_pipeline.run(
                        question=requirement.question,
                        sources=sources,
                        permitted_scope=scope,
                    )
                    valid_map = {
                        fact.fact_id: fact for fact in validation.valid_facts
                    }
                    rejections = list(validation.rejected_facts)
                source_map = {str(source["id"]): source for source in sources}
                metric_hints = metric_hints_by_requirement.get(
                    requirement.requirement_id, ()
                )

                def targeted_sources_for(
                    group: EvidenceGroup, recovery_scope: Scope
                ) -> list[dict]:
                    row_terms = [
                        "total revenue" if hint.casefold() == "revenue" else
                        re.sub(r"\s+expenses?$", "", hint, flags=re.IGNORECASE)
                        for hint in metric_hints
                    ]
                    years = sorted(
                        {item.fiscal_year for item in requirement.groups if item.ticker == group.ticker}
                    )
                    targeted_search = self.document_search.execute(
                        DocumentSearchRequest(
                            query=" ".join([*(row_terms or [requirement.question]), *years]),
                            scope=recovery_scope,
                            purpose=_purpose(requirement),
                            preserve_all=True,
                        ),
                        permitted_scope=permitted_scope,
                    )
                    found = merger.add(targeted_search.bundle)
                    role = {
                        "requirement_id": requirement.requirement_id,
                        "evidence_type": requirement.evidence_type.value,
                        "document_type": requirement.document_type,
                    }
                    found_ids = {str(source["id"]) for source in found}
                    for merged_source in merger.sources:
                        if str(merged_source["id"]) in found_ids:
                            roles = list(merged_source.get("evidence_requirements") or [])
                            if role not in roles:
                                roles.append(role)
                            merged_source["evidence_requirements"] = roles
                    known_ids = {str(source["id"]) for source in sources}
                    for source in found:
                        source_map[str(source["id"])] = source
                        if str(source["id"]) not in known_ids:
                            sources.append(source)
                            known_ids.add(str(source["id"]))
                    return found

                off_metric_facts: list[ValidatedFinancialFact] = []
                def matches_planned_reference(
                    fact: ValidatedFinancialFact,
                ) -> bool:
                    if not planned_references:
                        return not metric_hints or any(
                            _fact_matches_metric(fact, hint) for hint in metric_hints
                        )
                    source_group = (
                        fact.ticker,
                        str(source_map.get(fact.source_id, {}).get("fiscal_year")),
                    )
                    group_references = [
                        reference
                        for reference in planned_references
                        if (reference.ticker, reference.fiscal_year) == source_group
                    ]
                    return any(
                        (
                            not reference.metric_hint
                            or _fact_matches_metric(fact, reference.metric_hint)
                        )
                        and _fact_matches_period(
                            fact,
                            reference.period or reference.fiscal_year,
                            allow_unknown=False,
                        )
                        for reference in group_references
                    )

                if planned_references or metric_hints:
                    off_metric_facts = [
                        fact
                        for fact in valid_map.values()
                        if not matches_planned_reference(fact)
                    ]
                    for fact in off_metric_facts:
                        valid_map.pop(fact.fact_id, None)

                def fact_group_keys() -> set[tuple[str, str]]:
                    values_by_group: dict[tuple[str, str], set[Decimal]] = {}
                    references = references_by_requirement.get(
                        requirement.requirement_id, ()
                    )
                    for fact in valid_map.values():
                        group_key = (
                            fact.ticker,
                            str(source_map.get(fact.source_id, {}).get("fiscal_year")),
                        )
                        group_references = [
                            reference
                            for reference in references
                            if (reference.ticker, reference.fiscal_year) == group_key
                        ]
                        if not group_references or any(
                            _fact_matches_period(
                                fact,
                                reference.period or reference.fiscal_year,
                                allow_unknown=False,
                            )
                            for reference in group_references
                        ):
                            values_by_group.setdefault(group_key, set()).add(
                                fact.base_value
                            )
                    return {
                        group_key for group_key, values in values_by_group.items()
                        if len(values) == 1
                    }

                # A model can omit one row when many groups share a context.
                # Retry only the missing group against its already-retrieved
                # sources; validation and scope checks remain identical.
                force_focused_extraction = (
                    "effective tax rate" in requirement.question.casefold()
                )
                initial_missing = [] if structured_rows else [
                    group
                    for group in requirement.groups
                    if force_focused_extraction
                    or group.key not in fact_group_keys()
                ]
                for group in initial_missing:
                    group_sources = [
                        source
                        for source in sources
                        if str(source.get("ticker", "")).upper() == group.ticker
                        and str(source.get("fiscal_year", "")) == group.fiscal_year
                        and _normalize_doc_type(source.get("doc_type"))
                        == requirement.document_type
                    ]
                    recovery_scope = Scope(
                        tickers=(group.ticker,),
                        years=(group.fiscal_year,),
                        doc_type=requirement.document_type,
                        requested_groups=(group.key,),
                    )
                    searched_targeted = not group_sources
                    if searched_targeted:
                        group_sources = targeted_sources_for(group, recovery_scope)
                    if not group_sources:
                        continue
                    recovery = self.fact_pipeline.run(
                        question=(
                            f"{requirement.question} Return only the exact requested "
                            f"fact for {group.ticker} source fiscal year "
                            f"{group.fiscal_year}; do not convert its scale. "
                            f"{_numeric_search_instruction(requirement.question)}"
                        ),
                        sources=_focused_numeric_sources(
                            group_sources,
                            requirement.question,
                            metric_hints,
                        ),
                        permitted_scope=recovery_scope,
                    )
                    for fact in recovery.valid_facts:
                        if matches_planned_reference(fact):
                            valid_map[fact.fact_id] = fact
                        else:
                            off_metric_facts.append(fact)
                    rejections.extend(recovery.rejected_facts)
                    # If compression/ranking omitted the exact row, perform one
                    # bounded search for only this missing company/year. This is
                    # a retry within the same evidence requirement, not a new
                    # planner-controlled hop.
                    if not searched_targeted and (
                        force_focused_extraction
                        or group.key not in fact_group_keys()
                    ):
                        targeted_sources = targeted_sources_for(group, recovery_scope)
                        if targeted_sources:
                            targeted_recovery = self.fact_pipeline.run(
                                question=(
                                    f"Return only the exact requested fact for "
                                    f"{group.ticker} source fiscal year "
                                    f"{group.fiscal_year}. "
                                    f"{_numeric_search_instruction(requirement.question)}"
                                ),
                                sources=_focused_numeric_sources(
                                    targeted_sources,
                                    requirement.question,
                                    metric_hints,
                                ),
                                permitted_scope=recovery_scope,
                            )
                            for fact in targeted_recovery.valid_facts:
                                if matches_planned_reference(fact):
                                    valid_map[fact.fact_id] = fact
                                else:
                                    off_metric_facts.append(fact)
                            rejections.extend(targeted_recovery.rejected_facts)
                # Existing primary-statement HTML can correct omitted currency
                # metadata or resolve an ambiguous segment-versus-total row.
                exact_row_recovery = getattr(
                    self.fact_pipeline, "recover_exact_table_row", None
                )
                if metric_hints and callable(exact_row_recovery):
                    for group in requirement.groups:
                        recovery_scope = Scope(
                            tickers=(group.ticker,),
                            years=(group.fiscal_year,),
                            doc_type=requirement.document_type,
                            requested_groups=(group.key,),
                        )
                        group_sources = [
                            source for source in sources
                            if str(source.get("ticker", "")).upper() == group.ticker
                            and str(source.get("fiscal_year", "")) == group.fiscal_year
                            and _normalize_doc_type(source.get("doc_type"))
                            == requirement.document_type
                        ]
                        exact_result = exact_row_recovery(
                            metric_hint=metric_hints[0],
                            period_year=group.fiscal_year,
                            sources=_focused_numeric_sources(
                                group_sources, requirement.question, metric_hints
                            ),
                            permitted_scope=recovery_scope,
                        )
                        for fact in exact_result.valid_facts:
                            if matches_planned_reference(fact):
                                valid_map[fact.fact_id] = fact
                            else:
                                off_metric_facts.append(fact)
                        rejections.extend(exact_result.rejected_facts)
                # Search ranking can miss an indexed statement parent. Read
                # its requested row/year directly. Exact indexed statement
                # cells replace model-extracted variants for the same group.
                if (
                    self.statement_lookup is not None
                    and requirement.document_type == "10K"
                ):
                    for group in requirement.groups:
                        recovery_scope = Scope(
                            tickers=(group.ticker,), years=(group.fiscal_year,),
                            doc_type=requirement.document_type,
                            requested_groups=(group.key,),
                        )
                        group_rows = (
                            tuple(
                                row for row in structured_rows
                                if str(row.source.get("ticker") or "").upper()
                                == group.ticker
                                and row.year == group.fiscal_year
                            )
                            if structured_rows
                            else self.statement_lookup.statement_rows(
                                requirement.question, recovery_scope
                            )
                        )
                        for row in group_rows:
                            indexed_bundle = RetrievalBundle(
                                row.source["evidence_text"], [row.source],
                                recovery_scope, 0,
                            )
                            indexed_source = merger.add(indexed_bundle)[0]
                            source_id = str(indexed_source["id"])
                            source_map[source_id] = indexed_source
                            if not any(source["id"] == source_id for source in sources):
                                sources.append(indexed_source)
                            role = {
                                "requirement_id": requirement.requirement_id,
                                "evidence_type": requirement.evidence_type.value,
                                "document_type": requirement.document_type,
                            }
                            roles = list(indexed_source.get("evidence_requirements") or [])
                            if role not in roles:
                                roles.append(role)
                            indexed_source["evidence_requirements"] = roles
                            recovered = self.fact_pipeline.recover_exact_table_row(
                                metric_hint=row.label,
                                period_year=group.fiscal_year,
                                sources=[indexed_source],
                                permitted_scope=recovery_scope,
                            )
                            exact_facts = [
                                fact for fact in recovered.valid_facts
                                if matches_planned_reference(fact)
                            ]
                            if exact_facts:
                                for fact_id, fact in tuple(valid_map.items()):
                                    source = source_map.get(fact.source_id, {})
                                    fact_group = (
                                        fact.ticker,
                                        str(source.get("fiscal_year", "")),
                                    )
                                    if (
                                        fact_group == group.key
                                        and _fact_matches_metric(fact, row.label)
                                    ):
                                        valid_map.pop(fact_id, None)
                                for fact in exact_facts:
                                    valid_map[fact.fact_id] = fact
                            off_metric_facts.extend(
                                fact for fact in recovered.valid_facts
                                if fact not in exact_facts
                            )
                            rejections.extend(recovered.rejected_facts)
                recovered_groups.extend(
                    list(group.key) for group in initial_missing
                    if group.key in fact_group_keys()
                )

                facts = tuple(valid_map.values())
                rejected_count = len(rejections)
                rejected_facts = [
                    {
                        "source_id": rejected.source_id,
                        "metric": rejected.metric,
                        "reason": rejected.reason,
                        "period": rejected.period,
                        "raw_value": rejected.raw_value,
                        "row_label": rejected.row_label,
                        "column_label": rejected.column_label,
                    }
                    for rejected in rejections
                ]
                rejected_facts.extend(
                    {
                        "source_id": fact.source_id,
                        "metric": fact.metric,
                        "reason": "fact_does_not_match_requested_metric_or_period",
                        "period": fact.period,
                        "raw_value": fact.raw_value,
                        "row_label": fact.row_label or "",
                        "column_label": fact.column_label or "",
                    }
                    for fact in off_metric_facts
                )
                rejected_count += len(off_metric_facts)
                facts_by_requirement[requirement.requirement_id] = facts
                for fact in facts:
                    all_facts[fact.fact_id] = fact
                if not facts:
                    issues.append(f"{requirement.requirement_id}:no_validated_facts")
                else:
                    fact_groups = fact_group_keys()
                    missing_fact_groups = [
                        group for group in requirement.groups if group.key not in fact_groups
                    ]
                    if missing_fact_groups and requirement.requires_complete_coverage:
                        labels = ",".join(
                            f"{group.ticker}-{group.fiscal_year}"
                            for group in missing_fact_groups
                        )
                        issues.append(
                            f"{requirement.requirement_id}:missing_validated_fact_groups:"
                            f"{labels}"
                        )
            else:
                facts_by_requirement[requirement.requirement_id] = ()

            missing = _missing_groups(requirement, sources)
            if missing and requirement.requires_complete_coverage:
                labels = ",".join(
                    f"{group.ticker}-{group.fiscal_year}" for group in missing
                )
                issues.append(
                    f"{requirement.requirement_id}:missing_evidence_groups:{labels}"
                )
            covered_groups = tuple(
                group for group in requirement.groups if group not in missing
            )
            requirement_results.append(
                RequirementResult(
                    requirement_id=requirement.requirement_id,
                    covered_groups=covered_groups,
                    source_ids=tuple(
                        dict.fromkeys(str(source["id"]) for source in sources)
                    ),
                    fact_ids=tuple(fact.fact_id for fact in facts),
                )
            )
            requirement_traces.append(
                {
                    "requirement_id": requirement.requirement_id,
                    "evidence_type": requirement.evidence_type.value,
                    "document_type": requirement.document_type,
                    "purpose": search.purpose.value,
                    "retrieval_mode": (
                        "structured_statement" if structured_rows else "hybrid"
                    ),
                    "candidate_count": search.candidate_count,
                    "source_ids": [str(source["id"]) for source in sources],
                    "fact_ids": [fact.fact_id for fact in facts],
                    "facts": [
                        {
                            "fact_id": fact.fact_id,
                            "ticker": fact.ticker,
                            "metric": fact.metric,
                            "period": fact.period,
                            "raw_value": fact.raw_value,
                            "value_type": fact.value_type.value,
                            "normalized_unit": fact.normalized_unit,
                            "currency": fact.currency,
                            "accounting_basis": fact.accounting_basis,
                            "source_id": fact.source_id,
                            "row_label": fact.row_label,
                            "column_label": fact.column_label,
                        }
                        for fact in facts
                    ],
                    "rejected_fact_count": rejected_count,
                    "rejected_facts": rejected_facts,
                    "recovered_fact_groups": recovered_groups,
                    "missing_groups": [list(group.key) for group in missing],
                }
            )

        source_map = {str(source["id"]): source for source in merger.sources}
        calculation_run = run_calculations(
            plan.calculations,
            facts_by_requirement=facts_by_requirement,
            source_map=source_map,
            available_facts=all_facts,
            calculator=self.calculator,
        )
        issues.extend(calculation_run.issues)

        facts = tuple(all_facts.values())
        calculations = calculation_run.completed
        bundle = merger.build(
            scope=permitted_scope,
            facts=facts,
            calculations=calculations,
        )
        return MultiHopExecutionResult(
            bundle=bundle,
            requirement_results=tuple(requirement_results),
            facts=facts,
            calculations=calculations,
            issues=tuple(dict.fromkeys(issues)),
            trace={
                "requirements": requirement_traces,
                "calculations": list(calculation_run.trace),
                "merged_source_ids": [source["id"] for source in bundle.sources],
                "candidate_count": bundle.candidate_count,
            },
        )
