"""Direct, bounded execution of multi-hop financial research plans."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal

from src.financial.calculator import CalculationRequest, CalculationResult
from src.financial.fact_pipeline import FinancialFactPipeline
from src.financial.models import ValidatedFinancialFact, ValueType
from src.orchestration.models import (
    EvidenceGroup,
    EvidenceRequirement,
    EvidenceType,
    FactReference,
    MultiHopPlan,
    RequirementResult,
)
from src.schemas import RetrievalBundle, Scope
from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.tools.document_search import (
    DocumentSearchPurpose,
    DocumentSearchRequest,
    DocumentSearchTool,
)
from src.tools.financial_calculator import FinancialCalculatorTool


WORD_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


@dataclass(frozen=True)
class ExecutedCalculation:
    calculation_id: str
    label: str
    result: CalculationResult


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


def _normalize_doc_type(value: object) -> str:
    return str(value or "").strip().upper().replace("-", "")


def _source_identity(source: dict) -> tuple[str, ...]:
    parent_id = str(source.get("parent_id") or "").strip()
    source_hash = str(source.get("source_hash") or "").strip()
    if parent_id:
        return (
            "parent",
            source_hash,
            parent_id,
            str(source.get("ticker") or "").upper(),
            str(source.get("fiscal_year") or ""),
            _normalize_doc_type(source.get("doc_type")),
        )
    evidence_hash = hashlib.sha256(
        str(source.get("evidence_text") or "").encode("utf-8")
    ).hexdigest()[:16]
    return (
        "fallback",
        str(source.get("ticker") or "").upper(),
        str(source.get("fiscal_year") or ""),
        _normalize_doc_type(source.get("doc_type")),
        str(source.get("source") or ""),
        str(source.get("section") or ""),
        str(source.get("pdf_page") or ""),
        evidence_hash,
    )


def _merge_text(existing: object, incoming: object) -> str:
    left = str(existing or "").strip()
    right = str(incoming or "").strip()
    if not left:
        return right
    if not right or right in left:
        return left
    if left in right:
        return right
    return f"{left}\n\n{right}"


def _source_header(source: dict) -> str:
    fields = [
        f"SOURCE {source['id']}",
        f"TICKER: {source.get('ticker')}",
        f"FISCAL YEAR: {source.get('fiscal_year')}",
        f"DOCUMENT PERIOD ONLY: {source.get('fiscal_period')} (not the period of every fact)",
        f"TYPE: {source.get('doc_type')}",
        f"FILE: {source.get('source')}",
        f"SECTION: {source.get('section')}",
        f"PDF PAGE: {source.get('pdf_page', 'not available')}",
    ]
    for key, label in (
        ("period_end_date", "PERIOD END"),
        ("fiscal_q4_months", "FISCAL Q4 MONTHS"),
        ("call_date", "CALL DATE"),
        ("speaker", "SPEAKER"),
        ("speaker_role", "SPEAKER ROLE"),
    ):
        if source.get(key):
            fields.append(f"{label}: {source[key]}")
    roles = source.get("evidence_requirements") or []
    if roles:
        role_labels = ", ".join(
            f"{role.get('requirement_id')} ({role.get('evidence_type')})"
            for role in roles
        )
        fields.append(f"EVIDENCE ROLES: {role_labels}")
    return "[" + " | ".join(fields) + "]"


class _EvidenceMerger:
    """Deduplicate parent evidence and assign globally unique citation IDs."""

    def __init__(self) -> None:
        self.sources: list[dict] = []
        self._index_by_identity: dict[tuple[str, ...], int] = {}
        self.candidate_count = 0

    def add(self, bundle: RetrievalBundle) -> list[dict]:
        self.candidate_count += bundle.candidate_count
        remapped: list[dict] = []
        for original in bundle.sources:
            source = dict(original)
            identity = _source_identity(source)
            index = self._index_by_identity.get(identity)
            if index is None:
                source["id"] = f"S{len(self.sources) + 1}"
                self._index_by_identity[identity] = len(self.sources)
                self.sources.append(source)
                remapped.append(source)
                continue

            existing = self.sources[index]
            existing["evidence_text"] = _merge_text(
                existing.get("evidence_text"), source.get("evidence_text")
            )
            existing["full_evidence_text"] = _merge_text(
                existing.get("full_evidence_text"),
                source.get("full_evidence_text"),
            )
            source["id"] = existing["id"]
            source["evidence_text"] = existing.get("evidence_text", "")
            source["full_evidence_text"] = existing.get("full_evidence_text", "")
            remapped.append(source)
        return remapped

    def build(
        self,
        *,
        scope: Scope,
        facts: tuple[ValidatedFinancialFact, ...],
        calculations: tuple[ExecutedCalculation, ...],
    ) -> RetrievalBundle:
        # Validation may need many candidate parents, but generation should
        # see only the sources that support selected facts/calculations plus a
        # small, balanced set of narrative evidence for each company/year.
        selected_ids = {
            source_id
            for item in calculations
            for source_id in item.result.source_ids
        }
        if not selected_ids:
            selected_ids = {fact.source_id for fact in facts}
        narrative_counts: dict[tuple[str, str], int] = {}
        for source in self.sources:
            if not any(
                role.get("evidence_type") == "narrative"
                for role in source.get("evidence_requirements") or []
            ):
                continue
            key = (
                str(source.get("ticker")),
                str(source.get("fiscal_year")),
            )
            if narrative_counts.get(key, 0) < 3:
                selected_ids.add(str(source["id"]))
                narrative_counts[key] = narrative_counts.get(key, 0) + 1
        selected_sources = (
            [source for source in self.sources if source["id"] in selected_ids]
            if selected_ids
            else list(self.sources)
        )
        context_parts = [
            f"{_source_header(source)}\n{str(source.get('evidence_text') or '').strip()}"
            for source in selected_sources
            if str(source.get("evidence_text") or "").strip()
        ]
        if facts:
            lines = ["[APPLICATION-VALIDATED FINANCIAL FACTS]"]
            for fact in facts:
                lines.append(
                    f"{fact.fact_id}: {fact.ticker} | {fact.metric} | {fact.period} | "
                    f"raw value {fact.raw_value} | source {fact.source_id}"
                )
            context_parts.append("\n".join(lines))
        if calculations:
            lines = ["[APPLICATION-VERIFIED CALCULATIONS]"]
            for item in calculations:
                result = item.result
                source_ids = ", ".join(result.source_ids)
                lines.append(
                    f"{item.calculation_id}: {item.label} | {result.formula} | "
                    f"{result.substituted_formula} = {_display(result.result)} "
                    f"{_result_unit_label(result)} | supporting sources: {source_ids}"
                )
            context_parts.append("\n".join(lines))
        covered = tuple(
            sorted(
                {
                    (str(source.get("ticker")), str(source.get("fiscal_year")))
                    for source in selected_sources
                }
            )
        )
        return RetrievalBundle(
            context="\n\n---\n\n".join(context_parts),
            sources=selected_sources,
            scope=scope,
            candidate_count=self.candidate_count,
            covered_groups=covered,
        )


def _display(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _result_unit_label(result: CalculationResult) -> str:
    if result.result_unit == "currency":
        return " ".join(
            value for value in (result.currency, result.scale) if value
        ) or "currency"
    return result.result_unit


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


def _terms(value: str) -> set[str]:
    terms = set(WORD_RE.findall(value.casefold()))
    normalized = {
        term[:-1] if term.endswith("s") and len(term) > 4 else term
        for term in terms
    }
    aliases = {
        "operating": "operation",
    }
    normalized = {aliases.get(term, term) for term in normalized}
    return normalized - {
        "and",
        "amount",
        "expense",
        "financial",
        "from",
        "reported",
        "total",
        "value",
    }


MATERIAL_METRIC_QUALIFIERS = frozenset(
    {
        "adjusted",
        "automotive",
        "combined",
        "restricted",
        "segment",
        "short",
    }
)


def _fact_matches_metric(fact: ValidatedFinancialFact, metric_hint: str) -> bool:
    """Match common accounting aliases without accepting a materially broader row."""
    expected = _terms(metric_hint)
    # The extracted row label is source evidence. The model-authored metric name
    # is only a fallback and must not override a contradictory table row.
    actual = _terms(fact.row_label or fact.metric)
    if "rate" in expected and fact.value_type != ValueType.PERCENT:
        return False
    tax_rate_alias = (
        expected == {"effective", "tax", "rate"}
        and actual == {"effective", "rate"}
    )
    if not expected or (not tax_rate_alias and not expected.issubset(actual)):
        return False
    unexpected_qualifiers = (actual & MATERIAL_METRIC_QUALIFIERS) - expected
    return not unexpected_qualifiers


def _fact_matches_period(
    fact: ValidatedFinancialFact,
    requested_period: str,
    *,
    allow_unknown: bool,
) -> bool:
    actual = f"{fact.period} {fact.column_label or ''}"
    if requested_period.casefold() in actual.casefold():
        return True
    requested_years = YEAR_RE.findall(requested_period)
    actual_years = YEAR_RE.findall(actual)
    if requested_years:
        if all(year in actual_years for year in requested_years):
            return True
        return allow_unknown and not actual_years
    return False


def _numeric_search_instruction(question: str) -> str:
    lowered = question.casefold()
    base = (
        "Find the exact reported table row, year column, units, and scale."
    )
    if "operating income" in lowered:
        return (
            f'{base} Treat "income from operations" as the same requested row. '
            "Prefer the consolidated income statement or operations table."
        )
    if "cash and cash equivalents" in lowered:
        return (
            f"{base} Use the exact Cash and cash equivalents row from the "
            "Consolidated Balance Sheets under Assets / Current assets. Exclude "
            "fair-value footnotes, restricted cash, combined cash and restricted-cash "
            "reconciliations, and short-term investments."
        )
    if "effective tax rate" in lowered:
        return (
            f"{base} Prefer the exact effective-rate row in the income-tax rate "
            "reconciliation table containing the Federal statutory rate and Effect of "
            "rows, not a rounded narrative summary."
        )
    if any(
        term in lowered
        for term in (
            "revenue",
            "income",
            "loss",
            "expense",
            "profit",
        )
    ):
        return f"{base} Prefer the consolidated income statement or operations table."
    return base


def _with_full_evidence(sources: list[dict]) -> list[dict]:
    """Use the stored parent only for a focused numeric recovery attempt."""
    expanded: list[dict] = []
    for source in sources:
        item = dict(source)
        full_text = str(source.get("full_evidence_text") or "").strip()
        if full_text:
            item["evidence_text"] = full_text
        expanded.append(item)
    return expanded


def _focused_numeric_sources(
    sources: list[dict],
    question: str,
    metric_hints: tuple[str, ...] = (),
) -> list[dict]:
    """Prefer parents that contain the requested primary statement/table."""
    lowered = question.casefold()
    if any(
        hint.casefold() in {"revenue", "total revenue", "total revenues", "operating income", "research and development expense"}
        for hint in metric_hints
    ):
        statements = [
            source for source in sources
            if "financial statements and supplementary data"
            in str(source.get("section") or "").casefold()
            and re.search(
                r"(?:income statements|statements of operations)",
                str(source.get("full_evidence_text") or source.get("evidence_text") or ""),
                re.IGNORECASE,
            )
        ]
        if statements:
            return _with_full_evidence(statements)
    if "effective tax rate" in lowered:
        exact_reconciliations = [
            source
            for source in sources
            if "<td>effective rate</td>"
            in str(
                source.get("full_evidence_text")
                or source.get("evidence_text")
                or ""
            ).casefold()
            and "federal statutory rate"
            in str(
                source.get("full_evidence_text")
                or source.get("evidence_text")
                or ""
            ).casefold()
        ]
        if exact_reconciliations:
            return _with_full_evidence(exact_reconciliations)
    preferred: list[dict] = []
    exact_rows = []
    for source in sources:
        full_text = str(
            source.get("full_evidence_text") or source.get("evidence_text") or ""
        ).casefold()
        for hint in metric_hints:
            row_label = re.sub(r"\s+expenses?$", "", hint, flags=re.IGNORECASE)
            if re.search(
                rf"<td[^>]*>\s*{re.escape(row_label.casefold())}\s*</td>",
                full_text,
            ):
                exact_rows.append(source)
                break
        if "effective tax rate" in lowered:
            if "effective rate" in full_text and (
                "federal statutory rate" in full_text
                or "tax at statutory federal rate" in full_text
            ):
                preferred.append(source)
        elif "cash and cash equivalents" in lowered:
            if (
                "consolidated balance sheets" in full_text
                and "cash and cash equivalents" in full_text
            ):
                preferred.append(source)
    return _with_full_evidence(preferred or exact_rows or sources)


def _select_fact(
    reference: FactReference,
    facts_by_requirement: dict[str, tuple[ValidatedFinancialFact, ...]],
    source_map: dict[str, dict],
) -> ValidatedFinancialFact:
    candidates = [
        fact
        for fact in facts_by_requirement.get(reference.requirement_id, ())
        if fact.ticker == reference.ticker
        and str(source_map.get(fact.source_id, {}).get("fiscal_year"))
        == reference.fiscal_year
    ]
    if reference.metric_hint:
        candidates = [
            fact
            for fact in candidates
            if _fact_matches_metric(fact, reference.metric_hint)
        ]

    requested_period = reference.period or reference.fiscal_year
    matching_period = [
        fact
        for fact in candidates
        if _fact_matches_period(fact, requested_period, allow_unknown=False)
    ]
    if matching_period:
        candidates = matching_period
    else:
        candidates = [
            fact
            for fact in candidates
            if _fact_matches_period(fact, requested_period, allow_unknown=True)
        ]

    exact_rows = [
        fact for fact in candidates if "exact_table_row" in fact.validation_checks
    ]
    if exact_rows:
        candidates = exact_rows

    # A consolidated statement outranks segment disclosures for company-wide
    # metrics. Keep conflicting figures ambiguous when no primary row exists.
    primary: list[ValidatedFinancialFact] = []
    if reference.metric_hint and reference.metric_hint.casefold() in {
        "revenue", "total revenue", "total revenues", "operating income",
        "research and development expense",
    }:
        primary = [
            fact for fact in candidates
            if "financial statements and supplementary data"
            in str(source_map.get(fact.source_id, {}).get("section") or "").casefold()
            and re.search(
                r"(?:income statements|statements of operations)",
                str(source_map.get(fact.source_id, {}).get("full_evidence_text")
                    or source_map.get(fact.source_id, {}).get("evidence_text") or ""),
                re.IGNORECASE,
            )
        ]
        if primary:
            candidates = primary
    monetary = [fact for fact in candidates if fact.value_type == ValueType.CURRENCY]
    if primary and monetary and len({fact.numeric_value for fact in monetary}) == 1:
        candidates = monetary

    unique = {fact.fact_id: fact for fact in candidates}
    if len(unique) > 1:
        values = {fact.numeric_value for fact in unique.values()}
        if len(values) == 1:
            return next(iter(unique.values()))
        highest_precision = max(
            max(0, -fact.numeric_value.as_tuple().exponent)
            for fact in unique.values()
        )
        precise = [
            fact
            for fact in unique.values()
            if max(0, -fact.numeric_value.as_tuple().exponent) == highest_precision
        ]
        if len(precise) == 1:
            return precise[0]
    if len(unique) != 1:
        raise ValueError(
            f"Fact reference {reference.requirement_id}/{reference.ticker}/"
            f"{reference.fiscal_year}/{requested_period} matched {len(unique)} "
            "validated facts."
        )
    return next(iter(unique.values()))


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
            search_query = requirement.question
            if requirement.evidence_type == EvidenceType.NUMERIC:
                search_query = (
                    f"{search_query} {_numeric_search_instruction(requirement.question)}"
                )
            elif "liquidity" in search_query.casefold():
                search_query += " financing debt covenants funding investment portfolio credit market risk"
            elif "tax reconciliation" in search_query.casefold():
                search_query += " income taxes statutory rate tax credits valuation allowance"
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
                planned_references = references_by_requirement.get(
                    requirement.requirement_id, ()
                )

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
                initial_missing = [
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
                # its requested row/year directly before declaring a fact absent.
                if self.statement_lookup is not None:
                    for group in requirement.groups:
                        if group.key in fact_group_keys():
                            continue
                        recovery_scope = Scope(
                            tickers=(group.ticker,), years=(group.fiscal_year,),
                            doc_type=requirement.document_type,
                            requested_groups=(group.key,),
                        )
                        for row in self.statement_lookup.statement_rows(
                            requirement.question, recovery_scope
                        ):
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
                            for fact in recovered.valid_facts:
                                if matches_planned_reference(fact):
                                    valid_map[fact.fact_id] = fact
                                else:
                                    off_metric_facts.append(fact)
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
        calculation_results: list[ExecutedCalculation] = []
        calculation_traces: list[dict] = []
        for planned in plan.calculations:
            try:
                selected = tuple(
                    _select_fact(reference, facts_by_requirement, source_map)
                    for reference in planned.inputs
                )
                request = CalculationRequest(
                    operation=planned.operation,
                    input_fact_ids=tuple(fact.fact_id for fact in selected),
                    periods=planned.periods,
                )
                result = self.calculator.execute(
                    request,
                    available_facts=all_facts,
                )
            except ValueError as error:
                issues.append(f"{planned.calculation_id}:calculation_unavailable")
                calculation_traces.append(
                    {
                        "calculation_id": planned.calculation_id,
                        "status": "unavailable",
                        "reason": str(error),
                    }
                )
                continue
            calculation_results.append(
                ExecutedCalculation(
                    calculation_id=planned.calculation_id,
                    label=planned.label,
                    result=result,
                )
            )
            calculation_traces.append(
                {
                    "calculation_id": planned.calculation_id,
                    "status": "completed",
                    "operation": result.operation.value,
                    "input_fact_ids": list(result.input_fact_ids),
                    "source_ids": list(result.source_ids),
                    "result": str(result.result),
                    "result_unit": result.result_unit,
                }
            )

        facts = tuple(all_facts.values())
        calculations = tuple(calculation_results)
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
                "calculations": calculation_traces,
                "merged_source_ids": [source["id"] for source in bundle.sources],
                "candidate_count": bundle.candidate_count,
            },
        )
