"""Small composition boundary for extraction, normalization, and validation."""

from __future__ import annotations

import re
import json

from bs4 import BeautifulSoup

from src.financial.fact_extractor import FinancialFactExtractor
from src.financial.extracted_fact_validator import ExtractedFactValidator
from src.financial.models import (
    CandidateFinancialFact,
    FactValidationResult,
    RejectedFinancialFact,
)
from src.financial.metric_terms import (
    has_financial_metric_conflict,
    is_metric_match,
    normalize_financial_term,
)
from src.financial.coverage import fact_coverage, fact_matches_requirement, normalize_reported_period
from src.retrieval.structured_lookup import StructuredDocumentLookup, MONETARY_STATEMENT_RE
from src.observability import trace_component
from src.schemas import Scope


class FinancialFactPipeline:
    """Produce validated facts from retrieved evidence without doing calculations."""

    def __init__(
        self,
        *,
        extractor: FinancialFactExtractor | None = None,
        validator: ExtractedFactValidator | None = None,
    ) -> None:
        self.extractor = extractor or FinancialFactExtractor()
        self.validator = validator or ExtractedFactValidator()

    @trace_component("finrag.financial.fact_extraction_validation")
    def run(
        self,
        *,
        question: str,
        sources: list[dict],
        permitted_scope: Scope,
        requirements: tuple[dict, ...] = (),
    ) -> FactValidationResult:
        fact_requirements = tuple(
            item for item in requirements if item.get("kind", "fact") == "fact"
        )
        # Exact statement cells do not need probabilistic extraction. Recover
        # them first so large parent tables are not sent through an LLM when the
        # requested row and reported-period column are already deterministic.
        recovered = self.recover_table_facts(sources=sources, permitted_scope=permitted_scope,
            requirements=fact_requirements)
        valid_facts = {fact.fact_id: fact for fact in recovered.valid_facts}

        covered_ids = set(fact_coverage(
            fact_requirements, valid_facts.values(), sources,
        )["covered_requirement_ids"])
        pending_requirements = tuple(
            item for item in fact_requirements
            if item["requirement_id"] not in covered_ids
        )
        if fact_requirements and not pending_requirements:
            return FactValidationResult(valid_facts=tuple(valid_facts.values()))

        extraction_question = json.dumps({"question": question, "requirements":
            list(pending_requirements)}) if requirements else question
        candidates = self.extractor.extract(question=extraction_question, sources=sources)
        validation = self.validator.validate(
            candidates,
            sources,
            permitted_scope=permitted_scope,
            requirements=requirements,
        )
        valid_facts.update((fact.fact_id, fact) for fact in validation.valid_facts)
        unresolved = list(validation.rejected_facts)
        # Source IDs are routing references, not proof. Rebind a misplaced
        # candidate only when an actual retrieved source independently passes
        # all the same cell, period, quantity and scope checks.
        for candidate in candidates:
            rejection = next((item for item in unresolved if item.source_id == candidate.source_id
                and item.metric == candidate.metric and item.raw_value == candidate.raw_value
                and item.period == candidate.period and item.reason in {
                    "unknown_source_id", "raw_value_not_in_source", "metric_or_row_not_in_source",
                    "evidence_excerpt_not_in_source"}), None)
            if rejection is None:
                continue
            rebound = self.validator.validate(
                [candidate.model_copy(update={"source_id": str(source["id"])})
                 for source in sources if str(source["id"]) != candidate.source_id],
                sources, permitted_scope=permitted_scope, requirements=requirements,
            ).valid_facts
            cells = {(fact.ticker, fact.period, fact.base_value, fact.value_type, fact.currency,
                      fact.row_label, fact.column_label, fact.accounting_basis) for fact in rebound}
            if len(cells) == 1:
                valid_facts[rebound[0].fact_id] = rebound[0]
                unresolved.remove(rejection)
        rejected_facts: list[RejectedFinancialFact] = []
        recoveries: dict[tuple[str, str], FactValidationResult] = {}
        for rejected in unresolved:
            metric_hint = rejected.row_label or rejected.metric
            period_text = rejected.column_label or rejected.period or ""
            years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", period_text))
            if not metric_hint or len(years) != 1:
                rejected_facts.append(rejected)
                continue

            period_year = years.pop()
            key = (metric_hint.casefold(), period_year)
            recovered = recoveries.get(key)
            if recovered is None:
                recovered = self.recover_exact_table_row(
                    metric_hint=metric_hint,
                    period_year=period_year,
                    sources=sources,
                    permitted_scope=permitted_scope,
                    requirements=requirements,
                )
                recoveries[key] = recovered
            matching = [
                fact
                for fact in recovered.valid_facts
                if is_metric_match(fact.row_label or fact.metric, metric_hint)
            ]
            if matching:
                valid_facts.update((fact.fact_id, fact) for fact in matching)
            else:
                rejected_facts.extend(recovered.rejected_facts or (rejected,))

        # Recover omitted requirements from evidence already in memory as well
        # as rejected candidates; this adds no model or retrieval call.
        for requirement in requirements:
            if requirement.get("kind", "fact") != "fact":
                continue
            years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", requirement.get("reported_period") or "")
            if len(set(years)) == 1:
                recovered = self.recover_exact_table_row(
                    metric_hint=requirement["metric"], period_year=years[0],
                    sources=sources, permitted_scope=permitted_scope,
                    requirements=(requirement,),
                )
                if recovered.valid_facts:
                    valid_facts = {key: fact for key, fact in valid_facts.items()
                                   if fact.requirement_id != requirement["requirement_id"]}
                    valid_facts.update((fact.fact_id, fact) for fact in recovered.valid_facts)
        missing = set(fact_coverage(list(requirements), valid_facts.values(), sources)["missing_requirement_ids"])
        repair_ids = {candidate.requirement_id for candidate in candidates if candidate.requirement_id in missing
                      and any(item.source_id == candidate.source_id and item.metric == candidate.metric
                              and item.raw_value == candidate.raw_value for item in rejected_facts)}
        if repair_ids:
            repair_question = json.dumps({"question": question,
                "requirements": [item for item in requirements if item["requirement_id"] in repair_ids],
                "validation_feedback": [item.model_dump() for item in rejected_facts],
                "instruction": "Repair rejected extraction once using this SAME evidence. Copy literal values and their stated units; do not convert, invent missing facts, or change the request."})
            repaired = self.validator.validate(
                self.extractor.extract(question=repair_question, sources=sources), sources,
                permitted_scope=permitted_scope, requirements=requirements,
            )
            valid_facts.update((fact.fact_id, fact.model_copy(update={
                "validation_checks": (*fact.validation_checks, "repaired_extraction")})) for fact in repaired.valid_facts)
            covered = {fact.requirement_id for fact in repaired.valid_facts}
            resolved = {(candidate.source_id, candidate.metric, candidate.raw_value) for candidate in candidates
                        if candidate.requirement_id in covered}
            rejected_facts = [item for item in rejected_facts if (item.source_id, item.metric, item.raw_value) not in resolved]
            rejected_facts.extend(repaired.rejected_facts)
        return FactValidationResult(
            valid_facts=tuple(valid_facts.values()),
            rejected_facts=tuple(rejected_facts),
        )

    def recover_table_facts(self, *, sources: list[dict], permitted_scope: Scope,
                            requirements: tuple[dict, ...]) -> FactValidationResult:
        """Validate exact cells without invoking probabilistic extraction."""
        facts = {}
        for requirement in requirements:
            years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", requirement.get("reported_period") or ""))
            if requirement.get("kind", "fact") == "fact" and len(years) == 1:
                result = self.recover_exact_table_row(metric_hint=requirement["metric"],
                    period_year=next(iter(years)), sources=sources, permitted_scope=permitted_scope,
                    requirements=(requirement,))
                facts.update((fact.fact_id, fact) for fact in result.valid_facts)
        return FactValidationResult(valid_facts=tuple(facts.values()))

    def recover_exact_table_row(
        self,
        *,
        metric_hint: str,
        period_year: str,
        sources: list[dict],
        permitted_scope: Scope,
        requirements: tuple[dict, ...] = (),
    ) -> FactValidationResult:
        """Read a simple, unambiguous HTML row when model extraction omitted it."""
        label = re.sub(r"\s+expenses?$", "", metric_hint, flags=re.IGNORECASE)
        label = re.sub(rf"\b(?:FY\s*)?{re.escape(period_year)}\b", "", label, flags=re.I).strip()
        expected = normalize_financial_term(label)
        if not expected:
            return FactValidationResult(valid_facts=(), rejected_facts=())
        exact_candidates: list[CandidateFinancialFact] = []
        approximate_candidates: list[CandidateFinancialFact] = []
        statement_candidates: list[CandidateFinancialFact] = []

        def exact_label(value: str) -> str:
            tokens = re.findall(r"[a-z0-9]+", value.casefold())
            return " ".join(
                token[:-1]
                if len(token) > 4 and token.endswith("s") and not token.endswith("ss")
                else token
                for token in tokens
            )

        expected_exact = exact_label(label)
        for source in sources:
            section = str(source.get("section") or "").casefold()
            if not requirements and "financial statements and supplementary data" not in section:
                continue
            body = str(source.get("full_evidence_text") or source.get("evidence_text") or "")
            for match in re.finditer(r"<table\b.*?</table>", body, re.IGNORECASE | re.DOTALL):
                table = BeautifulSoup(match.group(0), "html.parser").find("table")
                if table is None:
                    continue
                heading = StructuredDocumentLookup.table_context(body, match.start())
                rows = table.find_all("tr")
                grid = StructuredDocumentLookup.table_grid(table)
                header_width, _ = StructuredDocumentLookup.reported_year_columns(table)
                columns = StructuredDocumentLookup.reported_value_columns(table, period_year, label)
                if not columns:
                    continue
                table_text = table.get_text(" ", strip=True)
                scale_match = re.search(
                    r"\b(?:millions|billions|thousands)\b",
                    f"{heading} {table_text}",
                    re.IGNORECASE,
                )
                section_label = ""
                for row, values in zip(rows, grid):
                    if values and len(set(values)) == 1 and values[0].strip() and not re.search(r"\d", values[0]):
                        section_label = values[0]
                    if not row.find("td", recursive=False) or len(values) - header_width not in {0, 1}:
                        continue
                    # A header without a label cell has one fewer column.
                    row_label = values[0]
                    requested_measure = " ".join(expected - normalize_financial_term(f"{section_label} {row_label}"))
                    columns = StructuredDocumentLookup.reported_value_columns(table, period_year, requested_measure)
                    measure_label = " ".join(dict.fromkeys(header[index] for header_row, header in zip(rows, grid)
                        if header_row.find("th", recursive=False) for index in columns if index < len(header)
                        and header[index].strip() and not re.search(r"(?:19|20)\d{2}", header[index])))
                    measured_metric = f"{row_label} {measure_label}".strip()
                    local_metric_label = f"{section_label} {measured_metric}".strip()
                    if (
                        has_financial_metric_conflict(local_metric_label, label)
                        and not is_metric_match(measured_metric, label)
                        and not is_metric_match(row_label, label)
                    ):
                        continue
                    row_terms = normalize_financial_term(row_label)
                    other_row_terms = set().union(*(normalize_financial_term(cells[0])
                        for tr, cells in zip(rows, grid) if tr.find("td", recursive=False)
                        and cells and cells[0] != row_label))
                    scoped_terms = (normalize_financial_term(f"{section_label} {measured_metric}")
                                    | normalize_financial_term(heading) - other_row_terms)
                    if not (is_metric_match(row_label, label) or is_metric_match(f"{section_label} {row_label}", label)
                            or is_metric_match(measured_metric, label)
                            or row_terms.issubset(expected) and expected.issubset(scoped_terms)):
                        continue
                    measurement_context = " ".join((
                        section_label, row_label, measure_label,
                    ))
                    is_per_share = bool(re.search(
                        r"\bper[\s-]+shares?\b", measurement_context, re.I,
                    ))
                    # A table-wide monetary scale applies to currency rows, but
                    # never to counts or rates. Per-share rows are monetary too,
                    # with an explicit unit and no millions/billions multiplier.
                    cells = [values[index + len(values) - header_width] for index in columns
                             if 0 < index + len(values) - header_width < len(values)
                             and re.search(r"\d", values[index + len(values) - header_width])]
                    if len(cells) != 1:
                        continue
                    raw = cells[0]
                    is_percent = "%" in raw
                    if not is_percent and (re.search(r"\b(?:units?|percent(?:age)?s?|rates?|ratios?|bps)\b", measurement_context, re.I)
                            or not is_per_share and re.search(r"\bshares?\b", measurement_context, re.I)
                            or scale_match is None or not (MONETARY_STATEMENT_RE.search(heading)
                                or any(item.get("value_type") == "currency" for item in requirements))):
                        continue
                    candidate = CandidateFinancialFact(
                        ticker=str(source["ticker"]),
                        metric=(requirements[0]["metric"] if len(requirements) == 1
                                else measured_metric if is_metric_match(measured_metric, label)
                                else row_label),
                        period=normalize_reported_period(period_year, heading + " " + " ".join(
                            " ".join(header) for header_row, header in zip(rows, grid) if header_row.find("th", recursive=False))),
                        raw_value=raw,
                        unit="percent" if is_percent else "per share" if is_per_share else "USD",
                        scale=None if is_per_share or is_percent else scale_match.group(0).casefold(),
                        currency=None if is_percent else "USD",
                        source_id=str(source["id"]),
                        evidence_excerpt=f"{row_label} {period_year} {raw}",
                        row_label=row_label,
                        column_label=f"{period_year} {measure_label}".strip(),
                        requirement_id=requirements[0]["requirement_id"] if len(requirements) == 1 else None,
                    )
                    target = (
                        exact_candidates
                        if exact_label(row_label) == expected_exact
                        else approximate_candidates
                    )
                    target.append(candidate)
                    if "financial statements" in section and MONETARY_STATEMENT_RE.search(heading):
                        statement_candidates.append(candidate)

        candidates = statement_candidates or exact_candidates or approximate_candidates
        result = self.validator.validate(
            candidates,
            sources,
            permitted_scope=permitted_scope,
            requirements=requirements,
        )
        if requirements:
            source_map = {str(source["id"]): source for source in sources}
            result = FactValidationResult(
                valid_facts=tuple(fact for fact in result.valid_facts
                    if any(fact_matches_requirement(fact, item, source_map[fact.source_id])
                           for item in requirements if item.get("kind", "fact") == "fact")),
                rejected_facts=result.rejected_facts,
            )
        values = {
            (
                fact.ticker,
                fact.base_value,
                fact.value_type,
                fact.normalized_unit,
                fact.currency,
                fact.scale,
            )
            for fact in result.valid_facts
        }
        if len(values) > 1:
            return FactValidationResult(
                rejected_facts=tuple(
                    RejectedFinancialFact(
                        source_id=candidate.source_id,
                        metric=candidate.metric,
                        reason="ambiguous_table_row_match",
                        period=candidate.period,
                        raw_value=candidate.raw_value,
                        row_label=candidate.row_label,
                        column_label=candidate.column_label,
                    )
                    for candidate in candidates
                )
            )
        return FactValidationResult(
            valid_facts=tuple(
                fact.model_copy(update={
                    "validation_checks": (*fact.validation_checks, "exact_table_row")
                })
                for fact in result.valid_facts
            ),
            rejected_facts=result.rejected_facts,
        )
