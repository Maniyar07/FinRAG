"""Small composition boundary for extraction, normalization, and validation."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.financial.fact_extractor import FinancialFactExtractor
from src.financial.extracted_fact_validator import ExtractedFactValidator
from src.financial.models import CandidateFinancialFact, FactValidationResult
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

    def run(
        self,
        *,
        question: str,
        sources: list[dict],
        permitted_scope: Scope,
    ) -> FactValidationResult:
        candidates = self.extractor.extract(question=question, sources=sources)
        return self.validator.validate(
            candidates,
            sources,
            permitted_scope=permitted_scope,
        )

    def recover_exact_table_row(
        self,
        *,
        metric_hint: str,
        period_year: str,
        sources: list[dict],
        permitted_scope: Scope,
    ) -> FactValidationResult:
        """Read a simple, unambiguous HTML row when model extraction omitted it."""
        label = re.sub(r"\s+expenses?$", "", metric_hint, flags=re.IGNORECASE)
        expected = " ".join(label.casefold().split())
        aliases = {
            "revenue": {"total revenue", "total revenues"},
            "operating income": {"operating income", "income from operations"},
        }
        accepted_labels = aliases.get(expected, {expected})
        candidates: list[CandidateFinancialFact] = []
        for source in sources:
            section = str(source.get("section") or "").casefold()
            if "financial statements and supplementary data" not in section:
                continue
            body = str(source.get("full_evidence_text") or source.get("evidence_text") or "")
            for match in re.finditer(r"<table\b.*?</table>", body, re.IGNORECASE | re.DOTALL):
                table = BeautifulSoup(match.group(0), "html.parser").find("table")
                if table is None:
                    continue
                heading = body[max(0, match.start() - 350):match.start()]
                if expected in {"revenue", "operating income", "research and development"}:
                    if not re.search(r"(?:income statements|statements of operations)", heading, re.IGNORECASE):
                        continue
                rows = table.find_all("tr")
                year_columns = [
                    index
                    for row in rows
                    for index, cell in enumerate(row.find_all("th", recursive=False))
                    if " ".join(cell.get_text(" ", strip=True).split()) == period_year
                ]
                if len(set(year_columns)) != 1:
                    continue
                column = year_columns[0]
                table_text = table.get_text(" ", strip=True)
                scale_match = re.search(
                    r"\b(?:millions|billions|thousands)\b",
                    f"{heading} {table_text}",
                    re.IGNORECASE,
                )
                if scale_match is None or "$" not in table_text:
                    continue
                for row in rows:
                    cells = row.find_all(["td", "th"], recursive=False)
                    if len(cells) <= column:
                        continue
                    row_label = " ".join(cells[0].get_text(" ", strip=True).split())
                    normalized_label = re.sub(
                        r"\s*\([^)]*\)\s*", " ", row_label
                    ).strip().casefold()
                    if normalized_label not in accepted_labels:
                        continue
                    raw = " ".join(cells[column].get_text(" ", strip=True).split())
                    candidates.append(
                        CandidateFinancialFact(
                            ticker=str(source["ticker"]),
                            metric=row_label,
                            period=period_year,
                            raw_value=raw,
                            unit="USD",
                            scale=scale_match.group(0).casefold(),
                            currency="USD",
                            source_id=str(source["id"]),
                            evidence_excerpt=f"{row_label} {period_year} {raw}",
                            row_label=row_label,
                            column_label=period_year,
                        )
                    )
        result = self.validator.validate(
            candidates,
            sources,
            permitted_scope=permitted_scope,
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
