from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup

from .models import (
    CalculationRecord,
    EvidenceBlock,
    GroundedAnswerTarget,
    GroundedClaimTarget,
    MasterExample,
    NumericFactRecord,
    ProvenanceRecord,
    QualityRecord,
    ScopeRecord,
)
from .utils import normalized_text, question_fingerprint, sha256_text, stable_id


TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"(?m)^\s*#{1,5}\s+([^\n]+)$")
PAGE_RE = re.compile(r"\[PAGE:\s*\d+\]", re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
NUMERIC_CELL_RE = re.compile(r"^\s*\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?%?\s*$")
COMPANY_NAMES = {"MSFT": "Microsoft", "TSLA": "Tesla", "JPM": "JPMorgan Chase"}


@dataclass(frozen=True)
class TableFact:
    ticker: str
    source_year: str
    source: str
    source_hash: str
    parent_id: str
    section: str
    title: str
    period: str
    label: str
    raw_value: str
    unit: str
    evidence_text: str
    table_markdown: str

    @property
    def signature(self) -> str:
        return sha256_text(
            "\x1f".join(
                (
                    self.source_hash,
                    normalized_text(self.title),
                    normalized_text(self.label),
                    self.period,
                    _number_key(self.raw_value) or self.raw_value,
                    self.unit,
                )
            )
        )


@dataclass(frozen=True)
class ParsedTable:
    facts: tuple[TableFact, ...]
    ticker: str
    source_year: str
    source: str
    source_hash: str
    parent_id: str
    section: str
    title: str
    unit: str
    evidence_text: str
    markdown: str
    column_count: int


def _clean(value: str) -> str:
    text = BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)
    return " ".join(text.replace("|", " ").split())


def _number_key(raw: str) -> str | None:
    value = str(raw).strip()
    negative = "(" in value and ")" in value
    cleaned = re.sub(r"[$,%\s()]", "", value).replace(",", "")
    if not cleaned:
        return None
    if negative and not cleaned.startswith("-"):
        cleaned = "-" + cleaned
    try:
        return format(Decimal(cleaned).normalize(), "f")
    except InvalidOperation:
        return None


def _display_value(raw: str, unit: str) -> str:
    value = " ".join(str(raw).replace("$", "").split())
    if unit == "%":
        return value if value.endswith("%") else f"{value}%"
    if unit == "USD per share":
        return f"USD {value} per share"
    if unit == "million shares":
        return f"{value} million shares"
    if unit.startswith("USD "):
        scale = unit.removeprefix("USD ").rstrip("s")
        return f"USD {value} {scale}"
    return f"{value} {unit}".strip()


def _expand_grid(rows: Iterable) -> list[list[str]]:
    grid: list[list[str]] = []
    active: dict[int, tuple[int, str]] = {}
    for row_node in rows:
        row: dict[int, str] = {column: text for column, (_, text) in active.items()}
        additions: dict[int, tuple[int, str]] = {}
        column = 0
        for cell in row_node.find_all(["th", "td"], recursive=False):
            while column in row:
                column += 1
            text = _clean(cell.get_text(" ", strip=True))
            try:
                colspan = max(1, int(cell.get("colspan", "1")))
                rowspan = max(1, int(cell.get("rowspan", "1")))
            except (TypeError, ValueError):
                colspan = rowspan = 1
            for offset in range(colspan):
                target = column + offset
                row[target] = text
                if rowspan > 1:
                    additions[target] = (rowspan - 1, text)
            column += colspan
        width = max(row, default=-1) + 1
        grid.append([row.get(index, "") for index in range(width)])
        active = {
            column: (remaining - 1, text)
            for column, (remaining, text) in active.items()
            if remaining > 1
        }
        active.update(additions)
    return grid


def _pad(grid: list[list[str]], width: int) -> list[list[str]]:
    return [row + [""] * (width - len(row)) for row in grid]


def _table_title(body: str, table_start: int, previous_end: int, section: str) -> str:
    headings = [match for match in HEADING_RE.finditer(body, previous_end, table_start)]
    heading = _clean(headings[-1].group(1)) if headings else _clean(section)
    start = headings[-1].end() if headings else previous_end
    between = PAGE_RE.sub("", body[start:table_start]).strip()
    paragraphs = [
        _clean(value)
        for value in re.split(r"\n\s*\n", between)
        if 4 <= len(_clean(value).split()) <= 35 and "<table" not in value.casefold()
    ]
    caption = paragraphs[-1].rstrip(".:;") if paragraphs else ""
    title = caption or heading or "reported financial table"
    return title[:220]


def _unit(header_text: str, label: str, value: str) -> str | None:
    lowered = header_text.casefold()
    label_lower = label.casefold()
    if value.strip().endswith("%"):
        return "%"
    if "per share" in label_lower and "except per share" in lowered:
        return "USD per share"
    if re.search(r"shares?\s+in\s+millions?", lowered):
        return "million shares"
    scale = re.search(r"\bin\s+(millions?|thousands?|billions?)\b", lowered)
    if scale:
        return "USD " + scale.group(1).casefold()
    return None


def _column_headers(header_grid: list[list[str]], width: int) -> list[str]:
    padded = _pad(header_grid, width)
    headers: list[str] = []
    for column in range(width):
        values: list[str] = []
        for row in padded:
            value = row[column].strip()
            if value and value not in values:
                values.append(value)
        headers.append(" | ".join(values))
    return headers


def _header_dimension(header: str, period: str) -> str:
    """Return the non-period metric dimension encoded in a column header."""
    values: list[str] = []
    for part in header.split("|"):
        cleaned = YEAR_RE.sub("", part)
        cleaned = re.sub(
            r"(?i)\b(?:as\s+of|year\s+ended|years\s+ended|january|february|march|"
            r"april|may|june|july|august|september|october|november|december|"
            r"in\s+millions?|in\s+thousands?|in\s+billions?)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"[^A-Za-z]+", " ", cleaned)
        cleaned = " ".join(cleaned.split()).strip()
        if cleaned.casefold() in {"", "amount", "date", "year"}:
            continue
        if cleaned not in values:
            values.append(cleaned)
    return " — ".join(values)


def _markdown(headers: list[str], rows: list[list[str]]) -> str:
    safe_headers = [(value or ("Metric" if index == 0 else f"Column {index + 1}"))
                    .replace("|", r"\|").replace("$", "USD ")
                    for index, value in enumerate(headers)]
    lines = [
        "| " + " | ".join(safe_headers) + " |",
        "| " + " | ".join("---" for _ in safe_headers) + " |",
    ]
    for row in rows:
        safe = [value.replace("|", r"\|").replace("$", "USD ") for value in row]
        lines.append("| " + " | ".join(safe) + " |")
    return "\n".join(lines)


def _parse_table(record: dict, raw_table: str, title: str) -> ParsedTable | None:
    metadata = record["metadata"]
    soup = BeautifulSoup(raw_table, "html.parser")
    table = soup.find("table")
    if table is None:
        return None
    thead = table.find("thead")
    tbody = table.find("tbody")
    header_nodes = thead.find_all("tr") if thead else [
        row for row in table.find_all("tr") if row.find("th") is not None
    ]
    body_nodes = tbody.find_all("tr") if tbody else [
        row for row in table.find_all("tr") if row.find("td") is not None
    ]
    if not header_nodes or not body_nodes:
        return None
    header_grid = _expand_grid(header_nodes)
    body_grid = _expand_grid(body_nodes)
    width = max((len(row) for row in header_grid + body_grid), default=0)
    if width < 2 or width > 8:
        return None
    header_grid = _pad(header_grid, width)
    body_grid = _pad(body_grid, width)
    headers = _column_headers(header_grid, width)
    periods: dict[int, str] = {}
    for column, header in enumerate(headers[1:], 1):
        years = list(dict.fromkeys(YEAR_RE.findall(header)))
        if len(years) == 1:
            periods[column] = years[0]
    if not periods:
        return None

    context_header = " ".join(value for row in header_grid for value in row)
    table_rows: list[list[str]] = []
    facts: list[TableFact] = []
    group_label = ""
    evidence_text = f"## {title}\n\n{raw_table.strip()}"
    source_hash = str(metadata.get("source_hash") or "")
    for row in body_grid:
        nonempty = [value for value in row if value.strip()]
        if not nonempty:
            continue
        numeric_columns = [
            column for column, value in enumerate(row)
            if column > 0 and NUMERIC_CELL_RE.fullmatch(value or "")
        ]
        label = row[0].strip()
        if len(nonempty) == 1 and label and not numeric_columns:
            group_label = label if len(label.split()) <= 12 else ""
            continue
        if not label or re.fullmatch(r"(?:19|20)\d{2}", label):
            continue
        base_label = f"{group_label} — {label}" if group_label else label
        base_label = re.sub(r"\s+", " ", base_label).strip(" —")
        if len(base_label) < 3 or len(base_label) > 180:
            continue
        accepted_row = False
        for column in numeric_columns:
            period = periods.get(column)
            if not period:
                continue
            dimension = _header_dimension(headers[column], period)
            semantic_label = (
                f"{dimension} — {base_label}" if dimension else base_label
            )
            semantic_label = semantic_label[:220]
            raw_value = row[column].strip()
            unit = _unit(context_header, semantic_label, raw_value)
            if not unit or _number_key(raw_value) is None:
                continue
            accepted_row = True
            facts.append(
                TableFact(
                    ticker=str(metadata["ticker"]),
                    source_year=str(metadata["fiscal_year"]),
                    source=str(metadata["source"]),
                    source_hash=source_hash,
                    parent_id=str(metadata["parent_id"]),
                    section=_clean(str(metadata.get("section") or "Unknown section")),
                    title=title,
                    period=period,
                    label=semantic_label,
                    raw_value=raw_value,
                    unit=unit,
                    evidence_text=evidence_text,
                    table_markdown="",
                )
            )
        if accepted_row:
            table_rows.append(row)
    if not facts or len(table_rows) < 2:
        return None
    markdown = _markdown(headers, table_rows)
    facts = [TableFact(**{**fact.__dict__, "table_markdown": markdown}) for fact in facts]
    units = {fact.unit for fact in facts}
    table_unit = next(iter(units)) if len(units) == 1 else "as reported"
    return ParsedTable(
        facts=tuple(facts),
        ticker=str(metadata["ticker"]),
        source_year=str(metadata["fiscal_year"]),
        source=str(metadata["source"]),
        source_hash=source_hash,
        parent_id=str(metadata["parent_id"]),
        section=_clean(str(metadata.get("section") or "Unknown section")),
        title=title,
        unit=table_unit,
        evidence_text=evidence_text,
        markdown=markdown,
        column_count=width,
    )


def load_tables(parent_dir: Path) -> list[ParsedTable]:
    parsed: dict[str, ParsedTable] = {}
    for path in sorted(parent_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        metadata = record.get("metadata") or {}
        if str(metadata.get("doc_type", "")).upper().replace("-", "") != "10K":
            continue
        body = str(record.get("page_content") or "")
        previous_end = 0
        for match in TABLE_RE.finditer(body):
            title = _table_title(
                body, match.start(), previous_end, str(metadata.get("section") or "")
            )
            previous_end = match.end()
            table = _parse_table(record, match.group(), title)
            if table is None:
                continue
            signature = sha256_text(
                "\x1f".join(
                    (table.source_hash, normalized_text(table.title), normalized_text(table.markdown))
                )
            )
            parsed.setdefault(signature, table)
    return sorted(
        parsed.values(),
        key=lambda table: (table.ticker, table.source_year, table.source, table.parent_id, table.title),
    )


def _evidence(table: ParsedTable, source_id: str = "S1") -> EvidenceBlock:
    return EvidenceBlock(
        source_id=source_id,
        parent_id=table.parent_id,
        ticker=table.ticker,
        fiscal_year=table.source_year,
        doc_type="10K",
        source=table.source,
        section=table.section,
        text=table.evidence_text,
        text_sha256=sha256_text(table.evidence_text),
        source_hash=table.source_hash,
    )


def _scope(table: ParsedTable) -> ScopeRecord:
    return ScopeRecord(
        tickers=[table.ticker], years=[table.source_year], doc_types=["10K"]
    )


def _provenance(table: ParsedTable, signature: str, generator: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        generator=generator,
        source_record_ids=[table.parent_id],
        source_hashes=[table.source_hash] if table.source_hash else [],
        canonical_fact_signature=signature,
    )


def _direct_example(table: ParsedTable, fact: TableFact) -> MasterExample:
    company = COMPANY_NAMES.get(fact.ticker, fact.ticker)
    question = (
        f"According to {company}'s {fact.source_year} 10-K, what value was reported "
        f"for {fact.label} in {fact.period} in the table \"{fact.title}\"? "
        "Preserve the reported units."
    )
    claim = (
        f"{company} reported {fact.label} of {_display_value(fact.raw_value, fact.unit)} "
        f"for {fact.period} in {fact.title}."
    )
    signature = fact.signature
    return MasterExample(
        example_id=stable_id("finrag_table_", signature, question, width=20),
        group_id=stable_id("grp_", signature),
        query_type="direct_extraction",
        difficulty="simple",
        usage="train_candidate",
        question=question,
        scope=_scope(table),
        evidence=[_evidence(table)],
        numeric_facts=[
            NumericFactRecord(
                label=fact.label,
                value=_number_key(fact.raw_value) or fact.raw_value,
                unit=fact.unit,
                period=fact.period,
                source_ids=["S1"],
            )
        ],
        target=GroundedAnswerTarget(
            claims=[GroundedClaimTarget(text=claim, source_ids=["S1"])],
            source_ids=["S1"],
        ),
        provenance=_provenance(table, signature, "deterministic_table_miner_v1"),
        quality=QualityRecord(review_status="pending"),
    )


def _yoy_example(table: ParsedTable, newer: TableFact, older: TableFact) -> MasterExample | None:
    new_value = Decimal(_number_key(newer.raw_value) or "NaN")
    old_value = Decimal(_number_key(older.raw_value) or "NaN")
    if not new_value.is_finite() or not old_value.is_finite() or old_value == 0:
        return None
    if newer.unit != older.unit:
        return None
    difference_places = min(
        6,
        max(0, -new_value.as_tuple().exponent, -old_value.as_tuple().exponent),
    )
    difference = (new_value - old_value).quantize(
        Decimal(1).scaleb(-difference_places), rounding=ROUND_HALF_UP
    )
    percentage = ((difference / abs(old_value)) * Decimal(100)).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )
    difference_text = format(difference.normalize(), "f")
    percentage_text = format(percentage, "f")
    company = COMPANY_NAMES.get(table.ticker, table.ticker)
    question = (
        f"Using {company}'s {table.source_year} 10-K table \"{table.title}\", compare "
        f"{newer.label} for {older.period} and {newer.period}. Calculate the absolute "
        "change and percentage change using the newer period minus the older period."
    )
    facts = [
        NumericFactRecord(
            label=newer.label,
            value=format(new_value, "f"),
            unit=newer.unit,
            period=newer.period,
            source_ids=["S1"],
        ),
        NumericFactRecord(
            label=older.label,
            value=format(old_value, "f"),
            unit=older.unit,
            period=older.period,
            source_ids=["S1"],
        ),
    ]
    calculations = [
        CalculationRecord(
            operation="difference",
            operands=[format(new_value, "f"), format(old_value, "f")],
            result=difference_text,
            unit=newer.unit,
            decimal_places=difference_places,
            explanation=(
                f"Calculated as {format(new_value, 'f')} - {format(old_value, 'f')}."
            ),
            source_ids=["S1"],
        ),
        CalculationRecord(
            operation="percentage_change",
            operands=[format(new_value, "f"), format(old_value, "f")],
            result=percentage_text,
            unit="%",
            decimal_places=1,
            explanation="Calculated as ((newer - older) / abs(older)) * 100.",
            source_ids=["S1"],
        ),
    ]
    application_context = (
        "[APPLICATION-VALIDATED FINANCIAL FACTS]\n"
        f"F1: {newer.label} | value={facts[0].value} | unit={newer.unit} | "
        f"period={newer.period} | sources=S1\n"
        f"F2: {older.label} | value={facts[1].value} | unit={older.unit} | "
        f"period={older.period} | sources=S1\n\n"
        "[APPLICATION-VERIFIED CALCULATIONS]\n"
        f"C1: difference({facts[0].value}, {facts[1].value}) = {difference_text} "
        f"{newer.unit}; sources=S1\n"
        f"C2: percentage_change({facts[0].value}, {facts[1].value}) = "
        f"{percentage_text}%; sources=S1"
    )
    first_claim = (
        f"{company} reported {newer.label} of {_display_value(newer.raw_value, newer.unit)} "
        f"for {newer.period}, compared with {_display_value(older.raw_value, older.unit)} "
        f"for {older.period}."
    )
    second_claim = (
        f"The absolute change was {_display_value(difference_text, newer.unit)}, and the "
        f"percentage change was {percentage_text}%."
    )
    signature = sha256_text(newer.signature + older.signature + "year_over_year")
    return MasterExample(
        example_id=stable_id("finrag_yoy_", signature, width=20),
        group_id=stable_id("grp_", signature),
        query_type="year_over_year",
        difficulty="moderate",
        usage="train_candidate",
        question=question,
        scope=_scope(table),
        evidence=[_evidence(table)],
        numeric_facts=facts,
        calculations=calculations,
        application_context=application_context,
        target=GroundedAnswerTarget(
            claims=[
                GroundedClaimTarget(text=first_claim, source_ids=["S1"]),
                GroundedClaimTarget(text=second_claim, source_ids=["S1"]),
            ],
            source_ids=["S1"],
        ),
        provenance=_provenance(table, signature, "deterministic_table_miner_v1"),
        quality=QualityRecord(review_status="pending"),
    )


def _complete_table_example(table: ParsedTable) -> MasterExample:
    company = COMPANY_NAMES.get(table.ticker, table.ticker)
    question = (
        f"Show the complete \"{table.title}\" table from {company}'s "
        f"{table.source_year} 10-K. Preserve all available rows, columns, and units."
    )
    unit_line = "" if table.unit == "as reported" else f"\n\nUnit: {table.unit}"
    answer = f"## {table.title}{unit_line}\n\n{table.markdown}\n\n[S1]"
    signature = sha256_text(
        table.source_hash + normalized_text(table.title) + normalized_text(table.markdown)
    )
    return MasterExample(
        example_id=stable_id("finrag_fulltable_", signature, width=20),
        group_id=stable_id("grp_", signature),
        query_type="complete_table",
        difficulty="moderate",
        usage="train_candidate",
        question=question,
        scope=_scope(table),
        evidence=[_evidence(table)],
        target=GroundedAnswerTarget(answer=answer, claims=[], source_ids=["S1"]),
        provenance=_provenance(table, signature, "deterministic_table_miner_v1"),
        quality=QualityRecord(review_status="pending"),
    )


def _targeted_table_example(
    table: ParsedTable, newer: TableFact, older: TableFact
) -> MasterExample | None:
    if newer.unit != older.unit:
        return None
    company = COMPANY_NAMES.get(table.ticker, table.ticker)
    question = (
        f"Using {company}'s {table.source_year} 10-K, show {newer.label} for "
        f"{older.period} and {newer.period} in a compact table. Preserve reported units."
    )
    markdown = (
        "| Period | Metric | Value |\n"
        "| --- | --- | --- |\n"
        f"| {older.period} | {older.label.replace('|', '/')} | "
        f"{_display_value(older.raw_value, older.unit).replace('$', 'USD ')} |\n"
        f"| {newer.period} | {newer.label.replace('|', '/')} | "
        f"{_display_value(newer.raw_value, newer.unit).replace('$', 'USD ')} |"
    )
    answer = f"## {newer.label}\n\nUnit: {newer.unit}\n\n{markdown}\n\n[S1]"
    signature = sha256_text(newer.signature + older.signature + "targeted_table")
    return MasterExample(
        example_id=stable_id("finrag_targettable_", signature, width=20),
        group_id=stable_id("grp_", signature),
        query_type="targeted_table",
        difficulty="moderate",
        usage="train_candidate",
        question=question,
        scope=_scope(table),
        evidence=[_evidence(table)],
        numeric_facts=[
            NumericFactRecord(
                label=older.label,
                value=_number_key(older.raw_value) or older.raw_value,
                unit=older.unit,
                period=older.period,
                source_ids=["S1"],
            ),
            NumericFactRecord(
                label=newer.label,
                value=_number_key(newer.raw_value) or newer.raw_value,
                unit=newer.unit,
                period=newer.period,
                source_ids=["S1"],
            ),
        ],
        target=GroundedAnswerTarget(answer=answer, claims=[], source_ids=["S1"]),
        provenance=_provenance(table, signature, "deterministic_table_miner_v1"),
        quality=QualityRecord(review_status="pending"),
    )


def _canonical_common_metric(label: str) -> str | None:
    cleaned = normalized_text(label).replace("\u2014", " ")
    cleaned = re.sub(r"[^a-z ]+", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    patterns = (
        (r"(?:total )?revenues?", "total revenue"),
        (r"net income", "net income"),
        (r"operating income", "operating income"),
        (r"total assets", "total assets"),
        (r"total liabilities", "total liabilities"),
        (r"total current assets", "total current assets"),
        (r"total current liabilities", "total current liabilities"),
        (r"cash and cash equivalents", "cash and cash equivalents"),
        (r"research and development", "research and development"),
    )
    for pattern, metric in patterns:
        if re.fullmatch(pattern, cleaned):
            return metric
    return None


def _fact_evidence(fact: TableFact, source_id: str) -> EvidenceBlock:
    return EvidenceBlock(
        source_id=source_id,
        parent_id=fact.parent_id,
        ticker=fact.ticker,
        fiscal_year=fact.source_year,
        doc_type="10K",
        source=fact.source,
        section=fact.section,
        text=fact.evidence_text,
        text_sha256=sha256_text(fact.evidence_text),
        source_hash=fact.source_hash,
    )


def _cross_company_example(left: TableFact, right: TableFact) -> MasterExample | None:
    metric = _canonical_common_metric(left.label)
    if (
        metric is None
        or metric != _canonical_common_metric(right.label)
        or left.ticker == right.ticker
        or left.period != right.period
        or left.source_year != right.source_year
        or left.unit != right.unit
    ):
        return None
    left_value = Decimal(_number_key(left.raw_value) or "NaN")
    right_value = Decimal(_number_key(right.raw_value) or "NaN")
    if not left_value.is_finite() or not right_value.is_finite():
        return None
    ordered = sorted((left, right), key=lambda fact: fact.ticker)
    left, right = ordered
    left_value = Decimal(_number_key(left.raw_value) or "NaN")
    right_value = Decimal(_number_key(right.raw_value) or "NaN")
    left_company = COMPANY_NAMES.get(left.ticker, left.ticker)
    right_company = COMPANY_NAMES.get(right.ticker, right.ticker)
    if left_value == right_value:
        conclusion = f"The two companies reported the same {metric}."
    else:
        winner = left_company if left_value > right_value else right_company
        conclusion = f"{winner} reported the higher {metric}."
    question = (
        f"Using the {left.source_year} 10-K filings, compare {metric} for "
        f"{left_company} and {right_company} for {left.period}, and identify which "
        "company reported the higher value. Preserve reported units."
    )
    claims = [
        GroundedClaimTarget(
            text=(
                f"{left_company} reported {metric} of "
                f"{_display_value(left.raw_value, left.unit)} for {left.period}."
            ),
            source_ids=["S1"],
        ),
        GroundedClaimTarget(
            text=(
                f"{right_company} reported {metric} of "
                f"{_display_value(right.raw_value, right.unit)} for {right.period}."
            ),
            source_ids=["S2"],
        ),
        GroundedClaimTarget(text=conclusion, source_ids=["S1", "S2"]),
    ]
    signature = sha256_text(left.signature + right.signature + "cross_company")
    return MasterExample(
        example_id=stable_id("finrag_compare_", signature, width=20),
        group_id=stable_id("grp_", signature),
        query_type="comparison",
        difficulty="moderate",
        usage="train_candidate",
        question=question,
        scope=ScopeRecord(
            tickers=[left.ticker, right.ticker],
            years=[left.source_year],
            doc_types=["10K"],
        ),
        evidence=[_fact_evidence(left, "S1"), _fact_evidence(right, "S2")],
        numeric_facts=[
            NumericFactRecord(
                label=metric,
                value=format(left_value, "f"),
                unit=left.unit,
                period=left.period,
                source_ids=["S1"],
            ),
            NumericFactRecord(
                label=metric,
                value=format(right_value, "f"),
                unit=right.unit,
                period=right.period,
                source_ids=["S2"],
            ),
        ],
        target=GroundedAnswerTarget(
            claims=claims, source_ids=["S1", "S2"]
        ),
        provenance=ProvenanceRecord(
            generator="deterministic_cross_company_miner_v1",
            source_record_ids=[left.parent_id, right.parent_id],
            source_hashes=list(dict.fromkeys([left.source_hash, right.source_hash])),
            canonical_fact_signature=signature,
        ),
        quality=QualityRecord(review_status="pending"),
    )


def _balanced_take(examples: Iterable[MasterExample], limit: int) -> list[MasterExample]:
    if limit <= 0:
        return []
    buckets: dict[tuple[str, str], deque[MasterExample]] = defaultdict(deque)
    for example in sorted(examples, key=lambda row: row.example_id):
        buckets[(example.scope.tickers[0], example.scope.years[0])].append(example)
    output: list[MasterExample] = []
    keys = sorted(buckets)
    while len(output) < limit and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(output) < limit:
                output.append(buckets[key].popleft())
    return output


def _unambiguous_questions(examples: Iterable[MasterExample]) -> list[MasterExample]:
    """Drop conflicting prompts and collapse source-identical duplicate prompts."""
    groups: dict[str, list[MasterExample]] = defaultdict(list)
    for example in examples:
        groups[question_fingerprint(example.question)].append(example)
    output: list[MasterExample] = []
    for rows in groups.values():
        if len(rows) == 1:
            output.append(rows[0])
            continue
        targets = {row.target.model_dump_json() for row in rows}
        if len(targets) == 1:
            output.append(sorted(rows, key=lambda row: row.example_id)[0])
        # Conflicting targets for the same natural-language question are unsafe;
        # all of them are excluded instead of selecting one silently.
    return output


def mine_table_candidates(
    parent_dir: Path,
    *,
    direct_limit: int = 500,
    yoy_limit: int = 150,
    targeted_table_limit: int = 50,
    comparison_limit: int = 60,
    complete_table_limit: int = 50,
) -> list[MasterExample]:
    tables = load_tables(parent_dir)
    direct: dict[str, MasterExample] = {}
    yoy: dict[str, MasterExample] = {}
    targeted: dict[str, MasterExample] = {}
    comparisons: dict[str, MasterExample] = {}
    complete: dict[str, MasterExample] = {}
    for table in tables:
        for fact in table.facts:
            example = _direct_example(table, fact)
            direct.setdefault(example.provenance.canonical_fact_signature, example)
        by_label: dict[str, list[TableFact]] = defaultdict(list)
        for fact in table.facts:
            by_label[normalized_text(fact.label)].append(fact)
        for facts in by_label.values():
            unique_by_period = {fact.period: fact for fact in facts}
            ordered = sorted(unique_by_period.values(), key=lambda fact: int(fact.period))
            if len(ordered) < 2:
                continue
            example = _yoy_example(table, ordered[-1], ordered[-2])
            if example is not None:
                yoy.setdefault(example.provenance.canonical_fact_signature, example)
            targeted_example = _targeted_table_example(table, ordered[-1], ordered[-2])
            if targeted_example is not None:
                targeted.setdefault(
                    targeted_example.provenance.canonical_fact_signature,
                    targeted_example,
                )
        table_example = _complete_table_example(table)
        complete.setdefault(table_example.provenance.canonical_fact_signature, table_example)

    common: dict[tuple[str, str, str, str], dict[str, TableFact]] = defaultdict(dict)
    for table in tables:
        for fact in table.facts:
            metric = _canonical_common_metric(fact.label)
            if metric is None:
                continue
            key = (fact.source_year, fact.period, fact.unit, metric)
            common[key].setdefault(fact.ticker, fact)
    for by_ticker in common.values():
        if {"MSFT", "TSLA"}.issubset(by_ticker):
            example = _cross_company_example(by_ticker["MSFT"], by_ticker["TSLA"])
            if example is not None:
                comparisons.setdefault(example.provenance.canonical_fact_signature, example)
    return (
        _balanced_take(_unambiguous_questions(direct.values()), direct_limit)
        + _balanced_take(_unambiguous_questions(yoy.values()), yoy_limit)
        + _balanced_take(_unambiguous_questions(targeted.values()), targeted_table_limit)
        + _balanced_take(_unambiguous_questions(comparisons.values()), comparison_limit)
        + _balanced_take(_unambiguous_questions(complete.values()), complete_table_limit)
    )
