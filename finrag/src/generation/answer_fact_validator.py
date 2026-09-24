"""Conservative cross-checks for values copied into generated answers.

Only a unique, unambiguous row/year match can invalidate an answer. If a
filing's table layout cannot be interpreted, generation continues unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser


METRICS = (
    "total automotive revenues",
    "intelligent cloud operating income",
    "cash and cash equivalents",
    "total current liabilities",
    "short-term investments",
    "research and development expense",
    "energy generation and storage revenue",
    "automotive regulatory credits revenue",
    "total assets",
    "total revenue",
    "net income",
    "operating income",
    "gross profit",
    "gross margin",
)
NUMBER_RE = re.compile(r"(?<![\w.])-?\(?\$?\s*\d[\d,]*(?:\.\d+)?\)?(?![\w.])")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
WORD_RE = re.compile(r"[a-z0-9]+")
CHANGE_QUESTION_RE = re.compile(
    r"\bhow did\s+(.+?\brevenue)\s+change\b", re.IGNORECASE
)


@dataclass(frozen=True)
class TableCheck:
    valid: bool
    reason: str


class _TableReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def _words(value: str) -> set[str]:
    words = set(WORD_RE.findall(value.casefold()))
    # Singular/plural variations occur between questions and filing row labels.
    return {word[:-1] if word.endswith("s") and len(word) > 4 else word for word in words}


def _amount(value: str) -> Decimal | None:
    compact = re.sub(r"[\s,$]", "", value)
    if compact.startswith("(") and compact.endswith(")"):
        compact = "-" + compact[1:-1]
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", compact):
        return None
    try:
        return Decimal(compact)
    except InvalidOperation:
        return None


def _answer_amounts(answer: str) -> list[tuple[Decimal, Decimal]]:
    """Return answer amounts in USD millions with their rounding tolerances."""
    amounts: list[tuple[Decimal, Decimal]] = []
    for match in NUMBER_RE.finditer(answer):
        amount = _amount(match.group())
        if amount is None:
            continue
        suffix = answer[match.end(): match.end() + 15].lstrip().casefold()
        scale = Decimal(1000) if suffix.startswith("billion") else Decimal(1)
        # A stated value such as 57.2 billion represents a rounded figure.
        decimals = len(match.group().split(".", 1)[1]) if "." in match.group() else 0
        tolerance = (Decimal("0.5") * (Decimal(10) ** -decimals) * scale)
        if scale == 1:
            tolerance = Decimal(0)
        amounts.append((amount * scale, tolerance))
    return amounts


def _question_metric(question: str) -> str | None:
    normalized = " ".join(WORD_RE.findall(question.casefold()))
    for metric in METRICS:
        if " ".join(WORD_RE.findall(metric)) in normalized:
            return metric
    return None


def _requested_year(question: str) -> str | None:
    years = YEAR_RE.findall(question)
    # A comparison year in the question is distinct from the filing year.
    if re.search(r"\bcomparative\b", question, re.IGNORECASE) and len(years) > 1:
        return years[-1]
    if len(set(years)) > 1:
        return None
    return years[-1] if years else None


def _table_values(evidence: str, metric: str, year: str) -> set[Decimal]:
    reader = _TableReader()
    reader.feed(evidence)
    expected_words = frozenset(_words(metric))
    values: set[Decimal] = set()
    for table in reader.tables:
        year_columns: dict[str, int] = {}
        section = ""
        for row in table:
            years = {
                cell: index for index, cell in enumerate(row)
                if YEAR_RE.fullmatch(cell)
            }
            if year in years and len(years) >= 2:
                year_columns = years
                continue
            if not row:
                continue
            if len(row) == 1 or (len(row) > 1 and all(not cell for cell in row[1:])):
                section = row[0]
                continue
            column = year_columns.get(year)
            if column is None or column >= len(row):
                continue
            label = row[0]
            # A segment row may be named "Intelligent Cloud" under an
            # "Operating Income" heading. Both must match the question.
            if expected_words not in {
                frozenset(_words(label)),
                frozenset(_words(f"{section} {label}")),
            }:
                continue
            value = _amount(row[column])
            if value is not None:
                values.add(value)
    return values


def validate_table_answer(question: str, answer: str, sources: list[dict]) -> TableCheck:
    """Reject only a clear contradiction of a unique source table value.

    Narrative answers, ambiguous tables and missing evidence are left for the
    existing citation guardrails and model to handle. This avoids turning a
    layout/parser limitation into an unnecessary abstention.
    """
    # A combined amount is derived from several rows; checking just one
    # component as though it were the requested answer would be incorrect.
    if re.search(r"\b(?:combined|sum of|total of)\b", question, re.IGNORECASE):
        return TableCheck(True, "table_check_composite_question")

    metric = _question_metric(question)
    year = _requested_year(question)
    if not metric or not year:
        return TableCheck(True, "table_check_not_applicable")

    answer_values = _answer_amounts(answer)
    if not answer_values:
        return TableCheck(True, "table_check_no_numeric_claim")

    checked = 0
    for source in sources:
        evidence = str(source.get("evidence_text") or source.get("full_evidence_text") or "")
        values = _table_values(evidence, metric, year)
        if len(values) != 1:
            continue
        checked += 1
        expected = next(iter(values))
        if any(abs(expected - amount) <= tolerance for amount, tolerance in answer_values):
            return TableCheck(True, "table_value_verified")

    if checked:
        return TableCheck(False, "requested_table_value_missing_from_answer")
    return TableCheck(True, "table_check_inconclusive")


def validate_revenue_change(question: str, answer: str, sources: list[dict]) -> TableCheck:
    """Reject a conflicting percentage stated for the exact requested metric."""
    requested = CHANGE_QUESTION_RE.search(question)
    if not requested:
        return TableCheck(True, "revenue_change_not_applicable")
    metric = " ".join(requested.group(1).split())
    pattern = re.compile(
        re.escape(metric)
        + r"\s+(?:increased|decreased|grew|declined)\b.{0,80}?\b(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE | re.DOTALL,
    )
    supported: set[Decimal] = set()
    for source in sources:
        evidence = str(source.get("evidence_text") or source.get("full_evidence_text") or "")
        for match in pattern.finditer(evidence):
            supported.add(Decimal(match.group(1)))
    if len(supported) != 1:
        return TableCheck(True, "revenue_change_inconclusive")

    claimed = {Decimal(match.group(1)) for match in pattern.finditer(answer)}
    expected = next(iter(supported))
    if claimed and claimed != {expected}:
        return TableCheck(False, f"requested_revenue_change_mismatch:{expected}%")
    return TableCheck(True, "revenue_change_verified" if claimed else "revenue_change_unstated")
