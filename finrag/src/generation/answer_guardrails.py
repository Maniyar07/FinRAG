from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.schemas import RetrievalBundle, Scope


CITATION_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
CITATION_VARIANT_RE = re.compile(
    r"(?:\[\s*(?:SOURCE\s+)?(S\d+)\s*\]|\(\s*(S\d+)\s*\))",
    re.IGNORECASE,
)
SOURCE_ID_RE = re.compile(r"^\[?\s*(S\d+)\s*\]?$", re.IGNORECASE)
TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
PLACEHOLDER_VALUE_RE = re.compile(r"(?:\$\s*\?{2,}|\b(?:USD|EUR|GBP)\s+\?{2,})", re.IGNORECASE)
MAX_MARKDOWN_TABLE_COLUMNS = 8
INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I cannot find sufficient evidence in the retrieved financial documents to answer this question."
)
UNVERIFIABLE_RESPONSE = (
    "Relevant evidence was retrieved, but the generated answer could not be verified. "
    "Please retry this request."
)


@dataclass(frozen=True)
class AnswerValidation:
    """Result of validating one model answer against the retrieved source IDs."""

    valid: bool
    answer: str
    reason: str
    source_ids: tuple[str, ...] = ()


def normalize_citations(answer: str) -> str:
    """Normalize harmless model variations such as ``(S1)`` or ``[Source S1]``."""

    def replacement(match: re.Match[str]) -> str:
        source_id = next(group for group in match.groups() if group)
        return f"[{source_id.upper()}]"

    return CITATION_VARIANT_RE.sub(replacement, answer)


def normalize_table_currency(answer: str) -> str:
    """Escape currency markers only in pipe tables so Streamlit renders them literally."""
    lines: list[str] = []
    for line in answer.splitlines():
        if line.strip().startswith("|"):
            line = re.sub(r"(?<!\\)\$", r"\\$", line)
        lines.append(line)
    return "\n".join(lines)


def _table_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    if not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped[1:-1])]


def validate_markdown_tables(answer: str, *, require_table: bool = False) -> str | None:
    """Return a validation error for malformed or overly wide pipe tables."""
    if "****" in answer:
        return "malformed_empty_bold_marker"
    if answer.count("`") % 2:
        return "unmatched_markdown_backtick"
    if PLACEHOLDER_VALUE_RE.search(answer):
        return "placeholder_financial_value"

    table_blocks: list[list[tuple[int, list[str]]]] = []
    active: list[tuple[int, list[str]]] = []
    for line_number, line in enumerate(answer.splitlines(), start=1):
        cells = _table_cells(line)
        if cells is None:
            # A pipe-rich line that is not a pipe-wrapped row is commonly a
            # collapsed Markdown table. Streamlit otherwise renders the whole
            # line as a heading or paragraph.
            if len(re.findall(r"(?<!\\)\|", line)) >= 2:
                return f"malformed_table_row:{line_number}"
            if active:
                table_blocks.append(active)
                active = []
            continue
        if not cells:
            return f"malformed_table_row:{line_number}"
        if "`" in line:
            return f"code_backtick_in_table:{line_number}"
        active.append((line_number, cells))
    if active:
        table_blocks.append(active)

    if require_table and not table_blocks:
        return "required_markdown_table_missing"

    for block in table_blocks:
        header_line, header = block[0]
        column_count = len(header)
        if column_count > MAX_MARKDOWN_TABLE_COLUMNS:
            return f"wide_markdown_table:{column_count}_columns"
        if column_count < 2:
            return f"malformed_table_header:{header_line}"
        if len(block) < 3:
            return f"incomplete_markdown_table:{header_line}"

        separator_line, separator = block[1]
        if len(separator) != column_count or not all(
            TABLE_SEPARATOR_CELL_RE.fullmatch(cell.replace(" ", ""))
            for cell in separator
        ):
            return f"invalid_table_separator:{separator_line}"

        for line_number, cells in block[2:]:
            if len(cells) != column_count:
                return (
                    f"inconsistent_table_columns:{line_number}:"
                    f"expected_{column_count}:found_{len(cells)}"
                )
    return None


def _normalize_declared_ids(values: Iterable[object] | None) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values or ():
        match = SOURCE_ID_RE.fullmatch(str(value))
        if not match:
            normalized.append(str(value).strip().upper())
            continue
        source_id = match.group(1).upper()
        if source_id not in normalized:
            normalized.append(source_id)
    return tuple(normalized)


def missing_comparison_groups(bundle: RetrievalBundle) -> tuple[tuple[str, ...], ...]:
    if not bundle.scope.is_comparison:
        return ()
    if bundle.scope.required_doc_types:
        required = {
            (ticker, year, document_type)
            for ticker, year in bundle.scope.groups
            for document_type in bundle.scope.required_doc_types
        }
        covered = {
            (str(source["ticker"]), str(source["fiscal_year"]), str(source["doc_type"]))
            for source in bundle.sources
        }
    else:
        required = set(bundle.scope.groups)
        covered = {
            (str(source["ticker"]), str(source["fiscal_year"]))
            for source in bundle.sources
        }
    return tuple(sorted(required - covered))


def validate_answer_payload(
    answer: str,
    sources: list[dict],
    declared_source_ids: Iterable[object] | None = None,
    *,
    require_markdown_table: bool = False,
) -> AnswerValidation:
    """Validate inline and structured citations without rejecting harmless variations."""

    cleaned = normalize_citations(answer.strip())
    if not cleaned:
        return AnswerValidation(False, UNVERIFIABLE_RESPONSE, "empty_answer")
    if cleaned == INSUFFICIENT_EVIDENCE_RESPONSE:
        return AnswerValidation(True, cleaned, "insufficient_evidence")

    table_error = validate_markdown_tables(
        cleaned,
        require_table=require_markdown_table,
    )
    if table_error:
        return AnswerValidation(False, UNVERIFIABLE_RESPONSE, table_error)
    cleaned = normalize_table_currency(cleaned)

    allowed = {str(source["id"]).upper() for source in sources}
    inline_ids = tuple(dict.fromkeys(value.upper() for value in CITATION_RE.findall(cleaned)))
    declared_ids = _normalize_declared_ids(declared_source_ids)

    invalid_inline = tuple(value for value in inline_ids if value not in allowed)
    if invalid_inline:
        return AnswerValidation(
            False,
            UNVERIFIABLE_RESPONSE,
            f"unknown_inline_source_ids:{','.join(invalid_inline)}",
        )

    invalid_declared = tuple(value for value in declared_ids if value not in allowed)
    if invalid_declared:
        return AnswerValidation(
            False,
            UNVERIFIABLE_RESPONSE,
            f"unknown_declared_source_ids:{','.join(invalid_declared)}",
        )

    if inline_ids:
        return AnswerValidation(True, cleaned, "valid_inline_citations", inline_ids)

    if declared_ids:
        appended = cleaned.rstrip() + " " + "".join(f"[{value}]" for value in declared_ids)
        return AnswerValidation(
            True,
            appended,
            "valid_structured_ids_appended",
            declared_ids,
        )

    return AnswerValidation(False, UNVERIFIABLE_RESPONSE, "missing_source_ids")


def validate_generated_answer(answer: str, sources: list[dict]) -> str:
    """Backward-compatible text-only validator used by existing callers and tests."""

    validation = validate_answer_payload(answer, sources)
    return validation.answer if validation.valid else UNVERIFIABLE_RESPONSE


def scope_coverage_message(missing: tuple[tuple[str, ...], ...], scope: Scope) -> str:
    del scope
    labels = ", ".join(" ".join(group) for group in missing)
    return (
        f"I cannot complete the requested comparison because relevant evidence was not "
        f"retrieved for: {labels}. No complete comparison was generated."
    )
