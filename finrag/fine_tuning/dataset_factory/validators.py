from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Iterable

from src.generation.answer_guardrails import (
    CITATION_RE,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    validate_markdown_tables,
)

from .models import CalculationRecord, MasterExample
from .utils import (
    has_mojibake,
    jaccard,
    normalized_text,
    question_fingerprint,
    sha256_text,
    token_ngrams,
)


NUMBER_RE = re.compile(
    r"(?<![\w.])(?:USD\s*|\$\s*)?-?\(?\d[\d,]*(?:\.\d+)?\)?%?",
    re.IGNORECASE,
)
SOURCE_RE = re.compile(r"\bS[1-9]\d*\b", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetValidation:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    per_example: dict[str, tuple[str, ...]]


def _number_key(raw: str) -> str | None:
    raw_text = str(raw).strip()
    cleaned = re.sub(r"(?i)USD|[$,%\s()]", "", raw_text).replace(",", "")
    if not cleaned:
        return None
    # Accounting negatives may be written as ``$(123)`` or ``USD (123)``.
    if "(" in raw_text and ")" in raw_text and not cleaned.startswith("-"):
        cleaned = "-" + cleaned
    try:
        return format(Decimal(cleaned).normalize(), "f")
    except InvalidOperation:
        return None


def number_keys(value: str) -> set[str]:
    return {
        key for match in NUMBER_RE.finditer(value)
        if (key := _number_key(match.group(0))) is not None
    }


def _calculate(record: CalculationRecord) -> Decimal:
    values = [Decimal(_number_key(value) or "NaN") for value in record.operands]
    if any(not value.is_finite() for value in values):
        raise ValueError("invalid calculation operand")
    if record.operation == "difference":
        result = values[0] - values[1]
    elif record.operation == "percentage_change":
        if values[1] == 0:
            raise ValueError("percentage-change denominator is zero")
        result = ((values[0] - values[1]) / abs(values[1])) * Decimal(100)
    elif record.operation == "percentage_of":
        if values[1] == 0:
            raise ValueError("percentage-of denominator is zero")
        result = (values[0] / values[1]) * Decimal(100)
    elif record.operation == "ratio":
        if values[1] == 0:
            raise ValueError("ratio denominator is zero")
        result = values[0] / values[1]
    elif record.operation == "sum":
        result = sum(values, Decimal(0))
    else:  # pragma: no cover - Pydantic forbids other operations
        raise ValueError(f"unsupported operation: {record.operation}")
    quantum = Decimal(1).scaleb(-record.decimal_places)
    return result.quantize(quantum, rounding=ROUND_HALF_UP)


def validate_example(example: MasterExample) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    evidence_ids = [block.source_id for block in example.evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("duplicate_evidence_source_id")
    evidence_set = set(evidence_ids)
    for block in example.evidence:
        if sha256_text(block.text) != block.text_sha256:
            errors.append(f"evidence_hash_mismatch:{block.source_id}")

    if example.target.claims:
        used_ids = {
            source_id
            for claim in example.target.claims
            for source_id in claim.source_ids
        }
        if set(example.target.source_ids) != used_ids:
            errors.append("top_level_source_ids_do_not_match_claims")
        if any(CITATION_RE.search(claim.text) for claim in example.target.claims):
            errors.append("claim_text_contains_citation_markup")
    else:
        used_ids = set(example.target.source_ids)
        table_error = validate_markdown_tables(
            example.target.answer,
            require_table=example.query_type in {"targeted_table", "complete_table"},
        )
        if table_error:
            errors.append(f"target_table:{table_error}")
        cited = {value.upper() for value in CITATION_RE.findall(example.target.answer)}
        if cited != used_ids:
            errors.append("answer_citations_do_not_match_source_ids")
    unknown = used_ids - evidence_set
    if unknown:
        errors.append("unknown_target_source_ids:" + ",".join(sorted(unknown)))
    unused = evidence_set - used_ids
    if unused and example.query_type != "insufficient_evidence":
        warnings.append("unused_evidence_source_ids:" + ",".join(sorted(unused)))

    if example.query_type == "insufficient_evidence":
        if example.target.answer.strip() != INSUFFICIENT_EVIDENCE_RESPONSE:
            errors.append("insufficient_evidence_target_must_use_canonical_response")
        if example.target.claims or example.target.source_ids:
            errors.append("insufficient_evidence_target_must_not_cite_sources")
        if example.calculations:
            errors.append("insufficient_evidence_target_must_not_include_calculations")
    elif example.target.answer.strip() == INSUFFICIENT_EVIDENCE_RESPONSE:
        errors.append("canonical_insufficient_response_requires_matching_query_type")
    if example.query_type == "calculation" and not example.calculations:
        errors.append("calculation_query_requires_verified_calculation")

    context = "\n".join(block.text for block in example.evidence)
    context += "\n" + example.application_context
    supported_numbers = number_keys(context) | number_keys(example.question)
    # Years in authoritative source metadata are also legitimate output tokens.
    supported_numbers.update(example.scope.years)
    target_text = example.target.answer + "\n" + "\n".join(
        claim.text for claim in example.target.claims
    )
    unsupported = number_keys(target_text) - supported_numbers
    if unsupported:
        errors.append("unsupported_target_numbers:" + ",".join(sorted(unsupported)))

    for index, calculation in enumerate(example.calculations, 1):
        try:
            expected = _calculate(calculation)
            actual_key = _number_key(calculation.result)
            actual = Decimal(actual_key) if actual_key is not None else Decimal("NaN")
        except (InvalidOperation, ValueError) as error:
            errors.append(f"calculation_{index}:{error}")
            continue
        if actual != expected:
            errors.append(f"calculation_{index}:wrong_result:{actual}:expected:{expected}")
        if not set(calculation.source_ids).issubset(evidence_set):
            errors.append(f"calculation_{index}:unknown_source_id")

    question_source_ids = SOURCE_RE.findall(example.question)
    if question_source_ids:
        errors.append("question_exposes_internal_source_ids")
    if len(example.question.split()) < 4:
        errors.append("question_too_short")
    supervised_text = example.question + "\n" + target_text
    if has_mojibake(supervised_text):
        errors.append("mojibake_in_supervised_question_or_target")
    if example.usage == "train_candidate" and example.quality.review_status == "approved":
        if not example.quality.reviewed_by.strip():
            errors.append("approved_training_example_missing_reviewer")
        try:
            reviewed_at = datetime.fromisoformat(
                example.quality.reviewed_at.replace("Z", "+00:00")
            )
            if reviewed_at.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError:
            errors.append("approved_training_example_invalid_review_timestamp")
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def validate_dataset(
    examples: Iterable[MasterExample], *, near_duplicate_threshold: float = 0.92
) -> DatasetValidation:
    rows = list(examples)
    global_errors: list[str] = []
    global_warnings: list[str] = []
    per_example: dict[str, tuple[str, ...]] = {}
    ids = [row.example_id for row in rows]
    for value, count in Counter(ids).items():
        if count > 1:
            global_errors.append(f"duplicate_example_id:{value}")

    fingerprints: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        fingerprints[question_fingerprint(row.question)].append(row.example_id)
        errors, warnings = validate_example(row)
        if errors:
            per_example[row.example_id] = tuple(errors)
        global_warnings.extend(f"{row.example_id}:{warning}" for warning in warnings)
    for values in fingerprints.values():
        if len(values) > 1:
            global_errors.append("duplicate_question:" + ",".join(sorted(values)))

    group_by_id = {row.example_id: row.group_id for row in rows}
    ngrams = [(row.example_id, token_ngrams(row.question)) for row in rows]
    for index, (left_id, left) in enumerate(ngrams):
        for right_id, right in ngrams[index + 1:]:
            score = jaccard(left, right)
            if score >= near_duplicate_threshold:
                finding = f"near_duplicate_question:{left_id}:{right_id}:{score:.3f}"
                # Different verified facts often share a deliberate question template.
                # They are not semantic duplicates.  Rewordings of the same fact group
                # are leakage risks and therefore remain hard failures.
                if group_by_id[left_id] == group_by_id[right_id]:
                    global_errors.append(finding)
                else:
                    global_warnings.append("shared_question_template:" + finding)

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.split:
            group_splits[row.group_id].add(row.split)
    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            global_errors.append(
                f"group_split_leakage:{group_id}:{','.join(sorted(splits))}"
            )

    for example_id, errors in per_example.items():
        global_errors.extend(f"{example_id}:{error}" for error in errors)
    return DatasetValidation(
        passed=not global_errors,
        errors=tuple(dict.fromkeys(global_errors)),
        warnings=tuple(dict.fromkeys(global_warnings)),
        per_example=per_example,
    )


def with_validation_status(examples: Iterable[MasterExample]) -> list[MasterExample]:
    output: list[MasterExample] = []
    for example in examples:
        errors, warnings = validate_example(example)
        quality = example.quality.model_copy(
            update={
                "automated_status": "failed" if errors else "passed",
                "issues": errors,
                "warnings": warnings,
            }
        )
        output.append(example.model_copy(update={"quality": quality}))
    return output
