from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from typing import Iterable

from src.generation.answer_guardrails import INSUFFICIENT_EVIDENCE_RESPONSE

from .models import (
    CalculationRecord,
    GroundedAnswerTarget,
    GroundedClaimTarget,
    MasterExample,
    ProvenanceRecord,
    QualityRecord,
)
from .utils import normalized_text, question_fingerprint, sha256_text, stable_id


COMPANY_NAMES = {"MSFT": "Microsoft", "TSLA": "Tesla"}
REVIEWED_BY = "project-owner-authorized-deterministic-review-v1"
REVIEWED_AT = "2026-10-02T17:30:00+05:30"
REVIEW_NOTES = (
    "The project owner explicitly authorized this deterministic augmentation. "
    "The source example was previously approved; schema, evidence availability, "
    "arithmetic, citation, duplicate, benchmark, and split-isolation gates must "
    "still pass before export."
)
SAFETY_CATEGORIES = (
    "unavailable_metric",
    "unavailable_period",
    "out_of_scope_company",
    "missing_transcript",
    "incomplete_comparison",
)
UNAVAILABLE_METRICS = (
    "dividend payout ratio",
    "external credit rating",
    "market share percentage",
    "employee satisfaction score",
    "Scope 3 emissions",
    "retail store count",
)
CALCULATION_PREFERRED_LABELS = (
    "revenue",
    "cost of revenue",
    "gross profit",
    "gross margin",
    "operating expenses",
    "net cash",
    "income taxes",
    "effective tax rate",
    "sales and marketing",
    "research and development",
    "buildings and improvements",
    "leasehold improvements",
)


def _approved_source(example: MasterExample) -> bool:
    return (
        example.usage == "train_candidate"
        and example.quality.automated_status == "passed"
        and example.quality.review_status == "approved"
        and not example.quality.benchmark_overlap
        and example.split in {"train", "validation", "test"}
    )


def _review() -> QualityRecord:
    return QualityRecord(
        automated_status="pending",
        review_status="approved",
        reviewed_by=REVIEWED_BY,
        reviewed_at=REVIEWED_AT,
        reviewer_notes=REVIEW_NOTES,
    )


def _split_quotas(target_count: int) -> dict[str, int]:
    if target_count <= 0:
        return {"train": 0, "validation": 0, "test": 0}
    if target_count < 5:
        return {"train": target_count, "validation": 0, "test": 0}
    validation = max(1, round(target_count * 0.08))
    test = max(1, round(target_count * 0.08))
    return {
        "train": target_count - validation - test,
        "validation": validation,
        "test": test,
    }


def _balanced_order(
    examples: Iterable[MasterExample], *, purpose: str, ranker=None
) -> list[MasterExample]:
    buckets: dict[tuple[str, str], deque[MasterExample]] = defaultdict(deque)
    for example in sorted(
        examples,
        key=lambda row: (
            ranker(row) if ranker is not None else (),
            sha256_text(f"{purpose}\x1f{row.example_id}"),
        ),
    ):
        key = (example.scope.tickers[0], example.scope.years[0])
        buckets[key].append(example)
    output: list[MasterExample] = []
    keys = sorted(buckets)
    while any(buckets.values()):
        for key in keys:
            if buckets[key]:
                output.append(buckets[key].popleft())
    return output


def _clean_calculation_source(example: MasterExample) -> bool:
    label = example.numeric_facts[0].label.strip()
    lowered = label.casefold()
    rejected_fragments = (
        "\ufffd",
        "[",
        "]",
        "$",
        "except ",
        " issuance",
        "gross amounts of derivatives",
    )
    periods = [fact.period for fact in example.numeric_facts[:2]]
    return (
        2 <= len(label) <= 80
        and not label[0].isdigit()
        and not any(value in lowered for value in rejected_fragments)
        and all(re.fullmatch(r"(?:19|20)\d{2}", value) for value in periods)
        and abs(int(periods[0]) - int(periods[1])) == 1
        and any(value in lowered for value in CALCULATION_PREFERRED_LABELS)
    )


def _calculation_rank(example: MasterExample) -> tuple[int, int]:
    label = normalized_text(example.numeric_facts[0].label)
    penalty = 0 if any(value in label for value in CALCULATION_PREFERRED_LABELS) else 2
    if label == "total":
        penalty += 8
    if any(value in label for value in ("derivative", "carrying amount", "shares")):
        penalty += 5
    if "adjusted" in label or "non-gaap" in label:
        penalty += 2
    return penalty, len(label)


def _table_name(question: str) -> str:
    match = re.search(r'table\s+"([^"]+)"', question, flags=re.IGNORECASE)
    return match.group(1).strip() if match else "the retrieved financial table"


def _calculation_question(
    base: MasterExample,
    calculation: CalculationRecord,
) -> str:
    newer, older = base.numeric_facts[:2]
    company = COMPANY_NAMES.get(base.scope.tickers[0], base.scope.tickers[0])
    table = _table_name(base.question)
    if calculation.operation == "difference":
        return (
            f"Use the verified {newer.label} operands in {company}'s "
            f"{base.scope.years[0]} 10-K table \"{table}\". What absolute change "
            f"does the calculator produce from {older.period} to {newer.period} "
            "when it subtracts the older value from the newer value? State the "
            "operands, result, and reported unit."
        )
    places = calculation.decimal_places
    return (
        f"For {newer.label} in {company}'s {base.scope.years[0]} 10-K table "
        f"\"{table}\", calculate the percentage change from {older.period} to "
        f"{newer.period} as ((newer - older) / abs(older)) * 100. Show both "
        f"verified operands and round the result to {places} decimal "
        f"place{'s' if places != 1 else ''}."
    )


def _calculation_claim(
    base: MasterExample,
    calculation: CalculationRecord,
) -> str:
    newer, older = base.numeric_facts[:2]
    if calculation.operation == "difference":
        return (
            f"Using the verified operands {newer.value} for {newer.period} and "
            f"{older.value} for {older.period}, the absolute change is "
            f"{calculation.result} {calculation.unit}, calculated as "
            f"{newer.value} - {older.value}."
        )
    return (
        f"Using the verified operands {newer.value} for {newer.period} and "
        f"{older.value} for {older.period}, the percentage change is "
        f"{calculation.result}%, calculated as (({newer.value} - "
        f"{older.value}) / abs({older.value})) * 100."
    )


def _calculation_context(
    base: MasterExample,
    calculation: CalculationRecord,
) -> str:
    fact_lines = [
        f"F{index}: {fact.label} | value={fact.value} | unit={fact.unit} | "
        f"period={fact.period} | sources={','.join(fact.source_ids)}"
        for index, fact in enumerate(base.numeric_facts[:2], 1)
    ]
    return (
        "[APPLICATION-VALIDATED FINANCIAL FACTS]\n"
        + "\n".join(fact_lines)
        + "\n\n[APPLICATION-VERIFIED CALCULATIONS]\n"
        + f"C1: {calculation.operation}({', '.join(calculation.operands)}) = "
        + f"{calculation.result} {calculation.unit}; "
        + f"sources={','.join(calculation.source_ids)}"
    )


def _calculation_example(
    base: MasterExample,
    calculation: CalculationRecord,
) -> MasterExample:
    if calculation.operation == "difference" and calculation.unit == "%":
        calculation = calculation.model_copy(update={"unit": "percentage points"})
    question = _calculation_question(base, calculation)
    signature = sha256_text(
        f"{base.provenance.canonical_fact_signature}\x1fcalculation\x1f"
        f"{calculation.operation}"
    )
    return MasterExample(
        example_id=stable_id("finrag_calc_", signature, question, width=20),
        group_id=base.group_id,
        query_type="calculation",
        difficulty="moderate",
        usage="train_candidate",
        question=question,
        scope=base.scope,
        evidence=base.evidence,
        numeric_facts=base.numeric_facts[:2],
        calculations=[calculation],
        application_context=_calculation_context(base, calculation),
        target=GroundedAnswerTarget(
            claims=[
                GroundedClaimTarget(
                    text=_calculation_claim(base, calculation),
                    source_ids=calculation.source_ids,
                )
            ],
            source_ids=calculation.source_ids,
        ),
        provenance=ProvenanceRecord(
            index_version=base.provenance.index_version,
            generator="deterministic_calculation_augmentation_v1",
            prompt_version=base.provenance.prompt_version,
            source_record_ids=base.provenance.source_record_ids,
            source_hashes=base.provenance.source_hashes,
            canonical_fact_signature=signature,
        ),
        quality=_review(),
        split=base.split,
    )


def build_calculation_candidates(
    examples: Iterable[MasterExample], *, target_count: int = 25
) -> list[MasterExample]:
    rows = [
        row
        for row in examples
        if _approved_source(row)
        and row.query_type == "year_over_year"
        and len(row.numeric_facts) >= 2
        and {item.operation for item in row.calculations}
        >= {"difference", "percentage_change"}
        and _clean_calculation_source(row)
    ]
    quotas = _split_quotas(target_count)
    existing_questions = {question_fingerprint(row.question) for row in examples}
    output: list[MasterExample] = []
    operation_counts: Counter[str] = Counter()
    for split in ("train", "validation", "test"):
        candidates = _balanced_order(
            (row for row in rows if row.split == split),
            purpose=f"calculation-{split}",
            ranker=_calculation_rank,
        )
        selected = 0
        for base in candidates:
            preferred = (
                "difference"
                if operation_counts["difference"] <= operation_counts["percentage_change"]
                else "percentage_change"
            )
            calculation = next(
                item for item in base.calculations if item.operation == preferred
            )
            candidate = _calculation_example(base, calculation)
            fingerprint = question_fingerprint(candidate.question)
            if fingerprint in existing_questions:
                continue
            output.append(candidate)
            existing_questions.add(fingerprint)
            operation_counts[preferred] += 1
            selected += 1
            if selected == quotas[split]:
                break
        if selected != quotas[split]:
            raise ValueError(
                f"Only {selected} distinct {split} calculation candidates were "
                f"available; {quotas[split]} are required"
            )
    if len(output) != target_count:
        raise ValueError(f"Expected {target_count} calculation candidates, got {len(output)}")
    return output


def _clean_safety_source(example: MasterExample) -> bool:
    label = example.numeric_facts[0].label.strip()
    lowered = label.casefold()
    return (
        2 <= len(label) <= 70
        and label.casefold() != "total"
        and not label[0].isdigit()
        and not any(
            value in lowered
            for value in ("\ufffd", "\u2014", "[", "]", "$", "except ", "designated as")
        )
    )


def _safety_rank(example: MasterExample) -> tuple[int, int]:
    label = normalized_text(example.numeric_facts[0].label)
    preferred = (
        "revenue",
        "net income",
        "operating income",
        "operating expenses",
        "research and development",
        "sales and marketing",
        "cash and cash equivalents",
        "net cash",
        "total assets",
        "total liabilities",
        "effective tax rate",
        "income taxes",
        "automotive sales",
        "automotive leasing",
    )
    return (0 if any(value in label for value in preferred) else 2, len(label))


def _missing_metric(evidence_text: str, *, seed: str) -> str:
    normalized = normalized_text(evidence_text)
    start = int(sha256_text(seed)[:8], 16) % len(UNAVAILABLE_METRICS)
    ordered = UNAVAILABLE_METRICS[start:] + UNAVAILABLE_METRICS[:start]
    for metric in ordered:
        if normalized_text(metric) not in normalized:
            return metric
    raise ValueError("Could not find a demonstrably absent metric")


def _unavailable_year(base: MasterExample) -> str:
    present = set(re.findall(r"\b(?:19|20)\d{2}\b", "\n".join(
        block.text for block in base.evidence
    )))
    candidate = max(int(value) for value in base.scope.years) + 3
    while str(candidate) in present:
        candidate += 1
    return str(candidate)


def _safety_question(base: MasterExample, category: str) -> str:
    fact = base.numeric_facts[0]
    ticker = base.scope.tickers[0]
    company = COMPANY_NAMES.get(ticker, ticker)
    year = base.scope.years[0]
    if category == "unavailable_metric":
        missing = _missing_metric(
            "\n".join(block.text for block in base.evidence), seed=base.example_id
        )
        return (
            f"Using only the retrieved evidence from {company}'s {year} 10-K, "
            f"what {missing} was reported for {fact.period}?"
        )
    if category == "unavailable_period":
        unavailable = _unavailable_year(base)
        return (
            f"According to the retrieved {company} filing evidence, what value was "
            f"reported for {fact.label} in {unavailable}?"
        )
    if category == "out_of_scope_company":
        return (
            f"Using only these retrieved documents, what value did Apple report for "
            f"{fact.label} in {fact.period}?"
        )
    if category == "missing_transcript":
        return (
            f"What did {company} management say in its {year} earnings-call "
            f"transcript about {fact.label}? Use only the retrieved evidence."
        )
    if category == "incomplete_comparison":
        other = "Tesla" if ticker == "MSFT" else "Microsoft"
        return (
            f"Compare {company} with {other} on {fact.label} for {fact.period} and "
            "identify which company reported the higher value, using only the "
            "retrieved evidence."
        )
    raise ValueError(f"Unsupported safety category: {category}")


def _safety_example(base: MasterExample, category: str) -> MasterExample:
    question = _safety_question(base, category)
    signature = sha256_text(
        f"{base.provenance.canonical_fact_signature}\x1finsufficient-evidence\x1f{category}"
    )
    return MasterExample(
        example_id=stable_id("finrag_safety_", signature, question, width=20),
        group_id=base.group_id,
        query_type="insufficient_evidence",
        difficulty="moderate" if category != "incomplete_comparison" else "complex",
        usage="train_candidate",
        question=question,
        scope=base.scope,
        evidence=base.evidence,
        target=GroundedAnswerTarget(
            answer=INSUFFICIENT_EVIDENCE_RESPONSE,
            claims=[],
            source_ids=[],
        ),
        provenance=ProvenanceRecord(
            index_version=base.provenance.index_version,
            generator=f"deterministic_evidence_safety_v1:{category}",
            prompt_version=base.provenance.prompt_version,
            source_record_ids=base.provenance.source_record_ids,
            source_hashes=base.provenance.source_hashes,
            canonical_fact_signature=signature,
        ),
        quality=_review(),
        split=base.split,
    )


def build_evidence_safety_candidates(
    examples: Iterable[MasterExample], *, target_count: int = 25
) -> list[MasterExample]:
    if target_count % len(SAFETY_CATEGORIES) != 0:
        raise ValueError(
            f"Safety target count must be divisible by {len(SAFETY_CATEGORIES)}"
        )
    rows = [
        row
        for row in examples
        if _approved_source(row)
        and row.query_type == "direct_extraction"
        and row.numeric_facts
        and row.scope.doc_types == ["10K"]
        and _clean_safety_source(row)
    ]
    quotas = _split_quotas(target_count)
    existing_questions = {question_fingerprint(row.question) for row in examples}
    category_limit = target_count // len(SAFETY_CATEGORIES)
    category_counts: Counter[str] = Counter()
    output: list[MasterExample] = []
    used_groups: set[str] = set()
    slot = 0
    for split in ("train", "validation", "test"):
        candidates = _balanced_order(
            (row for row in rows if row.split == split),
            purpose=f"evidence-safety-{split}",
            ranker=_safety_rank,
        )
        selected = 0
        for base in candidates:
            if base.group_id in used_groups:
                continue
            available_categories = [
                name
                for name in SAFETY_CATEGORIES
                if category_counts[name] < category_limit
            ]
            if not available_categories:
                break
            category = available_categories[slot % len(available_categories)]
            candidate = _safety_example(base, category)
            fingerprint = question_fingerprint(candidate.question)
            if fingerprint in existing_questions:
                continue
            output.append(candidate)
            used_groups.add(base.group_id)
            existing_questions.add(fingerprint)
            category_counts[category] += 1
            selected += 1
            slot += 1
            if selected == quotas[split]:
                break
        if selected != quotas[split]:
            raise ValueError(
                f"Only {selected} distinct {split} evidence-safety candidates were "
                f"available; {quotas[split]} are required"
            )
    if len(output) != target_count:
        raise ValueError(f"Expected {target_count} safety candidates, got {len(output)}")
    if set(category_counts.values()) != {category_limit}:
        raise ValueError(f"Safety categories are not balanced: {dict(category_counts)}")
    return output

