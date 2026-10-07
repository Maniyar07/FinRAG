from __future__ import annotations

from pathlib import Path

from .models import MasterExample
from .utils import jaccard, normalized_text, question_fingerprint, read_json, token_ngrams
from .validators import _number_key, number_keys


def _label_terms(value: str) -> set[str]:
    stop = {
        "according", "calculate", "company", "fiscal", "from", "reported",
        "state", "table", "using", "value", "what", "which", "with", "year",
    }
    return {
        token for token in normalized_text(value).replace("-", " ").split()
        if len(token) > 2 and token not in stop
    }


def mark_benchmark_overlaps(
    examples: list[MasterExample], benchmark_path: Path
) -> list[MasterExample]:
    """Conservatively prevent canonical evaluation facts from entering training."""

    records = read_json(benchmark_path)
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark must be a non-empty JSON array")
    benchmark_questions = {
        question_fingerprint(str(record.get("question") or "")) for record in records
    }
    benchmark_ngrams = [
        token_ngrams(str(record.get("question") or "")) for record in records
    ]
    fact_keys: set[tuple[str, str, str, str]] = set()
    loose_facts: list[tuple[set[str], set[str], set[str]]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        sources = {
            str(item.get("document") or "") for item in metadata.get("sources") or []
        }
        text = f"{record.get('question', '')} {record.get('reference', '')}"
        terms = _label_terms(text)
        values = {
            value for value in number_keys(str(record.get("reference") or ""))
            if value not in set(str(year) for year in range(1990, 2101))
        }
        loose_facts.append((sources, terms, values))
        for fact in metadata.get("numeric_facts") or []:
            value = _number_key(str(fact.get("value") or ""))
            if value is None:
                continue
            for source in sources:
                fact_keys.add(
                    (
                        source,
                        normalized_text(str(fact.get("label") or "")),
                        value,
                        normalized_text(str(fact.get("period") or "")),
                    )
                )

    protected: list[MasterExample] = []
    for example in examples:
        reasons: list[str] = []
        if question_fingerprint(example.question) in benchmark_questions:
            reasons.append("benchmark_question_exact_match")
        question_ngrams = token_ngrams(example.question)
        if any(jaccard(question_ngrams, values) >= 0.95 for values in benchmark_ngrams):
            reasons.append("benchmark_question_near_match")
        sources = {block.source for block in example.evidence}
        for fact in example.numeric_facts:
            value = _number_key(fact.value)
            if value is None:
                continue
            exact = {
                (source, normalized_text(fact.label), value, normalized_text(fact.period))
                for source in sources
            }
            if exact & fact_keys:
                reasons.append("benchmark_numeric_fact_exact_match")
                continue
            fact_terms = _label_terms(fact.label)
            for gold_sources, gold_terms, gold_values in loose_facts:
                if not (sources & gold_sources) or value not in gold_values:
                    continue
                if fact_terms and len(fact_terms & gold_terms) / len(fact_terms) >= 0.6:
                    reasons.append("benchmark_numeric_fact_probable_match")
                    break
        overlap = bool(reasons) or example.quality.benchmark_overlap
        warnings = list(dict.fromkeys([*example.quality.warnings, *reasons]))
        quality = example.quality.model_copy(
            update={"benchmark_overlap": overlap, "warnings": warnings}
        )
        protected.append(example.model_copy(update={"quality": quality}))
    return protected
