"""Dynamic normalization and comparison for reported financial metric labels."""

from __future__ import annotations

import re
from difflib import SequenceMatcher


_STOPWORDS = frozenset({"a", "an", "and", "for", "from", "of", "the", "total", "net",
                        "under", "with", "by", "at", "as", "in", "on", "to"})
_TOKEN_ALIASES = {
    # Filing vocabularies use both phrases for the same per-share concept.
    "earning": "income",
}


def canonicalize_financial_metric(term: str) -> str:
    """Expand supported financial abbreviations into one stable metric label.

    A bare EPS request is resolved to diluted EPS.  This is an application-wide
    reporting policy, not fuzzy evidence matching: explicit ``basic`` and
    ``diluted`` requests remain distinct so validation cannot swap their rows.
    """
    text = " ".join(str(term or "").strip().split())
    lowered = text.casefold()
    expanded = re.sub(
        r"\bearnings?\s+per\s+diluted\s+shares?\b",
        "diluted earnings per share",
        lowered,
    )
    expanded = re.sub(
        r"(?<![a-z0-9])e\.?p\.?s\.?(?![a-z0-9])",
        "earnings per share",
        expanded,
    )
    expanded = re.sub(r"\bearnings?\s+per[\s-]+shares?\b", "earnings per share", expanded)
    if expanded == lowered and "earnings per share" not in expanded:
        return text
    if (
        "earnings per share" in expanded
        and not re.search(r"\b(?:basic|diluted)\b", expanded)
    ):
        expanded = expanded.replace("earnings per share", "diluted earnings per share", 1)
    return " ".join(expanded.split())


def has_financial_metric_conflict(source_label: str, target_metric: str) -> bool:
    """Reject a locally related table measure with a different quantity type."""
    source_terms = normalize_financial_term(source_label)
    target_terms = normalize_financial_term(target_metric)
    if (
        "expense" in source_terms
        and "income" in target_terms
        and "expense" not in target_terms
    ):
        return True
    if not {"per", "share"}.issubset(target_terms):
        return False
    share_count = re.compile(
        r"\bweighted[\s-]+average\s+shares?\b|"
        r"\bshares?\s+(?:used\s+)?in\s+computing\b|"
        r"\bshares?\s+(?:issued\s+and\s+)?outstanding\b",
        re.IGNORECASE,
    )
    return bool(share_count.search(source_label)) and not share_count.search(target_metric)


def normalize_financial_term(term: str) -> set[str]:
    """Normalize a metric label into meaningful, lightly singularized tokens."""
    text = re.sub(r"\([^)]*\)", " ", str(term or "").casefold())
    tokens = re.findall(r"[a-z0-9]+", re.sub(r"[^a-z0-9]+", " ", text))
    return {
        _TOKEN_ALIASES.get(normalized, normalized)
        for token in tokens
        if token not in _STOPWORDS
        for normalized in (
            token[:-1]
            if len(token) > 4 and token.endswith("s") and not token.endswith("ss")
            else token,
        )
    }


def is_metric_match(
    extracted_label: str,
    target_metric: str,
    threshold: float = 0.75,
) -> bool:
    """Compare labels with token overlap and character similarity."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    extracted = normalize_financial_term(extracted_label)
    target = normalize_financial_term(target_metric)
    if not extracted or not target:
        return False
    # Match inflections one-to-one; unmatched qualifiers still lower overlap.
    pairs = sorted((SequenceMatcher(None, left, right).ratio(), left, right)
                   for left in extracted for right in target)
    matched_extracted, matched_target = set(), set()
    for score, left, right in reversed(pairs):
        if score >= threshold and left not in matched_extracted and right not in matched_target:
            matched_extracted.add(left)
            matched_target.add(right)
    matched = len(matched_extracted)
    overlap = 2 * matched / (len(extracted) + len(target))
    similarity = SequenceMatcher(
        None, " ".join(sorted(extracted)), " ".join(sorted(target))
    ).ratio()
    return min(overlap, similarity) >= threshold
