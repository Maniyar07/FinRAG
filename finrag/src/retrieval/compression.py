from __future__ import annotations

import re

from src.retrieval.lexical_index import tokenize
from src.retrieval.text_blocks import Block, blocks
from src.retrieval.token_budget import count_tokens


# Protect financial facts, reporting periods, units, and accounting qualifiers.
# Compression remains extractive: selected text is copied unchanged.
FINANCIAL_FACT_RE = re.compile(
    r"(?:[$€£]\s*\(?-?\d|\b\d[\d,]*(?:\.\d+)?\s*%|"
    r"\b\d[\d,]*(?:\.\d+)?\b|\b(?:19|20)\d{2}\b|"
    r"\bQ[1-4]\b|\bFY\s*\d{2,4}\b|"
    r"\b(?:USD|million|millions|billion|billions|thousand|thousands|"
    r"basis points?|bps|GAAP|non-GAAP)\b)",
    re.IGNORECASE,
)

FINANCIAL_TERM_RE = re.compile(
    r"\b(?:revenue|sales|income|loss|profit|margin|earnings|EPS|cash|"
    r"assets?|liabilit(?:y|ies)|equity|debt|borrowings?|expense|costs?|"
    r"capital expenditures?|capex|free cash flow|operating cash flow|"
    r"gross profit|net income|regulatory credits?|finance leases?)\b",
    re.IGNORECASE,
)

PERIOD_OR_QUALIFIER_RE = re.compile(
    r"\b(?:year ended|quarter ended|three months ended|six months ended|"
    r"nine months ended|as of|fiscal year|full[- ]year|quarterly|"
    r"approximately|includes?|excludes?|except|respectively|unaudited|"
    r"recast|reclassified|finance leases?|constant currency)\b",
    re.IGNORECASE,
)


def is_table_block(text: str) -> bool:
    """Return True when text contains a complete Markdown or HTML table."""

    return any(part.is_table for part in blocks(text))


def _stats(
    status: str,
    original: str,
    compressed: str | None = None,
    **extra: object,
) -> dict[str, object]:
    output = original if compressed is None else compressed
    result: dict[str, object] = {
        "status": status,
        "original_len": len(original),
        "compressed_len": len(output),
        "reduction_pct": round(
            max(0.0, (1.0 - len(output) / max(len(original), 1)) * 100.0),
            2,
        ),
    }
    result.update(extra)
    return result


def _query_overlap(part: Block, query_terms: set[str]) -> int:
    return len(query_terms.intersection(tokenize(part.text)))


def compress(
    text: str,
    query: str,
    *,
    keep_blocks: int = 6,
    preserve_all: bool = False,
) -> tuple[str, dict[str, object]]:
    """Safely reduce a retrieved parent without rewriting its evidence.

    Rules:
    - preserve_all returns the parent unchanged;
    - tables are indivisible, so rows and columns are never sliced;
    - table titles, period/unit headers, and nearby notes are retained;
    - numeric facts, dates, units, and qualifiers are retained;
    - narrative blocks are copied verbatim with neighboring context.

    Each source parent is compressed independently, so evidence from one
    company is not discarded merely because another company is requested.
    """

    if preserve_all:
        return text, _stats("bypassed_preserve_all", text)

    parts = blocks(text)
    if not parts:
        return text, _stats("bypassed_empty", text)

    keep_blocks = max(1, int(keep_blocks))
    if len(parts) <= keep_blocks:
        return text, _stats("bypassed_short_document", text)

    query_terms = set(tokenize(query))
    if not query_terms:
        return text, _stats("bypassed_empty_query", text)

    scores = [_query_overlap(part, query_terms) for part in parts]
    matching = [index for index, score in enumerate(scores) if score > 0]
    if not matching:
        # If the heuristic cannot identify evidence, returning the original is
        # safer than deleting possibly relevant text.
        return text, _stats("fallback_full_no_match", text)

    selected: set[int] = set()

    # Begin with the strongest query-matching blocks.
    ranked = sorted(matching, key=lambda index: (-scores[index], index))
    selected.update(ranked[:keep_blocks])

    # Keep one neighbor on each side for definitions and qualifiers.
    for index in tuple(selected):
        selected.update(range(max(0, index - 1), min(len(parts), index + 2)))

    # Preserve every table atomically, plus nearby title/unit/note blocks.
    table_indexes = [index for index, part in enumerate(parts) if part.is_table]
    for index in table_indexes:
        selected.update(range(max(0, index - 2), min(len(parts), index + 3)))

    # Lock factual evidence even when user wording differs from filing wording.
    for index, part in enumerate(parts):
        if (
            FINANCIAL_FACT_RE.search(part.text)
            or FINANCIAL_TERM_RE.search(part.text)
            or PERIOD_OR_QUALIFIER_RE.search(part.text)
        ):
            selected.add(index)

    # Keep page/section/speaker markers used for citations.
    for index, part in enumerate(parts):
        lowered = part.text.lower().lstrip()
        if lowered.startswith(("[page:", "[section:", "[speaker:", "[table")):
            selected.add(index)

    # Keep a heading when it introduces retained evidence.
    for index, part in enumerate(parts[:-1]):
        if part.text.lstrip().startswith("#") and (index + 1) in selected:
            selected.add(index)

    if len(selected) >= len(parts):
        return text, _stats(
            "bypassed_lossless_rules_kept_all",
            text,
            table_count=len(table_indexes),
        )

    # ``blocks`` preserves the source's line endings. Concatenating the
    # selected blocks therefore keeps their original separators; inserting
    # new separators here would make the reconstructed text larger.
    compressed = "".join(parts[index].text for index in sorted(selected)).strip()
    if len(compressed) >= len(text):
        return text, _stats(
            "bypassed_no_size_reduction",
            text,
            text,
            kept_blocks=len(selected),
            original_blocks=len(parts),
            table_count=len(table_indexes),
        )
    return compressed, _stats(
        "compressed_extractively",
        text,
        compressed,
        kept_blocks=len(selected),
        original_blocks=len(parts),
        table_count=len(table_indexes),
    )


def compress_to_budget(
    text: str,
    query: str,
    *,
    max_chars: int,
    max_tokens: int,
    keep_blocks: int = 6,
) -> tuple[str, dict[str, object]]:
    """Select complete, query-relevant blocks that fit the remaining budget.

    This is the last-resort path used when the normally compressed parent is
    still too large. It remains extractive and never slices a table, row, or
    narrative block. Returning a focused excerpt is preferable to silently
    discarding every source for a comparison group.
    """
    parts = blocks(text)
    query_terms = set(tokenize(query))
    if not parts or not query_terms or max_chars <= 0 or max_tokens <= 0:
        return "", _stats(
            "budget_excerpt_unavailable",
            text,
            "",
            max_chars=max(0, max_chars),
            max_tokens=max(0, max_tokens),
        )

    scores = [_query_overlap(part, query_terms) for part in parts]
    matching = [index for index, score in enumerate(scores) if score > 0]
    if not matching:
        return "", _stats(
            "budget_excerpt_no_query_match",
            text,
            "",
            max_chars=max_chars,
            max_tokens=max_tokens,
        )

    # Prefer blocks matching more distinct query terms. For equal overlap,
    # tables come first because they usually contain the requested row, unit,
    # and period headers together.
    ranked = sorted(
        matching,
        key=lambda index: (
            -scores[index],
            -int(parts[index].is_table),
            len(parts[index].text),
            index,
        ),
    )
    primary = ranked[: max(1, int(keep_blocks))]
    priority: list[int] = []
    for index in primary:
        priority.append(index)
        # Nearby headings and notes often carry the table scale or qualifier.
        priority.extend((index - 1, index + 1))
    priority.extend(ranked[len(primary) :])

    selected: set[int] = set()
    for index in priority:
        if index < 0 or index >= len(parts) or index in selected:
            continue
        proposed = "".join(
            parts[item].text for item in sorted((*selected, index))
        ).strip()
        if len(proposed) <= max_chars and count_tokens(proposed) <= max_tokens:
            selected.add(index)

    if not selected:
        return "", _stats(
            "budget_excerpt_no_block_fit",
            text,
            "",
            matching_blocks=len(matching),
            max_chars=max_chars,
            max_tokens=max_tokens,
        )

    compressed = "".join(parts[index].text for index in sorted(selected)).strip()
    return compressed, _stats(
        "compressed_to_budget",
        text,
        compressed,
        kept_blocks=len(selected),
        original_blocks=len(parts),
        matching_blocks=len(matching),
        table_count=sum(parts[index].is_table for index in selected),
        max_chars=max_chars,
        max_tokens=max_tokens,
        output_tokens=count_tokens(compressed),
    )
