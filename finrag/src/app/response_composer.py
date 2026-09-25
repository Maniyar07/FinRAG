"""Compose answers from facts and calculations that have already been verified."""

from __future__ import annotations

import re
from collections import OrderedDict

from src.financial.models import ValueType
from src.generation.citations import expand_citations
from src.orchestration.executor import MultiHopExecutionResult
from src.retrieval.structured_lookup import VerifiedTableRow
from src.schemas import Scope


CALCULATION_INTENT_RE = re.compile(
    r"\b(?:calculat(?:e|ed|es|ing)|comput(?:e|ed|es|ing)|percentage|percent|"
    r"differences?|changes?|growth|ratios?|margins?|increas(?:e|ed|es|ing)|"
    r"decreas(?:e|ed|es|ing))\b",
    re.IGNORECASE,
)
NEGATED_CALCULATION_RE = re.compile(
    r"\b(?:without|do\s+not|don't|no\s+need\s+to)\s+"
    r"(?:calculate|compute|calculating|computing)"
    r"(?:\s+(?:the\s+)?(?:absolute\s+|percentage\s+|percent\s+)?"
    r"(?:change|difference|growth|ratio|margin|increase|decrease))?\b",
    re.IGNORECASE,
)
NARRATIVE_INTENT_RE = re.compile(
    r"\b(?:why|explain|reasons?|drivers?|factors?|causes?|management|commentary|"
    r"risks?|outlook|guidance|profitability|said|say|affect(?:ed|s|ing)?|"
    r"impact(?:ed|s|ing)?|contribut(?:e|ed|es|ing)|discuss(?:ed|es|ing)?|"
    r"summari[sz](?:e|ed|es|ing)|m\s*d\s*&\s*a)\b",
    re.IGNORECASE,
)
DIRECT_STATEMENT_INTENT_RE = re.compile(
    r"\b(?:what\s+(?:was|is|were|are)|how\s+much|report(?:ed|s)?|compare|"
    r"show|give|provide|list)\b",
    re.IGNORECASE,
)


def _period(fact: object, source_years: dict[str, str]) -> str:
    candidates = (
        str(getattr(fact, "column_label", "") or ""),
        str(getattr(fact, "period", "") or ""),
        source_years.get(str(getattr(fact, "source_id", "")), ""),
    )
    for candidate in candidates:
        if match := re.search(r"\b(?:19|20)\d{2}\b", candidate):
            return match.group(0)
    return next((value.strip() for value in candidates if value.strip()), "Period")


def _fact_value(fact: object) -> str:
    value = str(getattr(fact, "raw_value", "")).strip()
    if getattr(fact, "value_type", None) != ValueType.CURRENCY:
        return value
    value = re.sub(r"^(?:[$]|USD|EUR|GBP|JPY)\s*", "", value, flags=re.I)
    currency = str(getattr(fact, "currency", "") or "")
    scale = str(getattr(fact, "scale", "") or "").rstrip("s")
    return " ".join(part for part in (currency, value, scale) if part)


def _metric(item: object, fact: object, ticker: str) -> str:
    metric = str(
        getattr(fact, "metric", "") or getattr(fact, "row_label", "") or ""
    ).strip()
    if metric:
        return metric
    metric = str(getattr(item, "label", "") or "result")
    metric = re.sub(r"\b(?:calculate|compute)\b", "", metric, flags=re.I)
    if ticker:
        metric = re.sub(rf"\b{re.escape(ticker)}\b", "", metric, flags=re.I)
    metric = re.sub(
        r"\b(?:absolute|percentage|percent|basis point|percentage point)\s+change\b|"
        r"\b(?:change|cagr|ratio)\b",
        "",
        metric,
        flags=re.I,
    )
    return " ".join(metric.split()) or "Result"


def _result_value(result: object, *, margin_ratio: bool) -> str:
    raw = result.result * 100 if margin_ratio else result.result
    value = f"{raw:,.2f}".rstrip("0").rstrip(".")
    if margin_ratio or result.result_unit == "percent":
        return f"{value}%"
    if result.result_unit == "currency":
        scale = str(result.scale or "").rstrip("s")
        return " ".join(part for part in (result.currency, value, scale) if part)
    unit = str(result.result_unit).replace("_", " ")
    return f"{value} {unit}".strip()


def _operation_label(result: object, *, margin_ratio: bool) -> str:
    if margin_ratio:
        return "Net profit margin"
    return str(result.operation.value).replace("_", " ").capitalize()


def verified_calculation_text(
    execution: MultiHopExecutionResult, question: str = ""
) -> str | None:
    """Format validated inputs once and group related deterministic calculations."""
    if not execution.calculations:
        return None

    facts = {fact.fact_id: fact for fact in execution.facts}
    bundle = getattr(execution, "bundle", None)
    source_years = {
        str(source.get("id")): str(source.get("fiscal_year") or "")
        for source in getattr(bundle, "sources", ())
    }
    blocks: OrderedDict[tuple, dict] = OrderedDict()

    for item in execution.calculations:
        result = item.result
        inputs = [facts.get(fact_id) for fact_id in result.input_fact_ids]
        if any(fact is None for fact in inputs):
            return None
        first_fact = inputs[0]
        ticker = str(getattr(first_fact, "ticker", "") or "")
        metric = _metric(item, first_fact, ticker)
        margin_ratio = result.result_unit == "ratio" and bool(
            re.search(r"\b(?:margin|percentage|percent)\b", f"{question} {item.label}", re.I)
        )
        title_metric = "Net profit margin" if margin_ratio else metric
        key = (ticker, title_metric.casefold(), tuple(result.input_fact_ids))
        block = blocks.setdefault(
            key,
            {"ticker": ticker, "metric": title_metric, "inputs": [], "results": []},
        )
        if not block["inputs"]:
            for fact in inputs:
                period = _period(fact, source_years)
                input_metric = str(
                    getattr(fact, "metric", "")
                    or getattr(fact, "row_label", "")
                    or "Value"
                ).strip()
                label = f"{input_metric} ({period})" if margin_ratio else period
                block["inputs"].append(
                    f"- **{label}:** {_fact_value(fact)} [{fact.source_id}]"
                )
        block["results"].append(
            f"- **{_operation_label(result, margin_ratio=margin_ratio)}:** "
            f"**{_result_value(result, margin_ratio=margin_ratio)}**"
        )

    lines = ["### Calculated results"]
    for block in blocks.values():
        title = " - ".join(
            part for part in (block["ticker"], str(block["metric"]).capitalize()) if part
        )
        body = "\n".join((*block["inputs"], *block["results"]))
        lines.append(f"#### {title}\n\n{body}")

    _append_comparison_conclusion(lines, execution, facts, question)
    return "\n\n".join(lines)


def verified_calculation_partial(
    execution: MultiHopExecutionResult,
    question: str = "",
    limitations: tuple[str, ...] = (),
) -> str | None:
    """Return verified numeric work while clearly identifying omitted evidence."""
    numeric_text = verified_calculation_text(execution, question)
    if numeric_text is None:
        return None
    details = tuple(dict.fromkeys(item for item in limitations if item))
    limitation_lines = (
        [f"- {item}" for item in details]
        if details
        else ["- Some requested evidence could not be verified, so it was omitted."]
    )
    answer = "\n\n".join(
        (numeric_text, "### Evidence limitations", "\n".join(limitation_lines))
    )
    return expand_citations(answer, execution.bundle.sources)


def _metric_key(label: str) -> tuple[str, ...]:
    """Normalize harmless label variants without defining financial aliases."""
    words = []
    for word in re.findall(r"[a-z0-9]+", label.casefold()):
        if word == "total":
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return tuple(words)


def _calculation_requested(question: str) -> bool:
    return bool(CALCULATION_INTENT_RE.search(NEGATED_CALCULATION_RE.sub("", question)))


def requires_verified_statement_values(question: str) -> bool:
    """Return whether indexed statement cells are part of the requested answer."""
    return bool(
        DIRECT_STATEMENT_INTENT_RE.search(question)
        and not NARRATIVE_INTENT_RE.search(question)
        and not _calculation_requested(question)
    )


def verified_statement_answer_text(
    rows: tuple[VerifiedTableRow, ...], scope: Scope, question: str
) -> str | None:
    """Render a complete one-metric answer directly from verified table cells."""
    groups = scope.groups
    if (
        not groups
        or not requires_verified_statement_values(question)
    ):
        return None

    rows_by_group: dict[tuple[str, str], list[VerifiedTableRow]] = {}
    for row in rows:
        group = (str(row.source.get("ticker") or ""), str(row.year))
        rows_by_group.setdefault(group, []).append(row)
    if any(len(rows_by_group.get(group, ())) != 1 for group in groups):
        return None

    selected = [rows_by_group[group][0] for group in groups]
    metric_keys = {_metric_key(row.label) for row in selected}
    if len(metric_keys) != 1:
        return None

    if len(selected) == 1:
        row = selected[0]
        scale = row.scale.rstrip("s")
        value = " ".join(part for part in (row.value, scale) if part)
        return f"**{row.label}:** {value} [{row.source['id']}]"

    lines = [
        f"### {selected[0].label} comparison",
        "",
        "| Company | Fiscal year | Reported value |",
        "| --- | ---: | ---: |",
    ]
    for row in selected:
        ticker = str(row.source.get("ticker") or "Unknown")
        scale = row.scale.rstrip("s")
        value = " ".join(part for part in (row.value, scale) if part)
        lines.append(f"| {ticker} | {row.year} | {value} [{row.source['id']}] |")
    return "\n".join(lines)


def _comparison_operation(calculations: list, question: str) -> str | None:
    available = {item.result.operation.value for item in calculations}
    preferences: list[str] = []
    # The requested conclusion determines the comparison basis. A question may
    # ask us to display both absolute and percentage changes, then ask which
    # company grew faster; that conclusion is a rate comparison.
    if re.search(r"\b(?:percent|percentage|growth|grew|faster|performed better)\b", question, re.I):
        preferences.append("percentage_change")
    if re.search(r"\bmargins?\b", question, re.I):
        preferences.append("ratio")
    if re.search(
        r"\b(?:absolute|amount|dollars?|larger\s+absolute\s+increase)\b",
        question,
        re.I,
    ):
        preferences.append("absolute_change")
    preferences.extend(("percentage_change", "ratio", "absolute_change", "difference"))
    return next((operation for operation in preferences if operation in available), None)


def _append_comparison_conclusion(
    lines: list[str], execution: MultiHopExecutionResult, facts: dict, question: str
) -> None:
    if not re.search(
        r"\b(?:which|identify|conclusion|performed\s+better|grew\s+more|faster|"
        r"larger\s+(?:one|increase|change)|higher\s+(?:one|change|growth)|compare)\b",
        question,
        re.I,
    ):
        return
    calculations = list(execution.calculations)
    operation = _comparison_operation(calculations, question)
    comparable = [item for item in calculations if item.result.operation.value == operation]
    tickers = {
        str(getattr(facts.get(item.result.input_fact_ids[0]), "ticker", "") or "")
        for item in comparable
    }
    if len(comparable) < 2 or len(tickers - {""}) < 2:
        return
    winner = max(comparable, key=lambda item: item.result.result)
    ticker = str(getattr(facts[winner.result.input_fact_ids[0]], "ticker", ""))
    if operation == "ratio" and re.search(r"\bmargins?\b", question, re.I):
        conclusion = f"{ticker} had the higher net profit margin."
    elif re.search(r"\bperformed\s+better\b", question, re.I):
        conclusion = f"{ticker} performed better based on the calculated change."
    elif re.search(r"\b(?:grew\s+more|faster)\b", question, re.I):
        conclusion = f"{ticker} grew faster based on the calculated percentage change."
    elif operation == "absolute_change":
        conclusion = f"{ticker} had the larger absolute increase."
    else:
        conclusion = f"{ticker} had the higher calculated change."
    lines.append(f"### Comparison\n\n**{conclusion}**")
