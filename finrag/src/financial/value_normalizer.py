"""Strict parsing and unit normalization for financial values."""

from __future__ import annotations

import html
import re
from decimal import Decimal, InvalidOperation

from src.financial.models import CandidateFinancialFact, NormalizedFinancialValue, ValueType


NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")
TAG_RE = re.compile(r"<[^>]+>")
NULL_VALUES = {"", "-", "--", "—", "–", "n/a", "na", "nm", "not meaningful"}
SCALE_ALIASES = {
    "one": "ones",
    "ones": "ones",
    "unit": "ones",
    "units": "ones",
    "thousand": "thousands",
    "thousands": "thousands",
    "million": "millions",
    "millions": "millions",
    "billion": "billions",
    "billions": "billions",
}
SCALE_MULTIPLIERS = {
    "ones": Decimal("1"),
    "thousands": Decimal("1000"),
    "millions": Decimal("1000000"),
    "billions": Decimal("1000000000"),
}
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}


class ValueNormalizationError(ValueError):
    """Raised when a financial value is blank, ambiguous, or unsupported."""


def _clean(value: object) -> str:
    text = html.unescape(TAG_RE.sub("", str(value)))
    return " ".join(text.replace("\u00a0", " ").split()).strip()


def _recognized_scale(value: str) -> str | None:
    lowered = value.casefold()
    for alias, normalized in SCALE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return normalized
    return None


def _scale_from(candidate: CandidateFinancialFact, text: str) -> str | None:
    declared = _recognized_scale(candidate.scale or "")
    presented = _recognized_scale(text)
    unit_scale = _recognized_scale(candidate.unit or "")
    if declared and presented and declared != presented:
        raise ValueNormalizationError("declared scale conflicts with the presented value")
    return presented or declared or unit_scale


def _presented_currency(text: str) -> str | None:
    for symbol, currency in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return currency
    match = re.search(r"\b(USD|EUR|GBP|JPY|CAD|AUD)\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _currency_from(candidate: CandidateFinancialFact, text: str) -> str | None:
    declared = candidate.currency.upper() if candidate.currency else None
    presented = _presented_currency(text)
    if declared and presented and declared != presented:
        raise ValueNormalizationError("declared currency conflicts with the presented value")
    return presented or declared


def _value_type(candidate: CandidateFinancialFact, text: str, currency: str | None) -> ValueType:
    hint = (candidate.unit or "").casefold()
    lowered = text.casefold()
    if "%" in text or "percent" in hint or "percentage" in hint:
        return ValueType.PERCENT
    if re.search(r"\b(?:bps?|basis points?)\b", f"{lowered} {hint}"):
        return ValueType.BASIS_POINTS
    if "per share" in hint or re.search(r"\bper share\b", lowered):
        return ValueType.PER_SHARE
    if "ratio" in hint or re.search(r"\d\s*x\b", lowered):
        return ValueType.RATIO
    if currency:
        return ValueType.CURRENCY
    if "count" in hint or "shares" in hint:
        return ValueType.COUNT
    return ValueType.NUMBER


class FinancialValueNormalizer:
    """Convert one presented value into an exact Decimal and canonical base value."""

    def normalize(self, candidate: CandidateFinancialFact) -> NormalizedFinancialValue:
        raw = _clean(candidate.raw_value)
        if raw.casefold() in NULL_VALUES:
            raise ValueNormalizationError("value is blank or not numerically meaningful")

        currency = _currency_from(candidate, raw)
        value_type = _value_type(candidate, raw, currency)
        scale = _scale_from(candidate, raw)
        if value_type in {ValueType.PERCENT, ValueType.BASIS_POINTS, ValueType.RATIO}:
            if currency:
                raise ValueNormalizationError(
                    "non-monetary value cannot declare a currency"
                )
            scale = None
        elif value_type == ValueType.PER_SHARE:
            scale = None
        matches = NUMBER_RE.findall(raw.replace(",", ""))
        if len(matches) != 1:
            raise ValueNormalizationError(
                "value must contain exactly one unambiguous number"
            )

        try:
            numeric = Decimal(matches[0])
        except InvalidOperation as error:
            raise ValueNormalizationError("value is not a valid decimal") from error

        parenthesized = "(" in raw and ")" in raw
        if parenthesized:
            numeric = -abs(numeric)

        multiplier = SCALE_MULTIPLIERS.get(scale or "ones", Decimal("1"))
        if value_type in {
            ValueType.PERCENT,
            ValueType.BASIS_POINTS,
            ValueType.PER_SHARE,
            ValueType.RATIO,
        }:
            base_value = numeric
        else:
            base_value = numeric * multiplier

        if value_type == ValueType.CURRENCY:
            normalized_unit = currency or "currency"
        else:
            normalized_unit = value_type.value

        return NormalizedFinancialValue(
            raw_value=candidate.raw_value,
            numeric_value=numeric,
            base_value=base_value,
            value_type=value_type,
            normalized_unit=normalized_unit,
            currency=currency,
            scale=scale,
        )

    def normalize_many(
        self, candidates: list[CandidateFinancialFact] | tuple[CandidateFinancialFact, ...]
    ) -> tuple[NormalizedFinancialValue, ...]:
        return tuple(self.normalize(candidate) for candidate in candidates)
