from __future__ import annotations

import re
from datetime import date

from src.constants import TICKER_ALIASES, TICKERS, YEARS
from src.schemas import QueryUnderstanding


ALL_TICKER_RE = re.compile(
    r"\ball\s+(company|companies|ticker|tickers)\b|"
    r"\b(?:each|every)\s+(?:of\s+)?(?:the\s+)?(?:three|3)\s+"
    r"(?:company|companies|ticker|tickers)\b|"
    r"\b(across|compare)\s+(?:all\s+)?(?:three|3)\b",
    re.IGNORECASE,
)
ALL_YEAR_RE = re.compile(
    r"\ball\s+(?:available\s+)?years\b|\b2024\s+(?:and|&)\s+2025\b",
    re.IGNORECASE,
)
COMPARISON_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|differ|higher|lower|between)\b",
    re.IGNORECASE,
)
OUT_OF_SCOPE_RE = re.compile(
    r"\b(weather|recipe|sports score|football score|movie review|write (?:me )?a (?:poem|story)|"
    r"translate this|medical advice|diagnos(?:e|is)|generate (?:python|java|javascript) code|"
    r"current stock price|stock price today|latest news|real[- ]time|live market|price target|"
    r"capital of|prime minister|president of|photosynthesis|periodic table)\b",
    re.IGNORECASE,
)
LIVE_MARKET_RE = re.compile(
    r"\b(?:"
    r"(?:current|latest|live|real[- ]time)\s+(?:stock|share|market)\s+price|"
    r"(?:stock|share|market)\s+price\s+(?:right\s+now|today|currently|now)|"
    r"trad(?:e|ing)\s+at\s+(?:right\s+now|today|currently|now)"
    r")\b",
    re.IGNORECASE,
)
DOMAIN_RE = re.compile(
    r"\b(company|companies|business|strategy|operations?|product|customer|employee|financial|"
    r"filing|report|earnings|revenue|income|profit|loss|expense|cash|liquidity|asset|debt|"
    r"capital|equity|margin|growth|risk|cybersecurity|legal|regulatory|segment|cloud|azure|"
    r"automotive|deliveries|banking|credit|loan|deposit|guidance|outlook|gaap|tax|dividend|"
    r"repurchase|shareholder|stockholder|executive|board|management|performance|results?|"
    r"nii|cet1|rotce|eps|capex|fsd|ai)\b",
    re.IGNORECASE,
)
STRONG_FOLLOWUP_RE = re.compile(
    r"^(?:and\s+|also\s+)?(?:what|how) about\b|"
    r"^(?:why|how|explain|elaborate|continue)(?:\s+(?:that|this|those|it))?\??$",
    re.IGNORECASE,
)
ANAPHORA_RE = re.compile(
    r"\b(that|those|it|its|they|their|them|the same|former|latter)\b",
    re.IGNORECASE,
)
EXPLICIT_TEN_K_RE = re.compile(
    r"\b(10[- ]?k(?:s|'s)?|annual reports?|item\s+\d+[a-c]?|footnotes?)\b",
    re.IGNORECASE,
)
SEC_FORM_RE = re.compile(
    r"\b(?:form\s+)?(?P<form>\d{1,2}\s*-\s*[a-z])(?:s|'s)?\b",
    re.IGNORECASE,
)
TRANSCRIPT_RE = re.compile(
    r"\b(transcripts?|earnings calls?|conference calls?|prepared remarks?|q\s*&\s*a|"
    r"questions?\s+(?:and|&)\s+answers?|speaker|who said|management said)\b",
    re.IGNORECASE,
)
MIXED_EVIDENCE_RE = re.compile(
    r"\bcalculat\w*\b.*\b(?:explain|summarize|commentary|reasons?|drivers?)\b|"
    r"\b(?:explain|summarize|commentary|reasons?|drivers?)\b.*\bcalculat\w*\b",
    re.IGNORECASE,
)
LATEST_YEAR_RE = re.compile(
    r"\b(most recent|latest available|newest)\s+(?:annual report|10[- ]?k|filing|"
    r"earnings call|transcript|year|results?)\b",
    re.IGNORECASE,
)
BOTH_DOC_RE = re.compile(
    r"\b(?:both\s+(?:document\s+types?|sources?)|"
    r"10[- ]?k(?:s)?\s+(?:and|&)\s+(?:the\s+)?transcripts?)\b|^both$",
    re.IGNORECASE,
)
DOC_TYPE_REPLY_RE = re.compile(
    r"^(?:the\s+)?(?:10[- ]?k|annual report|transcript|both)$",
    re.IGNORECASE,
)
UNSUPPORTED_COMPANIES = {
    "AAPL": ("AAPL", "APPLE"),
    "AMZN": ("AMZN", "AMAZON"),
    "GOOG": ("GOOG", "GOOGL", "GOOGLE", "ALPHABET"),
    "META": ("META", "FACEBOOK"),
    "NVDA": ("NVDA", "NVIDIA"),
}

TOPICS = (
    ("risk", r"\b(risk|risks|risk factors?|cybersecurity|litigation|regulatory)\b"),
    ("liquidity", r"\b(cash|liquidity|capital resources?|debt|borrowings?)\b"),
    ("performance", r"\b(revenue|income|profit|loss|margin|eps|growth|performance)\b"),
    ("segments", r"\b(segment|business line|division|automotive|cloud|banking)\b"),
    ("guidance", r"\b(guidance|outlook|expect(?:s|ed|ation)?|forecast)\b"),
    ("governance", r"\b(board|director|executive|governance|controls?|audit)\b"),
)

TABLE_OUTPUT_RE = re.compile(
    r"\b(tabular|table|structured\s+table|rows?\s+and\s+columns?)\b",
    re.IGNORECASE,
)
COMPLETE_TABLE_RE = re.compile(
    r"\b(complete|full|entire|whole)\b.{0,50}\b(table|statement|data)\b|"
    r"\b(all\s+(?:rows|columns|entries|data))\b",
    re.IGNORECASE,
)
FINANCIAL_STATEMENT_RE = re.compile(
    r"\b(balance\s+sheets?|statements?\s+of\s+(?:operations|income|cash\s+flows?|"
    r"stockholders['\u2019]?\s+equity|shareholders['\u2019]?\s+equity|equity)|"
    r"income\s+statements?|cash[- ]flow\s+statements?)\b",
    re.IGNORECASE,
)
DISPLAY_VERB_RE = re.compile(r"\b(show|give|provide|display|present|list)\b", re.IGNORECASE)
TABULAR_DISPLAY_RE = re.compile(
    r"\b(show|give|provide|display|present)\b.{0,60}\b(tabular|table)\b",
    re.IGNORECASE,
)
TABLE_PROJECTION_RE = re.compile(
    r"\b(avg\.?|average|mean|min\.?|minimum|max\.?|maximum|only|just)\b",
    re.IGNORECASE,
)

# Conservative corrections for high-value financial search terms. This is query
# normalization, not answer-specific prompting. Unknown words are left untouched.
DOMAIN_TOKEN_CORRECTIONS = {
    "balace": "balance",
    "balnce": "balance",
    "statment": "statement",
    "statments": "statements",
    "tabluar": "tabular",
    "microsft": "microsoft",
}


def normalize_query_text(query: str) -> str:
    """Canonicalize harmless spacing, punctuation, and common domain typos."""
    normalized = query.replace("\u2019", "'").replace("\u2018", "'")
    normalized = re.sub(r"(?i)([a-z])((?:19|20)\d{2})\b", r"\1 \2", normalized)
    normalized = re.sub(r"(?i)\b((?:19|20)\d{2})([a-z])", r"\1 \2", normalized)
    normalized = re.sub(r"\b10\s*[- ]?\s*k\b", "10-K", normalized, flags=re.IGNORECASE)
    for misspelling, correction in DOMAIN_TOKEN_CORRECTIONS.items():
        normalized = re.sub(
            rf"\b{re.escape(misspelling)}\b",
            correction,
            normalized,
            flags=re.IGNORECASE,
        )
    return " ".join(normalized.strip().split())


def _contains_alias(text: str, alias: str) -> bool:
    # Natural-language users often omit the apostrophe in "Microsofts" or
    # "Teslas". Accept that suffix for company-name aliases, but not tickers.
    bare_possessive = r"(?:['\u2019]?S)?" if len(alias.replace(" ", "")) >= 5 else ""
    return bool(
        re.search(
            rf"(?<![A-Z0-9]){re.escape(alias)}{bare_possessive}(?![A-Z0-9])",
            text,
        )
    )


def _paired_groups(
    text: str, tickers: tuple[str, ...], years: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    if len(tickers) < 2 or len(years) < 2:
        return ()
    year_positions = [
        (match.start(), match.group())
        for match in re.finditer(r"\b(?:2024|2025)\b", text)
    ]
    pairs_with_position: list[tuple[int, str, str]] = []
    for ticker in tickers:
        spans = []
        for alias in TICKER_ALIASES[ticker]:
            spans.extend(
                match.span()
                for match in re.finditer(
                    rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", text
                )
            )
        if not spans:
            return ()
        company_start, company_end = min(spans, key=lambda span: span[0])
        nearest = min(
            year_positions,
            key=lambda item: min(abs(item[0] - company_start), abs(item[0] - company_end)),
        )
        distance = min(abs(nearest[0] - company_start), abs(nearest[0] - company_end))
        if distance > 28:
            return ()
        pairs_with_position.append((company_start, ticker, nearest[1]))
    pairs = tuple((ticker, year) for _, ticker, year in sorted(pairs_with_position))
    if len({year for _, year in pairs}) != len(years):
        return ()
    return pairs


def _document_years(text: str) -> tuple[str, ...]:
    """Find years that label a requested source rather than a table column.

    A 2024 10-K commonly contains 2023 and 2022 comparison columns. Only the
    year attached to the filing/call should become a retrieval filter.
    """
    year = r"(?P<year>(?:19|20)\d{2})"
    document = (
        r"(?:form\s+)?\d{1,2}\s*-\s*[a-z](?:s|'s)?|annual reports?|sec filings?|"
        r"earnings transcripts?|earnings calls?|conference calls?|transcripts?"
    )
    patterns = (
        re.compile(rf"\b{year}\s+(?:{document})\b", re.IGNORECASE),
        re.compile(rf"\bq[1-4]\s+{year}\s+(?:{document})\b", re.IGNORECASE),
        re.compile(
            rf"\b(?:{document})\s+(?:for\s+|of\s+|in\s+)?(?:fy\s*)?{year}\b",
            re.IGNORECASE,
        ),
    )
    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        matches.extend((match.start(), match.group("year")) for match in pattern.finditer(text))
    return tuple(dict.fromkeys(value for _, value in sorted(matches)))


def understand_query(query: str) -> QueryUnderstanding:
    normalized = normalize_query_text(query)
    upper = normalized.upper()

    tickers = tuple(
        ticker
        for ticker in TICKERS
        if any(_contains_alias(upper, alias) for alias in TICKER_ALIASES[ticker])
    )
    mentioned_years = tuple(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", normalized)))
    if not mentioned_years and re.search(r"\blast year\b", normalized, re.IGNORECASE):
        mentioned_years = (str(date.today().year - 1),)
    source_years = _document_years(normalized)
    if source_years:
        years = tuple(year for year in source_years if year in YEARS)
        unsupported_years = tuple(year for year in source_years if year not in YEARS)
        reference_years = tuple(year for year in mentioned_years if year not in source_years)
    else:
        years = tuple(year for year in mentioned_years if year in YEARS)
        unsupported_years = tuple(year for year in mentioned_years if year not in YEARS)
        reference_years = ()

    unsupported_companies = tuple(
        ticker
        for ticker, aliases in UNSUPPORTED_COMPANIES.items()
        if any(_contains_alias(upper, alias) for alias in aliases)
    )
    requested_groups = _paired_groups(upper, tickers, years)

    transcript_signal = bool(TRANSCRIPT_RE.search(normalized))
    mixed_evidence_signal = (
        transcript_signal
        and len(mentioned_years) > 1
        and bool(MIXED_EVIDENCE_RE.search(normalized))
    )
    if mixed_evidence_signal:
        years = tuple(year for year in mentioned_years if year in YEARS)
        unsupported_years = tuple(year for year in mentioned_years if year not in YEARS)
        reference_years = ()
        requested_groups = _paired_groups(upper, tickers, years)
    sec_form_types = tuple(
        dict.fromkeys(
            re.sub(r"\W", "", match.group("form")).upper()
            for match in SEC_FORM_RE.finditer(normalized)
        )
    )
    explicit_ten_k_signal = bool(EXPLICIT_TEN_K_RE.search(normalized))
    risk_factor_signal = bool(
        re.search(r"\brisk factors?\b", normalized, re.IGNORECASE)
    )
    ten_k_signal = explicit_ten_k_signal or (
        risk_factor_signal and not transcript_signal and not sec_form_types
    ) or (
        mixed_evidence_signal
    )
    explicit_both_doc_types = bool(BOTH_DOC_RE.search(normalized))
    detected_doc_types = (
        sec_form_types
        + (("10K",) if ten_k_signal else ())
        + (("TRANSCRIPT",) if transcript_signal else ())
    )
    requested_doc_types = (
        ("10K", "TRANSCRIPT")
        if explicit_both_doc_types
        else tuple(dict.fromkeys(detected_doc_types))
    )
    doc_type = requested_doc_types[0] if len(requested_doc_types) == 1 else None

    topic = None
    for name, pattern in TOPICS:
        if re.search(pattern, normalized, re.IGNORECASE):
            topic = name
            break

    generic_followup = bool(STRONG_FOLLOWUP_RE.search(normalized)) or (
        not tickers and bool(ANAPHORA_RE.search(normalized))
    ) or bool(DOC_TYPE_REPLY_RE.fullmatch(normalized))

    statement_display_request = bool(DISPLAY_VERB_RE.search(normalized)) and bool(
        FINANCIAL_STATEMENT_RE.search(normalized)
    )
    tabular_display_request = bool(TABULAR_DISPLAY_RE.search(normalized))
    projected_table_request = bool(TABLE_PROJECTION_RE.search(normalized))
    wants_table = bool(TABLE_OUTPUT_RE.search(normalized)) or statement_display_request
    wants_complete_table = wants_table and (
        bool(COMPLETE_TABLE_RE.search(normalized))
        or (
            (statement_display_request or tabular_display_request)
            and not projected_table_request
        )
    )

    return QueryUnderstanding(
        tickers=tickers,
        years=years,
        doc_type=doc_type,
        all_tickers=bool(ALL_TICKER_RE.search(normalized)),
        all_years=bool(ALL_YEAR_RE.search(normalized)),
        comparison=bool(COMPARISON_RE.search(normalized)) or len(tickers) > 1 or len(years) > 1,
        explicit_out_of_scope=bool(
            OUT_OF_SCOPE_RE.search(normalized) or LIVE_MARKET_RE.search(normalized)
        ),
        topic=topic,
        unsupported_years=unsupported_years,
        unsupported_companies=unsupported_companies,
        domain_relevant=(
            bool(DOMAIN_RE.search(normalized))
            or bool(topic)
            or bool(tickers or years)
            or bool(requested_doc_types)
        ),
        generic_followup=generic_followup,
        requested_groups=requested_groups,
        requested_doc_types=requested_doc_types,
        reference_years=reference_years,
        latest_available_year=bool(LATEST_YEAR_RE.search(normalized)),
        wants_table=wants_table,
        wants_complete_table=wants_complete_table,
        normalized_query=normalized,
    )
