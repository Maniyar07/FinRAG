from __future__ import annotations

TICKERS = ("JPM", "MSFT", "TSLA")
YEARS = ("2024", "2025")
DOC_TYPES = ("10K", "TRANSCRIPT")

COMPANY_NAMES = {
    "JPM": "JPMorgan Chase & Co.",
    "MSFT": "Microsoft Corporation",
    "TSLA": "Tesla, Inc.",
}

TICKER_ALIASES = {
    "JPM": ("JPM", "JPMORGAN", "JPMORGAN CHASE", "JP MORGAN"),
    "MSFT": ("MSFT", "MICROSOFT"),
    "TSLA": ("TSLA", "TESLA"),
}

METADATA_SCHEMA_VERSION = "3.1"
CHUNK_SCHEMA_VERSION = "3.1"
INDEX_LAYOUT_VERSION = "2.0"
TRANSCRIPT_QUARTER = "Q4"

# These labels prevent the model from silently treating every company's Q4 as
# the same calendar period. They describe the standard fiscal calendar used by
# the supplied sources; the filing/call text remains authoritative.
FISCAL_CALENDARS = {
    "JPM": {
        "fiscal_year_end": "December 31",
        "fiscal_q4_months": "October-December",
    },
    "MSFT": {
        "fiscal_year_end": "June 30",
        "fiscal_q4_months": "April-June",
    },
    "TSLA": {
        "fiscal_year_end": "December 31",
        "fiscal_q4_months": "October-December",
    },
}

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm"}
