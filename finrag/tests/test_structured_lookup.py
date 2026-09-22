from __future__ import annotations

import json

from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.schemas import Scope


def _parent(
    root, name: str, *, body: str, item: str, section: str,
    ticker: str = "MSFT", year: str = "2025",
) -> None:
    (root / f"{name}.json").write_text(
        json.dumps(
            {
                "page_content": body,
                "metadata": {
                    "ticker": ticker,
                    "fiscal_year": year,
                    "doc_type": "10K",
                    "item": item,
                    "section": section,
                    "source": f"{ticker}_{year}_10K.pdf",
                    "pdf_page_start": 10,
                },
            }
        ),
        encoding="utf-8",
    )


def test_exact_balance_sheet_uses_complete_matching_source_table(tmp_path) -> None:
    _parent(
        tmp_path,
        "statement",
        item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "[PAGE: 12]\n## <u>PART II</u> Item 8 **BALANCE SHEETS**\n(In millions)\n"
            "<table><tr><th>June 30</th><th>2025</th><th>2024</th></tr>"
            "<tr><td>Cash</td><td>30,242</td><td>18,315</td></tr>"
            "<tr><td>Accounts receivable</td><td>69,905</td><td></td></tr>"
            "<tr><td></td><td></td><td>56,924</td></tr>"
            "<tr><td>Total assets</td><td>619,003</td><td>512,163</td></tr></table>"
            "\n## CASH FLOWS STATEMENTS\n"
            "<table><tr><th>Year</th><th>2025</th></tr>"
            "<tr><td>Net income</td><td>101,832</td></tr></table>"
        ),
    )
    lookup = StructuredDocumentLookup(tmp_path)
    result = lookup.answer(
        "Give me the exact balance sheet table in the MSFT 2025 10-K",
        Scope(tickers=("MSFT",), years=("2025",), doc_type="10K"),
        wants_complete_table=True,
    )

    assert result is not None
    assert result.mode == "exact_table"
    assert result.answer.startswith("## BALANCE SHEETS\n")
    assert "| Accounts receivable | 69,905 | 56,924 |" in result.answer
    assert "| Total assets | 619,003 | 512,163 |" in result.answer
    assert "Net income" not in result.answer
    assert result.sources[0]["pdf_page"] == "12"


def test_any_matching_table_heading_can_be_returned_without_statement_rules(tmp_path) -> None:
    _parent(
        tmp_path, "revenue", ticker="TSLA", year="2024", item="Note 2",
        section="Note 2. REVENUE",
        body=(
            "[PAGE: 31]\n## Revenue by source\n## (in millions)\n"
            "<table><tr><th>Source</th><th>2024</th></tr>"
            "<tr><td>Automotive sales</td><td>72,480</td></tr></table>"
        ),
    )
    lookup = StructuredDocumentLookup(tmp_path)
    scope = Scope(tickers=("TSLA",), years=("2024",), doc_type="10K")

    result = lookup.answer(
        "Show the complete revenue by source table in the TSLA 2024 10-K",
        scope, wants_complete_table=True,
    )

    assert result is not None
    assert result.answer.startswith("## Revenue by source\n")
    assert "| Automotive sales | 72,480 |" in result.answer
    assert result.sources[0]["pdf_page"] == "31"
    assert lookup.answer("Show the complete table in the TSLA 2024 10-K", scope,
                         wants_complete_table=True) is None


def test_income_statement_matches_operations_title_without_other_income_table(tmp_path) -> None:
    _parent(
        tmp_path, "statements", ticker="TSLA", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS",
        body=(
            "## Consolidated Statements of Comprehensive Income\n"
            "<table><tr><th>Metric</th><th>2025</th></tr>"
            "<tr><td>Other comprehensive income</td><td>12</td></tr></table>\n"
            "## Consolidated Statements of Operations\n"
            "<table><tr><th>Metric</th><th>2025</th></tr>"
            "<tr><td>Revenue</td><td>100</td></tr></table>"
        ),
    )
    result = StructuredDocumentLookup(tmp_path).answer(
        "Show the full income statement table in TSLA's 2025 10-K",
        Scope(tickers=("TSLA",), years=("2025",), doc_type="10K"),
        wants_complete_table=True,
    )

    assert result is not None
    assert result.answer.startswith("## Consolidated Statements of Operations\n")
    assert "| Revenue | 100 |" in result.answer
    assert "Other comprehensive income" not in result.answer


def test_requested_statement_rows_use_matching_year_and_metric(tmp_path) -> None:
    _parent(
        tmp_path, "operations", ticker="TSLA", year="2025", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "## Consolidated Statements of Operations\n(in millions)\n"
            "<table><tr><th>Metric</th><th>2025</th><th>2024</th></tr>"
            "<tr><td>Total revenues</td><td>94,827</td><td>97,690</td></tr>"
            "<tr><td>Net income</td><td>3,855</td><td>7,153</td></tr>"
            "<tr><td>Net income attributable to common stockholders</td>"
            "<td>3,794</td><td>7,091</td></tr></table>"
        ),
    )
    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "Give revenue and net income, then summarize a risk factor",
        Scope(tickers=("TSLA",), years=("2025",), doc_type="10K"),
    )

    assert {(row.label, row.value) for row in rows} == {
        ("Total revenues", "94,827"), ("Net income", "3,855")
    }
    assert all(row.year == "2025" and row.scale == "millions" for row in rows)


def test_all_items_come_from_document_metadata_in_order(tmp_path) -> None:
    _parent(
        tmp_path, "late", body="## ITEM 10. DIRECTORS", item="Item 10",
        section="Item 10. DIRECTORS",
    )
    _parent(
        tmp_path, "early", body="## ITEM 1. BUSINESS", item="Item 1",
        section="Item 1. BUSINESS",
    )
    _parent(
        tmp_path, "note", body="## NOTE 1", item="Note 1",
        section="Note 1. BUSINESS",
    )
    result = StructuredDocumentLookup(tmp_path).answer(
        "List all section items in MSFT's 2025 10-K",
        Scope(tickers=("MSFT",), years=("2025",), doc_type="10K"),
        wants_complete_table=False,
    )

    assert result is not None
    assert result.mode == "document_items"
    assert result.answer.index("Item 1. BUSINESS") < result.answer.index("Item 10. DIRECTORS")
    assert len(result.sources) == 2
