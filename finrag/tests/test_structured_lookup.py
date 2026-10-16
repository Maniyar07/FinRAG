from __future__ import annotations

import json
import pytest

from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.schemas import Scope


@pytest.mark.parametrize("separator", ["***", "---", "___"])
def test_table_identity_ignores_formatting_separators_and_unrelated_body_keywords(tmp_path, separator):
    _parent(tmp_path, "cash", ticker="ACME", item="Item 8", section="Statements",
        body=f"## Cash Flows Statements\n\n**(In millions)**\n\n{separator}\n\n"
             "<table><tr><th>Metric</th><th>2025</th></tr>"
             "<tr><td>Operating activities</td><td>100</td></tr></table>")
    _parent(tmp_path, "derivatives", ticker="ACME", item="Note 8", section="Notes",
        body="## Gains recognized in income statements on derivatives\n"
             "<table><tr><th>Metric</th><th>2025</th></tr>"
             "<tr><td>Cash flow hedges</td><td>999</td></tr></table>")
    lookup = StructuredDocumentLookup(tmp_path)
    result = lookup.answer("cash flow statement", Scope(("ACME",), ("2025",), "10K"), wants_complete_table=True)
    assert result and "| Operating activities | 100 |" in result.answer
    assert "999" not in result.answer
    assert lookup.answer("income statement", Scope(("ACME",), ("2025",), "10K"), wants_complete_table=True) is not None


def test_row_values_cannot_supply_a_missing_table_subject(tmp_path):
    _parent(tmp_path, "derivatives", ticker="ACME", item="Note 8", section="Notes",
        body="## Gains recognized in income statements on derivatives\n"
             "<table><tr><th>Metric</th><th>2025</th></tr>"
             "<tr><td>Cash flow hedges</td><td>999</td></tr></table>")
    assert StructuredDocumentLookup(tmp_path).answer("cash flow statement",
        Scope(("ACME",), ("2025",), "10K"), wants_complete_table=True) is None


def test_tables_without_marked_headers_keep_the_first_data_row_as_data(tmp_path):
    _parent(tmp_path, "definitions", ticker="ACME", item="Item 1", section="Business",
        body="## Product metrics\n<table><tr><td>First metric</td><td>First definition</td></tr>"
             "<tr><td>Second metric</td><td>Second definition</td></tr></table>")
    result = StructuredDocumentLookup(tmp_path).answer("product metrics", Scope(("ACME",), ("2025",), "10K"), wants_complete_table=True)
    assert result and result.answer.splitlines()[2] == "| Column 1 | Column 2 |"
    assert "| First metric | First definition |" in result.answer
    assert result.sources[0]["table_completeness"]["generated_header"]
    assert result.sources[0]["table_completeness"]["source_rows"] == result.sources[0]["table_completeness"]["returned_rows"] == 2


def test_caption_selects_supplement_instead_of_similar_statement(tmp_path) -> None:
    _parent(
        tmp_path, "leases", ticker="ACME", item="Item 8", section="Notes",
        body=(
            "## BALANCE SHEETS\n(In millions)\n"
            "<table><tr><th>Assets</th><th>2025</th></tr>"
            "<tr><td>Operating lease assets</td><td>100</td></tr></table>\n"
            "## NOTE 7 - LEASES\n"
            "Lease expense was as follows:\n\n"
            "<table><tr><th>Expense</th><th>2025</th></tr>"
            "<tr><td>Operating leases</td><td>10</td></tr></table>\n\n"
            "Supplemental balance sheet information related to leases was as follows:\n\n"
            "<table><tr><th>Metric</th><th>2025</th></tr>"
            "<tr><td>Finance lease liabilities</td><td>80</td></tr></table>"
        ),
    )
    result = StructuredDocumentLookup(tmp_path).answer(
        "Give me the exact table of supplemental balance sheet information "
        "related to leases exactly from ACME 2025 10-K",
        Scope(("ACME",), ("2025",), "10K"), wants_complete_table=True,
    )

    assert result is not None
    assert "| Finance lease liabilities | 80 |" in result.answer
    assert "| Operating lease assets | 100 |" not in result.answer
    assert "| Operating leases | 10 |" not in result.answer


def test_table_selection_rejects_missing_subject_and_ambiguous_matches(tmp_path) -> None:
    for number, value in enumerate((100, 200)):
        _parent(
            tmp_path, f"balance-{number}", item="Item 8", section="Statements",
            body=("## BALANCE SHEETS\n"
                  "<table><tr><th>Metric</th><th>2025</th></tr>"
                  f"<tr><td>Total assets</td><td>{value}</td></tr></table>"),
        )
    lookup = StructuredDocumentLookup(tmp_path)
    scope = Scope(("MSFT",), ("2025",), "10K")
    for question in (
        "Show the exact supplemental balance sheet table related to leases",
        "Show the exact balance sheet table",
    ):
        assert lookup.answer(question, scope, wants_complete_table=True) is None


def test_plain_caption_matches_a_different_table_subject(tmp_path) -> None:
    _parent(
        tmp_path, "maturities", item="Item 8", section="Notes",
        body=("Future debt maturities were as follows:\n\n"
              "<table><tr><th>Year</th><th>Amount</th></tr>"
              "<tr><td>2026</td><td>75</td></tr></table>"),
    )
    result = StructuredDocumentLookup(tmp_path).answer(
        "Show the complete future debt maturities table",
        Scope(("MSFT",), ("2025",), "10K"), wants_complete_table=True,
    )
    assert result is not None
    assert "| 2026 | 75 |" in result.answer


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


def test_statement_rows_select_requested_comparative_column(tmp_path) -> None:
    _parent(
        tmp_path, "comparative", ticker="TSLA", year="2024", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "## Consolidated Statements of Operations\n(in millions)\n"
            "<table><tr><th>Metric</th><th>2024</th><th>2023</th></tr>"
            "<tr><td>Total automotive revenues</td>"
            "<td>77,070</td><td>82,419</td></tr></table>"
        ),
    )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "What were total automotive revenues in comparative year 2023?",
        Scope(tickers=("TSLA",), years=("2024",), doc_type="10K"),
        value_years=("2023",),
    )

    assert [(row.label, row.year, row.value) for row in rows] == [
        ("Total automotive revenues", "2023", "82,419")
    ]


def test_statement_rows_never_treat_multirow_headers_as_financial_facts(tmp_path) -> None:
    _parent(
        tmp_path, "combined-cash", ticker="MSFT", year="2025", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "## Balance Sheets\n(in millions)\n"
            "<table><thead>"
            "<tr><th rowspan='2'>(In millions, except percentages and per share amounts)</th>"
            "<th></th><th></th></tr>"
            "<tr><th>2025</th><th>2024</th></tr></thead><tbody>"
            "<tr><td>Total cash, cash equivalents, and short-term investments</td>"
            "<td>94,565</td><td>75,543</td></tr></tbody></table>"
        ),
    )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "What was the combined amount of cash, cash equivalents, and "
        "short-term investments in 2025?",
        Scope(tickers=("MSFT",), years=("2025",), doc_type="10K"),
    )

    assert [(row.label, row.year, row.value) for row in rows] == [
        ("Total cash, cash equivalents, and short-term investments", "2025", "94,565")
    ]


def test_statement_rows_use_group_heading_to_disambiguate_metric(tmp_path) -> None:
    _parent(
        tmp_path, "segments", item="Item 7", section="Item 7. MD&A",
        body=(
            "## Segment Results of Operations\n(in millions)\n"
            "<table><tr><th>Metric</th><th>2025</th></tr>"
            "<tr><td colspan='2'>Intelligent Cloud</td></tr>"
            "<tr><td>Operating income</td><td>44,589</td></tr>"
            "<tr><td colspan='2'>Total</td></tr>"
            "<tr><td>Operating income</td><td>128,528</td></tr></table>"
        ),
    )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "What was Intelligent Cloud operating income in 2025?",
        Scope(tickers=("MSFT",), years=("2025",), doc_type="10K"),
    )

    assert [(row.label, row.value) for row in rows] == [
        ("Intelligent Cloud Operating income", "44,589")
    ]


def test_statement_rows_accept_date_and_year_in_one_header_cell(tmp_path) -> None:
    _parent(
        tmp_path, "balance", ticker="TSLA", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "## Consolidated Balance Sheets\n(in millions)\n"
            "<table><tr><th>Metric</th><th>December 31, 2025</th>"
            "<th>December 31, 2024</th></tr>"
            "<tr><td>Total assets</td><td>137,806</td><td>122,070</td>"
            "</tr></table>"
        ),
    )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "What were total assets in 2025?",
        Scope(tickers=("TSLA",), years=("2025",), doc_type="10K"),
    )

    assert [(row.label, row.value) for row in rows] == [
        ("Total assets", "137,806")
    ]


def test_statement_rows_ignore_numeric_period_structure(tmp_path) -> None:
    _parent(
        tmp_path, "operations-period-row", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "## Income Statements\n(in millions)\n"
            "<table><tr><th>Metric</th><th>2025</th><th>2024</th></tr>"
            "<tr><td>Year Ended June 30,</td><td>2025</td><td>2024</td></tr>"
            "<tr><td>Total revenue</td><td>281,724</td><td>245,122</td></tr>"
            "</table>"
        ),
    )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "What was total revenue for the fiscal year ended June 30, 2025?",
        Scope(tickers=("MSFT",), years=("2025",), doc_type="10K"),
    )

    assert [(row.label, row.value) for row in rows] == [
        ("Total revenue", "281,724")
    ]


def test_statement_rows_prefer_specific_metric_over_generic_match(tmp_path) -> None:
    _parent(
        tmp_path, "automotive-revenue", ticker="TSLA", year="2024", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "## Consolidated Statements of Operations\n(in millions)\n"
            "<table><tr><th>Metric</th><th>2024</th><th>2023</th></tr>"
            "<tr><td>Total revenues</td><td>97,690</td><td>96,773</td></tr>"
            "<tr><td>Total automotive revenues</td><td>77,070</td><td>82,419</td></tr>"
            "</table>"
        ),
    )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "What were total automotive revenues for the year ended December 31, 2024?",
        Scope(tickers=("TSLA",), years=("2024",), doc_type="10K"),
    )

    assert [(row.label, row.value) for row in rows] == [
        ("Total automotive revenues", "77,070")
    ]


def test_statement_rows_keep_broad_metric_when_separately_requested(tmp_path) -> None:
    _parent(
        tmp_path, "two-revenue-levels", ticker="TSLA", year="2024", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "## Consolidated Statements of Operations\n(in millions)\n"
            "<table><tr><th>Metric</th><th>2024</th><th>2023</th></tr>"
            "<tr><td>Total revenues</td><td>97,690</td><td>96,773</td></tr>"
            "<tr><td>Total automotive revenues</td><td>77,070</td><td>82,419</td></tr>"
            "</table>"
        ),
    )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "Report both total automotive revenues and total company revenues for 2024.",
        Scope(tickers=("TSLA",), years=("2024",), doc_type="10K"),
    )

    assert {(row.label, row.value) for row in rows} == {
        ("Total revenues", "97,690"),
        ("Total automotive revenues", "77,070"),
    }


def test_operating_income_matches_income_loss_from_operations(tmp_path) -> None:
    _parent(
        tmp_path, "operating-income", ticker="TSLA", year="2025", item="Item 8",
        section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        body=(
            "## Consolidated Statements of Operations\n(in millions)\n"
            "<table><tr><th>Metric</th><th>2025</th><th>2024</th></tr>"
            "<tr><td>Income (loss) from operations</td>"
            "<td>6,625</td><td>7,076</td></tr></table>"
        ),
    )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "What was TSLA operating income in 2025?",
        Scope(tickers=("TSLA",), years=("2025",), doc_type="10K"),
    )

    assert [(row.label, row.value) for row in rows] == [
        ("Income (loss) from operations", "6,625")
    ]


def test_statement_rows_cover_each_requested_company_group(tmp_path) -> None:
    for ticker, value in (("MSFT", "281,724"), ("TSLA", "94,827")):
        _parent(
            tmp_path, f"{ticker}-revenue", ticker=ticker, year="2025", item="Item 8",
            section="Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
            body=(
                "## Consolidated Statements of Operations\n(in millions)\n"
                "<table><tr><th>Metric</th><th>2025</th></tr>"
                f"<tr><td>Total revenues</td><td>{value}</td></tr></table>"
            ),
        )

    rows = StructuredDocumentLookup(tmp_path).statement_rows(
        "Compare MSFT and TSLA total revenue in 2025.",
        Scope(
            tickers=("MSFT", "TSLA"), years=("2025",), doc_type="10K",
            requested_groups=(("MSFT", "2025"), ("TSLA", "2025")),
        ),
    )

    assert {(row.source["ticker"], row.value) for row in rows} == {
        ("MSFT", "281,724"), ("TSLA", "94,827")
    }


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
