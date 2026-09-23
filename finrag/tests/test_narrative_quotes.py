from __future__ import annotations

import unittest

from src.generation.narrative_quotes import NarrativeQuoteSelector


class FakeChain:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def invoke(self, values: dict) -> dict:
        assert "Sources:" not in values["context"]
        return self.payload


def source(source_id: str, text: str, ticker: str = "MSFT") -> dict:
    return {
        "id": source_id,
        "ticker": ticker,
        "fiscal_year": "2025",
        "doc_type": "TRANSCRIPT",
        "evidence_text": text,
        "evidence_requirements": [
            {"requirement_id": "drivers", "evidence_type": "narrative"}
        ],
    }


class NarrativeQuoteSelectorTests(unittest.TestCase):
    def test_accepts_only_exact_passage_from_correct_source(self) -> None:
        passage = "Revenue increased because demand for cloud services grew during the year."
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [
                {"source_id": "S1", "quote": passage},
                {"source_id": "S2", "quote": "Tesla revenue rose because of something not in the source."},
            ]
        }))
        quotes = selector.select(
            question="Compare revenue and explain the reasons.",
            task="revenue: explain the reasons management gave",
            sources=[source("S1", passage), source("S2", "Tesla discussed deliveries.", "TSLA")],
            requirement_ids={"drivers"},
        )
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].source_id, "S1")
        self.assertEqual(quotes[0].text, passage)

    def test_rejects_future_outlook_as_historical_reason(self) -> None:
        passage = "We expect future demand for our products to grow substantially next year."
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [{"source_id": "S1", "quote": passage}]
        }))
        quotes = selector.select(
            question="Why did revenue change in 2025?",
            task="explain the reasons management gave",
            sources=[source("S1", passage)],
            requirement_ids={"drivers"},
        )
        self.assertEqual(quotes, ())

    def test_segment_revenue_fact_is_not_a_reason_for_total_revenue_change(self) -> None:
        passage = "Energy revenue increased 26.6% year over year across all regions."
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [{"source_id": "S1", "quote": passage}]
        }))
        quotes = selector.select(
            question="Explain the reasons for total revenue change.",
            task="revenue: explain the reasons management gave",
            sources=[source("S1", passage)],
            requirement_ids={"drivers"},
        )
        self.assertEqual(quotes, ())

    def test_liquidity_excludes_unrelated_operational_risk(self) -> None:
        passage = "Unauthorized access to our products could damage consumer confidence."
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [{"source_id": "S1", "quote": passage}]
        }))
        quotes = selector.select(
            question="Summarize liquidity risks.",
            task="cash and cash equivalents: summarize principal liquidity risks",
            sources=[source("S1", passage)],
            requirement_ids={"drivers"},
        )
        self.assertEqual(quotes, ())

    def test_liquidity_excludes_balance_sheet_cash_row(self) -> None:
        row = "Cash and cash equivalents 30,242 18,315"
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [{"source_id": "S1", "quote": row}]
        }))
        quotes = selector.select(
            question="Summarize liquidity risks.",
            task="cash and cash equivalents: summarize principal liquidity risks",
            sources=[source("S1", row)],
            requirement_ids={"drivers"},
        )
        self.assertEqual(quotes, ())

    def test_liquidity_falls_back_to_exact_risk_sentence(self) -> None:
        passage = (
            "These investments are subject to credit and liquidity risks, "
            "which may be exacerbated by market downturns."
        )
        candidate = source("S1", passage)
        candidate["section"] = "Item 1A. RISK FACTORS"
        selector = NarrativeQuoteSelector(chain=FakeChain({"quotes": []}))
        quotes = selector.select(
            question="Summarize liquidity risks.",
            task="summarize principal liquidity risks",
            sources=[candidate],
            requirement_ids={"drivers"},
        )
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].text, passage)

    def test_risk_selection_fills_a_company_group_omitted_by_model(self) -> None:
        msft_passage = (
            "Our competitors may develop products that gain greater market acceptance, "
            "which could reduce our revenue and market share."
        )
        tsla_passage = (
            "Increased competition could result in lower vehicle sales, revenue "
            "shortfalls and loss of market share."
        )
        msft = source("S1", msft_passage, "MSFT")
        tsla = source("S2", tsla_passage, "TSLA")
        for candidate in (msft, tsla):
            candidate["section"] = "Item 1A. RISK FACTORS"
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [{"source_id": "S2", "quote": tsla_passage}]
        }))

        quotes = selector.select(
            question="Compare Microsoft and Tesla competition risks.",
            task="Compare competition risks for both companies.",
            sources=[msft, tsla],
            requirement_ids={"drivers"},
        )

        self.assertEqual({quote.ticker for quote in quotes}, {"MSFT", "TSLA"})
        self.assertEqual(
            next(quote.text for quote in quotes if quote.ticker == "MSFT"),
            msft_passage,
        )

    def test_transcript_commentary_is_recovered_when_model_returns_none(self) -> None:
        passage = (
            "On the energy front, we ended the year with nearly $12.8 billion in "
            "revenue, a 26.6% year-over-year growth. This was the result of high "
            "deployments in all regions and continued strength in demand."
        )
        candidate = source("S1", passage, "TSLA")
        selector = NarrativeQuoteSelector(chain=FakeChain({"quotes": []}))

        quotes = selector.select(
            question="Summarize management's revenue commentary.",
            task="Summarize management's revenue commentary from the transcript.",
            sources=[candidate],
            requirement_ids={"drivers"},
        )

        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].text, passage)

    def test_liquidity_requires_an_adverse_risk_not_just_a_covenant_fact(self) -> None:
        passage = "Under certain circumstances we are required to maintain liquidity."
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [{"source_id": "S1", "quote": passage}]
        }))
        quotes = selector.select(
            question="Summarize liquidity risks.",
            task="summarize principal liquidity risks",
            sources=[source("S1", passage)],
            requirement_ids={"drivers"},
        )
        self.assertEqual(quotes, ())

    def test_rd_commentary_rejects_product_announcement(self) -> None:
        passage = "We launched over one hundred new product capabilities this year."
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [{"source_id": "S1", "quote": passage}]
        }))
        quotes = selector.select(
            question="Explain research and development expense commentary.",
            task="research and development expense: explain management commentary",
            sources=[source("S1", passage)],
            requirement_ids={"drivers"},
        )
        self.assertEqual(quotes, ())

    def test_tax_table_text_can_be_quoted_exactly(self) -> None:
        table = "<table><tr><td>Changes in valuation allowances</td><td>389</td><td>7.5</td></tr></table>"
        selector = NarrativeQuoteSelector(chain=FakeChain({
            "quotes": [{"source_id": "S1", "quote": "Changes in valuation allowances 389 7.5"}]
        }))
        quotes = selector.select(
            question="Explain tax reconciliation factors.",
            task="effective tax rates: explain tax reconciliation factors",
            sources=[source("S1", table)],
            requirement_ids={"drivers"},
        )
        self.assertEqual(len(quotes), 1)

    def test_tax_reconciliation_table_is_used_when_model_omits_it(self) -> None:
        table = (
            "<table><tr><th>Amount</th><th>Percent</th></tr>"
            "<tr><td>U.S. federal statutory tax rate</td><td>1,108</td><td>21.0%</td></tr>"
            "<tr><td>Foreign tax credits</td><td>(327)</td><td>(6.2)</td></tr>"
            "<tr><td>Changes in valuation allowances</td><td>389</td><td>7.5</td></tr>"
            "<tr><td>Effective tax rate</td><td>1,423</td><td>27.0%</td></tr></table>"
        )
        candidate = source("S1", table)
        candidate["full_evidence_text"] = table
        selector = NarrativeQuoteSelector(chain=FakeChain({"quotes": []}))
        quotes = selector.select(
            question="Explain tax reconciliation factors.",
            task="effective tax rates: explain tax reconciliation factors",
            sources=[candidate],
            requirement_ids={"drivers"},
        )
        self.assertEqual(len(quotes), 1)
        self.assertIn("Foreign tax credits", quotes[0].text)
        self.assertIn("Changes in valuation allowances", quotes[0].text)
        self.assertEqual(
            quotes[0].factors,
            (("Changes in valuation allowances", "7.5"), ("Foreign tax credits", "-6.2")),
        )

    def test_tax_filing_explanation_is_recovered_verbatim(self) -> None:
        passage = (
            "The decrease in our effective tax rate for fiscal year 2025 "
            "compared to 2024 was due to changes in the mix of earnings."
        )
        candidate = source("S1", passage)
        selector = NarrativeQuoteSelector(chain=FakeChain({"quotes": []}))
        quotes = selector.select(
            question="Explain tax reconciliation factors.",
            task="effective tax rates: explain tax reconciliation factors",
            sources=[candidate],
            requirement_ids={"drivers"},
        )
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].text, passage)


if __name__ == "__main__":
    unittest.main()
