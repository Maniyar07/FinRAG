from __future__ import annotations

import unittest

from src.generation.answer_fact_validator import (
    validate_revenue_change,
    validate_table_answer,
)


TESLA_REVENUE = """
<table>
<tr><th>Year ended December 31</th><th>2024</th><th>2023</th></tr>
<tr><td>Total automotive revenues</td><td>77,070</td><td>82,419</td></tr>
<tr><td>Total automotive cost of revenues</td><td>62,873</td><td>66,389</td></tr>
<tr><td>Total revenues</td><td>97,690</td><td>96,773</td></tr>
</table>
"""

MICROSOFT_SEGMENTS = """
<table>
<tr><th>Year ended June 30</th><th>2024</th><th>2023</th></tr>
<tr><th>Revenue</th><th></th><th></th></tr>
<tr><td>Intelligent Cloud</td><td>105,362</td><td>87,907</td></tr>
<tr><th>Operating Income</th><th></th><th></th></tr>
<tr><td>Intelligent Cloud</td><td>49,584</td><td>37,884</td></tr>
</table>
"""


class AnswerFactValidatorTests(unittest.TestCase):
    def test_rejects_adjacent_automotive_total(self) -> None:
        result = validate_table_answer(
            "What were total automotive revenues for 2024?",
            "Total automotive revenues were $87,604 million [S1].",
            [{"id": "S1", "evidence_text": TESLA_REVENUE}],
        )
        self.assertFalse(result.valid)

    def test_accepts_exact_automotive_row(self) -> None:
        result = validate_table_answer(
            "What were total automotive revenues for 2024?",
            "Total automotive revenues were $77,070 million [S1].",
            [{"id": "S1", "evidence_text": TESLA_REVENUE}],
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "table_value_verified")

    def test_selects_comparative_year_and_operating_income_section(self) -> None:
        question = (
            "According to Microsoft's 2024 10-K, what was Intelligent Cloud "
            "operating income for the comparative fiscal year 2023?"
        )
        sources = [{"id": "S1", "evidence_text": MICROSOFT_SEGMENTS}]
        self.assertTrue(
            validate_table_answer(question, "$37,884 million [S1]", sources).valid
        )
        self.assertFalse(
            validate_table_answer(question, "$49,584 million [S1]", sources).valid
        )

    def test_accepts_a_rounded_billion_value(self) -> None:
        source = """<table>
        <tr><th></th><th>2024</th><th>2023</th></tr>
        <tr><td>Short-term investments</td><td>57,228</td><td>76,552</td></tr>
        </table>"""
        result = validate_table_answer(
            "What were short-term investments in 2024?",
            "Short-term investments were $57.2 billion [S1].",
            [{"id": "S1", "evidence_text": source}],
        )
        self.assertTrue(result.valid)

    def test_inconclusive_layout_does_not_reject_answer(self) -> None:
        result = validate_table_answer(
            "What were total automotive revenues for 2024?",
            "Total automotive revenues were $77,070 million [S1].",
            [{"id": "S1", "evidence_text": "No parseable table."}],
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "table_check_inconclusive")

    def test_composite_question_is_not_checked_as_a_single_row(self) -> None:
        result = validate_table_answer(
            "What was the combined cash and short-term investments in 2024?",
            "The combined amount was $94,565 million [S1].",
            [{"id": "S1", "evidence_text": TESLA_REVENUE}],
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "table_check_composite_question")

    def test_two_year_comparison_is_not_treated_as_one_column(self) -> None:
        result = validate_table_answer(
            "Compare total automotive revenues in 2024 and 2023.",
            "They were $77,070 million and $82,419 million [S1].",
            [{"id": "S1", "evidence_text": TESLA_REVENUE}],
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.reason, "table_check_not_applicable")

    def test_rejects_broader_revenue_percentage_for_exact_metric(self) -> None:
        question = "How did Xbox content and services revenue change in fiscal 2025?"
        sources = [{
            "id": "S1",
            "evidence_text": "Xbox content and services revenue increased 16% in 2025.",
        }]
        result = validate_revenue_change(
            question,
            "Xbox content and services revenue increased $2 billion or 9%. "
            "Xbox content and services revenue increased 16% [S1].",
            sources,
        )
        self.assertFalse(result.valid)
        self.assertIn("16%", result.reason)

    def test_accepts_exact_revenue_percentage(self) -> None:
        result = validate_revenue_change(
            "How did Xbox content and services revenue change in fiscal 2025?",
            "Xbox content and services revenue increased 16% [S1].",
            [{"id": "S1", "evidence_text":
              "Xbox content and services revenue increased 16% in 2025."}],
        )
        self.assertTrue(result.valid)


if __name__ == "__main__":
    unittest.main()
