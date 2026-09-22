from __future__ import annotations

import unittest

from src.ingestion.document_processing import format_transcript, split_sections


class DocumentProcessingTests(unittest.TestCase):
    def test_section_split_preserves_item_identity(self) -> None:
        text = "Intro\n\nITEM 1A. RISK FACTORS\nSupply constraints may affect results."
        fragments, inherited = split_sections(text, "Front Matter")
        self.assertEqual(fragments[-1].item, "Item 1A")
        self.assertIn("Risk Factors", fragments[-1].section.title())
        self.assertEqual(inherited, fragments[-1].section)

    def test_inline_transcript_speaker(self) -> None:
        text, speakers, sections = format_transcript(
            ["Prepared Remarks", "Operator: Welcome to the call.", "QUESTION AND ANSWER SECTION"]
        )
        self.assertIn("[SPEAKER: Operator]", text)
        self.assertIn("Operator", speakers)
        self.assertIn("Question and Answer", sections)

    def test_multiline_speaker_and_role(self) -> None:
        text, speakers, _ = format_transcript(
            ["Jeremy Barnum", "Chief Financial Officer", "Revenue increased during the year."]
        )
        self.assertIn("[SPEAKER: Jeremy Barnum | ROLE: Chief Financial Officer]", text)
        self.assertIn("Jeremy Barnum", speakers)

    def test_financial_label_is_not_misclassified_as_speaker(self) -> None:
        text, speakers, _ = format_transcript(
            ["Revenue: $64.7 billion", "Satya Nadella: Demand remained strong."]
        )
        self.assertNotIn("Revenue", speakers)
        self.assertIn("Satya Nadella", speakers)
        self.assertIn("Revenue: $64.7 billion", text)

    def test_margin_badges_are_removed(self) -> None:
        text, _, _ = format_transcript(["Q", "Jane Doe", "Research Analyst", "Question text."])
        self.assertNotIn("\n\nQ\n\n", text)


if __name__ == "__main__":
    unittest.main()
