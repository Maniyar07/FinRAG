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

    def test_explicit_call_boundary_excludes_editorial_names_and_definitions(self) -> None:
        text, speakers, _ = format_transcript([
            "Image source: A financial publisher.", "Call participants", "Chief Executive Officer - Jane Doe",
            "Program Orion: A stock compensation plan.", "Full Conference Call Transcript",
            "Jane Doe: Storage deployments grew. We expect demand to remain strong.",
            "John Smith", "Chief Financial Officer", "Our capacity expansion requires investment.",
        ])
        self.assertEqual(speakers, {"Jane Doe", "John Smith"})
        self.assertNotIn("financial publisher", text)
        self.assertNotIn("stock compensation plan", text)
        self.assertIn("Storage deployments grew", text)
        self.assertIn("capacity expansion requires investment", text)

    def test_boundary_words_inside_spoken_sentences_do_not_trim_the_call(self) -> None:
        text, speakers, _ = format_transcript([
            "Operator: Welcome. A full conference call transcript will be available later.",
            "Jane Doe: Here are our prepared remarks and financial results.",
        ])
        self.assertEqual(speakers, {"Operator", "Jane Doe"})
        self.assertIn("Welcome", text)

    def test_toc_is_skipped_until_remarks_with_a_role_qualified_speaker(self) -> None:
        text, speakers, _ = format_transcript([
            "Image source: Publisher.", "Contents:", "Prepared Remarks", "Questions and Answers", "Call Participants",
            "Prepared Remarks:", "Jane Doe -- Chief Executive Officer", "Our investments support new factory capacity.",
            "Questions and Answers", "John Smith -- Analyst", "What is the demand outlook?",
            "Call Participants", "Board participants not speaking in this call.",
        ])
        self.assertEqual(speakers, {"Jane Doe", "John Smith"})
        self.assertIn("[SPEAKER: Jane Doe | ROLE: Chief Executive Officer]", text)
        self.assertIn("What is the demand outlook?", text)
        self.assertNotIn("Publisher", text)
        self.assertNotIn("Board participants", text)

    def test_attributed_customer_quote_is_text_under_the_current_speaker(self) -> None:
        text, speakers, _ = format_transcript([
            "Jane Doe: Demand for our products increased.",
            "As one physician put it: The service saves time.",
        ])
        self.assertEqual(speakers, {"Jane Doe"})
        self.assertIn("As one physician put it: The service saves time.", text)


if __name__ == "__main__":
    unittest.main()
