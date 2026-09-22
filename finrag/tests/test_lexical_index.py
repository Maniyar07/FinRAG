from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval.lexical_index import LexicalIndex
from src.schemas import Scope


class LexicalIndexTests(unittest.TestCase):
    def test_exact_abbreviation_and_metadata_filter(self) -> None:
        records = [
            {
                "child_id": "c1",
                "parent_id": "p1",
                "text": "JPM CET1 ratio increased during 2024.",
                "metadata": {
                    "ticker": "JPM",
                    "fiscal_year": "2024",
                    "doc_type": "10K",
                },
            },
            {
                "child_id": "c2",
                "parent_id": "p2",
                "text": "Tesla automotive revenue and gross margin.",
                "metadata": {
                    "ticker": "TSLA",
                    "fiscal_year": "2024",
                    "doc_type": "10K",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            index = LexicalIndex(path)
            hits = index.search("CET1", Scope(("JPM",), ("2024",), "10K"), k=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].record["child_id"], "c1")
        self.assertEqual(hits[0].coverage, 1.0)


if __name__ == "__main__":
    unittest.main()

