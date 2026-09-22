from __future__ import annotations

import json
import unittest
from pathlib import Path


class RagasGoldAlignmentTests(unittest.TestCase):
    def test_cross_company_questions_match_their_references(self) -> None:
        path = Path(__file__).resolve().parents[1] / "evals" / "ragas_gold.json"
        rows = {row["id"]: row for row in json.loads(path.read_text(encoding="utf-8"))}

        for row_id, metric in ((38, "total revenue"), (41, "total current liabilities")):
            with self.subTest(row_id=row_id):
                self.assertIn(metric, rows[row_id]["question"].casefold())
                self.assertIn(metric, rows[row_id]["reference"].casefold())


if __name__ == "__main__":
    unittest.main()
