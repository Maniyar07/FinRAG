from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from src.retrieval.context_builder import ContextBuilder
from src.schemas import RankedChild, Scope
from src.retrieval import token_budget


class ContextBuilderBudgetTests(unittest.TestCase):
    def test_token_counter_initializes_through_the_library_model_mapping(self):
        encoding = Mock()
        encoding.name = "test"
        encoding.encode.return_value = [1, 2]
        token_budget._encoding.cache_clear()
        try:
            with patch.object(token_budget.tiktoken, "encoding_for_model", return_value=encoding) as load:
                self.assertEqual(token_budget.count_tokens("evidence"), 2)
                self.assertEqual(token_budget.token_counter_name(), "tiktoken:test")
                load.assert_called_once_with(token_budget.LLM_MODEL)
        finally:
            token_budget._encoding.cache_clear()

    def test_token_counter_reports_its_conservative_offline_fallback(self):
        token_budget._encoding.cache_clear()
        try:
            with patch.object(token_budget.tiktoken, "encoding_for_model", side_effect=KeyError("unknown model")):
                with self.assertLogs(token_budget.__name__, level="WARNING"):
                    self.assertEqual(token_budget.count_tokens("€"), len("€".encode("utf-8")))
                self.assertEqual(token_budget.token_counter_name(), "utf8_bytes_conservative_fallback")
        finally:
            token_budget._encoding.cache_clear()

    def setUp(self) -> None:
        self.parent_id = "a" * 32
        self.full_parent = "Revenue was 100 million.\n\nOperating income was 20 million."
        self.parent = Document(
            page_content=self.full_parent,
            metadata={
                "ticker": "MSFT",
                "fiscal_year": "2025",
                "doc_type": "10K",
                "source": "annual-report.pdf",
            },
        )
        store = type("MemoryParentStore", (), {
            "get_many": lambda _store, ids: [
                self.parent if parent_id == self.parent_id else None
                for parent_id in ids
            ]
        })()
        with patch("src.retrieval.context_builder.ParentStore", return_value=store):
            self.builder = ContextBuilder(Path.cwd())
        self.scope = Scope(("MSFT",), ("2025",), "10K")
        self.children = [
            RankedChild("child-a", self.parent_id, "Revenue was 100 million.", {
                "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K",
            }, fused_score=0.9),
            RankedChild("child-b", self.parent_id, "Operating income was 20 million.", {
                "ticker": "MSFT", "fiscal_year": "2025", "doc_type": "10K",
            }, fused_score=0.8),
        ]

    def test_disjoint_child_hits_return_the_complete_parent_once(self) -> None:
        bundle = self.builder.build(
            self.children,
            self.scope,
            max_parents=8,
            max_chars=48_000,
            max_tokens=2_000,
            query="revenue operating income",
            preserve_all=False,
        )

        self.assertEqual(len(bundle.sources), 1)
        self.assertEqual(bundle.sources[0]["parent_id"], self.parent_id)
        self.assertTrue(bundle.sources[0]["parent_complete"])
        self.assertIn("Revenue was 100 million.", bundle.context)
        self.assertIn("Operating income was 20 million.", bundle.context)

    def test_parent_is_excluded_only_after_it_was_returned_complete(self) -> None:
        bundle = self.builder.build(
            self.children,
            self.scope,
            max_parents=8,
            max_chars=48_000,
            max_tokens=2_000,
            query="revenue operating income",
            preserve_all=True,
            exclude_parent_ids=frozenset({self.parent_id}),
        )

        self.assertEqual(bundle.sources, [])
        self.assertEqual(bundle.context, "")
        self.assertEqual(bundle.context_tokens, 0)

    def test_token_budget_never_truncates_an_indivisible_parent(self) -> None:
        bundle = self.builder.build(
            self.children,
            self.scope,
            max_parents=8,
            max_chars=48_000,
            max_tokens=1,
            query="revenue operating income",
            preserve_all=True,
        )

        self.assertEqual(bundle.sources, [])
        self.assertEqual(bundle.context, "")
        self.assertLessEqual(bundle.context_tokens, 1)

    def test_oversized_parent_returns_a_complete_relevant_block(self) -> None:
        self.parent.page_content = (
            "Background unrelated to the requested metric. " * 80
            + "\n\n"
            + "<table><tr><th>Year</th><th>Revenue</th></tr>"
            + "<tr><td>2025</td><td>$100 million</td></tr></table>\n"
        )

        bundle = self.builder.build(
            self.children[:1],
            self.scope,
            max_parents=1,
            max_chars=700,
            max_tokens=700,
            query="total revenue 2025 supporting row",
            preserve_all=False,
        )

        self.assertEqual(len(bundle.sources), 1)
        self.assertEqual(bundle.sources[0]["evidence_mode"], "budget_excerpt")
        self.assertFalse(bundle.sources[0]["parent_complete"])
        self.assertIn("$100 million", bundle.context)
        self.assertNotIn("Background unrelated", bundle.context)
        self.assertEqual(
            bundle.diagnostics["selected_source_count"], 1
        )

    def test_preserving_a_parent_never_silently_discards_disjoint_evidence(self) -> None:
        self.parent.page_content = self.full_parent * 40
        bundle = self.builder.build(self.children, self.scope, max_parents=1,
                                    max_chars=200, max_tokens=200, query="Revenue", preserve_all=True)
        self.assertEqual(bundle.sources, [])
        self.assertEqual(bundle.context, "")


if __name__ == "__main__":
    unittest.main()
