from __future__ import annotations

import unittest

from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lexical_index import LexicalHit
from src.schemas import RetrievalBundle, Scope


def _metadata(child_id: str, parent_id: str) -> dict:
    return {
        "child_id": child_id,
        "parent_id": parent_id,
        "ticker": "MSFT",
        "fiscal_year": "2025",
        "doc_type": "10K",
    }


class FakeVectorStore:
    def similarity_search_with_score(self, *, query, k, filter):
        return [(Document(page_content="baseline", metadata=_metadata("c1", "p1")), 0.9)]


class FakeLexicalIndex:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query, scope, *, k):
        self.queries.append(query)
        if query == "remaining performance obligations":
            return [
                LexicalHit(
                    record={
                        **_metadata("c2", "p2"),
                        "text": "remaining performance obligations were disclosed",
                        "metadata": _metadata("c2", "p2"),
                    },
                    score=3.0,
                    coverage=1.0,
                )
            ]
        return []


class FakeContextBuilder:
    def __init__(self) -> None:
        self.ranked = []
        self.kwargs = {}

    def build(self, ranked, scope, **kwargs):
        self.ranked = ranked
        self.kwargs = kwargs
        return RetrievalBundle("context", [], scope, len(ranked))


class HybridExpansionTests(unittest.TestCase):
    def test_expansion_is_lexical_only_lower_priority_and_preserves_tables(self) -> None:
        lexical = FakeLexicalIndex()
        context = FakeContextBuilder()
        retriever = HybridRetriever(
            FakeVectorStore(),
            lexical_index=lexical,
            context_builder=context,
        )

        retriever.retrieve(
            "contracted but unrecognized revenue",
            Scope(("MSFT",), ("2025",), "10K"),
            query_expansions=("remaining performance obligations",),
            preserve_all=True,
        )

        self.assertEqual(
            lexical.queries,
            ["contracted but unrecognized revenue", "remaining performance obligations"],
        )
        self.assertEqual([child.child_id for child in context.ranked], ["c1", "c2"])
        self.assertGreater(context.ranked[0].fused_score, context.ranked[1].fused_score)
        self.assertEqual(context.kwargs["query"], "contracted but unrecognized revenue")
        self.assertTrue(context.kwargs["preserve_all"])


if __name__ == "__main__":
    unittest.main()
