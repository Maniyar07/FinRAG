from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

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
        self.limits: list[int] = []

    def search(self, query, scope, *, k):
        self.queries.append(query)
        self.limits.append(k)
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
        self.assertTrue(
            all(
                child.metadata["rrf_score"] == child.fused_score
                for child in context.ranked
            )
        )
        self.assertEqual(context.kwargs["query"], "contracted but unrecognized revenue")
        self.assertTrue(context.kwargs["preserve_all"])

    def test_expansions_share_one_lexical_candidate_budget(self) -> None:
        lexical = FakeLexicalIndex()
        retriever = HybridRetriever(
            FakeVectorStore(),
            lexical_index=lexical,
            context_builder=FakeContextBuilder(),
        )

        retriever.retrieve(
            "baseline",
            Scope(("MSFT",), ("2025",), "10K"),
            query_expansions=("expansion one", "expansion two", "expansion three"),
        )

        self.assertEqual(lexical.limits[0], 48)
        self.assertEqual(lexical.limits[1:], [16, 16, 16])

    def test_request_context_limits_are_forwarded_to_parent_builder(self) -> None:
        context = FakeContextBuilder()
        retriever = HybridRetriever(
            FakeVectorStore(),
            lexical_index=FakeLexicalIndex(),
            context_builder=context,
        )

        retriever.retrieve(
            "total revenue",
            Scope(("MSFT",), ("2025",), "10K"),
            max_parents=4,
            max_chars=24_000,
        )

        self.assertEqual(context.kwargs["max_parents"], 4)
        self.assertEqual(context.kwargs["max_chars"], 24_000)


def test_independent_observations_share_one_embedding_batch_across_locked_groups():
    scope = Scope(("MSFT", "TSLA"), ("2025",), "10K")
    embeddings = SimpleNamespace(embed_documents=Mock(return_value=[[1.0], [2.0], [3.0]]))
    def vector_search(*, embedding, k, filter):
        ticker = next(condition.match.value for condition in filter.must if condition.key == "metadata.ticker")
        child = f"{ticker}-{int(embedding[0])}"
        return [(Document(page_content="Operating drivers and investment evidence.", metadata={
            **_metadata(child, child), "ticker": ticker}), 0.9)]
    vectors = SimpleNamespace(embeddings=embeddings,
        similarity_search_with_score_by_vector=Mock(side_effect=vector_search),
        similarity_search_with_score=Mock(side_effect=AssertionError("Do not embed each group separately")))
    context = FakeContextBuilder()
    retriever = HybridRetriever(vectors, lexical_index=SimpleNamespace(search=lambda *args, **kwargs: []), context_builder=context)
    retriever.retrieve("costs and investment", scope,
        requirement_queries=("cost drivers", "investment needs", "cost drivers"), max_parents=4)
    embeddings.embed_documents.assert_called_once_with(["costs and investment", "cost drivers", "investment needs"])
    assert vectors.similarity_search_with_score_by_vector.call_count == 6
    for ticker in scope.tickers:
        for index, query in ((2, "cost drivers"), (3, "investment needs")):
            hit = next(child for child in context.ranked if child.parent_id == f"{ticker}-{index}")
            assert hit.metadata["evidence_queries"] == (query,)
    assert context.kwargs["max_parents"] == 4


def test_facet_fusion_recovers_operating_evidence_over_keyword_only_allocation():
    scope = Scope(("MSFT",), ("2025",), "10K")
    operating = Document(page_content="Costs include computing infrastructure, labor and development.", metadata=_metadata("op", "operations"))
    allocation = {"child_id": "allocation", "text": "Our allocation structure assigns reported costs.",
                  "metadata": _metadata("allocation", "allocation")}
    baseline = Document(page_content="Other corporate disclosures.", metadata=_metadata("base", "baseline"))
    def lexical(query, scope, *, k):
        return [LexicalHit(allocation, 10.0, 1.0), LexicalHit({"child_id": "op", "text": operating.page_content,
            "metadata": operating.metadata}, 2.0, 0.5)] if query == "cost structure" else []
    vectors = SimpleNamespace(embeddings=SimpleNamespace(embed_documents=lambda queries: [[float(index)] for index in range(len(queries))]),
        similarity_search_with_score_by_vector=lambda *, embedding, **kwargs: [(operating if embedding[0] else baseline, 0.9)])
    context = FakeContextBuilder()
    retriever = HybridRetriever(vectors, lexical_index=SimpleNamespace(search=lexical), context_builder=context)
    retriever.retrieve("business comparison", scope, requirement_queries=("cost structure",))
    best = max(context.ranked, key=lambda child: child.fused_score)
    assert best.parent_id == "operations" and best.metadata["evidence_queries"] == ("cost structure",)


if __name__ == "__main__":
    unittest.main()
