from __future__ import annotations

import unittest

from src.schemas import RetrievalBundle, Scope
from src.tools.document_search import (
    DocumentSearchPurpose,
    DocumentSearchRequest,
    DocumentSearchTool,
)


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, query, scope, **kwargs):
        self.calls.append((query, scope, kwargs))
        return RetrievalBundle(
            context="retrieved evidence",
            sources=[{"id": "S1", "evidence_text": "evidence"}],
            scope=scope,
            candidate_count=3,
            covered_groups=scope.groups,
        )


class DocumentSearchToolTests(unittest.TestCase):
    def test_delegates_to_existing_retriever_without_changing_behavior(self) -> None:
        retriever = FakeRetriever()
        tool = DocumentSearchTool(retriever)
        scope = Scope(("MSFT",), ("2024",), "10K")
        request = DocumentSearchRequest(
            query="  Why   did revenue increase? ",
            scope=scope,
            purpose=DocumentSearchPurpose.MANAGEMENT_EXPLANATION,
            query_expansions=("revenue drivers",),
            preserve_all=True,
        )

        result = tool.execute(request, permitted_scope=scope)

        self.assertEqual(result.query, "Why did revenue increase?")
        self.assertEqual(
            result.purpose, DocumentSearchPurpose.MANAGEMENT_EXPLANATION
        )
        self.assertEqual(result.context, "retrieved evidence")
        self.assertEqual(result.candidate_count, 3)
        self.assertEqual(
            retriever.calls,
            [
                (
                    "Why did revenue increase?",
                    scope,
                    {
                        "query_expansions": ("revenue drivers",),
                        "preserve_all": True,
                    },
                )
            ],
        )

    def test_allows_a_focused_subset_of_the_permitted_scope(self) -> None:
        retriever = FakeRetriever()
        tool = DocumentSearchTool(retriever)
        permitted = Scope(
            tickers=("MSFT", "TSLA"),
            years=("2024", "2025"),
            required_doc_types=("10K", "TRANSCRIPT"),
        )
        requested = Scope(("MSFT",), ("2024",), "10K")

        result = tool.execute(
            DocumentSearchRequest("Microsoft risk factors", requested),
            permitted_scope=permitted,
        )

        self.assertEqual(result.bundle.scope, requested)
        self.assertEqual(len(retriever.calls), 1)

    def test_rejects_company_or_year_scope_expansion(self) -> None:
        tool = DocumentSearchTool(FakeRetriever())
        permitted = Scope(("MSFT",), ("2024",), "10K")
        requested = Scope(("TSLA",), ("2024",), "10K")

        with self.assertRaisesRegex(ValueError, "company/year"):
            tool.execute(
                DocumentSearchRequest("Tesla risk factors", requested),
                permitted_scope=permitted,
            )

    def test_rejects_document_type_scope_expansion(self) -> None:
        tool = DocumentSearchTool(FakeRetriever())
        permitted = Scope(("MSFT",), ("2024",), "10K")
        requested = Scope(("MSFT",), ("2024",))

        with self.assertRaisesRegex(ValueError, "document-type"):
            tool.execute(
                DocumentSearchRequest("Microsoft risks", requested),
                permitted_scope=permitted,
            )

    def test_rejects_invalid_query_expansions_before_retrieval(self) -> None:
        retriever = FakeRetriever()
        tool = DocumentSearchTool(retriever)
        scope = Scope(("MSFT",), ("2024",), "10K")

        with self.assertRaisesRegex(ValueError, "at most 3"):
            DocumentSearchRequest(
                "Microsoft risks",
                scope,
                query_expansions=("one", "two", "three", "four"),
            )

        self.assertEqual(retriever.calls, [])


if __name__ == "__main__":
    unittest.main()
