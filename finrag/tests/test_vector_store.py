from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.retrieval.vector_store import FinancialVectorStore


class FinancialVectorStoreTests(unittest.TestCase):
    def test_reuses_local_schema_validation_without_remote_dummy_embedding(self) -> None:
        store = FinancialVectorStore.__new__(FinancialVectorStore)
        store.client = Mock(name="qdrant_client")
        store.collection_name = "finrag_documents"
        store.embeddings = Mock(name="openai_embeddings")

        with patch("src.retrieval.vector_store.QdrantVectorStore") as vectorstore_type:
            result = store.get_vectorstore()

        vectorstore_type.assert_called_once_with(
            client=store.client,
            collection_name=store.collection_name,
            embedding=store.embeddings,
            validate_collection_config=False,
        )
        self.assertIs(result, vectorstore_type.return_value)


if __name__ == "__main__":
    unittest.main()
