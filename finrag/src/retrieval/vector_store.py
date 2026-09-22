from __future__ import annotations

import atexit
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    get_index_paths,
    validate_runtime_config,
)


class VectorSchemaError(RuntimeError):
    """Raised when an index was built with incompatible embedding settings."""


class FinancialVectorStore:
    """Open one isolated local-Qdrant index and validate its dense-vector schema."""

    def __init__(self, *, index_version: str | None = None, create_if_missing: bool = False):
        validate_runtime_config(require_openai=True)
        self.index_paths = get_index_paths(index_version)
        self.collection_name = COLLECTION_NAME
        self.client: QdrantClient | None = None

        if not self.index_paths.qdrant_dir.exists() and not create_if_missing:
            raise FileNotFoundError(
                f"Index '{self.index_paths.version}' does not exist. Run ingestion first."
            )
        self.index_paths.qdrant_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            api_key=OPENAI_API_KEY,
        )

        try:
            self.client = QdrantClient(path=str(self.index_paths.qdrant_dir))
            names = {item.name for item in self.client.get_collections().collections}
            exists = self.collection_name in names
            if not exists and not create_if_missing:
                raise FileNotFoundError(
                    f"Collection '{self.collection_name}' is missing from index "
                    f"'{self.index_paths.version}'."
                )
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSIONS, distance=Distance.COSINE
                    ),
                )
            self._validate_schema()
        except Exception:
            self.close()
            raise

        atexit.register(self.close)

    def _validate_schema(self) -> None:
        if self.client is None:
            raise RuntimeError("Qdrant client is closed.")
        collection = self.client.get_collection(self.collection_name)
        vectors: Any = collection.config.params.vectors
        if isinstance(vectors, dict):
            raise VectorSchemaError("FinRAG expects one unnamed dense vector.")
        actual_size = getattr(vectors, "size", None)
        actual_distance = getattr(vectors, "distance", None)
        distance_value = getattr(actual_distance, "value", str(actual_distance)).lower()
        if actual_size != EMBEDDING_DIMENSIONS:
            raise VectorSchemaError(
                f"Embedding dimensions differ: index={actual_size}, "
                f"configuration={EMBEDDING_DIMENSIONS}. Build a new index version."
            )
        if distance_value != Distance.COSINE.value.lower():
            raise VectorSchemaError(
                f"Distance differs: index={actual_distance}, expected={Distance.COSINE.value}."
            )

    def get_vectorstore(self) -> QdrantVectorStore:
        if self.client is None:
            raise RuntimeError("Qdrant client is closed.")
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def point_count(self) -> int:
        if self.client is None:
            raise RuntimeError("Qdrant client is closed.")
        return int(
            self.client.count(collection_name=self.collection_name, exact=True).count
        )

    def close(self) -> None:
        client = self.client
        if client is None:
            return
        try:
            client.close()
        except (Exception, ModuleNotFoundError, TypeError, AttributeError):
            pass
        finally:
            self.client = None

