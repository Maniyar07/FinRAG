from __future__ import annotations

import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHILD_OVERLAP_10K,
    CHILD_TOKENS_10K,
    PARENT_OVERLAP_10K,
    PARENT_TOKENS_10K,
    TS_CHILD_OVERLAP,
    TS_CHILD_TOKENS,
    TS_PARENT_OVERLAP,
    TS_PARENT_TOKENS,
)
from src.constants import CHUNK_SCHEMA_VERSION
from src.ingestion.document_processing import PAGE_MARKER_RE, page_range
from src.ingestion.lexical_store import write_lexical_index
from src.ingestion.parent_store import ParentStore


ID_KEY = "parent_id"
TEN_K_SEPARATORS = ["\n## ", "\n### ", "\n|", "\n\n", "\n", ". ", " "]
TRANSCRIPT_SEPARATORS = ["\n[SECTION:", "\n[SPEAKER:", "\n\n", "\n", ". ", " "]
SECTION_TAG_RE = re.compile(r"\[SECTION:\s*([^\]]+)\]")
SPEAKER_TAG_RE = re.compile(r"\[SPEAKER:\s*([^\]|]+?)(?:\s*\|\s*ROLE:\s*([^\]]+))?\]")
REQUIRED_SOURCE_METADATA = {
    "document_id",
    "source_segment_id",
    "ticker",
    "fiscal_year",
    "fiscal_period",
    "doc_type",
    "source",
    "source_hash",
    "section",
    "metadata_schema_version",
}


def _stable_uuid(*parts: object) -> str:
    return str(uuid5(NAMESPACE_URL, "|".join(map(str, parts))))


def _splitter(
    size: int, overlap: int, separators: list[str]
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=size,
        chunk_overlap=overlap,
        separators=separators,
        keep_separator=True,
    )


def _content_metadata(document: Document) -> dict:
    metadata = dict(document.metadata)
    metadata.pop("speakers", None)
    metadata.pop("speakers_in_chunk", None)
    has_page_marker = bool(PAGE_MARKER_RE.search(document.page_content))
    inherited_page = metadata.get("pdf_page_start")
    if has_page_marker or inherited_page is not None:
        start, end = page_range(
            document.page_content,
            int(inherited_page or 1),
            int(metadata.get("pdf_page_end") or inherited_page or 1),
        )
        metadata["pdf_page_start"] = start
        metadata["pdf_page_end"] = end

    sections = SECTION_TAG_RE.findall(document.page_content)
    if sections:
        metadata["section"] = sections[0].strip()
    speakers = SPEAKER_TAG_RE.findall(document.page_content)
    if speakers:
        unique_speakers = tuple(dict.fromkeys(value[0].strip() for value in speakers))
        metadata["speakers_in_chunk"] = list(unique_speakers)
        if len(unique_speakers) == 1:
            metadata["speaker"] = unique_speakers[0]
            roles = tuple(
                dict.fromkeys(value[1].strip() for value in speakers if value[1].strip())
            )
            if len(roles) == 1:
                metadata["speaker_role"] = roles[0]
        else:
            metadata.pop("speaker", None)
            metadata.pop("speaker_role", None)
    return metadata


class FinancialChunker:
    """Create deterministic parent chunks and small hybrid-search children."""

    def __init__(self, vectorstore, *, parent_store_path: Path, lexical_index_path: Path):
        self.vectorstore = vectorstore
        self.parent_store_path = Path(parent_store_path)
        self.lexical_index_path = Path(lexical_index_path)
        self.parent_store = ParentStore(self.parent_store_path, create=True)

        self.k10_parent = _splitter(
            PARENT_TOKENS_10K, PARENT_OVERLAP_10K, TEN_K_SEPARATORS
        )
        self.k10_child = _splitter(CHILD_TOKENS_10K, CHILD_OVERLAP_10K, TEN_K_SEPARATORS)
        self.ts_parent = _splitter(
            TS_PARENT_TOKENS, TS_PARENT_OVERLAP, TRANSCRIPT_SEPARATORS
        )
        self.ts_child = _splitter(TS_CHILD_TOKENS, TS_CHILD_OVERLAP, TRANSCRIPT_SEPARATORS)

    def _splitters_for(self, doc_type: str):
        if doc_type == "10K":
            return self.k10_parent, self.k10_child
        if doc_type == "TRANSCRIPT":
            return self.ts_parent, self.ts_child
        raise ValueError(f"Unsupported doc_type '{doc_type}'.")

    @staticmethod
    def _embedding_prefix(metadata: dict) -> str:
        fields = [
            metadata.get("ticker"),
            metadata.get("fiscal_year"),
            metadata.get("doc_type"),
            metadata.get("section"),
            metadata.get("speaker"),
        ]
        return "[" + " | ".join(str(value) for value in fields if value) + "]\n"

    def build_chunks(
        self, documents: list[Document]
    ) -> tuple[list[tuple[str, Document]], list[Document], list[str]]:
        parents: list[tuple[str, Document]] = []
        children: list[Document] = []
        child_ids: list[str] = []

        for source in documents:
            missing_metadata = sorted(
                key for key in REQUIRED_SOURCE_METADATA if source.metadata.get(key) in (None, "")
            )
            if missing_metadata:
                raise ValueError(
                    "Source segment is missing required metadata: "
                    + ", ".join(missing_metadata)
                )
            doc_type = str(source.metadata.get("doc_type", "")).upper()
            parent_splitter, child_splitter = self._splitters_for(doc_type)
            source_id = source.metadata.get("source_segment_id")
            if not source_id:
                raise ValueError("Every source segment requires source_segment_id metadata.")

            for parent_index, parent in enumerate(parent_splitter.split_documents([source])):
                parent_id = _stable_uuid(source_id, "parent", parent_index, CHUNK_SCHEMA_VERSION)
                parent.metadata = _content_metadata(parent)
                parent.metadata.update(
                    {
                        "parent_id": parent_id,
                        "parent_index": parent_index,
                        "chunk_type": "parent",
                        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                        "word_count": len(parent.page_content.split()),
                    }
                )
                parents.append((parent_id, parent))

                for child_index, child in enumerate(child_splitter.split_documents([parent])):
                    child_id = _stable_uuid(
                        parent_id, "child", child_index, CHUNK_SCHEMA_VERSION
                    )
                    child.metadata = _content_metadata(child)
                    child.metadata.update(
                        {
                            ID_KEY: parent_id,
                            "child_id": child_id,
                            "child_index": child_index,
                            "chunk_type": "child",
                            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                            "word_count": len(child.page_content.split()),
                        }
                    )
                    child.page_content = self._embedding_prefix(child.metadata) + child.page_content
                    children.append(child)
                    child_ids.append(child_id)

        if not parents or not children:
            raise ValueError("Chunking produced no searchable content.")
        parent_ids = [parent_id for parent_id, _ in parents]
        if len(parent_ids) != len(set(parent_ids)):
            raise RuntimeError("Deterministic parent ID collision detected.")
        if len(child_ids) != len(set(child_ids)):
            raise RuntimeError("Deterministic child ID collision detected.")
        return parents, children, child_ids

    def process_and_add_documents(
        self, documents: list[Document], *, batch_size: int = 128
    ) -> dict[str, int]:
        if not documents:
            raise ValueError("No source documents were provided for chunking.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        parents, children, child_ids = self.build_chunks(documents)
        self.parent_store.put_many(parents)
        write_lexical_index(self.lexical_index_path, children)
        for start in range(0, len(children), batch_size):
            end = min(start + batch_size, len(children))
            self.vectorstore.add_documents(children[start:end], ids=child_ids[start:end])

        return {
            "source_segments": len(documents),
            "parent_chunks": len(parents),
            "child_chunks": len(children),
        }
