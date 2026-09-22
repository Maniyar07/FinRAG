from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.constants import INDEX_LAYOUT_VERSION


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SOURCE_DATA_DIR = BASE_DIR / "Data"
RUNTIME_DATA_DIR = BASE_DIR / "data"
INDEXES_DIR = RUNTIME_DATA_DIR / "indexes"
LOGS_DIR = RUNTIME_DATA_DIR / "logs"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

COLLECTION_NAME = os.getenv("FINRAG_QDRANT_COLLECTION", "finrag_documents")
EMBEDDING_MODEL = os.getenv("FINRAG_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("FINRAG_EMBEDDING_DIMENSIONS", "1536"))
LLM_MODEL = os.getenv("FINRAG_LLM_MODEL", "gpt-4o-mini")
SEMANTIC_FALLBACK_ENABLED = os.getenv(
    "FINRAG_SEMANTIC_FALLBACK_ENABLED", "false"
).strip().casefold() in {"1", "true", "yes", "on"}
SEMANTIC_FALLBACK_MODEL = os.getenv(
    "FINRAG_SEMANTIC_FALLBACK_MODEL", LLM_MODEL
).strip()
SEMANTIC_FALLBACK_TIMEOUT_SECONDS = float(
    os.getenv("FINRAG_SEMANTIC_FALLBACK_TIMEOUT_SECONDS", "10")
)

PARENT_TOKENS_10K = int(os.getenv("FINRAG_10K_PARENT_TOKENS", "3200"))
PARENT_OVERLAP_10K = int(os.getenv("FINRAG_10K_PARENT_OVERLAP", "400"))
CHILD_TOKENS_10K = int(os.getenv("FINRAG_10K_CHILD_TOKENS", "800"))
CHILD_OVERLAP_10K = int(os.getenv("FINRAG_10K_CHILD_OVERLAP", "200"))

TS_PARENT_TOKENS = int(os.getenv("FINRAG_TS_PARENT_TOKENS", "2000"))
TS_PARENT_OVERLAP = int(os.getenv("FINRAG_TS_PARENT_OVERLAP", "250"))
TS_CHILD_TOKENS = int(os.getenv("FINRAG_TS_CHILD_TOKENS", "400"))
TS_CHILD_OVERLAP = int(os.getenv("FINRAG_TS_CHILD_OVERLAP", "100"))

DENSE_CANDIDATE_K = int(os.getenv("FINRAG_DENSE_CANDIDATE_K", "48"))
LEXICAL_CANDIDATE_K = int(os.getenv("FINRAG_LEXICAL_CANDIDATE_K", "48"))
MAX_CONTEXT_PARENTS = int(os.getenv("FINRAG_MAX_CONTEXT_PARENTS", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("FINRAG_MAX_CONTEXT_CHARS", "48000"))
MIN_DENSE_SCORE = float(os.getenv("FINRAG_MIN_DENSE_SCORE", "0.20"))
MIN_LEXICAL_COVERAGE = float(os.getenv("FINRAG_MIN_LEXICAL_COVERAGE", "0.20"))

INDEX_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ACTIVE_INDEX_VERSION = os.getenv("FINRAG_INDEX_VERSION", "").strip()
DEBUG_TRACES = os.getenv("FINRAG_DEBUG_TRACES", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
MULTIHOP_ENABLED = os.getenv(
    "FINRAG_MULTIHOP_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
MULTIHOP_MAX_SEARCHES = int(os.getenv("FINRAG_MULTIHOP_MAX_SEARCHES", "4"))
MULTIHOP_MAX_CALCULATIONS = int(
    os.getenv("FINRAG_MULTIHOP_MAX_CALCULATIONS", "6")
)


@dataclass(frozen=True)
class IndexPaths:
    version: str
    root: Path
    qdrant_dir: Path
    parent_docstore_dir: Path
    lexical_index_path: Path
    manifest_path: Path


def normalize_index_version(value: str | None) -> str:
    version = (value or ACTIVE_INDEX_VERSION).strip()
    if not version:
        raise ValueError(
            "No index version selected. Set FINRAG_INDEX_VERSION or pass --index-version."
        )
    if not INDEX_VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            "Index version must start with a letter or number and contain at most "
            "64 letters, numbers, dots, underscores, or hyphens."
        )
    return version


def get_index_paths(index_version: str | None = None) -> IndexPaths:
    version = normalize_index_version(index_version)
    root = INDEXES_DIR / version
    return IndexPaths(
        version=version,
        root=root,
        qdrant_dir=root / "qdrant",
        parent_docstore_dir=root / "parents",
        lexical_index_path=root / "lexical_chunks.jsonl",
        manifest_path=root / "manifest.json",
    )


def validate_runtime_config(
    *, require_openai: bool = False, require_llama: bool = False
) -> None:
    missing = []
    if require_openai and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if require_llama and not LLAMA_CLOUD_API_KEY:
        missing.append("LLAMA_CLOUD_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    if EMBEDDING_DIMENSIONS <= 0:
        raise ValueError("FINRAG_EMBEDDING_DIMENSIONS must be positive.")
    if not EMBEDDING_MODEL.strip() or not LLM_MODEL.strip():
        raise ValueError("Embedding and LLM model names cannot be empty.")
    if not SEMANTIC_FALLBACK_MODEL:
        raise ValueError("FINRAG_SEMANTIC_FALLBACK_MODEL cannot be empty.")
    if SEMANTIC_FALLBACK_TIMEOUT_SECONDS <= 0:
        raise ValueError("FINRAG_SEMANTIC_FALLBACK_TIMEOUT_SECONDS must be positive.")
    if not 0 <= MIN_DENSE_SCORE <= 1:
        raise ValueError("FINRAG_MIN_DENSE_SCORE must be between 0 and 1.")
    if not 0 <= MIN_LEXICAL_COVERAGE <= 1:
        raise ValueError("FINRAG_MIN_LEXICAL_COVERAGE must be between 0 and 1.")
    positive_values = (
        ("FINRAG_DENSE_CANDIDATE_K", DENSE_CANDIDATE_K),
        ("FINRAG_LEXICAL_CANDIDATE_K", LEXICAL_CANDIDATE_K),
        ("FINRAG_MAX_CONTEXT_PARENTS", MAX_CONTEXT_PARENTS),
        ("FINRAG_MAX_CONTEXT_CHARS", MAX_CONTEXT_CHARS),
        ("FINRAG_MULTIHOP_MAX_SEARCHES", MULTIHOP_MAX_SEARCHES),
        ("FINRAG_MULTIHOP_MAX_CALCULATIONS", MULTIHOP_MAX_CALCULATIONS),
    )
    for name, value in positive_values:
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    splitters = (
        ("10-K parent", PARENT_TOKENS_10K, PARENT_OVERLAP_10K),
        ("10-K child", CHILD_TOKENS_10K, CHILD_OVERLAP_10K),
        ("transcript parent", TS_PARENT_TOKENS, TS_PARENT_OVERLAP),
        ("transcript child", TS_CHILD_TOKENS, TS_CHILD_OVERLAP),
    )
    for label, size, overlap in splitters:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError(f"Invalid {label} split: size={size}, overlap={overlap}.")


def runtime_signature() -> dict[str, str | int]:
    return {
        "index_layout_version": INDEX_LAYOUT_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "llm_model": LLM_MODEL,
    }

COMPRESSION_ENABLED = os.getenv(
    "FINRAG_COMPRESSION_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "on"}

COMPRESSION_KEEP_BLOCKS = int(
    os.getenv("FINRAG_COMPRESSION_KEEP_BLOCKS", "6")
)
