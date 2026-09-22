from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, IndexPaths, runtime_signature
from src.constants import CHUNK_SCHEMA_VERSION, METADATA_SCHEMA_VERSION
from src.ingestion.document_processing import file_hash, package_version
from src.ingestion.source_inventory import CorpusInventory


class ManifestError(RuntimeError):
    """Raised when an index manifest is missing or incompatible."""


def build_manifest(
    *, index_paths: IndexPaths, inventory: CorpusInventory, counts: dict[str, int]
) -> dict:
    return {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "index_version": index_paths.version,
        "runtime": runtime_signature(),
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "counts": counts,
        "sources": [
            {
                "ticker": source.ticker,
                "fiscal_year": source.fiscal_year,
                "doc_type": source.doc_type,
                "source": source.path.name,
                "source_hash": file_hash(source.path),
            }
            for source in inventory.sources
        ],
        "missing_sources": [list(key) for key in inventory.missing],
        "dependencies": {
            "langchain-core": package_version("langchain-core"),
            "langchain-qdrant": package_version("langchain-qdrant"),
            "llama-cloud": package_version("llama-cloud"),
            "qdrant-client": package_version("qdrant-client"),
        },
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)


def load_manifest(index_paths: IndexPaths) -> dict:
    if not index_paths.manifest_path.is_file():
        raise ManifestError(
            f"Index '{index_paths.version}' has no completed manifest. Run ingestion first."
        )
    manifest = json.loads(index_paths.manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ManifestError(f"Index '{index_paths.version}' is not marked complete.")
    runtime = manifest.get("runtime", {})
    if runtime.get("index_layout_version") != runtime_signature()["index_layout_version"]:
        raise ManifestError("Index layout differs from this application. Build a new version.")
    if runtime.get("embedding_model") != EMBEDDING_MODEL:
        raise ManifestError("Embedding model differs from the selected index. Build a new version.")
    if runtime.get("embedding_dimensions") != EMBEDDING_DIMENSIONS:
        raise ManifestError(
            "Embedding dimensions differ from the selected index. Build a new version."
        )
    if manifest.get("metadata_schema_version") != METADATA_SCHEMA_VERSION:
        raise ManifestError("Metadata schema differs from the selected index. Build a new version.")
    if manifest.get("chunk_schema_version") != CHUNK_SCHEMA_VERSION:
        raise ManifestError("Chunk schema differs from the selected index. Build a new version.")
    return manifest


def available_keys(manifest: dict) -> set[tuple[str, str, str]]:
    return {
        (str(source["ticker"]), str(source["fiscal_year"]), str(source["doc_type"]))
        for source in manifest.get("sources", [])
    }
