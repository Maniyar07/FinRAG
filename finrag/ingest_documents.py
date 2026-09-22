from __future__ import annotations

import argparse
from pathlib import Path

from src.config import SOURCE_DATA_DIR, get_index_paths
from src.ingestion.chunker import FinancialChunker
from src.ingestion.document_loader import FinancialDocumentLoader
from src.ingestion.manifest import build_manifest, write_manifest
from src.ingestion.source_inventory import CorpusValidationError, discover_sources
from src.retrieval.vector_store import FinancialVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one isolated FinRAG index version.")
    parser.add_argument("--index-version", required=True, help="New immutable index version.")
    parser.add_argument("--data-dir", type=Path, default=SOURCE_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--require-complete-corpus",
        action="store_true",
        help="Fail unless all 12 expected company/year/type combinations exist.",
    )
    parser.add_argument(
        "--require-complete-year",
        action="append",
        choices=("2024", "2025"),
        default=[],
        help="Fail unless all three companies have a 10-K and transcript for this year.",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Validate and report source availability without parsing or embedding.",
    )
    return parser.parse_args()


def print_inventory(inventory) -> None:
    print(f"Validated sources: {len(inventory.sources)}")
    for source in inventory.sources:
        print(f"  {source.ticker} {source.fiscal_year} {source.doc_type}: {source.path.name}")
    if inventory.missing:
        print(f"Missing expected sources: {len(inventory.missing)}")
        for ticker, year, doc_type in inventory.missing:
            print(f"  {ticker} {year} {doc_type}")


def main() -> int:
    args = parse_args()
    inventory = discover_sources(
        args.data_dir, require_complete=args.require_complete_corpus
    )
    print_inventory(inventory)
    missing_required_years = [
        key for key in inventory.missing if key[1] in set(args.require_complete_year)
    ]
    if missing_required_years:
        raise CorpusValidationError(
            "Required test-year corpus is incomplete; missing="
            f"{missing_required_years}."
        )
    loader = FinancialDocumentLoader()
    loader.preflight_sources(inventory)
    print("Local source preflight: PASS")
    if args.inventory_only:
        return 0

    paths = get_index_paths(args.index_version)
    if paths.root.exists() and any(paths.root.iterdir()):
        raise RuntimeError(
            f"Index version '{paths.version}' already contains data. Choose a new version."
        )

    documents = loader.load_documents(
        args.data_dir,
        require_complete=args.require_complete_corpus,
        preflight=False,
    )
    manager: FinancialVectorStore | None = None
    try:
        manager = FinancialVectorStore(
            index_version=paths.version, create_if_missing=True
        )
        chunker = FinancialChunker(
            manager.get_vectorstore(),
            parent_store_path=paths.parent_docstore_dir,
            lexical_index_path=paths.lexical_index_path,
        )
        counts = chunker.process_and_add_documents(documents, batch_size=args.batch_size)
        point_count = manager.point_count()
        if point_count != counts["child_chunks"]:
            raise RuntimeError(
                f"Qdrant verification failed: expected {counts['child_chunks']} points, "
                f"found {point_count}."
            )
        if loader.inventory is None:
            raise RuntimeError("Source inventory was not retained by the loader.")
        manifest = build_manifest(
            index_paths=paths, inventory=loader.inventory, counts=counts
        )
        write_manifest(paths.manifest_path, manifest)
    finally:
        if manager is not None:
            manager.close()

    print(f"Index '{paths.version}' is complete.")
    print(
        f"Segments={counts['source_segments']} Parents={counts['parent_chunks']} "
        f"Children={counts['child_chunks']}"
    )
    print(f"Set FINRAG_INDEX_VERSION={paths.version} before starting the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
