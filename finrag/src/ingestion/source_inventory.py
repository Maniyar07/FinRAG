from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.constants import DOC_TYPES, SUPPORTED_EXTENSIONS, TICKERS, YEARS
from src.schemas import SourceFile


SOURCE_FILE_RE = re.compile(
    r"^(?P<ticker>JPM|MSFT|TSLA)_(?P<year>2024|2025)_"
    r"(?P<doc_type>10K|TRANSCRIPT)\.(?P<extension>pdf|html|htm)$",
    re.IGNORECASE,
)


class CorpusValidationError(ValueError):
    """Raised when source identity is ambiguous or unsafe."""


@dataclass(frozen=True)
class CorpusInventory:
    sources: tuple[SourceFile, ...]
    missing: tuple[tuple[str, str, str], ...]

    @property
    def available_keys(self) -> set[tuple[str, str, str]]:
        return {source.key for source in self.sources}


def expected_corpus() -> set[tuple[str, str, str]]:
    return {
        (ticker, year, doc_type)
        for ticker in TICKERS
        for year in YEARS
        for doc_type in DOC_TYPES
    }


def identify_source(path: Path) -> SourceFile:
    match = SOURCE_FILE_RE.fullmatch(path.name)
    if not match:
        raise CorpusValidationError(
            f"Invalid source name '{path.name}'. Expected "
            "<TICKER>_<YEAR>_10K.pdf or <TICKER>_<YEAR>_TRANSCRIPT.pdf/html."
        )

    ticker = match.group("ticker").upper()
    year = match.group("year")
    doc_type = match.group("doc_type").upper()
    extension = f".{match.group('extension').lower()}"
    if doc_type == "10K" and extension != ".pdf":
        raise CorpusValidationError(f"10-K sources must be PDF: '{path.name}'.")
    if doc_type == "TRANSCRIPT" and extension not in {".pdf", ".html", ".htm"}:
        raise CorpusValidationError(f"Unsupported transcript format: '{path.name}'.")

    folder_ticker = path.parent.name.upper()
    if folder_ticker in TICKERS and folder_ticker != ticker:
        raise CorpusValidationError(
            f"Folder/file ticker mismatch for '{path}': {folder_ticker} != {ticker}."
        )
    return SourceFile(path=path, ticker=ticker, fiscal_year=year, doc_type=doc_type)


def discover_sources(root: str | Path, *, require_complete: bool = False) -> CorpusInventory:
    data_root = Path(root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: '{data_root}'.")

    candidates = sorted(
        (path for path in data_root.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.relative_to(data_root).as_posix().lower(),
    )
    if not candidates:
        raise CorpusValidationError(f"No supported source documents found in '{data_root}'.")

    sources: list[SourceFile] = []
    errors: list[str] = []
    for path in candidates:
        try:
            sources.append(identify_source(path))
        except CorpusValidationError as error:
            errors.append(str(error))
    if errors:
        raise CorpusValidationError("Corpus validation failed:\n- " + "\n- ".join(errors))

    counts = Counter(source.key for source in sources)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise CorpusValidationError(f"Duplicate logical sources found: {duplicates}.")

    missing = tuple(sorted(expected_corpus() - set(counts)))
    if require_complete and missing:
        raise CorpusValidationError(f"Corpus is incomplete; missing={list(missing)}.")
    return CorpusInventory(tuple(sources), missing)

