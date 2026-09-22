from __future__ import annotations

import tempfile
import unittest
import hashlib
from pathlib import Path

from src.ingestion.parse_cache import ParseCache, ParseCacheError, ParseCacheIdentity


PARSE_TIER = "cost_effective"
PARSE_VERSION = "latest"
PARSE_PROMPT_VERSION = "1.0"
PARSING_INSTRUCTION = "Preserve the complete document faithfully."


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeParser:
    def __init__(self, text: str = "ITEM 1. BUSINESS\nUsable parsed content.") -> None:
        self.text = text
        self.calls = 0

    def load_data(self, path: str) -> list[dict]:
        del path
        self.calls += 1
        return [{"text": self.text, "metadata": {"page_number": 1}}]


def identity_for(path: Path, *, prompt_version: str = PARSE_PROMPT_VERSION) -> ParseCacheIdentity:
    return ParseCacheIdentity.create(
        source_hash=source_hash(path),
        tier=PARSE_TIER,
        parser_version=PARSE_VERSION,
        prompt_version=prompt_version,
        prompt=PARSING_INSTRUCTION,
    )


def load_or_parse(
    cache: ParseCache,
    source: Path,
    parser: FakeParser,
    *,
    refresh: bool = False,
) -> list[dict]:
    identity = identity_for(source)
    if not refresh:
        cached = cache.load(source, identity)
        if cached is not None:
            return cached
    pages = parser.load_data(str(source))
    cache.save(source, identity, pages)
    return pages


class ParseCacheTests(unittest.TestCase):
    def test_second_load_uses_cache_without_calling_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MSFT_2024_10K.pdf"
            source.write_bytes(b"%PDF-1.7\nsource bytes")
            parser = FakeParser()
            cache = ParseCache(root / "cache")

            first = load_or_parse(cache, source, parser)
            second = load_or_parse(cache, source, parser)

            self.assertEqual(parser.calls, 1)
            self.assertEqual(first, second)

    def test_changed_source_hash_creates_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "TSLA_2024_10K.pdf"
            source.write_bytes(b"%PDF-1.7\nfirst revision")
            parser = FakeParser()
            cache = ParseCache(root / "cache")

            load_or_parse(cache, source, parser)
            source.write_bytes(b"%PDF-1.7\nsecond revision")
            load_or_parse(cache, source, parser)

            self.assertEqual(parser.calls, 2)

    def test_prompt_version_changes_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "JPM_2024_10K.pdf"
            source.write_bytes(b"%PDF-1.7\ntest")

            first = identity_for(source, prompt_version="1.0")
            second = identity_for(source, prompt_version="2.0")

            self.assertNotEqual(first.cache_key, second.cache_key)

    def test_corrupted_cache_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "MSFT_2025_10K.pdf"
            source.write_bytes(b"%PDF-1.7\ntest")
            cache = ParseCache(root / "cache")
            identity = identity_for(source)
            cache_path = cache.path_for(source, identity)
            cache_path.parent.mkdir(parents=True)
            cache_path.write_text("{not valid json", encoding="utf-8")

            with self.assertRaises(ParseCacheError):
                cache.load(source, identity)

    def test_refresh_replaces_cache_and_calls_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "TSLA_2025_10K.pdf"
            source.write_bytes(b"%PDF-1.7\ntest")
            cache = ParseCache(root / "cache")
            first_parser = FakeParser("ITEM 1. BUSINESS\nFirst parse.")
            load_or_parse(cache, source, first_parser)

            refreshed_parser = FakeParser("ITEM 1. BUSINESS\nRefreshed parse.")
            pages = load_or_parse(cache, source, refreshed_parser, refresh=True)

            self.assertEqual(refreshed_parser.calls, 1)
            self.assertIn("Refreshed parse", pages[0]["text"])


if __name__ == "__main__":
    unittest.main()
