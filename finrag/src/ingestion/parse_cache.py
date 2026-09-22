from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


PARSE_CACHE_SCHEMA_VERSION = "1.0"
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ParseCacheError(RuntimeError):
    """Raised when an existing cache entry is unreadable or incompatible."""


@dataclass(frozen=True)
class ParseCacheIdentity:
    source_hash: str
    tier: str
    parser_version: str
    prompt_version: str
    prompt_hash: str

    @classmethod
    def create(
        cls,
        *,
        source_hash: str,
        tier: str,
        parser_version: str,
        prompt_version: str,
        prompt: str,
    ) -> "ParseCacheIdentity":
        return cls(
            source_hash=source_hash,
            tier=tier,
            parser_version=parser_version,
            prompt_version=prompt_version,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )

    @property
    def cache_key(self) -> str:
        serialized = json.dumps(
            {
                "schema_version": PARSE_CACHE_SCHEMA_VERSION,
                "source_hash": self.source_hash,
                "tier": self.tier,
                "parser_version": self.parser_version,
                "prompt_version": self.prompt_version,
                "prompt_hash": self.prompt_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, str]:
        return {
            "source_hash": self.source_hash,
            "tier": self.tier,
            "parser_version": self.parser_version,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "cache_key": self.cache_key,
        }


class ParseCache:
    """Persistent, content-addressed cache for page-level LlamaParse Markdown."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, source_path: str | Path, identity: ParseCacheIdentity) -> Path:
        source_name = SAFE_NAME_RE.sub("_", Path(source_path).stem).strip("._-")
        source_name = source_name or "document"
        return self.root / f"{source_name}_{identity.cache_key[:24]}.json"

    def load(
        self,
        source_path: str | Path,
        identity: ParseCacheIdentity,
    ) -> list[dict] | None:
        cache_path = self.path_for(source_path, identity)
        if not cache_path.is_file():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ParseCacheError(
                f"Parse cache entry is unreadable: '{cache_path}'. "
                "Run ingestion once with --refresh-parse-cache to replace it."
            ) from error

        self._validate_payload(payload, identity, cache_path)
        return [
            {
                "text": str(page["text"]),
                "metadata": dict(page.get("metadata") or {}),
            }
            for page in payload["pages"]
        ]

    def save(
        self,
        source_path: str | Path,
        identity: ParseCacheIdentity,
        pages: Iterable[Mapping[str, object]],
    ) -> Path:
        normalized_pages = [
            {
                "text": str(page.get("text") or ""),
                "metadata": dict(page.get("metadata") or {}),
            }
            for page in pages
        ]
        if not normalized_pages or not any(page["text"].strip() for page in normalized_pages):
            raise ParseCacheError("Refusing to cache an empty parsing result.")

        cache_path = self.path_for(source_path, identity)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": PARSE_CACHE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": Path(source_path).name,
            "identity": identity.as_dict(),
            "page_count": len(normalized_pages),
            "pages": normalized_pages,
        }

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=f".{cache_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_name = destination.name
                json.dump(payload, destination, ensure_ascii=False, indent=2)
                destination.write("\n")
                destination.flush()
                os.fsync(destination.fileno())
            Path(temporary_name).replace(cache_path)
        except Exception:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            raise
        return cache_path

    @staticmethod
    def _validate_payload(
        payload: object,
        identity: ParseCacheIdentity,
        cache_path: Path,
    ) -> None:
        if not isinstance(payload, dict):
            raise ParseCacheError(f"Invalid parse cache object in '{cache_path}'.")
        if payload.get("schema_version") != PARSE_CACHE_SCHEMA_VERSION:
            raise ParseCacheError(
                f"Unsupported parse cache schema in '{cache_path}'. "
                "Run ingestion once with --refresh-parse-cache to replace it."
            )
        cached_identity = payload.get("identity")
        if not isinstance(cached_identity, dict) or cached_identity != identity.as_dict():
            raise ParseCacheError(f"Parse cache identity mismatch in '{cache_path}'.")
        pages = payload.get("pages")
        if not isinstance(pages, list) or not pages:
            raise ParseCacheError(f"Parse cache contains no pages: '{cache_path}'.")
        for index, page in enumerate(pages, start=1):
            if not isinstance(page, dict) or not isinstance(page.get("text"), str):
                raise ParseCacheError(
                    f"Invalid cached page {index} in '{cache_path}'."
                )
            metadata = page.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ParseCacheError(
                    f"Invalid metadata for cached page {index} in '{cache_path}'."
                )
        if not any(str(page["text"]).strip() for page in pages):
            raise ParseCacheError(f"All cached pages are empty in '{cache_path}'.")
