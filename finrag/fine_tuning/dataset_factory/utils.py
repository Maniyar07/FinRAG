from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

from .models import MasterExample


WORD_RE = re.compile(r"[a-z0-9]+")
MOJIBAKE_LEADS = {"\u00c2", "\u00c3", "\u00e2", "\u00f0"}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_text(value: str) -> str:
    translations = str.maketrans(
        {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
         "\u2013": "-", "\u2014": "-", "\u00a0": " "}
    )
    return " ".join(value.translate(translations).casefold().split())


def has_mojibake(value: str) -> bool:
    return "\ufffd" in value or any(character in value for character in MOJIBAKE_LEADS)


def _legacy_byte(character: str) -> int | None:
    try:
        encoded = character.encode("cp1252")
    except UnicodeEncodeError:
        codepoint = ord(character)
        return codepoint if codepoint <= 255 else None
    return encoded[0] if len(encoded) == 1 else None


def repair_mojibake(value: str) -> str:
    """Repair UTF-8 byte sequences that were accidentally decoded as Windows-1252."""
    output: list[str] = []
    index = 0
    while index < len(value):
        first = _legacy_byte(value[index])
        expected = (
            2 if first is not None and 0xC2 <= first <= 0xDF
            else 3 if first is not None and 0xE0 <= first <= 0xEF
            else 4 if first is not None and 0xF0 <= first <= 0xF4
            else 0
        )
        if value[index] in MOJIBAKE_LEADS and expected and index + expected <= len(value):
            candidate = value[index:index + expected]
            raw_values = [_legacy_byte(character) for character in candidate]
            if all(item is not None for item in raw_values):
                try:
                    repaired = bytes(item for item in raw_values if item is not None).decode("utf-8")
                except UnicodeDecodeError:
                    repaired = ""
                if repaired:
                    output.append(repaired)
                    index += expected
                    continue
        output.append(value[index])
        index += 1
    return "".join(output)


def question_fingerprint(value: str) -> str:
    return sha256_text(" ".join(WORD_RE.findall(normalized_text(value))))


def token_ngrams(value: str, width: int = 3) -> set[tuple[str, ...]]:
    tokens = WORD_RE.findall(normalized_text(value))
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + width]) for index in range(len(tokens) - width + 1)}


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right) if left | right else 0.0


def stable_id(prefix: str, *parts: str, width: int = 16) -> str:
    digest = sha256_text("\x1f".join(normalized_text(part) for part in parts))[:width]
    return f"{prefix}{digest}"


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def read_jsonl(path: Path) -> list[MasterExample]:
    examples: list[MasterExample] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            examples.append(MasterExample.model_validate_json(line))
        except Exception as error:
            raise ValueError(f"Invalid example at {path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"Dataset is empty: {path}")
    return examples


def _atomic_replace(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8", newline="\n")
    # Antivirus/indexing tools can briefly hold a just-written file on Windows.
    # Keep the operation atomic, but tolerate a short transient lock.
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2**attempt))


def write_json(path: Path, payload: Any) -> None:
    _atomic_replace(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, examples: Iterable[MasterExample]) -> None:
    rows = [example.model_dump_json() for example in examples]
    _atomic_replace(path, "\n".join(rows) + ("\n" if rows else ""))
