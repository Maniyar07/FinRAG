from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


SECTION_RE = re.compile(
    r"(?im)^\s{0,4}(?:#{1,6}\s*)?(?:\*\*)?"
    r"(?P<label>ITEM\s+(?:1[0-6]|1[ABC]?|7A|9[ABC]?|[2-8])|NOTE\s+\d{1,2})"
    r"\.?\s*(?:\*\*)?\s*(?:[-:\u2013\u2014]\s*)?"
    r"(?P<title>[^\n|]{0,160})$"
)
PAGE_MARKER_RE = re.compile(r"\[PAGE:\s*(\d+)\]")
INLINE_SPEAKER_RE = re.compile(
    r"^\s*(?P<speaker>[A-Za-z][A-Za-z0-9 .,'\u2019&()/-]{1,100}?)\s*:\s*(?P<body>.*)$"
)
DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+20\d{2}\b",
    re.IGNORECASE,
)
NON_SPEAKER_LABEL_RE = re.compile(
    r"^(revenue|revenues|net income|operating income|gross profit|earnings per share|eps|"
    r"total assets|total liabilities|cash flow|fiscal year|year ended|quarter ended|"
    r"question|answer|prepared remarks|forward-looking statements?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SectionFragment:
    text: str
    section: str
    item: str | None
    subsection: str | None


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(
        r"<(?:style|script|noscript)\b[^>]*>.*?</(?:style|script|noscript)>",
        "",
        text,
        flags=re.I | re.S,
    )
    text = html.unescape(text).replace("\u00a0", " ").replace("\u200b", "")
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def parser_page_number(metadata: dict, fallback: int) -> int:
    for key in ("page_number", "page", "page_label"):
        match = re.search(r"\d+", str(metadata.get(key, "")))
        if match:
            return int(match.group())
    return fallback


def page_range(text: str, fallback_start: int, fallback_end: int) -> tuple[int, int]:
    pages = [int(value) for value in PAGE_MARKER_RE.findall(text)]
    return (min(pages), max(pages)) if pages else (fallback_start, fallback_end)


def split_sections(text: str, inherited: str) -> tuple[list[SectionFragment], str]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        item = inherited.split(".", 1)[0] if inherited.startswith(("Item ", "Note ")) else None
        return [SectionFragment(text, inherited, item, None)], inherited

    if len(matches) >= 4 and "table of contents" in text.lower():
        return [SectionFragment(text, "Table of Contents", None, None)], inherited

    fragments: list[SectionFragment] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        inherited_item = inherited.split(".", 1)[0] if inherited.startswith(("Item ", "Note ")) else None
        fragments.append(SectionFragment(prefix, inherited, inherited_item, None))

    current = inherited
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw_label = re.sub(r"\s+", " ", match.group("label").upper())
        label = raw_label.replace("ITEM ", "Item ").replace("NOTE ", "Note ")
        title = re.sub(r"\s+", " ", match.group("title")).strip(" *#.-:\u2013\u2014")
        current = f"{label}. {title}" if title else label
        body = text[match.start() : end].strip()
        fragments.append(SectionFragment(body, current, label, title or None))
    return fragments, current


def extract_first_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    return match.group(0) if match else None


def _section_name(text: str) -> str | None:
    compact = re.sub(r"[*#:_\-\u2013\u2014]+", " ", text).strip()
    if re.search(r"\b(question(?:s)?\s+(?:and|&)\s+answer|q\s*&\s*a)\b", compact, re.I):
        return "Question and Answer"
    if re.search(r"\b(management discussion|prepared remarks?)\b", compact, re.I):
        return "Prepared Remarks"
    if re.search(r"\b(?:move|open|go)\s+(?:over\s+|up\s+)?to\s+q\s*&?\s*a\b", compact, re.I):
        return "Question and Answer"
    return None


def _looks_like_role(text: str) -> bool:
    return bool(
        re.search(
            r"\b(chief|officer|president|chairman|analyst|securities|operator|investor relations|"
            r"director|treasurer|controller|executive|founder|partner|managing)\b",
            text,
            re.I,
        )
    )


def _looks_like_name(text: str) -> bool:
    compact = text.strip(" -*#")
    words = compact.split()
    if not 1 <= len(words) <= 8 or len(compact) > 100 or compact.endswith(('.', '?', '!')):
        return False
    if compact.upper() in {"QUESTION AND ANSWER SECTION", "MANAGEMENT DISCUSSION SECTION"}:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z .,'\u2019&()-]+", compact))


def _looks_like_inline_speaker(text: str) -> bool:
    compact = text.strip(" -*#[]")
    if NON_SPEAKER_LABEL_RE.fullmatch(compact):
        return False
    if compact.lower() in {"operator", "coordinator", "unidentified analyst"}:
        return True
    return len(compact.split()) >= 2 and _looks_like_name(compact)


def format_transcript(blocks: list[str]) -> tuple[str, set[str], set[str]]:
    """Normalize inline and multi-line speaker turns while preserving source order."""
    cleaned = [clean_text(block) for block in blocks]
    cleaned = [block for block in cleaned if block]
    output = ["[SECTION: Prepared Remarks]"]
    speakers: set[str] = set()
    sections = {"Prepared Remarks"}
    current_section = "Prepared Remarks"
    index = 0

    while index < len(cleaned):
        block = cleaned[index]
        if block.strip().upper() in {"Q", "A", "Q.", "A."}:
            index += 1
            continue
        section = _section_name(block)
        if section and len(block.split()) <= 18:
            if section != current_section:
                output.append(f"[SECTION: {section}]")
                sections.add(section)
                current_section = section
            index += 1
            continue

        inline = INLINE_SPEAKER_RE.match(block)
        if inline and _looks_like_inline_speaker(inline.group("speaker")):
            speaker = inline.group("speaker").strip()
            body = inline.group("body").strip()
            speakers.add(speaker)
            output.append(f"[SPEAKER: {speaker}]" + (f"\n{body}" if body else ""))
            index += 1
            continue

        next_block = cleaned[index + 1] if index + 1 < len(cleaned) else ""
        if _looks_like_name(block) and _looks_like_role(next_block):
            speaker = block.strip(" -*#")
            role = next_block.strip()
            speakers.add(speaker)
            output.append(f"[SPEAKER: {speaker} | ROLE: {role}]")
            index += 2
            continue

        output.append(block)
        index += 1

    return "\n\n".join(output), speakers, sections
