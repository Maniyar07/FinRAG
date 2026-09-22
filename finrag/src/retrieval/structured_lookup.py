"""Read complete source structures that relevance search cannot enumerate safely."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from src.schemas import Scope


ITEM_LIST_RE = re.compile(
    r"\b(?:all|every|complete|full)\b.{0,60}\b(?:items?|sections?)\b|"
    r"\b(?:list|enlist)\b.{0,60}\b(?:items?|sections?)\b",
    re.IGNORECASE,
)
ITEM_RE = re.compile(r"^Item\s+(\d+)([A-C]?)$", re.IGNORECASE)
HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"(?m)^\s*#{1,5}\s+([^\n]+)$")
PAGE_RE = re.compile(r"\[PAGE:\s*(\d+)\]", re.IGNORECASE)
TABLE_STOP_WORDS = frozenset(
    "a an and all as at by complete data document exact filing financial for from "
    "full give in inside into of on presented provide report requested show table the "
    "this to whole with year".split()
)


@dataclass(frozen=True)
class DirectAnswer:
    answer: str
    sources: list[dict]
    mode: str


@dataclass(frozen=True)
class VerifiedTableRow:
    label: str
    year: str
    value: str
    scale: str
    source: dict


class StructuredDocumentLookup:
    """Use stored parent records for exact tables and complete 10-K item lists."""

    def __init__(self, parent_dir: Path):
        self.parent_dir = Path(parent_dir)

    @lru_cache(maxsize=12)
    def _documents(self, ticker: str, year: str) -> tuple[tuple[str, str, dict], ...]:
        records = []
        for path in self.parent_dir.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            metadata = record["metadata"]
            if (
                metadata.get("ticker") == ticker
                and str(metadata.get("fiscal_year")) == year
                and str(metadata.get("doc_type", "")).upper().replace("-", "") == "10K"
            ):
                records.append((path.stem, record["page_content"], metadata))
        return tuple(records)

    @staticmethod
    def _group(scope: Scope) -> tuple[str, str] | None:
        if len(scope.groups) != 1:
            return None
        if scope.doc_type != "10K" and scope.required_doc_types != ("10K",):
            return None
        return scope.groups[0]

    @staticmethod
    def _source(
        source_id: str, parent_id: str, metadata: dict, evidence: str, page: str | None = None
    ) -> dict:
        return {
            "id": source_id,
            "parent_id": parent_id,
            "ticker": metadata.get("ticker"),
            "fiscal_year": metadata.get("fiscal_year"),
            "doc_type": metadata.get("doc_type"),
            "source": metadata.get("source"),
            "section": metadata.get("section"),
            "pdf_page": page or metadata.get("pdf_page_start") or "not available",
            "evidence_text": evidence,
        }

    @staticmethod
    def _markdown_table(html: str) -> str | None:
        table = BeautifulSoup(html, "html.parser").find("table")
        if table is None:
            return None
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"], recursive=False)
            if not cells:
                continue
            row: list[str] = []
            for cell in cells:
                value = " ".join(cell.stripped_strings).replace("|", r"\|")
                row.append(value)
                span = cell.get("colspan", "1")
                try:
                    width = max(1, int(span))
                except (TypeError, ValueError):
                    width = 1
                row.extend("" for _ in range(width - 1))
            # Some extracted PDFs split the last value onto a blank next row.
            if rows and row and all(not value for value in row[:-1]) and row[-1]:
                previous = rows[-1]
                if len(previous) == len(row) and previous[0] and not previous[-1]:
                    previous[-1] = row[-1]
                    continue
            rows.append(row)
        if len(rows) < 2:
            return None
        width = max(len(row) for row in rows)
        if width < 2:
            return None
        lines = []
        for index, row in enumerate(rows):
            lines.append("| " + " | ".join(row + [""] * (width - len(row))) + " |")
            if index == 0:
                lines.append("| " + " | ".join(["---"] * width) + " |")
        return "\n".join(lines)

    @staticmethod
    def _heading_title(raw: str) -> str:
        title = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
        title = re.sub(r"[*_`]+", "", title).strip()
        return re.sub(r"^(?:PART\s+[IVX]+\s+)?Item\s+\d+[A-C]?\s*", "", title, flags=re.I).strip()

    @staticmethod
    def _terms(value: str) -> set[str]:
        def singular(word: str) -> str:
            if len(word) > 5 and word.endswith("ies"):
                return word[:-3] + "y"
            if len(word) > 4 and word.endswith(("sses", "xes", "ches", "shes", "zes")):
                return word[:-2]
            if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
                return word[:-1]
            return word

        words = re.findall(r"[A-Za-z]{3,}", value.casefold())
        terms = {
            singular(word)
            for word in words
            if word not in TABLE_STOP_WORDS
        }
        # Filings use "statements of operations" and "income statements" for
        # the same primary statement.
        if "statement" in terms and "operation" in terms:
            terms.remove("operation")
            terms.add("income")
        return terms

    @staticmethod
    def _unit_heading(value: str) -> bool:
        return bool(re.match(r"^\(?in\s+(?:millions?|thousands?|billions?)\b", value, re.I))

    def _table(self, question: str, scope: Scope) -> DirectAnswer | None:
        group = self._group(scope)
        if group is None:
            return None
        requested_terms = self._terms(question)
        if not requested_terms:
            return None
        candidates = []
        for parent_id, body, metadata in self._documents(*group):
            headings = list(HEADING_RE.finditer(body))
            for index, heading in enumerate(headings):
                title = self._heading_title(heading.group(1))
                heading_terms = self._terms(title)
                matched = requested_terms & heading_terms
                if not matched or len(matched) / len(heading_terms) < 0.5:
                    continue
                table_match = HTML_TABLE_RE.search(body, heading.end())
                if table_match is None or table_match.start() - heading.end() > 500:
                    continue
                intervening = headings[index + 1 :]
                if any(
                    other.start() < table_match.start()
                    and not self._unit_heading(self._heading_title(other.group(1)))
                    for other in intervening
                ):
                    continue
                markdown = self._markdown_table(table_match.group())
                if markdown is None:
                    continue
                pages = PAGE_RE.findall(body[: heading.start()])
                page = pages[-1] if pages else None
                between = body[heading.end() : table_match.start()]
                unit = next(
                    (line.strip(" *") for line in between.splitlines() if "million" in line.casefold() or "thousand" in line.casefold()),
                    "",
                )
                source = self._source("S1", parent_id, metadata, table_match.group(), page)
                answer = f"## {title}\n\n{unit}\n\n{markdown}\n\n[S1]" if unit else f"## {title}\n\n{markdown}\n\n[S1]"
                score = (len(matched) / len(heading_terms), len(matched), len(markdown))
                candidates.append((score, DirectAnswer(answer, [source], "exact_table")))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _items(self, scope: Scope) -> DirectAnswer | None:
        group = self._group(scope)
        if group is None:
            return None
        chosen: dict[str, tuple[int, str, str, dict]] = {}
        for parent_id, _, metadata in self._documents(*group):
            item = str(metadata.get("item") or "").strip()
            if not ITEM_RE.fullmatch(item):
                continue
            section = re.sub(r"<[^>]+>", "", str(metadata.get("section") or "")).strip()
            if not re.match(rf"^{re.escape(item)}[.\s]", section, re.IGNORECASE):
                continue
            title = re.sub(r"\s+\d+$", "", section)
            score = int(title == title.upper())
            if item not in chosen or score > chosen[item][0]:
                chosen[item] = (score, title, parent_id, metadata)
        if not chosen:
            return None
        ordered = sorted(
            chosen,
            key=lambda item: (int(ITEM_RE.fullmatch(item).group(1)), ITEM_RE.fullmatch(item).group(2)),
        )
        sources = []
        lines = [f"Items in the {group[0]} {group[1]} 10-K:", ""]
        for item in ordered:
            _, title, parent_id, metadata = chosen[item]
            source_id = f"S{len(sources) + 1}"
            lines.append(f"- {title} [{source_id}]")
            sources.append(self._source(source_id, parent_id, metadata, title))
        return DirectAnswer("\n".join(lines), sources, "document_items")

    def statement_rows(self, question: str, scope: Scope) -> tuple[VerifiedTableRow, ...]:
        """Find requested row/year cells in indexed 10-K statement tables."""
        group = self._group(scope)
        if group is None:
            return ()
        requested = self._terms(question) - self._terms(group[0])
        chosen: dict[frozenset[str], VerifiedTableRow] = {}
        ambiguous: set[frozenset[str]] = set()
        for parent_id, body, metadata in self._documents(*group):
            if str(metadata.get("item", "")).casefold() != "item 8":
                continue
            for table_match in HTML_TABLE_RE.finditer(body):
                heading_matches = list(HEADING_RE.finditer(body[: table_match.start()]))
                heading = next(
                    (match for match in reversed(heading_matches)
                     if not self._unit_heading(self._heading_title(match.group(1)))),
                    None,
                )
                if heading is None or table_match.start() - heading.end() > 500:
                    continue
                table = BeautifulSoup(table_match.group(), "html.parser").find("table")
                if table is None:
                    continue
                rows = table.find_all("tr")
                year_columns = {
                    index for tr in rows for index, cell in
                    enumerate(tr.find_all("th", recursive=False))
                    if cell.get_text(" ", strip=True) == group[1]
                }
                if len(year_columns) != 1:
                    continue
                column = next(iter(year_columns))
                scale_match = re.search(
                    r"\b(?:millions?|billions?|thousands?)\b",
                    body[heading.end(): table_match.start()], re.IGNORECASE,
                )
                if scale_match is None:
                    continue
                scale = scale_match.group().lower()
                evidence = body[heading.start(): table_match.end()]
                for tr in rows:
                    cells = tr.find_all(["td", "th"], recursive=False)
                    if len(cells) <= column:
                        continue
                    label = " ".join(cells[0].stripped_strings)
                    terms = self._terms(label) - {"total"}
                    if not terms or not terms.issubset(requested):
                        continue
                    value = " ".join(cells[column].stripped_strings)
                    if not re.fullmatch(r"\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?", value):
                        continue
                    key = frozenset(terms)
                    source = self._source("S1", parent_id, metadata, evidence)
                    source["full_evidence_text"] = evidence
                    source["source_hash"] = metadata.get("source_hash")
                    row = VerifiedTableRow(label, group[1], value, scale, source)
                    previous = chosen.get(key)
                    if previous and previous.value.replace("$", "") != value.replace("$", ""):
                        ambiguous.add(key)
                    elif previous is None:
                        chosen[key] = row
        return tuple(row for key, row in chosen.items() if key not in ambiguous)

    def answer(
        self, question: str, scope: Scope, *, wants_complete_table: bool
    ) -> DirectAnswer | None:
        if wants_complete_table:
            return self._table(question, scope)
        if ITEM_LIST_RE.search(question):
            return self._items(scope)
        return None
