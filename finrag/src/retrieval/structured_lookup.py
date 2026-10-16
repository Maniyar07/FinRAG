"""Read complete source structures that relevance search cannot enumerate safely."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from src.constants import COMPANY_NAMES
from src.schemas import Scope
from src.observability import trace_component


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
    "full give in inside into of on presented primary provide report requested show table the "
    "this to whole with year exactly please me related was follows following "
    "could would can you it also".split()
)
PERIOD_ROW_TERMS = frozenset(
    "ended ending fiscal quarter quarters month months day days january february "
    "march april may june july august september october november december".split()
)
MONETARY_STATEMENT_RE = re.compile(
    r"(?:income statements?|statements? of (?:operations|cash flows?|comprehensive income)|cash flows? statements?|balance sheets?)",
    re.IGNORECASE,
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
    value: str | None
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
            "evidence_id": parent_id + ":" + hashlib.blake2b(evidence.encode(), digest_size=8).hexdigest(),
            "ticker": metadata.get("ticker"),
            "fiscal_year": metadata.get("fiscal_year"),
            "doc_type": metadata.get("doc_type"),
            "source": metadata.get("source"),
            "section": metadata.get("section"),
            "pdf_page": page or metadata.get("pdf_page_start") or "not available",
            "evidence_text": evidence,
        }

    @staticmethod
    def markdown_table(html: str) -> str | None:
        table = BeautifulSoup(html, "html.parser").find("table")
        if table is None:
            return None
        rows: list[list[str]] = []
        for values in StructuredDocumentLookup.table_grid(table):
            if not values:
                continue
            row = [value.replace("|", r"\|") for value in values]
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
        if table.find("th") is None:
            rows.insert(0, [f"Column {index + 1}" for index in range(width)])
        lines = []
        for index, row in enumerate(rows):
            lines.append("| " + " | ".join(row + [""] * (width - len(row))) + " |")
            if index == 0:
                lines.append("| " + " | ".join(["---"] * width) + " |")
        return "\n".join(lines)

    @staticmethod
    def _heading_title(raw: str) -> str:
        title = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
        title = re.sub(r"^(?:\s*[-*_]){3,}\s*$", "", title)
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
        terms = {"operation" if term == "operating" else term for term in terms}
        if {"income", "operation"}.issubset(terms):
            terms.discard("loss")
        # Filings use "statements of operations" and "income statements" for
        # the same primary statement.
        if "statement" in terms and "operation" in terms:
            terms.remove("operation")
            terms.add("income")
        return terms

    @staticmethod
    def reported_value_columns(table, year: str, column_label: str = "") -> tuple[int, ...]:
        """Intersect a reported-year span with a supplied subordinate column label."""
        _, years = StructuredDocumentLookup.reported_year_columns(table)
        columns = years.get(year, ())
        terms = StructuredDocumentLookup._terms(column_label) - PERIOD_ROW_TERMS
        if len(columns) <= 1 or not terms:
            return columns
        matched = {column for row, values in zip(table.find_all("tr"), StructuredDocumentLookup.table_grid(table))
                   if row.find("th", recursive=False) for column in columns
                   if column < len(values)
                   for header_terms in (StructuredDocumentLookup._terms(values[column]) - PERIOD_ROW_TERMS,)
                   if header_terms and (terms.issubset(header_terms) or header_terms.issubset(terms))}
        return tuple(column for column in columns if column in matched) if matched else columns

    @staticmethod
    def table_context(body: str, table_start: int) -> str:
        """Return the adjacent table caption and units, excluding preceding tables."""
        prefix = re.split(r"</table\s*>", body[:table_start], flags=re.I)[-1]
        paragraphs = [BeautifulSoup(part, "html.parser").get_text(" ", strip=True)
                      for part in re.split(r"\n\s*\n", prefix)]
        return " ".join([part for part in paragraphs if part.strip()][-2:])[-1500:]

    @staticmethod
    def _unit_heading(value: str) -> bool:
        return bool(re.match(r"^\(?in\s+(?:millions?|thousands?|billions?)\b", value, re.I))

    @staticmethod
    def table_grid(table) -> list[list[str]]:
        """Expand HTML row/column spans so header and value columns stay aligned."""
        grid: list[list[str]] = []
        active: dict[int, tuple[int, str]] = {}
        for tr in table.find_all("tr"):
            row: dict[int, str] = {
                column: text for column, (_, text) in active.items()
            }
            additions: dict[int, tuple[int, str]] = {}
            column = 0
            for cell in tr.find_all(["th", "td"], recursive=False):
                while column in row:
                    column += 1
                text = " ".join(cell.stripped_strings)
                try:
                    colspan = max(1, int(cell.get("colspan", "1")))
                    rowspan = max(1, int(cell.get("rowspan", "1")))
                except (TypeError, ValueError):
                    colspan = rowspan = 1
                for offset in range(colspan):
                    target = column + offset
                    row[target] = text
                    if rowspan > 1:
                        additions[target] = (rowspan - 1, text)
                column += colspan
            width = max(row, default=-1) + 1
            grid.append([row.get(index, "") for index in range(width)])
            active = {
                column: (remaining - 1, text)
                for column, (remaining, text) in active.items()
                if remaining > 1
            }
            active.update(additions)
        return grid

    @staticmethod
    def reported_year_columns(table) -> tuple[int, dict[str, tuple[int, ...]]]:
        """Resolve header spans, including a separate currency-symbol column."""
        headers = []
        for row, values in zip(table.find_all("tr"), StructuredDocumentLookup.table_grid(table)):
            if not row.find("th", recursive=False):
                continue
            columns = {}
            for index, text in enumerate(values):
                years = set(re.findall(r"(?:19|20)\d{2}", text))
                if len(years) == 1:
                    columns.setdefault(next(iter(years)), []).append(index)
            if columns:
                headers.append((len(columns), len(values), columns))
        if not headers:
            return 0, {}
        _, width, columns = max(headers, key=lambda item: item[:2])
        return width, {year: tuple(indices) for year, indices in columns.items()}

    def _table(self, question: str, scope: Scope) -> DirectAnswer | None:
        group = self._group(scope)
        if group is None:
            return None
        scope_terms = self._terms(f"{group[0]} {COMPANY_NAMES.get(group[0], '')}")
        requested_terms = self._terms(question) - scope_terms
        if not requested_terms:
            return None
        candidates = {}
        for parent_id, body, metadata in self._documents(*group):
            headings = list(HEADING_RE.finditer(body))
            previous_end = 0
            for table_match in HTML_TABLE_RE.finditer(body):
                heading = next(
                    (value for value in reversed(headings)
                     if previous_end <= value.start() < table_match.start()
                     and not self._unit_heading(self._heading_title(value.group(1)))),
                    None,
                )
                start = heading.end() if heading else previous_end
                between = body[start:table_match.start()]
                previous_end = table_match.end()
                paragraphs = [
                    self._heading_title(re.sub(r"(?m)^\s*#{1,5}\s+", "", value)) for value in
                    re.split(r"\n\s*\n", PAGE_RE.sub("", between).strip())
                    if value.strip()
                ]
                captions = [value for value in paragraphs if value
                            and not self._unit_heading(value)]
                title = captions[-1] if captions else (
                    self._heading_title(heading.group(1)) if heading else ""
                )
                title_terms = self._terms(title)
                table = BeautifulSoup(table_match.group(), "html.parser").find("table")
                header_terms = self._terms(" ".join(cell.get_text(" ", strip=True) for cell in table.find_all("th")))
                matched = requested_terms & title_terms
                if not matched or not requested_terms <= title_terms | header_terms:
                    continue
                markdown = self.markdown_table(table_match.group())
                if markdown is None:
                    continue
                pages = PAGE_RE.findall(body[: table_match.start()])
                page = pages[-1] if pages else None
                unit = next(
                    (line.strip(" *") for line in between.splitlines() if re.search(r"\b(?:million|thousand|billion)s?\b", line, re.I)),
                    "",
                )
                source = self._source("S1", parent_id, metadata,
                                      f"{title}\n{unit}\n{table_match.group()}", page)
                grid = self.table_grid(table)
                header_text = " ".join(
                    cell.get_text(" ", strip=True) for row in table.find_all("tr")
                    for cell in row.find_all("th")
                )
                periods = sorted(set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", header_text)))
                if not periods:
                    periods = sorted(set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", " ".join(" ".join(row) for row in grid[:2]))))
                source["table_completeness"] = {
                    "complete": all(" ".join(cell.stripped_strings) in markdown.replace(r"\|", "|")
                                    for cell in table.find_all(["td", "th"])),
                    "scope": "selected_indexed_table", "source_rows": len(grid),
                    "rows": len(markdown.splitlines()) - 1 - int(table.find("th") is None),
                    "returned_rows": len(markdown.splitlines()) - 1 - int(table.find("th") is None),
                    "generated_header": table.find("th") is None,
                    "period_columns": periods, "truncated": False,
                    "reported_periods": sorted(set(periods) | set(re.findall(
                        r"\b(?:year|quarter|period)\s+ended[^.;]*?\b((?:19|20)\d{2})\b", title + " " + unit, re.I))),
                }
                answer = f"## {title}\n\n{unit}\n\n{markdown}\n\n[S1]" if unit else f"## {title}\n\n{markdown}\n\n[S1]"
                score = (len(matched) / len(requested_terms), len(matched) / len(title_terms))
                candidate = (score, DirectAnswer(answer, [source], "exact_table"))
                if markdown not in candidates or score > candidates[markdown][0]:
                    candidates[markdown] = candidate
        ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
            return None
        return ranked[0][1]

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

    def statement_rows(
        self,
        question: str,
        scope: Scope,
        *,
        value_years: tuple[str, ...] = (),
        qualifiers: tuple[str, ...] = (),
        include_ambiguous: bool = False,
        metric: str | None = None,
    ) -> tuple[VerifiedTableRow, ...]:
        """Find requested row/year cells in indexed 10-K statement tables."""
        # Financial validation also imports this reader; defer its shared matcher
        # until the reader is initialized, avoiding a package import cycle.
        from src.financial.metric_terms import (
            has_financial_metric_conflict,
            is_metric_match,
            normalize_financial_term,
        )

        group = self._group(scope)
        if group is None and len(scope.groups) > 1 and scope.doc_type == "10K":
            return tuple(
                row
                for ticker, year in scope.groups
                for row in self.statement_rows(
                    question,
                    Scope(
                        tickers=(ticker,),
                        years=(year,),
                        doc_type="10K",
                        requested_groups=((ticker, year),),
                    ),
                    value_years=value_years,
                    qualifiers=qualifiers,
                    include_ambiguous=include_ambiguous,
                    metric=metric,
                )
            )
        if group is None:
            return ()
        requested_years = tuple(
            dict.fromkeys(
                year
                for year in value_years
                if re.fullmatch(r"(?:19|20)\d{2}", str(year))
            )
        ) or (group[1],)
        requested = (normalize_financial_term(" ".join((metric, *qualifiers))) if metric
                     else self._terms(question) - self._terms(group[0])) - {"total", "combined", "amount"}
        chosen: dict[tuple[str, frozenset[str]], VerifiedTableRow] = {}
        ambiguous: set[tuple[str, frozenset[str]]] = set()
        parent_matches = []
        for parent_id, body, metadata in self._documents(*group):
            for table_match in HTML_TABLE_RE.finditer(body):
                heading_matches = list(HEADING_RE.finditer(body[: table_match.start()]))
                heading = next(
                    (match for match in reversed(heading_matches)
                     if not self._unit_heading(self._heading_title(match.group(1)))),
                    None,
                )
                table = BeautifulSoup(table_match.group(), "html.parser").find("table")
                if table is None:
                    continue
                rows = table.find_all("tr")
                grid = self.table_grid(table)
                _, year_columns = self.reported_year_columns(table)
                resolved_columns = {year: columns for year, columns in year_columns.items()
                                    if year in requested_years}
                if not resolved_columns:
                    continue
                caption = self.table_context(body, table_match.start())
                scale_match = re.search(
                    r"\b(?:millions?|billions?|thousands?)\b",
                    caption + " " + table.get_text(" ", strip=True),
                    re.IGNORECASE,
                )
                if scale_match is None:
                    continue
                scale = scale_match.group().lower()
                evidence = body[heading.start() if heading else 0: table_match.end()]
                group_terms: set[str] = set()
                group_label = ""
                for tr, values in zip(rows, grid):
                    cells = tr.find_all(["td", "th"], recursive=False)
                    if not cells:
                        continue
                    # Column headers can contain year-shaped numeric cells and a
                    # unit label that overlaps words in the question. They define
                    # table structure; they are never reported financial rows.
                    if tr.find("td", recursive=False) is None:
                        continue
                    label = " ".join(cells[0].stripped_strings)
                    nonempty = [
                        " ".join(cell.stripped_strings)
                        for cell in cells
                        if " ".join(cell.stripped_strings)
                    ]
                    if (
                        label
                        and len(nonempty) == 1
                        and (
                            int(cells[0].get("colspan", "1") or "1") > 1
                            or all(
                                len(values) <= column or not values[column]
                                for columns in resolved_columns.values() for column in columns
                            )
                        )
                    ):
                        group_label = label
                        group_terms = normalize_financial_term(label) if metric else self._terms(label)
                        continue
                    terms = (normalize_financial_term(label) if metric else self._terms(label)) - {"total"}
                    if (
                        group_terms & requested
                        and "cost" in group_terms
                        and "cost" not in requested
                    ):
                        continue
                    contextual_terms = group_terms & requested
                    semantic_terms = {
                        term
                        for term in terms | contextual_terms
                        if term not in PERIOD_ROW_TERMS and not term.isdigit()
                    }
                    # Parser-produced period rows can look numeric (for example,
                    # "Year Ended June 30 | 2025") but are column structure, not
                    # financial facts. Only rows with a semantic label are eligible.
                    if not semantic_terms or not (all(any(is_metric_match(term, target) for target in requested)
                            for term in semantic_terms) if metric else semantic_terms.issubset(requested)):
                        continue
                    local_label = " ".join((label, group_label,
                        self._heading_title(heading.group(1)) if heading else ""))
                    if (
                        metric
                        and has_financial_metric_conflict(local_label, metric)
                        and not is_metric_match(label, metric)
                    ):
                        continue
                    local_terms = normalize_financial_term(local_label) if metric else self._terms(local_label)
                    if any(not self._terms(qualifier).issubset(local_terms) for qualifier in qualifiers):
                        continue
                    subject_terms = normalize_financial_term(metric) - {"total", "combined", "amount"} if metric else set()
                    if subject_terms and not all(any(is_metric_match(term, actual)
                            for actual in local_terms | self._terms(caption)) for term in subject_terms):
                        continue
                    if not metric and qualifiers and (requested - set().union(*(self._terms(q) for q in qualifiers))).isdisjoint(
                            local_terms | self._terms(caption)):
                        continue
                    display_label = (
                        f"{group_label} {label}"
                        if contextual_terms - terms and group_label
                        else label
                    )
                    for year, columns in resolved_columns.items():
                        numeric_cells = [values[column] for column in columns if len(values) > column
                                         and re.fullmatch(r"\$?\s*\(?-?\d[\d,]*(?:\.\d+)?\)?", values[column])]
                        if not numeric_cells or len(numeric_cells) != 1 and not include_ambiguous:
                            continue
                        value = numeric_cells[0] if len(numeric_cells) == 1 else None
                        key = (year, frozenset(semantic_terms))
                        source = self._source("S1", parent_id, metadata, evidence)
                        source["table_context"] = caption
                        source["full_evidence_text"] = evidence
                        source["source_hash"] = metadata.get("source_hash")
                        row = VerifiedTableRow(
                            display_label, year, value, scale, source
                        )
                        if include_ambiguous:
                            parent_matches.append((year, frozenset(semantic_terms), row))
                            continue
                        previous = chosen.get(key)
                        if (
                            previous
                            and previous.value.replace("$", "")
                            != value.replace("$", "")
                        ):
                            ambiguous.add(key)
                        elif previous is None:
                            chosen[key] = row
        eligible = {
            key: row for key, row in chosen.items() if key not in ambiguous
        }
        question_terms = [
            term
            for word in re.findall(r"[A-Za-z]{3,}", question.casefold())
            for term in self._terms(word)
        ]

        def separately_requested(
            broad: frozenset[str], specific: frozenset[str]
        ) -> bool:
            """Detect a broad metric mention separate from its specific variant."""
            if re.search(
                r"\b(?:combined|sum|total)\s+(?:amount\s+)?of\b",
                question,
                re.IGNORECASE,
            ):
                return False
            extras = specific - broad
            width = len(broad) + 2
            return any(
                question_terms[end - 1] in broad
                and broad.issubset(window)
                and not extras.intersection(window)
                for end in range(1, len(question_terms) + 1)
                for window in (
                    set(question_terms[max(0, end - width):end]),
                )
            )

        # When both a generic and a more specific matching row exist, retain the
        # specific row unless the question names the broad metric separately.
        eligible_rows = parent_matches if include_ambiguous else [(year, key, row) for (year, key), row in eligible.items()]
        return tuple(sorted((
            row
            for year, key, row in eligible_rows
            if not any(
                year == other_year
                and key < other
                and not separately_requested(key, other)
                for other_year, other, _ in eligible_rows
            )
        ), key=lambda row: not ("financial statements" in str(row.source.get("section", "")).casefold()
                                and MONETARY_STATEMENT_RE.search(str(row.source.get("table_context", ""))))))

    @trace_component("finrag.retrieval.structured_lookup")
    def answer(
        self, question: str, scope: Scope, *, wants_complete_table: bool
    ) -> DirectAnswer | None:
        if wants_complete_table:
            return self._table(question, scope)
        if ITEM_LIST_RE.search(question):
            return self._items(scope)
        return None
