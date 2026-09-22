from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from llama_cloud import LlamaCloud

from src.config import LLAMA_CLOUD_API_KEY, SOURCE_DATA_DIR, validate_runtime_config
from src.constants import (
    COMPANY_NAMES,
    FISCAL_CALENDARS,
    METADATA_SCHEMA_VERSION,
    TRANSCRIPT_QUARTER,
)
from src.ingestion.document_processing import (
    clean_text,
    extract_first_date,
    file_hash,
    format_transcript,
    package_version,
    parser_page_number,
    split_sections,
    stable_id,
)
from src.ingestion.source_inventory import CorpusInventory, discover_sources
from src.schemas import SourceFile


FISCAL_END_RE = re.compile(
    r"fiscal year ended\s+"
    r"(?P<date>(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+20\d{2})",
    re.IGNORECASE,
)
HTML_TRANSCRIPT_CUE_RE = re.compile(
    r"\b(transcript|operator\s*:|question\s+(?:and|&)\s+answer|prepared remarks?)\b",
    re.IGNORECASE,
)
HTML_SPEAKER_CUE_RE = re.compile(r"\b[A-Z][A-Z .'-]{2,}:\s")


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.title(), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


@dataclass(frozen=True)
class ParsedPage:
    text: str
    metadata: dict


class LlamaCloudMarkdownParser:
    """Expose the unified LlamaCloud Parse API through the loader's page contract."""

    def __init__(self, api_key: str) -> None:
        self.client = LlamaCloud(api_key=api_key)

    def load_data(self, path: str) -> list[ParsedPage]:
        uploaded = self.client.files.create(file=Path(path), purpose="parse")
        result = self.client.parsing.parse(
            file_id=uploaded.id,
            tier="agentic",
            version="latest",
            expand=["markdown"],
        )
        markdown = getattr(result, "markdown", None)
        pages = list(getattr(markdown, "pages", None) or [])
        parsed: list[ParsedPage] = []
        for fallback, page in enumerate(pages, start=1):
            text = str(
                getattr(page, "markdown", None)
                or getattr(page, "text", None)
                or ""
            )
            # The ordered Markdown page list maps directly to physical PDF pages.
            # Using a local one-based counter avoids SDK-version differences in
            # optional zero/one-based page-number fields.
            parsed.append(ParsedPage(text=text, metadata={"page_number": fallback}))
        if not parsed:
            raise ValueError(f"LlamaCloud returned no Markdown pages for '{Path(path).name}'.")
        return parsed
class FinancialDocumentLoader:
    """Parse the supported corpus into page-aware, normalized source segments."""

    def __init__(self, *, parser: Any | None = None) -> None:
        self.parser = parser
        self.corpus_root: Path | None = None
        self.inventory: CorpusInventory | None = None

    def _get_parser(self) -> Any:
        if self.parser is None:
            validate_runtime_config(require_llama=True)
            self.parser = LlamaCloudMarkdownParser(str(LLAMA_CLOUD_API_KEY))
        return self.parser

    def load_documents(
        self,
        data_dir: str | Path | None = None,
        *,
        require_complete: bool = False,
        preflight: bool = True,
    ) -> list[Document]:
        self.corpus_root = Path(data_dir or SOURCE_DATA_DIR).resolve()
        self.inventory = discover_sources(self.corpus_root, require_complete=require_complete)
        if preflight:
            self.preflight_sources(self.inventory)

        documents: list[Document] = []
        for source in self.inventory.sources:
            documents.extend(self.load_source(source))
        if not documents:
            raise ValueError("No usable source segments were produced.")
        return documents

    def preflight_sources(self, inventory: CorpusInventory) -> None:
        """Fail before paid parsing if a local source is empty or clearly unusable."""
        errors: list[str] = []
        for source in inventory.sources:
            try:
                if source.path.stat().st_size < 1024:
                    raise ValueError("file is unexpectedly small")
                if source.path.suffix.lower() == ".pdf":
                    with source.path.open("rb") as stream:
                        if stream.read(5) != b"%PDF-":
                            raise ValueError("file does not have a valid PDF signature")
                elif source.doc_type == "TRANSCRIPT":
                    blocks = self._extract_html_blocks(source.path)
                    formatted, speakers, _ = format_transcript(blocks)
                    words = len(formatted.split())
                    has_cue = bool(HTML_TRANSCRIPT_CUE_RE.search(" ".join(blocks)))
                    if words < 100 or (not speakers and not has_cue):
                        raise ValueError(
                            "HTML does not contain enough readable transcript text or speaker cues"
                        )
            except (OSError, ValueError) as error:
                errors.append(f"{source.path.name}: {error}")
        if errors:
            raise ValueError("Source preflight failed:\n- " + "\n- ".join(errors))

    def load_source(self, source: SourceFile) -> list[Document]:
        if source.doc_type == "10K":
            return self._load_10k(source)
        if source.path.suffix.lower() in {".html", ".htm"}:
            return self._load_html_transcript(source)
        return self._load_pdf_transcript(source)

    def _base_metadata(self, source: SourceFile) -> dict:
        digest = file_hash(source.path)
        try:
            source_path = source.path.relative_to(self.corpus_root).as_posix()
        except (TypeError, ValueError):
            source_path = source.path.name

        metadata = {
            "corpus": "finrag_financial_filings",
            "metadata_schema_version": METADATA_SCHEMA_VERSION,
            "document_id": stable_id(source.ticker, source.fiscal_year, source.doc_type, digest),
            "ticker": source.ticker,
            "company_name": COMPANY_NAMES[source.ticker],
            "doc_type": source.doc_type,
            "fiscal_year": source.fiscal_year,
            "fiscal_period": "FY" if source.doc_type == "10K" else TRANSCRIPT_QUARTER,
            "source": source.path.name,
            "source_path": source_path,
            "source_hash": digest,
            "fiscal_year_end": FISCAL_CALENDARS[source.ticker]["fiscal_year_end"],
            "fiscal_q4_months": FISCAL_CALENDARS[source.ticker]["fiscal_q4_months"],
        }
        return metadata

    def _load_10k(self, source: SourceFile) -> list[Document]:
        parsed_pages = self._get_parser().load_data(str(source.path))
        base = self._base_metadata(source)
        base.update(
            {
                "extraction_method": "llamaparse_markdown",
                "parser_version": package_version("llama-cloud"),
            }
        )

        raw_fragments: list[tuple[str, str, str | None, str | None, int]] = []
        inherited = "Front Matter"
        first_pages: list[str] = []
        for index, page in enumerate(parsed_pages, start=1):
            text = clean_text(getattr(page, "text", ""))
            if not text:
                continue
            if len(first_pages) < 8:
                first_pages.append(text)
            pdf_page = parser_page_number(getattr(page, "metadata", {}) or {}, index)
            fragments, inherited = split_sections(text, inherited)
            for fragment in fragments:
                if len(fragment.text) >= 40:
                    raw_fragments.append(
                        (fragment.text, fragment.section, fragment.item, fragment.subsection, pdf_page)
                    )

        front_text = "\n".join(first_pages)
        fiscal_match = FISCAL_END_RE.search(front_text)
        base["period_end_date"] = _iso_date(fiscal_match.group("date")) if fiscal_match else None

        documents: list[Document] = []
        active_key: str | None = None
        active_section = "Front Matter"
        active_item: str | None = None
        active_subsection: str | None = None
        active_pages: list[int] = []
        active_text: list[str] = []

        def flush() -> None:
            nonlocal active_key, active_section, active_item, active_subsection
            nonlocal active_pages, active_text
            if not active_text:
                return
            page_start, page_end = min(active_pages), max(active_pages)
            content = "\n\n".join(active_text)
            segment_index = len(documents)
            metadata = {
                **base,
                "section": active_section,
                "item": active_item,
                "subsection": active_subsection,
                "pdf_page_start": page_start,
                "pdf_page_end": page_end,
                "source_segment_id": stable_id(
                    base["document_id"], segment_index, active_key, page_start, page_end
                ),
                "source_segment_word_count": len(content.split()),
            }
            documents.append(Document(page_content=content, metadata=metadata))
            active_key = None
            active_section = "Front Matter"
            active_item = None
            active_subsection = None
            active_pages = []
            active_text = []

        for text, section, item, subsection, pdf_page in raw_fragments:
            key = item or section
            if active_key is not None and key != active_key:
                flush()
            if active_key is None:
                active_key = key
                active_section = section
                active_item = item
                active_subsection = subsection
            if not active_pages or active_pages[-1] != pdf_page:
                active_pages.append(pdf_page)
                active_text.append(f"[PAGE: {pdf_page}]")
            active_text.append(text)
        flush()

        if not documents:
            raise ValueError(f"No usable 10-K content parsed from '{source.path.name}'.")
        return documents

    @staticmethod
    def _html_content_root(soup: BeautifulSoup):
        specific_candidates = soup.select(
            "[id*='transcript' i], [class*='transcript' i], [id*='earnings' i], "
            "[class*='earnings' i], main .cmp-text, main .field--name-body"
        )
        candidates = specific_candidates or [
            candidate
            for candidate in (soup.find("article"), soup.find("main"), soup.body, soup)
            if candidate is not None
        ]
        scored_candidates = []
        for candidate in candidates:
            text = candidate.get_text(" ", strip=True)
            if not text:
                continue
            score = len(text)
            if HTML_TRANSCRIPT_CUE_RE.search(text):
                score += 100_000
            score += 10_000 * len(HTML_SPEAKER_CUE_RE.findall(text))
            scored_candidates.append((score, candidate))
        if scored_candidates:
            return max(scored_candidates, key=lambda item: item[0])[1]
        return soup

    @staticmethod
    def _unique_blocks(values: list[str]) -> list[str]:
        blocks: list[str] = []
        previous = ""
        for value in values:
            block = clean_text(value)
            if not block or block == previous:
                continue
            blocks.append(block)
            previous = block
        return blocks

    @classmethod
    def _html_blocks(cls, root) -> list[str]:
        """Extract transcript blocks from semantic tags or common div-based layouts."""
        semantic = [
            tag.get_text(" ", strip=True)
            for tag in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"])
            if tag.get_text(" ", strip=True)
        ]
        semantic = cls._unique_blocks(semantic)
        if semantic:
            return semantic

        leaf_blocks = []
        for tag in root.find_all(["div", "section", "article", "pre"]):
            if tag.find(["div", "section", "article", "p", "li"]):
                continue
            text = tag.get_text("\n", strip=True)
            leaf_blocks.extend(line for line in text.splitlines() if line.strip())
        leaf_blocks = cls._unique_blocks(leaf_blocks)
        if leaf_blocks:
            return leaf_blocks

        plain_text = root.get_text("\n", strip=True)
        return cls._unique_blocks([line for line in plain_text.splitlines() if line.strip()])

    @staticmethod
    def _script_transcript_payloads(soup: BeautifulSoup) -> list[str]:
        payloads = []
        for script in soup.find_all("script"):
            raw = script.string or script.get_text(" ", strip=True)
            if not raw or len(raw) < 200 or not HTML_TRANSCRIPT_CUE_RE.search(raw):
                continue
            decoded = html.unescape(raw)
            decoded = decoded.replace("\\n", "\n").replace("\\u003c", "<").replace("\\u003e", ">")
            payloads.append(decoded)
        return payloads

    def _extract_html_blocks(self, path: Path) -> list[str]:
        markup = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(markup, "html.parser")
        script_payloads = self._script_transcript_payloads(soup)
        for tag in soup.find_all(
            ["script", "style", "noscript", "nav", "header", "footer", "form", "button"]
        ):
            tag.decompose()
        root = self._html_content_root(soup)
        blocks = self._html_blocks(root)
        if len(" ".join(blocks).split()) < 25:
            for payload in script_payloads:
                payload_blocks = self._html_blocks(BeautifulSoup(payload, "html.parser"))
                if len(" ".join(payload_blocks).split()) > len(" ".join(blocks).split()):
                    blocks = payload_blocks
        if not blocks:
            raise ValueError(
                f"No readable transcript text found in '{path.name}'. "
                "Provide an HTML file containing the transcript text, not a webpage shell."
            )
        return blocks

    def _load_html_transcript(self, source: SourceFile) -> list[Document]:
        blocks = self._extract_html_blocks(source.path)
        return self._build_transcript_document(source, blocks, "beautifulsoup_html")

    def _load_pdf_transcript(self, source: SourceFile) -> list[Document]:
        blocks: list[str] = []
        for index, page in enumerate(self._get_parser().load_data(str(source.path)), start=1):
            text = clean_text(getattr(page, "text", ""))
            if not text:
                continue
            pdf_page = parser_page_number(getattr(page, "metadata", {}) or {}, index)
            blocks.append(f"[PAGE: {pdf_page}]")
            blocks.extend(text.splitlines())
        return self._build_transcript_document(source, blocks, "llamaparse_markdown")

    def _build_transcript_document(
        self, source: SourceFile, blocks: list[str], extraction_method: str
    ) -> list[Document]:
        text, speakers, sections = format_transcript(blocks)
        if len(text.split()) < 100:
            raise ValueError(
                f"Too little usable transcript content parsed from '{source.path.name}'."
            )

        base = self._base_metadata(source)
        call_date = _iso_date(extract_first_date("\n".join(blocks[:80])))
        metadata = {
            **base,
            "section": f"{TRANSCRIPT_QUARTER} Earnings Call",
            "call_date": call_date,
            "transcript_sections": sorted(sections),
            "speakers": sorted(speakers),
            "speaker_count": len(speakers),
            "extraction_method": extraction_method,
            "parser_version": package_version(
                "beautifulsoup4" if extraction_method == "beautifulsoup_html" else "llama-cloud"
            ),
            "source_segment_id": stable_id(base["document_id"], "full_transcript"),
            "source_segment_word_count": len(text.split()),
        }
        return [Document(page_content=text, metadata=metadata)]
