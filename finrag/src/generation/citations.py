from __future__ import annotations

import re


SOURCE_ID_RE = re.compile(r"\[(S\d+)\]")
DISPLAY_CITATION_RE = re.compile(r"\[S\d+:[^\]]+\]")
SOURCE_LIST_RE = re.compile(
    r"\n{2,}\*\*Sources\*\*\s*\n.*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _document_label(value: object) -> str:
    return "10-K" if str(value) == "10K" else "Transcript"


def citation_label(source: dict) -> str:
    """Build one compact citation from already-validated source metadata."""
    identity = (
        f"[{source['id']}] {source.get('ticker', 'Unknown')} "
        f"{source.get('fiscal_year', 'Unknown')} "
        f"{_document_label(source.get('doc_type'))}"
    )
    details = [identity]
    section = source.get("section")
    if section:
        details.append(str(section))
    page = source.get("pdf_page")
    if page and page != "not available":
        details.append(f"p. {page}")
    elif source.get("call_date"):
        details.append(f"call {source['call_date']}")
    filename = source.get("source")
    if filename:
        details.append(str(filename))
    return " — ".join((details[0], "; ".join(details[1:]))) if len(details) > 1 else details[0]


def expand_citations(answer: str, sources: list[dict]) -> str:
    """Keep compact inline IDs and append each full citation exactly once."""
    used_ids = tuple(dict.fromkeys(SOURCE_ID_RE.findall(answer)))
    if not used_ids:
        return answer

    source_by_id = {str(source["id"]): source for source in sources}
    citation_lines = [
        f"- {citation_label(source_by_id[source_id])}"
        for source_id in used_ids
        if source_id in source_by_id
    ]
    if not citation_lines:
        return answer
    return answer.rstrip() + "\n\n**Sources**\n\n" + "\n".join(citation_lines)


def escape_currency_for_markdown(text: str) -> str:
    """Prevent Streamlit from interpreting currency dollars as LaTeX delimiters."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def strip_citations(text: str) -> str:
    """Remove old source IDs from history so they cannot leak into a new answer."""
    text = SOURCE_LIST_RE.sub("", text)
    text = DISPLAY_CITATION_RE.sub("", text)
    return SOURCE_ID_RE.sub("", text)
