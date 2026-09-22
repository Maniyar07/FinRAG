"""Lossless block boundaries for Markdown/HTML source serialization."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    text: str
    is_table: bool = False


def blocks(text):
    # Recognize serialization grammar, never company names/financial topics.
    lines = text.splitlines(keepends=True)
    result = []
    buf = []
    table = False
    html = False

    def flush():
        nonlocal buf, table
        if buf:
            result.append(Block("".join(buf), table))
            buf = []
            table = False

    for line in lines:
        if "<table" in line.lower():
            flush()
            html = True
            table = True
        if html:
            buf.append(line)
            if "</table>" in line.lower():
                html = False
                flush()
            continue
        pipe = line.lstrip().startswith("|")
        if pipe:
            if buf and not table:
                flush()
            table = True
            buf.append(line)
            continue
        if table:
            flush()
        buf.append(line)
        if not line.strip():
            flush()
    flush()
    return result
