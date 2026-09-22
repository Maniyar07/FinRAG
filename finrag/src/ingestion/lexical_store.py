from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.documents import Document


def write_lexical_index(path: Path, documents: list[Document]) -> None:
    """Atomically persist searchable child text for local lexical retrieval."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for document in documents:
            record = {
                "child_id": document.metadata["child_id"],
                "parent_id": document.metadata["parent_id"],
                "text": document.page_content,
                "metadata": document.metadata,
            }
            destination.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)

