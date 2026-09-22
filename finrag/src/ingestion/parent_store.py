from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.documents import Document


class ParentStore:
    """Simple readable document store keyed by deterministic parent ID."""

    def __init__(self, root: Path, *, create: bool = False):
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Parent store not found: '{self.root}'.")

    def _path(self, parent_id: str) -> Path:
        if not parent_id or any(character not in "0123456789abcdef-" for character in parent_id.lower()):
            raise ValueError("Invalid parent ID.")
        return self.root / f"{parent_id}.json"

    def put_many(self, documents: list[tuple[str, Document]]) -> None:
        for parent_id, document in documents:
            destination = self._path(parent_id)
            temporary = destination.with_suffix(".tmp")
            record = {"page_content": document.page_content, "metadata": document.metadata}
            with temporary.open("w", encoding="utf-8") as target:
                json.dump(record, target, ensure_ascii=True, sort_keys=True)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            temporary.replace(destination)

    def get_many(self, parent_ids: list[str]) -> list[Document | None]:
        documents: list[Document | None] = []
        for parent_id in parent_ids:
            path = self._path(parent_id)
            if not path.is_file():
                documents.append(None)
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            documents.append(
                Document(page_content=record["page_content"], metadata=record["metadata"])
            )
        return documents

