"""BM25 keyword retrieval over the persisted child-chunk index."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.schemas import Scope


# This tokenizer keeps financial terms such as 10-K, non-GAAP, CET1 and 18.2%.
TOKEN_RE = re.compile(r"[^\W_]+(?:[.&'-][^\W_]+)*|\d+(?:\.\d+)?%?", re.UNICODE)

# Only ordinary question words are removed. Financial terms, company names,
# years, metrics and abbreviations remain searchable.
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "why",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    """Return deterministic, case-insensitive terms for BM25."""
    return [
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if token.casefold() not in STOPWORDS
    ]


@dataclass(frozen=True)
class LexicalHit:
    record: dict
    score: float
    coverage: float


class LexicalIndex:
    """Load child records once and search them with standard BM25 Okapi."""

    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"Lexical index not found: '{path}'.")

        self.records: list[dict] = []
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid lexical-index JSON at line {line_number}."
                    ) from exc
                if not all(key in record for key in ("child_id", "text", "metadata")):
                    raise ValueError(
                        f"Lexical-index line {line_number} is missing required fields."
                    )
                self.records.append(record)

        if not self.records:
            raise ValueError(f"Lexical index is empty: '{path}'.")

        self.tokens = [tokenize(str(record["text"])) for record in self.records]
        self.token_sets = [set(tokens) for tokens in self.tokens]

        # Empty chunks are represented by a private token because rank-bm25
        # requires a tokenized corpus. Real queries never generate this token.
        corpus = [tokens or ["__empty_chunk__"] for tokens in self.tokens]
        # BM25Okapi's documented defaults are k1=1.5 and b=0.75. Keeping the
        # library defaults makes this a drop-in replacement with no config edit.
        self.bm25 = BM25Okapi(corpus)

    @staticmethod
    def _matches_scope(metadata: dict, scope: Scope) -> bool:
        """Apply metadata constraints before results are returned."""
        ticker = str(metadata.get("ticker", ""))
        year = str(metadata.get("fiscal_year", ""))
        doc_type = str(metadata.get("doc_type", ""))

        if scope.tickers and ticker not in scope.tickers:
            return False
        if scope.years and year not in scope.years:
            return False
        if scope.doc_type and doc_type != scope.doc_type:
            return False
        return True

    def search(self, query: str, scope: Scope, *, k: int) -> list[LexicalHit]:
        """Return the top metadata-eligible BM25 hits and query-term coverage."""
        if k <= 0:
            return []

        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return []

        eligible_ids = [
            index
            for index, record in enumerate(self.records)
            if self._matches_scope(record["metadata"], scope)
        ]
        if not eligible_ids:
            return []

        scores = self.bm25.get_batch_scores(query_terms, eligible_ids)
        query_term_set = set(query_terms)
        hits: list[LexicalHit] = []

        for record_id, score in zip(eligible_ids, scores):
            matched_terms = query_term_set.intersection(self.token_sets[record_id])
            if not matched_terms:
                continue
            hits.append(
                LexicalHit(
                    record=self.records[record_id],
                    score=float(score),
                    coverage=len(matched_terms) / len(query_term_set),
                )
            )

        # child_id makes ties deterministic across runs and operating systems.
        hits.sort(key=lambda hit: (-hit.score, str(hit.record["child_id"])))
        return hits[:k]
