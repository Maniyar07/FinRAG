"""Structured extraction of requested financial values from retrieved evidence."""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.financial.models import CandidateFinancialFact, FactExtractionPayload


FACT_EXTRACTION_SYSTEM_PROMPT = """You extract financial facts from supplied evidence.

Rules:
1. Extract only values needed to answer the question.
2. Use only supplied source IDs and evidence. Never use memory or calculate a value.
3. Copy raw_value from the exact source table cell. Never convert scale, round,
   calculate, or rewrite it (for example, keep $97,690 million rather than changing
   it to $97.69 billion). evidence_excerpt may join the visible row label, period,
   and value from the same table, but every word and number must occur in that source.
4. Preserve the exact metric, row label, column label, period, sign, currency, scale,
   unit, and accounting basis. Parentheses indicate a negative presented value.
5. A filing fiscal year is not automatically the period of every table column.
6. Do not substitute a subtotal, segment, adjusted measure, or adjacent column.
7. Return no fact when the requested value is absent or ambiguous.
"""


def _evidence_context(sources: list[dict]) -> str:
    blocks: list[str] = []
    for source in sources:
        source_id = str(source.get("id", "")).strip().upper()
        if not source_id:
            continue
        header = (
            f"[SOURCE {source_id} | TICKER: {source.get('ticker')} | "
            f"FISCAL YEAR: {source.get('fiscal_year')} | "
            f"TYPE: {source.get('doc_type')} | SECTION: {source.get('section')} | "
            f"PAGE: {source.get('pdf_page')}]"
        )
        evidence = str(source.get("evidence_text") or "").strip()
        if evidence:
            blocks.append(f"{header}\n{evidence}")
    return "\n\n---\n\n".join(blocks)


class FinancialFactExtractor:
    """Ask a structured model for candidate facts; validation remains separate."""

    def __init__(self, *, chain: Any | None = None) -> None:
        if chain is None:
            from src.generation.llm_engine import get_llm_engine

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", FACT_EXTRACTION_SYSTEM_PROMPT),
                    (
                        "human",
                        "Question:\n{question}\n\nRetrieved evidence:\n{context}",
                    ),
                ]
            )
            structured = get_llm_engine().with_structured_output(
                FactExtractionPayload,
                method="json_schema",
            )
            chain = prompt | structured
        self.chain = chain

    def extract(
        self,
        *,
        question: str,
        sources: list[dict],
    ) -> tuple[CandidateFinancialFact, ...]:
        cleaned_question = " ".join(question.split())
        if not cleaned_question:
            raise ValueError("Financial fact extraction question cannot be empty.")
        context = _evidence_context(sources)
        if not context:
            return ()

        payload = self.chain.invoke(
            {
                "question": cleaned_question,
                "context": context,
            }
        )
        if isinstance(payload, FactExtractionPayload):
            return tuple(payload.facts)
        if isinstance(payload, dict):
            return tuple(FactExtractionPayload.model_validate(payload).facts)
        raise TypeError("Financial fact extractor returned an unsupported payload.")
