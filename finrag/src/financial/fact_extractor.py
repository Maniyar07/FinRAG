"""Structured extraction of requested financial values from retrieved evidence."""

from __future__ import annotations

import json

from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.financial.models import CandidateFinancialFact, FactExtractionPayload
from src.observability import trace_component


FACT_EXTRACTION_SYSTEM_PROMPT = """You extract financial facts from supplied evidence.

Rules:
1. Extract only values needed to answer the question.
2. Use only supplied source IDs and evidence. Never use memory or calculate a value.
3. Copy raw_value from the exact source table cell. Never convert scale, round,
   calculate, or rewrite it (for example, keep $97,690 million rather than changing
   it to $97.69 billion). evidence_excerpt may join the visible row label, period,
   and value from the same table, but every word and number must occur in that source.
   For prose, copy the contiguous literal amount including its stated scale.
   Never inherit the scale of an adjacent table or convert the prose amount to it.
4. Preserve the exact metric, row label, column label, period, sign, currency, scale,
   unit, and accounting basis. Parentheses indicate a negative presented value.
5. A filing fiscal year is not automatically the period of every table column.
6. Do not substitute a subtotal, segment, adjusted measure, or adjacent column.
   For period-end balances use the reported carrying amount, not a cost-basis,
   unrealized-gain or loss column. Use another measurement basis only when asked.
   Where a date spans multiple value columns, copy the leaf column label; a date
   alone does not identify the measure. Derive monetary scale from the table's
   unit declaration; exceptions for units, shares and per-share values still apply.
7. Return no fact when the requested value is absent or ambiguous.
8. A table row inherits its governing section heading. Preserve that financial
   meaning in metric; never relabel a cost, subtotal, or segment as another measure.
   A caption describes all rows, not each row individually. A component cannot
   satisfy an aggregate, even when both share words. Check the complete row AND
   its selected leaf column: neither may narrow the requested quantity. Prefer
   the primary statement's exact reported total when available. For notes with
   several valuation columns, use the source's reporting basis and preserve its
   actual leaf label and accounting_basis; never select the first numeric cell.
9. When requirements are supplied, extract ONLY numeric fact requirements and
   attach their requirement_id. Preserve source wording for metric; row_label
   and column_label must copy actual source labels. For prose, metric must copy
   the actual measurement's noun phrase, not request wording such as 'guidance'.
   Labels can differ semantically
   from the request (abbreviations, paraphrases); evidence must establish the same
   quantity and every qualifier. Do not turn status, timing, or explanations into
   numeric facts. For prose, put the literal source measurement phrase in row_label,
   including source abbreviations; metric may paraphrase the requested measurement.
   Copy a concise measurement noun phrase, not an entire sentence as row_label.
   If a value refers back to an earlier measurement, include that antecedent and
   the literal value together in the contiguous evidence_excerpt.
   Copy a contiguous narrative excerpt, or actual table labels and
   cells, supporting the value and its basis. For a forecast preserve the requested
   future period and the source's forecast wording. Never derive a value from a
   percentage or compute an amount. Copy thresholds literally: never change
   'exceeded' to 'over' or prefix a qualifier absent from the selected source.
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
        self.chain = chain

    @trace_component("finrag.financial.fact_extract")
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

        chain = self.chain
        if chain is None:
            from src.generation.llm_engine import get_llm_engine
            schema = FactExtractionPayload.model_json_schema()
            schema["$defs"]["CandidateFinancialFact"]["properties"]["source_id"] = {"anyOf": [
                {"type": "string", "enum": [str(source["id"]) for source in sources]}, {"type": "null"}]}
            try:
                request = json.loads(question)
            except json.JSONDecodeError:
                request = {}
            requirements = request.get("requirements", []) if isinstance(request, dict) else []
            if requirements:
                schema["$defs"]["CandidateFinancialFact"]["properties"].pop("requirement_id")
                ids = [item["requirement_id"] for item in requirements if item.get("kind", "fact") == "fact"]
                schema["properties"]["facts"] = {"type": "object", "properties": {key: {"anyOf": [
                    {"$ref": "#/$defs/CandidateFinancialFact"}, {"type": "null"}]} for key in ids},
                    "required": ids, "additionalProperties": False}
            prompt = ChatPromptTemplate.from_messages([
                ("system", FACT_EXTRACTION_SYSTEM_PROMPT + " With a plan, fill each requirement's own keyed slot with its measurement, or null when absent. Do not move an amount to another slot or return placeholders."),
                ("human", "Question:\n{question}\n\nRetrieved evidence:\n{context}"),
            ])
            chain = prompt | get_llm_engine().with_structured_output(schema, method="json_schema", strict=True)
        payload = chain.invoke({"question": cleaned_question, "context": context})
        if isinstance(payload, dict) and isinstance(payload.get("facts"), dict):
            return tuple(CandidateFinancialFact.model_validate({**value, "requirement_id": key})
                         for key, value in payload["facts"].items() if value is not None)
        if isinstance(payload, FactExtractionPayload):
            return tuple(payload.facts)
        if isinstance(payload, dict):
            return tuple(FactExtractionPayload.model_validate(payload).facts)
        raise TypeError("Financial fact extractor returned an unsupported payload.")
