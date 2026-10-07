from __future__ import annotations

import json

from src.generation.answer_guardrails import INSUFFICIENT_EVIDENCE_RESPONSE
from src.generation.prompt_templates import FINANCIAL_SYSTEM_PROMPT

from .models import MasterExample, ScopeRecord
from .utils import sha256_text


def scope_label(scope: ScopeRecord) -> str:
    groups = " vs ".join(
        f"{ticker} {year}" for ticker in scope.tickers for year in scope.years
    )
    document_types = " and ".join(
        "10-K" if value == "10K" else "transcript" for value in scope.doc_types
    )
    return f"{groups} | {document_types}"


def render_context(example: MasterExample) -> str:
    blocks: list[str] = []
    for evidence in example.evidence:
        header = (
            f"[SOURCE {evidence.source_id} | ticker={evidence.ticker} | "
            f"fiscal_year={evidence.fiscal_year} | doc_type={evidence.doc_type} | "
            f"source={evidence.source} | section={evidence.section} | "
            f"parent_id={evidence.parent_id}]"
        )
        blocks.append(f"{header}\n{evidence.text.strip()}")
    if example.application_context.strip():
        blocks.append(example.application_context.strip())
    return "\n\n".join(blocks)


def generation_instruction(example: MasterExample) -> str:
    if example.query_type == "complete_table":
        return (
            "Return the requested complete source table as valid Markdown, preserving "
            "its row and column order. Put citations outside the table."
        )
    if example.query_type in {"targeted_table", "comparison"}:
        return (
            "Return only the requested rows and columns. Use a valid Markdown table only "
            "when the question requests tabular presentation."
        )
    return (
        "Answer every requested part using only the retrieved evidence. Return atomic "
        "claims with supporting source IDs and preserve metric, period, unit, sign, and "
        "reported-versus-guidance status."
    )


def render_system_prompt(example: MasterExample) -> str:
    return FINANCIAL_SYSTEM_PROMPT.format(
        insufficient_evidence_response=INSUFFICIENT_EVIDENCE_RESPONSE,
        generation_instruction=generation_instruction(example),
        scope=scope_label(example.scope),
        history="None",
        context=render_context(example),
    )


def prompt_sha256(example: MasterExample) -> str:
    return sha256_text(render_system_prompt(example))


def training_record(example: MasterExample, *, include_metadata: bool = False) -> dict:
    record = {
        "prompt": [
            {"role": "system", "content": render_system_prompt(example)},
            {"role": "user", "content": f"Question: {example.question}"},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps(
                    example.target.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        ],
    }
    if include_metadata:
        record["metadata"] = {
            "example_id": example.example_id,
            "group_id": example.group_id,
            "query_type": example.query_type,
            "split": example.split,
        }
    return record
