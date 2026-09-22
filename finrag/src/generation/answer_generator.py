from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.generation.answer_guardrails import (
    AnswerValidation,
    CITATION_RE,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    UNVERIFIABLE_RESPONSE,
    validate_answer_payload,
)
from src.generation.citations import expand_citations, strip_citations
from src.generation.financial_fact_validator import (
    validate_revenue_change,
    validate_table_answer,
)
from src.schemas import RetrievalBundle


class GroundedAnswer(BaseModel):
    """Structured model output used to separate prose from citation identity."""

    answer: str = Field(description="Grounded Markdown answer with inline [S#] citations.")
    source_ids: list[str] = Field(
        default_factory=list,
        description="Unique retrieved source IDs used in the answer, for example S1 and S3.",
    )


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    attempts: int
    validation_reason: str
    raw_output_previews: tuple[str, ...] = ()


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "None"
    lines = []
    for message in history[-4:]:
        role = str(message.get("role", "user")).upper()
        content = strip_citations(str(message.get("content", "")))
        content = " ".join(content.split())[:1200]
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or "None"


class AnswerGenerator:
    def __init__(self, *, chain: Any | None = None) -> None:
        if chain is None:
            from src.generation.llm_engine import get_llm_engine
            from src.generation.prompt_templates import get_financial_prompt_template

            structured_llm = get_llm_engine().with_structured_output(
                GroundedAnswer,
                method="json_schema",
            )
            chain = get_financial_prompt_template() | structured_llm
        self.chain = chain

    @staticmethod
    def _payload_values(payload: Any) -> tuple[str, list[str]]:
        if isinstance(payload, GroundedAnswer):
            return payload.answer, list(payload.source_ids)
        if isinstance(payload, dict):
            return str(payload.get("answer", "")), list(payload.get("source_ids") or [])
        return str(payload), []

    @staticmethod
    def _preview(answer: str) -> str:
        return " ".join(answer.split())[:2000]

    @staticmethod
    def _compound_answer_requires_inline_citations(answer: str) -> bool:
        """Declared IDs are insufficient for a multi-section synthesized answer."""
        nonempty_lines = [line for line in answer.splitlines() if line.strip()]
        return len(answer) > 400 or len(nonempty_lines) >= 6

    @staticmethod
    def _uncited_substantive_bullet(answer: str) -> bool:
        narrative_terms = re.compile(
            r"\b(?:attribut|because|commentary|discuss|driver|due to|explain|"
            r"highlight|indicat|management|reason|risk)\w*\b",
            re.IGNORECASE,
        )
        for line in answer.splitlines():
            stripped = line.strip()
            if not stripped.startswith(("- ", "* ")):
                continue
            substantive = (
                len(stripped) >= 70
                or bool(re.search(r"\d", stripped))
                or bool(narrative_terms.search(stripped))
            )
            if substantive and not CITATION_RE.search(stripped):
                return True
        return False

    @staticmethod
    def _requested_transcript_citation_error(
        question: str,
        answer: str,
        sources: list[dict],
    ) -> str | None:
        """Require company-specific transcript support for management commentary."""
        if "transcript" not in question.casefold():
            return None
        source_map = {str(source.get("id", "")).upper(): source for source in sources}
        company_names = {
            "MSFT": ("msft", "microsoft"),
            "TSLA": ("tsla", "tesla"),
            "JPM": ("jpm", "jpmorgan", "jp morgan"),
        }
        narrative_terms = (
            "attribut",
            "commentary",
            "discuss",
            "driver",
            "due to",
            "explain",
            "highlight",
            "indicat",
            "management",
            "reason",
        )
        for line in answer.splitlines():
            lowered = line.casefold()
            if not any(term in lowered for term in narrative_terms):
                continue
            for ticker, names in company_names.items():
                if not any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in names):
                    continue
                cited = [value.upper() for value in CITATION_RE.findall(line)]
                if not cited or not any(
                    str(source_map.get(source_id, {}).get("ticker", "")).upper()
                    == ticker
                    and str(
                        source_map.get(source_id, {}).get("doc_type", "")
                    ).upper().replace("-", "")
                    == "TRANSCRIPT"
                    for source_id in cited
                ):
                    return f"requested_transcript_citation_missing:{ticker}"
        return None

    @staticmethod
    def _forward_looking_as_historical_cause_error(answer: str) -> str | None:
        """Reject an obvious future-outlook claim used to explain past results."""
        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            lowered = sentence.casefold()
            if (
                re.search(r"\b(?:declin|increas|decreas|growth|grew|fell)\w*\b", lowered)
                and re.search(r"\b(?:attribut|because|driven|due to|caus)\w*\b", lowered)
                and re.search(r"\b(?:expect|forecast|outlook|will|future)\w*\b", lowered)
            ):
                return "forward_looking_as_historical_cause"
        return None

    @staticmethod
    def _narrative_coverage_error(
        cited_ids: tuple[str, ...],
        sources: list[dict],
    ) -> str | None:
        narrative_groups = {
            (str(source.get("ticker")), str(source.get("fiscal_year")))
            for source in sources
            if any(
                role.get("evidence_type") == "narrative"
                for role in source.get("evidence_requirements") or []
            )
        }
        if not narrative_groups:
            return None
        cited = {value.upper() for value in cited_ids}
        covered = {
            (str(source.get("ticker")), str(source.get("fiscal_year")))
            for source in sources
            if str(source.get("id", "")).upper() in cited
            and any(
                role.get("evidence_type") == "narrative"
                for role in source.get("evidence_requirements") or []
            )
        }
        missing = sorted(narrative_groups - covered)
        if missing:
            labels = ",".join(f"{ticker}-{year}" for ticker, year in missing)
            return f"missing_narrative_citation_groups:{labels}"
        return None

    @staticmethod
    def _extractive_fallback(bundle: RetrievalBundle) -> str:
        """Return short, cited evidence excerpts if both synthesis attempts are invalid."""
        blocks = bundle.context.split("\n\n---\n\n")
        excerpts: list[str] = []
        for source, block in zip(bundle.sources[:2], blocks[:2]):
            body = block.split("\n", 1)[1] if "\n" in block else block
            compact = " ".join(body.split())
            if not compact:
                continue
            excerpt = compact[:600]
            if len(compact) > 600:
                sentence_end = max(excerpt.rfind(". "), excerpt.rfind("; "))
                if sentence_end >= 200:
                    excerpt = excerpt[: sentence_end + 1]
                else:
                    excerpt = excerpt.rstrip() + "…"
            excerpt = re.sub(r"\[(?:SOURCE|SECTION|SPEAKER)[^\]]*\]", "", excerpt).strip()
            excerpts.append(f"- {excerpt} [{source['id']}]")
        if not excerpts:
            return UNVERIFIABLE_RESPONSE
        return (
            "The generated synthesis did not pass format or citation validation. "
            "The most relevant retrieved evidence is:\n\n" + "\n\n".join(excerpts)
        )

    def _invoke(
        self,
        *,
        question: str,
        bundle: RetrievalBundle,
        history: list[dict] | None,
        generation_instruction: str,
    ) -> tuple[str, list[str]]:
        payload = self.chain.invoke(
            {
                "question": question,
                "context": bundle.context,
                "scope": bundle.scope.label(),
                "history": _format_history(history),
                "generation_instruction": generation_instruction,
                "insufficient_evidence_response": INSUFFICIENT_EVIDENCE_RESPONSE,
            }
        )
        return self._payload_values(payload)

    def generate_with_trace(
        self,
        question: str,
        bundle: RetrievalBundle,
        *,
        history: list[dict] | None = None,
        wants_table: bool = False,
        wants_complete_table: bool = False,
        fail_closed_on_invalid: bool = False,
    ) -> GenerationResult:
        allowed_ids = ", ".join(str(source["id"]) for source in bundle.sources)
        narrative_groups: dict[tuple[str, str], list[str]] = {}
        for source in bundle.sources:
            if not any(
                role.get("evidence_type") == "narrative"
                for role in source.get("evidence_requirements") or []
            ):
                continue
            key = (str(source.get("ticker")), str(source.get("fiscal_year")))
            narrative_groups.setdefault(key, []).append(str(source["id"]))
        narrative_guide = "; ".join(
            f"{ticker}-{year}: {', '.join(source_ids)}"
            for (ticker, year), source_ids in sorted(narrative_groups.items())
        ) or "not separately tagged"
        previews: list[str] = []
        last_validation = AnswerValidation(False, UNVERIFIABLE_RESPONSE, "not_attempted")

        if wants_complete_table:
            first_instruction = (
                "The user requested a complete financial-statement table. Return a valid "
                "Markdown table preserving every row and column available in the retrieved "
                "source table. Do not replace it with a summary or mix rows from another table. "
                "If the retrieved context does not contain the complete table, return the exact "
                "insufficient-evidence response."
            )
        elif wants_table:
            first_instruction = (
                "Return the requested evidence as a valid Markdown table. Keep the table title "
                "on a separate line and preserve the relevant source row labels and columns."
            )
        else:
            first_instruction = (
                "Answer every requested part using only retrieved evidence. For a financial "
                "value, identify the exact table row and requested year column before writing "
                "the answer. Do not substitute a segment, subtotal, adjacent row, or broader "
                "metric. Omit unrelated figures and cite each factual bullet or paragraph "
                "inline with its supporting source IDs; source_ids alone are not sufficient. "
                "Describe transcript items as causes of an annual change only when management "
                "explicitly connects them to that change. Otherwise label them as relevant "
                "management commentary without asserting causation. Do not add a table for a "
                "single requested metric unless the user asks for one."
            )

        instructions = [
            first_instruction,
            "The previous output failed format, citation, or financial-table validation. "
            "Check the requested metric's exact row, year column, and units, then regenerate "
            "using only "
            f"these available IDs: {allowed_ids}. Include inline [S#] citations and repeat the "
            "same IDs in source_ids. In a company comparison, cite every company's numeric and "
            "narrative subsection separately with that company's own sources. Preserve the "
            "requested document type: management commentary requested from a transcript must "
            "cite that company's transcript, not its 10-K. The application-tagged narrative "
            f"evidence IDs are: {narrative_guide}. Preserve the "
            "original source columns when the user asks for "
            "a complete table of up to eight columns. Split only a table wider than eight columns "
            "into compact tables that repeat the identifying first column. "
            "Every table row must have the same number of cells. Put units and citations outside "
            "the table. Put the title on its own line, then a blank line, and begin every table "
            "row on a new line with a leading and trailing pipe. Write currency names instead of "
            "dollar signs inside cells, and do not use "
            "code backticks, empty bold markers, or placeholder values.",
        ]
        if fail_closed_on_invalid:
            instructions.append(
                "Write a short answer covering every requested company. Use the "
                "[APPLICATION-VALIDATED FINANCIAL FACTS] and "
                "[APPLICATION-VERIFIED CALCULATIONS] exactly for numbers. Give "
                "each company's numeric comparison in its own sentence with inline "
                "10-K citations. Cite every numeric bullet, including each calculated "
                "change and the comparison winner, with its input 10-K sources. Then "
                "give each company's requested qualitative "
                "explanation in its own sentence, citing that company's relevant "
                "narrative source inline. Narrative evidence IDs by group: "
                f"{narrative_guide}. A citation list at the end is not enough. "
                "Only describe a factor as causing a historical change if the "
                "source explicitly says so; otherwise identify it as related "
                "commentary or say the retrieved evidence does not establish a "
                "cause. Do not invent a reason to fill a missing explanation."
            )
        for attempt, instruction in enumerate(instructions, start=1):
            if attempt == 3:
                instruction += f" Previous failure: {last_validation.reason}."
            if attempt > 1 and last_validation.reason.startswith(
                "requested_revenue_change_mismatch:"
            ):
                expected_change = last_validation.reason.split(":", 1)[1]
                instruction = (
                    f"The earlier answer mixed metrics. The retrieved text reports "
                    f"{expected_change} for the exact revenue metric in the question. "
                    "Return one sentence stating only that metric's change, the fiscal year, "
                    f"and an inline citation from these IDs: {allowed_ids}. Do not add "
                    "drivers, another percentage, or a broader segment metric."
                )
            try:
                answer, declared_ids = self._invoke(
                    question=question,
                    bundle=bundle,
                    history=history,
                    generation_instruction=instruction,
                )
            except Exception as error:
                error_type = type(error).__name__
                previews.append(f"<{error_type}: structured answer was not produced>")
                last_validation = AnswerValidation(
                    False,
                    UNVERIFIABLE_RESPONSE,
                    f"generation_error:{error_type}",
                )
                continue
            previews.append(self._preview(answer))
            last_validation = validate_answer_payload(
                answer,
                bundle.sources,
                declared_ids,
                require_markdown_table=wants_table,
            )
            if last_validation.valid:
                coverage_error = self._narrative_coverage_error(
                    last_validation.source_ids,
                    bundle.sources,
                )
                if coverage_error:
                    last_validation = AnswerValidation(
                        False,
                        UNVERIFIABLE_RESPONSE,
                        coverage_error,
                    )
                    continue
            if (
                last_validation.valid
                and last_validation.reason == "valid_structured_ids_appended"
                and (fail_closed_on_invalid or self._compound_answer_requires_inline_citations(answer))
            ):
                last_validation = AnswerValidation(
                    False,
                    UNVERIFIABLE_RESPONSE,
                    "compound_answer_requires_inline_citations",
                )
                continue
            if (
                last_validation.valid
                and (fail_closed_on_invalid or attempt == 1)
                and self._compound_answer_requires_inline_citations(answer)
                and self._uncited_substantive_bullet(answer)
            ):
                last_validation = AnswerValidation(
                    False,
                    UNVERIFIABLE_RESPONSE,
                    "uncited_substantive_bullet",
                )
                continue
            if last_validation.valid and (fail_closed_on_invalid or attempt == 1):
                transcript_error = self._requested_transcript_citation_error(
                    question,
                    answer,
                    bundle.sources,
                )
                if transcript_error:
                    last_validation = AnswerValidation(
                        False,
                        UNVERIFIABLE_RESPONSE,
                        transcript_error,
                    )
                    continue
            if last_validation.valid and fail_closed_on_invalid:
                temporal_error = self._forward_looking_as_historical_cause_error(answer)
                if temporal_error:
                    last_validation = AnswerValidation(
                        False,
                        UNVERIFIABLE_RESPONSE,
                        temporal_error,
                    )
                    continue
            if last_validation.valid:
                if last_validation.answer != INSUFFICIENT_EVIDENCE_RESPONSE:
                    table_check = validate_table_answer(
                        question, last_validation.answer, bundle.sources
                    )
                    if not table_check.valid:
                        last_validation = AnswerValidation(
                            False, UNVERIFIABLE_RESPONSE, table_check.reason
                        )
                        continue
                    change_check = validate_revenue_change(
                        question, last_validation.answer, bundle.sources
                    )
                    if not change_check.valid:
                        last_validation = AnswerValidation(
                            False, UNVERIFIABLE_RESPONSE, change_check.reason
                        )
                        continue
                final_answer = last_validation.answer
                if final_answer != INSUFFICIENT_EVIDENCE_RESPONSE:
                    final_answer = expand_citations(final_answer, bundle.sources)
                return GenerationResult(
                    final_answer,
                    attempt,
                    last_validation.reason,
                    tuple(previews),
                )

        # A confirmed wrong financial value should not be replaced with a
        # potentially unrelated extractive excerpt.
        if last_validation.reason == "requested_table_value_missing_from_answer" or (
            last_validation.reason.startswith("requested_revenue_change_mismatch:")
        ):
            return GenerationResult(
                INSUFFICIENT_EVIDENCE_RESPONSE,
                len(instructions),
                last_validation.reason,
                tuple(previews),
            )

        if fail_closed_on_invalid:
            return GenerationResult(
                INSUFFICIENT_EVIDENCE_RESPONSE,
                len(instructions),
                f"generation_validation_failed:{last_validation.reason}",
                tuple(previews),
            )

        fallback = self._extractive_fallback(bundle)
        if fallback != UNVERIFIABLE_RESPONSE:
            fallback = expand_citations(fallback, bundle.sources)
            reason = f"extractive_fallback_after:{last_validation.reason}"
        else:
            reason = last_validation.reason
        return GenerationResult(fallback, len(instructions), reason, tuple(previews))

    def generate(
        self,
        question: str,
        bundle: RetrievalBundle,
        *,
        history: list[dict] | None = None,
        wants_table: bool = False,
        wants_complete_table: bool = False,
    ) -> str:
        """Backward-compatible convenience API returning only the final answer text."""
        return self.generate_with_trace(
            question,
            bundle,
            history=history,
            wants_table=wants_table,
            wants_complete_table=wants_complete_table,
        ).answer
