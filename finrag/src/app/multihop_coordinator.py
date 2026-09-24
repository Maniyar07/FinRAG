"""Coordinate planning, evidence execution, and grounded multi-hop answers."""

from __future__ import annotations

import re
from collections.abc import Callable

from src.app.response_composer import (
    verified_calculation_partial,
    verified_calculation_text,
)
from src.generation.answer_generator import AnswerGenerator
from src.generation.answer_guardrails import (
    INSUFFICIENT_EVIDENCE_RESPONSE,
    UNVERIFIABLE_RESPONSE,
)
from src.generation.citations import expand_citations
from src.generation.narrative_quotes import NarrativeQuoteSelector
from src.orchestration.executor import MultiHopExecutor
from src.orchestration.models import EvidenceType
from src.orchestration.planner import MultiHopPlanner
from src.schemas import ChatResult, Decision, Scope
from src.tools.document_search import DocumentSearchTool


class MultiHopCoordinator:
    """Run a multi-hop plan without owning retrieval or generation resources."""

    def __init__(
        self,
        *,
        planner: MultiHopPlanner,
        executor: MultiHopExecutor,
        document_search_tool: DocumentSearchTool,
        generator: AnswerGenerator,
        narrative_quote_selector: NarrativeQuoteSelector | None,
        record: Callable[[dict], None],
        retrieval_source_trace: Callable[[dict], dict],
    ) -> None:
        self.multihop_planner = planner
        self.multihop_executor = executor
        self.document_search_tool = document_search_tool
        self.generator = generator
        self.narrative_quote_selector = narrative_quote_selector
        self._record = record
        self._retrieval_source_trace = retrieval_source_trace

    def run(
        self,
        *,
        question: str,
        scope: Scope,
        inherited_fields: tuple[str, ...],
        history: list[dict] | None,
        query_expansions: tuple[str, ...],
        wants_table: bool,
        trace: dict,
    ) -> ChatResult:
        planner = getattr(self, "multihop_planner", None)
        executor = getattr(self, "multihop_executor", None)
        if planner is None or executor is None:
            raise RuntimeError("Multi-hop orchestration is enabled but not initialized.")

        try:
            plan = planner.plan(question=question, permitted_scope=scope)
            trace["multihop"]["plan"] = plan.model_dump(mode="json")
            execution = executor.execute(
                plan,
                permitted_scope=scope,
                query_expansions=query_expansions,
            )
            trace["multihop"]["execution"] = execution.trace
            trace["multihop"]["issues"] = list(execution.issues)
        except Exception as error:
            trace["error_type"] = type(error).__name__
            trace["error_detail"] = " ".join(str(error).split())[:500]
            self._record({**trace, "decision": Decision.ERROR.value})
            return ChatResult(
                Decision.ERROR,
                "The multi-step document analysis could not be completed. Please try "
                "again or ask a narrower comparison question.",
                scope,
                inherited_fields=inherited_fields,
                trace=trace,
            )

        bundle = execution.bundle
        trace["retrieval"] = {
            "tool": self.document_search_tool.name,
            "mode": "multi_hop",
            "candidate_count": bundle.candidate_count,
            "source_ids": [source["id"] for source in bundle.sources],
            "source_groups": [
                [source["ticker"], source["fiscal_year"], source["doc_type"]]
                for source in bundle.sources
            ],
            "sources": [
                self._retrieval_source_trace(source) for source in bundle.sources
            ],
        }

        numeric_text = verified_calculation_text(execution, question)
        planned_calculations = len(getattr(plan, "calculations", ()))
        calculations_complete = bool(planned_calculations) and (
            len(execution.calculations) == planned_calculations
        )
        if not execution.complete and numeric_text is not None and not calculations_complete:
            limitations = tuple(
                f"A requested evidence hop was not completed: {issue}."
                for issue in execution.issues
            )
            answer = verified_calculation_partial(
                execution, question, limitations=limitations
            )
            trace["generation"] = {
                "mode": "verified_partial_calculations",
                "verified_calculation_partial": True,
            }
            result = ChatResult(
                Decision.INSUFFICIENT_EVIDENCE,
                answer or INSUFFICIENT_EVIDENCE_RESPONSE,
                scope,
                bundle.sources,
                inherited_fields,
                trace,
            )
            self._record({**trace, "decision": result.decision.value})
            return result
        if not execution.complete and numeric_text is None:
            result = ChatResult(
                Decision.INSUFFICIENT_EVIDENCE,
                INSUFFICIENT_EVIDENCE_RESPONSE,
                scope,
                bundle.sources,
                inherited_fields,
                trace,
            )
            self._record({**trace, "decision": result.decision.value})
            return result

        if numeric_text is not None:
            result = self._compose_verified_result(
                question=question,
                plan=plan,
                execution=execution,
                numeric_text=numeric_text,
                scope=scope,
                inherited_fields=inherited_fields,
                trace=trace,
            )
            self._record({**trace, "decision": result.decision.value})
            return result

        try:
            generation = self.generator.generate_with_trace(
                question,
                bundle,
                history=history,
                wants_table=wants_table,
                wants_complete_table=False,
                fail_closed_on_invalid=True,
            )
            answer = generation.answer
            trace["generation"] = {
                "attempts": generation.attempts,
                "validation_reason": generation.validation_reason,
                "raw_output_previews": list(generation.raw_output_previews),
            }
        except Exception as error:
            trace["error_type"] = type(error).__name__
            self._record({**trace, "decision": Decision.ERROR.value})
            return ChatResult(
                Decision.ERROR,
                "The answer model could not convert the verified multi-step evidence "
                "into an answer; please try again.",
                scope,
                bundle.sources,
                inherited_fields,
                trace,
            )

        decision = (
            Decision.VALIDATION_FAILED
            if answer == UNVERIFIABLE_RESPONSE
            or generation.validation_reason.startswith("generation_validation_failed:")
            else Decision.INSUFFICIENT_EVIDENCE
            if answer == INSUFFICIENT_EVIDENCE_RESPONSE
            else Decision.ANSWERED
        )
        if decision != Decision.ANSWERED:
            partial = verified_calculation_partial(execution, question)
            if partial is not None:
                answer = partial
                trace["generation"]["verified_calculation_partial"] = True
        result = ChatResult(
            decision,
            answer,
            scope,
            bundle.sources,
            inherited_fields,
            trace,
        )
        self._record({**trace, "decision": result.decision.value})
        return result

    def _compose_verified_result(
        self,
        *,
        question: str,
        plan: object,
        execution: object,
        numeric_text: str,
        scope: Scope,
        inherited_fields: tuple[str, ...],
        trace: dict,
    ) -> ChatResult:
        """Combine deterministic calculations with extractive narrative evidence."""
        bundle = execution.bundle
        requirements = [
            item
            for item in plan.requirements
            if item.evidence_type == EvidenceType.NARRATIVE
        ]
        missing: list[str] = []
        quote_ids: list[str] = []
        quote_texts: list[str] = []
        quote_sections: list[str] = []
        selector = getattr(self, "narrative_quote_selector", None)

        for requirement in requirements:
            try:
                quotes = (
                    selector.select(
                        question=question,
                        task=requirement.question,
                        sources=bundle.sources,
                        requirement_ids={requirement.requirement_id},
                    )
                    if selector is not None
                    else ()
                )
            except Exception as error:
                trace["narrative_error_type"] = type(error).__name__
                quotes = ()
            by_group = {(quote.ticker, quote.fiscal_year): quote for quote in quotes}
            historical_reason = bool(
                re.search(r"\b(?:reasons?|drivers?)\b", requirement.question, re.I)
            )
            for group in requirement.groups:
                quote = by_group.get(group.key)
                if quote is None:
                    missing.append(
                        f"{group.ticker} {group.fiscal_year} {requirement.document_type}"
                    )
                    continue
                quote_ids.append(quote.source_id)
                quote_texts.append(quote.text)
                if quote.factors:
                    factors = "\n".join(
                        f"- {label}: {rate} percentage points [{quote.source_id}]"
                        for label, rate in quote.factors
                    )
                    quote_sections.append(
                        f"#### {quote.ticker} {quote.fiscal_year} reconciliation\n\n{factors}"
                    )
                else:
                    quote_sections.append(
                        f"- **{quote.ticker} {quote.fiscal_year} {quote.doc_type}:** "
                        f"{quote.text} [{quote.source_id}]"
                    )
                if historical_reason and not re.search(
                    r"\b(?:attribut\w*|because|driven by|due to|primarily|reflect\w*)\b",
                    quote.text,
                    re.I,
                ):
                    missing.append(
                        f"a direct historical cause for {group.ticker} {group.fiscal_year}"
                    )

        if re.search(
            r"\b(?:directly\s+connect|connection)\b.{0,100}\bchanges?\b",
            question,
            re.I,
        ) and quote_ids:
            direct_cause = all(
                re.search(
                    r"\b(?:attribut\w*|because|driven by|due to|primarily|reflect\w*)\b",
                    text,
                    re.I,
                )
                for text in quote_texts
            )
            citations = "".join(f"[{item}]" for item in dict.fromkeys(quote_ids))
            if direct_cause:
                quote_sections.append(
                    "The retrieved passages directly attribute the reported changes "
                    f"to the discussed factors. {citations}"
                )
            else:
                quote_sections.append(
                    "The retrieved risk passages describe possible business or revenue "
                    "effects, but do not establish that those risks caused the calculated "
                    f"historical changes. {citations}"
                )

        sections = [numeric_text]
        if quote_sections:
            sections.extend(("### Source-supported explanation", *quote_sections))
        limitations = list(
            dict.fromkeys(
                f"No directly supported explanation was found for {item}; no "
                "explanation was inferred."
                for item in missing
            )
        )
        if not execution.complete:
            limitations.extend(
                f"A requested evidence hop was not completed: {issue}."
                for issue in execution.issues
            )
        if limitations:
            sections.extend(
                ("### Evidence limitations", "\n".join(f"- {item}" for item in limitations))
            )
        answer = expand_citations("\n\n".join(sections), bundle.sources)
        trace["generation"] = {
            "mode": "verified_calculations_and_exact_quotes",
            "quote_source_ids": quote_ids,
            "missing_narrative": missing,
            "verified_calculation_partial": bool(missing or not execution.complete),
        }
        return ChatResult(
            Decision.ANSWERED,
            answer,
            scope,
            bundle.sources,
            inherited_fields,
            trace,
        )
