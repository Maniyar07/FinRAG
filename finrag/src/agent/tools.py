"""Request-scoped LangChain tools for financial filing research."""

from __future__ import annotations

import json
import hashlib
import re
from typing import Callable, Literal, get_args

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import (
    AGENT_MAX_ACTIONS,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_PARENTS,
    MAX_CONTEXT_TOKENS,
)
from src.financial.calculator import CalculationRequest, FinancialCalculator
from src.financial.fact_pipeline import FinancialFactPipeline
from src.financial.metric_terms import (
    canonicalize_financial_metric,
    is_metric_match,
    normalize_financial_term,
)
from src.financial.coverage import fact_coverage, fact_matches_requirement
from src.financial.models import FactValidationResult, RejectedFinancialFact, ValidatedFinancialFact, ValueType
from src.generation.evidence_answer import RequirementAnswer, render_evidence_answer, resolve_evidence_answers
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.structured_lookup import StructuredDocumentLookup
from src.retrieval.token_budget import count_tokens
from src.runtime_errors import tool_error_response
from src.schemas import RetrievalBundle, Scope


DocumentType = Literal["10K", "TRANSCRIPT"]
StatementType = Literal[
    "income_statement",
    "balance_sheet",
    "cash_flow_statement",
]
CalculationOperationName = Literal[
    "absolute_change",
    "percentage_change",
    "percentage_point_change",
    "basis_point_change",
    "difference",
    "ratio",
    "percentage_of",
    "sum",
    "product",
    "cagr",
    "absolute_value",
    "percent_complement",
]

TICKER_RE = re.compile(r"^[A-Z0-9.]{1,10}$")
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


class StrictToolInput(BaseModel):
    """Base schema: reject extra keys and Pydantic's normal type coercions."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    @model_validator(mode="before")
    @classmethod
    def omit_injected_runtime_during_schema_validation(cls, value: object) -> object:
        """Keep LangGraph's injected runtime out of the strict public schema.

        LangChain validates the raw tool arguments before forwarding injected
        arguments to the Python function. Remove only an actual ToolRuntime
        object for that validation; unexpected user-supplied keys remain errors.
        """
        if isinstance(value, dict) and isinstance(value.get("runtime"), ToolRuntime):
            value = dict(value)
            value.pop("runtime")
        return value


class SearchFilingsInput(StrictToolInput):
    ticker: str = Field(min_length=1, max_length=10)
    filing_year: str = Field(
        pattern=r"^(?:19|20)\d{2}$",
        description=(
            "Fiscal year of the source filing or transcript. A comparative value "
            "inside that source may have a different reported period."
        ),
    )
    doc_type: DocumentType
    query: str = Field(min_length=3, max_length=6_000)
    unresolved_requirement_ids: list[str] = Field(
        default_factory=list,
        max_length=40,
        description="Planned requirement IDs that the previous result did not satisfy; omit on the first search.",
    )

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("ticker must be a string")
        ticker = value.strip().upper()
        if not TICKER_RE.fullmatch(ticker):
            raise ValueError("ticker must be a valid symbol")
        return ticker

    @field_validator("filing_year", mode="before")
    @classmethod
    def validate_filing_year(cls, value: object) -> str:
        if not isinstance(value, str) or not YEAR_RE.fullmatch(value.strip()):
            raise ValueError(
                "filing_year must be a four-digit year from 1900 through 2099"
            )
        return value.strip()

    @field_validator("doc_type", mode="before")
    @classmethod
    def normalize_doc_type(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("doc_type must be a string")
        doc_type = value.strip().upper().replace("-", "")
        if doc_type not in {"10K", "TRANSCRIPT"}:
            raise ValueError("doc_type must be 10K or TRANSCRIPT")
        return doc_type


class PlannedRequirement(StrictToolInput):
    object_id: str | None = Field(default=None, pattern=r"^o[1-9][0-9]*$",
        description="Reference the actual user-requested object; rows and columns of one source table share its object ID.")
    subject: str | None = Field(default=None, min_length=1, max_length=160,
        description="Resolved asset/business/product scope shared with related observations; null for company-wide metrics. Observations such as backlog or guidance are not business subjects.")
    requirement_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$",
        description="Short unique ID such as r1 or r2. Each company/period/metric/qualifier requirement needs its own ID; do not encode long metric names.")
    ticker: str = Field(min_length=1, max_length=10)
    source_year: str = Field(pattern=r"^(?:19|20)\d{2}$")
    doc_type: DocumentType
    kind: Literal["fact", "narrative", "statement"] = Field(
        default="fact", description="Classify the user request first: narrative for reported changes, lists, status or explanations; fact for a single numeric value or calculator operand; statement for ANY explicitly requested source table, including primary statements, footnote tables and schedules. Preserve its requested title/subject and qualifiers. An explanation or value found in a table is not a request for the full table.",
    )
    reported_period: str | None = Field(default=None, max_length=80,
        description="Required for numeric facts: reported fiscal period WITH year, e.g. FY2025 or Q4 2025. A source_year does not replace this field; use the period stated in the question.")
    metric: str = Field(min_length=2, max_length=240,
        description="One requested financial/product subject, retaining business/segment context. Do not prefix company, source or year (separate typed fields); retain names belonging to an actual product/segment. Independently requested observations need separate requirements.")
    retrieval_query: str | None = Field(default=None, max_length=240,
        description="Concise search terms for this subject, including equivalent financial terminology where useful. Preserve business context; do not supply an answer or guess numerical values.")
    evidence_question: str | None = Field(default=None, max_length=500,
        description="Standalone question for this requirement, preserving yes/no polarity and every requested detail. It must ask only what this source can establish, not the entire multi-source question.")
    qualifiers: list[str] = Field(default_factory=list, max_length=12,
        description="Evidence qualifiers such as geography, business/product scope, segment or accounting basis. Inherit the original business/product subject in later clauses even when the source year changes. Put dates and periods in reported_period, not qualifiers.")
    value_type: ValueType | None = Field(default=None,
        description="Type when explicit in the question: currency, percent, basis_points, per_share, ratio, count, or number. Leave null if units are ambiguous; retrieved evidence establishes the type. Never infer percent just from 'margin'. Amounts and growth rates are distinct quantities.")

    @field_validator("metric", mode="before")
    @classmethod
    def canonicalize_metric(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return canonicalize_financial_metric(value)

    @field_validator("qualifiers")
    @classmethod
    def separate_period_qualifiers(cls, values):
        # Reported periods already have a typed field. A transition phrase is
        # not an accounting/geographic condition that must occur in a table row.
        return [value for value in values if re.sub(
            r"\b(?:from|to|through|between|and|in|as|of|at|end|ended|fiscal|calendar|year|years|quarter|annual|FY\s*\d{4}|Q[1-4]|(?:19|20)\d{2})\b|[\s,./-]",
            "", value, flags=re.I)]

    @model_validator(mode="after")
    def remove_non_evidence_qualifiers(self) -> "PlannedRequirement":
        """Keep business/accounting constraints, not output-format instructions."""
        metric_terms = normalize_financial_term(self.metric)
        presentation_terms = {
            "amount", "balance", "citation", "currency", "date", "document",
            "filing", "period", "report", "scale", "sheet", "source", "unit",
            "value",
        }
        qualifiers = []
        for value in self.qualifiers:
            if re.match(
                r"^(?:currency|scale|units?|balance[- ]sheet date|reported period|"
                r"source(?: document| filing)?|citation)\s*:",
                value.strip(),
                flags=re.I,
            ):
                continue
            terms = normalize_financial_term(value)
            terms -= set(re.findall(
                r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|"
                r"august|september|october|november|december)\b|\b\d{1,2}\b",
                value.casefold(),
            ))
            if not terms or terms.issubset(metric_terms) or terms.issubset(presentation_terms):
                continue
            qualifiers.append(value)
        object.__setattr__(self, "qualifiers", list(dict.fromkeys(qualifiers)))
        if (
            "earnings per share" in self.metric.casefold()
            and not normalize_financial_term(self.metric).intersection({
                "change", "decrease", "growth", "increase", "rate",
            })
        ):
            object.__setattr__(self, "value_type", ValueType.PER_SHARE)
        return self

    @field_validator("value_type", mode="before")
    @classmethod
    def normalize_value_type(cls, value):
        if isinstance(value, str) and value.strip().casefold() in {"", "null", "none"}:
            return None
        return ValueType(value) if isinstance(value, str) else value

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_plan_ticker(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("ticker must be a string")
        ticker = value.strip().upper()
        if not TICKER_RE.fullmatch(ticker):
            raise ValueError("ticker must be a valid symbol")
        return ticker

    @field_validator("source_year", mode="before")
    @classmethod
    def validate_plan_year(cls, value: object) -> str:
        if not isinstance(value, str) or not YEAR_RE.fullmatch(value.strip()):
            raise ValueError("source_year must be a four-digit year")
        return value.strip()


class PlannedCalculation(StrictToolInput):
    operation: CalculationOperationName = Field(description="percentage_of means X as a percentage of Y (numerator then denominator, same period allowed); ratio divides X by Y. difference subtracts second from first. absolute_change, percentage_change and CAGR take earlier then later. All binary operations require two distinct requirement IDs.")
    input_requirement_ids: list[str] = Field(min_length=1, max_length=20,
        description="Ordered distinct requirement IDs already in requirements. Temporal binary changes need two separately planned reported-period facts.")
    periods: int | None = Field(default=None, gt=0)
    decimal_places: int | None = Field(default=None, ge=0, le=12,
        description="Requested rounding for the calculated result; null when not specified.")

    @model_validator(mode="after")
    def validate_formula_shape(self) -> "PlannedCalculation":
        if len(set(self.input_requirement_ids)) != len(self.input_requirement_ids):
            raise ValueError("A calculation must reference distinct requirements.")
        try:
            CalculationRequest(
                operation=self.operation,
                input_fact_ids=tuple(self.input_requirement_ids),
                periods=self.periods,
                decimal_places=self.decimal_places,
            )
        except ValueError as error:
            raise ValueError(str(error)) from error
        return self


class RequestPlanInput(StrictToolInput):
    calculation_intent: str | None = Field(default=None, max_length=600,
        description="First distinguish source-reported observations/changes from NEW arithmetic requested by the user. Identify only the clauses requiring derived results and their operands; otherwise say no new arithmetic is requested. Stating values does not require absolute_value. Do not infer units before evidence.")
    narrative_subject: str | None = Field(default=None, min_length=2, max_length=160,
        description="ONLY the shared business/product noun phrase for a coherent narrative comparison. Exclude company, dates and requested actions such as commentary, outlook or deployment. Later clauses inherit this subject unless explicitly changed. Null for independent topics or purely numeric requests; never copy a whole requirement here.")
    requirements: list[PlannedRequirement] = Field(default_factory=list, max_length=40)
    calculations: list[PlannedCalculation] = Field(default_factory=list, max_length=20,
        description="Only NEW arithmetic requested in calculation_intent, using the declared numeric requirement IDs. Source-reported changes and stating values require no calculations. Include every requested difference, ratio or computed change, with compatible evidence-backed operands.")
    synthesis: str | None = Field(default=None, max_length=500,
        description="Requested interpretation across the planned evidence, if any. Preserve the user's synthesis question here; it is not clarification or a separate source fact.")
    clarification: str | None = Field(default=None, max_length=500,
        description="Null for a resolved request. Only ask for a missing user parameter; never place caveats, the original research question, or a requested synthesis here.")

    @model_validator(mode="after")
    def validate_plan_references(self) -> "RequestPlanInput":
        ids = [item.requirement_id for item in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement_id values must be unique.")
        if not self.clarification and not self.requirements:
            raise ValueError("A plan needs requested evidence requirements.")
        unknown = {
            requirement_id
            for calculation in self.calculations
            for requirement_id in calculation.input_requirement_ids
            if requirement_id not in set(ids)
        }
        if unknown:
            raise ValueError(
                "Calculations reference unknown requirements: "
                + ", ".join(sorted(unknown))
            )
        calculation_inputs = {
            requirement_id
            for calculation in self.calculations
            for requirement_id in calculation.input_requirement_ids
        }
        if calculation_inputs:
            object.__setattr__(self, "requirements", [
                item.model_copy(update={"kind": "fact"})
                if item.requirement_id in calculation_inputs
                else item
                for item in self.requirements
            ])
        canonical, aliases, requirements = {}, {}, []
        for item in self.requirements:
            key = (item.ticker, item.source_year, item.doc_type, item.kind, item.reported_period,
                   item.metric.casefold(), tuple(sorted(item.qualifiers)), item.value_type,
                   item.evidence_question if item.kind != "fact" else None)
            if key in canonical:
                aliases[item.requirement_id] = canonical[key]
            else:
                canonical[key] = item.requirement_id
                requirements.append(item)
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "calculations", [item.model_copy(update={"input_requirement_ids":
            [aliases.get(key, key) for key in item.input_requirement_ids]}) for item in self.calculations])
        by_id = {item.requirement_id: item for item in self.requirements}
        if any(by_id[key].kind != "fact" for calculation in self.calculations
               for key in calculation.input_requirement_ids):
            raise ValueError("Calculation inputs must reference numeric fact requirements.")
        for calculation in self.calculations:
            types = {by_id[key].value_type for key in calculation.input_requirement_ids
                     if by_id[key].value_type is not None}
            if calculation.operation not in {"product", "absolute_value", "percent_complement"} and len(types) > 1:
                raise ValueError("Calculation operands must describe compatible quantity types; do not compare a monetary amount with a percentage. "
                                 "Temporal changes need the metric's earlier and later values, not an amount paired with a change rate.")
            if (calculation.operation in {"absolute_change", "percentage_change", "cagr"}
                    and len({by_id[key].ticker for key in calculation.input_requirement_ids}) > 1):
                raise ValueError("Temporal changes require one company; use difference for a cross-company subtraction.")
        if not self.clarification and any(
            not re.search(
                r"(?<!\d)(?:19|20)\d{2}(?!\d)",
                item.reported_period or "",
            )
            for item in self.requirements if item.kind == "fact"
        ):
            raise ValueError("Numeric requirements must specify a reported period with its fiscal year.")
        ordered_calculations = list(self.calculations)
        for index, calculation in enumerate(self.calculations):
            if calculation.operation in {"absolute_change", "percentage_change", "percentage_point_change", "basis_point_change", "cagr"}:
                if len({by_id[key].reported_period for key in calculation.input_requirement_ids}) != len(calculation.input_requirement_ids):
                    raise ValueError("Temporal calculations require separate earlier and later reported-period facts. Plan their actual metric values, not the computed changes as inputs.")
                period_order = {}
                for key in calculation.input_requirement_ids:
                    reported = by_id[key].reported_period or ""
                    year = re.search(r"(?:19|20)\d{2}", reported)
                    quarter = re.search(r"\bQ([1-4])\b", reported, re.I)
                    if not year:
                        break
                    period_order[key] = (int(year.group()), int(quarter.group(1)) if quarter else 4)
                if len(period_order) == len(calculation.input_requirement_ids):
                    ordered_calculations[index] = calculation.model_copy(update={
                        "input_requirement_ids": sorted(calculation.input_requirement_ids, key=period_order.get)})
        object.__setattr__(self, "calculations", list({
            (item.operation, tuple(item.input_requirement_ids), item.periods, item.decimal_places): item for item in ordered_calculations
        }.values()))
        return self

    def complete_explicit_calculations(self, question: str) -> "RequestPlanInput":
        """Compile explicitly requested arithmetic from typed numeric requirements."""
        text = " ".join(question.casefold().split())
        operations: list[str] = []
        if re.search(r"\b(?:percentage|percent)\s+(?:increase|decrease|change)\b|\bpercentage terms?\b", text):
            operations.append("percentage_change")
        if re.search(r"\b(?:dollar|absolute)\s+(?:increase|decrease|change|difference)\b|\b(?:increase|decrease|change)\s+in\s+(?:both\s+)?dollars?\b", text):
            operations.append("absolute_change")
        if re.search(r"\b(?:calculate|compute|find|show|state)\b.{0,80}\bdifference\b", text):
            operations.append(
                "difference"
                if len({item.ticker for item in self.requirements}) > 1
                else "absolute_change"
            )
        if re.search(r"\bas\s+(?:a\s+)?percentage\s+of\b", text):
            operations.append("percentage_of")
        if re.search(r"\b(?:calculate|compute|find)\b.{0,60}\bratio\b", text):
            operations.append("ratio")
        if re.search(r"\b(?:calculate|compute|find)?\s*cagr\b|compound annual growth rate", text):
            operations.append("cagr")
        operations = list(dict.fromkeys(operations))
        if not operations:
            return self

        facts = [item for item in self.requirements if item.kind == "fact"]
        calculations = list(self.calculations)
        rounding = re.search(r"\b(\d{1,2}|one|two|three|four|five|six)\s+decimal places?\b", text)
        decimal_places = None
        if rounding:
            words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
            decimal_places = words.get(rounding.group(1), int(rounding.group(1)) if rounding.group(1).isdigit() else None)

        temporal_groups: dict[tuple[str, frozenset[str]], list[PlannedRequirement]] = {}
        for item in facts:
            temporal_groups.setdefault(
                (item.ticker, frozenset(normalize_financial_term(item.metric))), []
            ).append(item)
        temporal_pairs = []
        for items in temporal_groups.values():
            ordered = sorted(items, key=lambda item: tuple(
                int(value) for value in re.findall(r"(?:19|20)\d{2}|(?<=Q)[1-4]", item.reported_period or "")
            ) or (0,))
            periods = {item.reported_period for item in ordered}
            if len(periods) > 1:
                temporal_pairs.append((ordered[0], ordered[-1]))

        cross_company_groups: dict[tuple[frozenset[str], str], list[PlannedRequirement]] = {}
        for item in facts:
            cross_company_groups.setdefault(
                (frozenset(normalize_financial_term(item.metric)), item.reported_period or ""), []
            ).append(item)
        cross_company_pairs = [
            (items[0], items[1]) for items in cross_company_groups.values()
            if len({item.ticker for item in items}) > 1
        ]

        for operation in operations:
            pairs = (
                cross_company_pairs
                if operation == "difference" and cross_company_pairs
                else temporal_pairs
                if operation in {"absolute_change", "percentage_change", "cagr"}
                else [(facts[0], facts[1])]
                if operation in {"percentage_of", "ratio"} and len(facts) == 2
                else []
            )
            for earlier, later in pairs:
                input_ids = [earlier.requirement_id, later.requirement_id]
                periods = None
                if operation == "cagr":
                    years = [int(value) for item in (earlier, later)
                             for value in re.findall(r"(?:19|20)\d{2}", item.reported_period or "")]
                    if len(years) != 2 or years[1] <= years[0]:
                        continue
                    periods = years[1] - years[0]
                if any(item.operation == operation and item.input_requirement_ids == input_ids
                       for item in calculations):
                    continue
                calculations.append(PlannedCalculation(
                    operation=operation,
                    input_requirement_ids=input_ids,
                    periods=periods,
                    decimal_places=decimal_places,
                ))
        if not calculations:
            raise ValueError(
                "The request explicitly asks for arithmetic, but the plan does not contain compatible numeric operands."
            )
        return self.model_copy(update={
            "calculation_intent": self.calculation_intent or question,
            "calculations": calculations,
        })

    def validate_scope(self, scope: Scope, *, bind_single_source: bool = False) -> "RequestPlanInput":
        """Use one source-scope invariant during planning and tool execution."""
        if self.clarification:
            return self
        requirements = []
        for item in self.requirements:
            years = {year for ticker, year, document_type in scope.retrieval_groups
                     if ticker == item.ticker and (document_type is None or document_type == item.doc_type)}
            requirements.append(item.model_copy(update={"source_year": next(iter(years))})
                                if bind_single_source and len(years) == 1 and item.source_year not in years else item)
        plan = self.model_copy(update={"requirements": requirements})
        represented = {(item.ticker, item.source_year, item.doc_type) for item in plan.requirements}
        allowed = {(ticker, year, document_type) for ticker, year in scope.groups
                   for document_type in _scope_doc_types(scope)}
        if not represented.issubset(allowed):
            raise ValueError("The request plan exceeds the locked source scope. Source years must come from the locked groups; reported-period years can differ.")
        if {(ticker, year) for ticker, year, _ in represented} != set(scope.groups):
            raise ValueError("The plan must include a requirement for every locked company and source year.")
        if (scope.doc_type or scope.required_doc_types) and represented != {
            (ticker, year, _normalized_doc_type(document_type) or "")
            for ticker, year, document_type in scope.retrieval_groups
        }:
            raise ValueError("The plan must include a requirement for every locked source group.")
        return plan


class FinalAnswerInput(StrictToolInput):
    # Unknown presentation fields cannot influence values rendered from state.
    # Search and calculator inputs retain strict rejection of extra arguments.
    model_config = ConfigDict(extra="ignore", strict=True)
    answers: list[RequirementAnswer] = Field(default_factory=list, max_length=40,
        description="Only narrative quotes, statement selections, or fact selections needed to resolve ambiguity. Unambiguous numeric answers are assembled automatically.")
    calculation_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("answers", "calculation_ids", mode="before")
    @classmethod
    def empty_selections(cls, value):
        return [] if value is None else value


class PlanRequestInput(StrictToolInput):
    """The planner reads the original question and source scope from graph state."""
    clarification: str | None = Field(default=None, max_length=500,
        description="Normally omit. Set only to ask about unresolved user intent before retrieval.")


class EvidenceFinalizationInput(StrictToolInput):
    """The production graph selects evidence from its validated state."""
    ready: Literal[True] = Field(default=True, description="Finalize the validated request plan from stored evidence. Normally omit; no facts, prose, quotes or calculator IDs need to be repeated.")


class CalculatorInput(StrictToolInput):
    # Use fact_id_1/fact_id_2 for unary and binary formulas. The list form lets
    # SUM use the calculator's full supported input range without a wider schema.
    fact_id_1: str | None = Field(default=None, min_length=1, max_length=100)
    fact_id_2: str | None = Field(
        default=None,
        max_length=100,
        description="Second validated fact ID for a binary operation.",
    )
    fact_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="Validated fact IDs for SUM when adding more than two facts.",
    )
    operation: CalculationOperationName
    periods: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Number of years/periods elapsed. Required only when operation is cagr."
        ),
    )
    decimal_places: int | None = Field(
        default=None,
        ge=0,
        le=12,
        description="Optional result rounding requested by the user.",
    )

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "CalculatorInput":
        if self.fact_ids is not None:
            if self.fact_id_1 is not None or self.fact_id_2 is not None:
                raise ValueError("Use fact_ids or fact_id_1/fact_id_2, not both.")
            input_fact_ids = tuple(self.fact_ids)
        else:
            if self.fact_id_1 is None:
                raise ValueError("fact_id_1 is required unless fact_ids is provided.")
            input_fact_ids = (self.fact_id_1,) + (
                (self.fact_id_2,) if self.fact_id_2 is not None else ()
            )
        if (
            len(set(input_fact_ids)) != len(input_fact_ids)
            and self.operation in {
                "absolute_change", "percentage_change", "percentage_point_change",
                "basis_point_change", "difference", "ratio", "percentage_of",
                "cagr", "product",
            }
        ):
            return self
        try:
            CalculationRequest(
                operation=self.operation,
                input_fact_ids=input_fact_ids,
                periods=self.periods,
                decimal_places=self.decimal_places,
            )
        except ValueError as error:
            raise ValueError(str(error)) from error
        return self


class StatementLookupInput(StrictToolInput):
    ticker: str = Field(min_length=1, max_length=10)
    filing_year: str = Field(
        pattern=r"^(?:19|20)\d{2}$",
        description="Fiscal year of the 10-K source; the table may include earlier periods.",
    )
    statement_type: str = Field(min_length=2, max_length=240,
        description="Exact requested table title or specific subject, including qualifiers: any indexed 10-K table or schedule, not just primary financial statements. Copy the planned metric. Existing primary-statement names remain accepted.")
    unresolved_requirement_ids: list[str] = Field(default_factory=list, max_length=40)

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("ticker must be a string")
        ticker = value.strip().upper()
        if not TICKER_RE.fullmatch(ticker):
            raise ValueError("ticker must be a valid symbol")
        return ticker

    @field_validator("filing_year", mode="before")
    @classmethod
    def validate_filing_year(cls, value: object) -> str:
        if not isinstance(value, str) or not YEAR_RE.fullmatch(value.strip()):
            raise ValueError(
                "filing_year must be a four-digit year from 1900 through 2099"
            )
        return value.strip()

    @field_validator("statement_type", mode="before")
    @classmethod
    def normalize_statement_type(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("statement_type must be a string")
        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        terms = StructuredDocumentLookup._terms(normalized.replace("_", " "))
        matching = [name for name in get_args(StatementType)
                    if StructuredDocumentLookup._terms(name.replace("_", " ")) == terms - {"consolidated"}]
        return matching[0] if len(matching) == 1 else value.strip()


_SOURCE_FIELDS = (
    "id",
    "parent_id",
    "evidence_id",
    "ticker",
    "fiscal_year",
    "doc_type",
    "source",
    "section",
    "pdf_page",
    "score",
    "parent_complete",
    "evidence_mode",
)


def _json_text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _normalized_doc_type(value: str | None) -> str | None:
    if value is None:
        return None
    return value.upper().replace("-", "")


def _scope_doc_types(scope: Scope) -> frozenset[str]:
    if scope.required_doc_types:
        return frozenset(
            value.upper().replace("-", "") for value in scope.required_doc_types
        )
    if scope.doc_type:
        return frozenset({_normalized_doc_type(scope.doc_type) or ""})
    return frozenset({"10K", "TRANSCRIPT"})


def _ensure_scope_is_permitted(
    requested: Scope, permitted: Scope | None
) -> None:
    if permitted is None:
        return
    if not permitted.complete:
        raise ValueError("The permitted company and year scope must be complete")
    if not set(requested.groups).issubset(set(permitted.groups)):
        raise ValueError("Tool request exceeds the permitted company/year scope")
    if not _scope_doc_types(requested).issubset(_scope_doc_types(permitted)):
        raise ValueError("Tool request exceeds the permitted document-type scope")


def _locked_scope_from_runtime(runtime: ToolRuntime) -> Scope | None:
    """Read the server-resolved scope attached to the latest user message."""
    messages = runtime.state.get("messages", ())
    if not isinstance(messages, (list, tuple)):
        return None
    prefix = "Application-resolved scope (locked):"
    for message in reversed(messages):
        if isinstance(message, dict):
            role = str(message.get("type") or message.get("role") or "").casefold()
            content = message.get("content", "")
        else:
            role = str(getattr(message, "type", "")).casefold()
            content = getattr(message, "content", "")
        if role not in {"human", "user"}:
            continue
        if isinstance(content, list):
            content = " ".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str) or prefix not in content:
            continue
        encoded_scope = content.split(prefix, 1)[1].splitlines()[0].strip()
        try:
            payload = json.loads(encoded_scope)
            groups = tuple(
                (str(group[0]), str(group[1]))
                for group in payload["requested_groups"]
                if isinstance(group, (list, tuple)) and len(group) == 2
            )
            document_types = tuple(
                str(value).upper().replace("-", "")
                for value in payload.get("required_doc_types", ())
            )
            doc_type = payload.get("doc_type")
            if not groups:
                raise ValueError("The locked scope has no company/year groups")
            return Scope(
                requested_groups=groups,
                doc_type=(
                    str(doc_type).upper().replace("-", "")
                    if doc_type
                    else None
                ),
                required_doc_types=document_types,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("The application-resolved search scope is invalid") from error
    return None


def _latest_user_question(runtime: ToolRuntime) -> str:
    """Return the latest user question, excluding the application's scope header."""
    messages = runtime.state.get("messages", ())
    if not isinstance(messages, (list, tuple)):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict):
            role = str(message.get("type") or message.get("role") or "").casefold()
            content = message.get("content", "")
        else:
            role = str(getattr(message, "type", "")).casefold()
            content = getattr(message, "content", "")
        if role not in {"human", "user"}:
            continue
        if isinstance(content, list):
            content = " ".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        if not isinstance(content, str):
            continue
        marker = "User question:"
        if marker in content:
            return content.rsplit(marker, 1)[1].strip()
        return content.strip()
    return ""


def _request_plan_from_runtime(runtime: ToolRuntime) -> RequestPlanInput | None:
    """Load the validated plan emitted earlier in this graph invocation."""
    messages = runtime.state.get("messages", ())
    if not isinstance(messages, (list, tuple)):
        return None
    calls: dict[str, str] = {}
    results: dict[str, str] = {}
    for message in messages:
        get = message.get if isinstance(message, dict) else lambda key, default=None: getattr(message, key, default)
        kind = str(get("type", get("role", ""))).casefold()
        if kind in {"ai", "assistant"}:
            for call in get("tool_calls", ()) or ():
                call_get = call.get if isinstance(call, dict) else lambda key, default=None: getattr(call, key, default)
                if call_get("name") == "plan_financial_request" and call_get("id"):
                    calls[str(call_get("id"))] = str(call_get("name"))
        elif kind == "tool" and get("name") == "plan_financial_request":
            call_id = get("tool_call_id")
            content = get("content", "")
            if call_id and isinstance(content, str):
                results[str(call_id)] = content
    for call_id in reversed(tuple(calls)):
        content = results.get(call_id)
        if content:
            try:
                payload = json.loads(content)
                if payload.get("plan_validated"):
                    return RequestPlanInput.model_validate(payload["plan"])
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return None


def _prior_search_state(runtime: ToolRuntime) -> tuple[dict, dict, int]:
    """Read this graph run's plans, retrieved parents, token use, and tool calls."""
    messages = runtime.state.get("messages", ())
    payloads: dict[tuple[str, str, str, str], dict] = {}
    parents: dict[tuple[str, str, str], set[str]] = {}
    complete_parents: dict[tuple[str, str, str], set[str]] = {}
    refined_requirements: set[str] = set()
    searched_groups: set[tuple[str, str, str]] = set()
    chars: dict[tuple[str, str, str], int] = {}
    tokens: dict[tuple[str, str, str], int] = {}
    completed_actions = 0
    if not isinstance(messages, (list, tuple)):
        return payloads, {
            "parents": parents,
            "complete_parents": complete_parents,
            "refined_requirements": refined_requirements,
            "searched_groups": searched_groups,
            "chars": chars,
            "tokens": tokens,
        }, completed_actions

    calls: dict[str, tuple[str, dict]] = {}
    results: dict[str, tuple[str, str]] = {}
    for message in messages:
        if isinstance(message, dict):
            kind = message.get("type", message.get("role"))
            tool_calls = message.get("tool_calls", ())
            name = message.get("name")
            call_id = message.get("tool_call_id")
            content = message.get("content", "")
        else:
            kind = getattr(message, "type", None)
            tool_calls = getattr(message, "tool_calls", ())
            name = getattr(message, "name", None)
            call_id = getattr(message, "tool_call_id", None)
            content = getattr(message, "content", "")
        if str(kind).casefold() in {"ai", "assistant"}:
            for call in tool_calls or ():
                if isinstance(call, dict):
                    name, args, call_id = call.get("name"), call.get("args", {}), call.get("id")
                else:
                    call_name = getattr(call, "name", None)
                    args = getattr(call, "args", {})
                    call_id = getattr(call, "id", None)
                    name = call_name
                if name and call_id:
                    calls[str(call_id)] = (
                        str(name), args if isinstance(args, dict) else {}
                    )
        elif str(kind).casefold() == "tool":
            if isinstance(content, str) and call_id:
                results[str(call_id)] = (str(name or ""), content)

    for call_id, (name, arguments) in calls.items():
        result = results.get(call_id)
        if result and result[0] in {
            "search_financial_documents",
            "get_financial_statement",
            "financial_calculator",
            "plan_financial_request",
        }:
            completed_actions += 1
        if not result or result[0] not in {
            "search_financial_documents", "get_financial_statement"
        }:
            continue
        try:
            payload = json.loads(result[1])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        ticker = str(payload.get("ticker", arguments.get("ticker", ""))).upper()
        year = str(payload.get("filing_year", arguments.get("filing_year", "")))
        doc_type = _normalized_doc_type(payload.get("doc_type", arguments.get("doc_type", ""))) or ""
        group = (ticker, year, doc_type)
        context = payload.get("context")
        if not isinstance(context, str):
            continue
        if group in searched_groups:
            refined_requirements.update(arguments.get("unresolved_requirement_ids") or ())
        searched_groups.add(group)
        chars[group] = chars.get(group, 0) + (len(context) if isinstance(context, str) else 0)
        tokens[group] = tokens.get(group, 0) + int(
            payload.get("context_tokens") or count_tokens(context)
            if isinstance(context, str)
            else 0
        )
        for source in payload.get("sources", ()):
            if isinstance(source, dict):
                parent_id = str(source.get("parent_id") or source.get("id") or "")
                if parent_id:
                    parents.setdefault(group, set()).add(parent_id)
                    if source.get("parent_complete") is True:
                        complete_parents.setdefault(group, set()).add(parent_id)
        if name == "search_financial_documents":
            key = (
                str(arguments.get("ticker", "")).upper(),
                str(arguments.get("filing_year", "")),
                _normalized_doc_type(arguments.get("doc_type", "")) or "",
                str(arguments.get("query", "")).strip().casefold(),
            )
            payloads[key] = payload
    return payloads, {
        "parents": parents,
        "complete_parents": complete_parents,
        "refined_requirements": refined_requirements,
        "searched_groups": searched_groups,
        "chars": chars,
        "tokens": tokens,
    }, completed_actions


def _parallel_group_call_count(
    runtime: ToolRuntime,
    *,
    ticker: str,
    filing_year: str,
    doc_type: str,
    request_scope: Scope,
) -> int:
    """Elect the first same-group search to retrieve all planned requirements."""
    call_id = getattr(runtime, "tool_call_id", None)
    messages = runtime.state.get("messages", ())
    if not call_id or not isinstance(messages, (list, tuple)):
        return 1
    for message in reversed(messages):
        calls = (
            message.get("tool_calls", ())
            if isinstance(message, dict)
            else getattr(message, "tool_calls", ())
        )
        if not any(
            str(call.get("id")) == str(call_id)
            if isinstance(call, dict)
            else str(getattr(call, "id", "")) == str(call_id)
            for call in calls or ()
        ):
            continue
        matching_call_ids = []
        for call in calls or ():
            if isinstance(call, dict):
                name, args = call.get("name"), call.get("args", {})
            else:
                name, args = getattr(call, "name", None), getattr(call, "args", {})
            if (
                name == "search_financial_documents"
                and isinstance(args, dict)
                and str(args.get("ticker", "")).upper() == ticker
                and _normalized_doc_type(args.get("doc_type", "10K" if name == "get_financial_statement" else None)) == doc_type
            ):
                requested_year = str(args.get("filing_year", ""))
                allowed_years = {
                    year
                    for group_ticker, year, group_type in request_scope.retrieval_groups
                    if group_ticker == ticker
                    and (
                        group_type is None
                        or _normalized_doc_type(group_type) == doc_type
                    )
                }
                effective_year = (
                    requested_year
                    if requested_year in allowed_years or len(allowed_years) != 1
                    else next(iter(allowed_years))
                )
                if effective_year == filing_year:
                    matching_call_ids.append(str(call.get("id") if isinstance(call, dict) else getattr(call, "id", "")))
        if str(call_id) not in matching_call_ids:
            return 1
        return len(matching_call_ids) if str(call_id) == matching_call_ids[0] else 0
    return 1


def _fact_summaries(
    facts: dict[str, ValidatedFinancialFact],
) -> list[dict[str, str]]:
    """Return compact, validated fact references that let the agent recover."""
    summaries = [
        {
            "fact_id": fact.fact_id,
            "ticker": fact.ticker,
            "metric": fact.metric,
            "period": fact.period,
            "raw_value": fact.raw_value,
            "source_id": fact.source_id,
            "value_type": fact.value_type.value,
            "normalized_unit": fact.normalized_unit,
            "currency": fact.currency,
            "scale": fact.scale,
            "row_label": fact.row_label,
            "column_label": fact.column_label,
        }
        for fact in facts.values()
    ]
    summaries.sort(
        key=lambda item: (
            item["ticker"],
            item["period"],
            item["metric"].casefold(),
            item["fact_id"],
        )
    )
    return summaries[:40]


def _source_metadata(source: dict) -> dict[str, object]:
    return {
        key: source[key]
        for key in _SOURCE_FIELDS
        if key in source and source[key] is not None
    }


def _bundle_payload(
    bundle: RetrievalBundle,
    *,
    ticker: str,
    filing_year: str,
    doc_type: str,
    facts: tuple[ValidatedFinancialFact, ...],
    rejected_facts: tuple[RejectedFinancialFact, ...],
) -> dict[str, object]:
    group = f"{ticker}|{filing_year}|{doc_type.upper().replace('-', '')}"
    source_id_map = {}
    for source in bundle.sources:
        source_id = str(source.get("id") or "")
        if not source_id:
            continue
        stable_source = source.get("evidence_id") or source.get("parent_id") or source.get("source_hash") or source_id
        identity = f"{group}|{stable_source}"
        digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
        source_id_map[source_id] = f"S{int.from_bytes(digest, 'big')}"
    sources = []
    for source in bundle.sources:
        metadata = _source_metadata(source)
        source_id = str(source.get("id") or "")
        if source_id in source_id_map:
            metadata["id"] = source_id_map[source_id]
        sources.append(metadata)
    context = re.sub(
        r"\[SOURCE\s+(S\d+)(?=\s*\|)",
        lambda match: f"[SOURCE {source_id_map.get(match.group(1), match.group(1))}",
        bundle.context,
    )
    remapped_facts = [
        fact.model_copy(
            update={"source_id": source_id_map.get(fact.source_id, fact.source_id)}
        ).model_dump(mode="json")
        for fact in facts
    ]
    remapped_rejections = [
        rejected.model_copy(
            update={
                "source_id": source_id_map.get(
                    rejected.source_id, rejected.source_id
                )
            }
        ).model_dump(mode="json")
        for rejected in rejected_facts
    ]
    return {
        "ticker": ticker,
        "filing_year": filing_year,
        "doc_type": doc_type,
        "context": context,
        "context_tokens": count_tokens(context),
        "sources": sources,
        "candidate_count": bundle.candidate_count,
        "covered_groups": [list(group) for group in bundle.covered_groups],
        "retrieval_diagnostics": bundle.diagnostics,
        "validated_facts": remapped_facts,
        "rejected_fact_count": len(remapped_rejections),
        "rejected_facts": remapped_rejections,
    }


def create_financial_tools(
    retriever: HybridRetriever,
    calculator: FinancialCalculator,
    *,
    permitted_scope: Scope | None = None,
    fact_pipeline: FinancialFactPipeline | None = None,
    plan_builder: Callable | None = None,
) -> list[BaseTool]:
    """Create request tools over the existing retriever and calculator.

    Set ``permitted_scope`` to enforce the server-resolved user scope. Validated
    facts are recovered from this conversation's LangGraph tool messages, so a
    compiled graph does not share mutable fact state between threads.
    """
    pipeline = fact_pipeline or FinancialFactPipeline()
    parent_store = getattr(getattr(retriever, "context_builder", None), "parent_store", None)
    structured_lookup = StructuredDocumentLookup(parent_store.root) if parent_store is not None else None

    @tool("plan_financial_request", args_schema=PlanRequestInput if plan_builder else RequestPlanInput)
    def plan_financial_request(
        requirements: list[PlannedRequirement] | None = None,
        calculations: list[PlannedCalculation] | None = None,
        clarification: str | None = None,
        *,
        runtime: ToolRuntime,
    ) -> str:
        """Build the request plan from the original question and locked scope. Call without arguments."""
        existing = _request_plan_from_runtime(runtime)
        if existing is not None:
            return _json_text({"plan_validated": True, "plan": existing.model_dump(mode="json")})
        try:
            locked = _locked_scope_from_runtime(runtime) or permitted_scope
            if locked is None and not clarification:
                raise ValueError("The application did not provide a locked source scope.")
            if plan_builder is not None and not clarification:
                plan = RequestPlanInput.model_validate(plan_builder(
                    _latest_user_question(runtime), locked,
                ))
                clarification = plan.clarification
            else:
                plan = RequestPlanInput(requirements=requirements or [], calculations=calculations or [],
                                        clarification=clarification)
            if not clarification:
                if locked is None:
                    raise ValueError("The application did not provide a locked source scope.")
                plan.validate_scope(locked)
            return _json_text({
                "plan_validated": True,
                "plan": plan.model_dump(mode="json"),
                "source_groups": [
                    {"ticker": ticker, "filing_year": year, "doc_type": doc_type,
                     "requirement_ids": [item.requirement_id for item in plan.requirements
                                         if (item.ticker, item.source_year, item.doc_type) == (ticker, year, doc_type)]}
                    for ticker, year, doc_type in dict.fromkeys(
                        (item.ticker, item.source_year, item.doc_type) for item in plan.requirements)
                ],
                "next_step": (
                    "Ask the user this clarification before retrieval."
                    if clarification
                    else "Retrieve once per planned source group and satisfy each requirement."
                ),
            })
        except ValueError as error:
            return _json_text({
                "plan_validated": False,
                "error": " ".join(str(error).split())[:300],
                "error_code": "invalid_request_plan", "retryable": False,
                "plan_attempt_exhausted": plan_builder is not None,
                "instruction": "Planning failed validation; do not retrieve or calculate from an invalid plan.",
            })
        except Exception as error:
            failure = tool_error_response("Request planning", error)
            attempts = sum((message.get("name") if isinstance(message, dict) else getattr(message, "name", None)) == "plan_financial_request"
                           for message in runtime.state.get("messages", ()))
            return _json_text({"plan_validated": False, **failure,
                               "plan_attempt_exhausted": not failure["retryable"] or attempts >= 1})

    def search(
        *,
        ticker: str,
        filing_year: str,
        doc_type: str,
        query: str,
        fact_question: str,
        unresolved_requirement_ids: tuple[str, ...] = (),
        preserve_all: bool = False,
        statement_subject: str | None = None,
        runtime: ToolRuntime | None = None,
    ) -> str:
        requested_filing_year = filing_year
        requested_query = query
        statement_type_label = statement_subject
        request_scope = None
        request_plan = _request_plan_from_runtime(runtime) if runtime is not None else None
        if runtime is not None:
            request_scope = _locked_scope_from_runtime(runtime)
            if request_scope is not None and request_plan is None:
                return _json_text({
                    "request_plan_required": True,
                    "message": "Validate the request plan before retrieving evidence.",
                })
            if request_plan is not None and request_plan.clarification:
                return _json_text({
                    "clarification_required": True,
                    "message": request_plan.clarification,
                })
            if request_scope is not None:
                allowed_years = {
                    group_year
                    for group_ticker, group_year, group_doc_type
                    in request_scope.retrieval_groups
                    if group_ticker == ticker
                    and (
                        group_doc_type is None
                        or _normalized_doc_type(group_doc_type)
                        == _normalized_doc_type(doc_type)
                    )
                }
                if filing_year not in allowed_years and len(allowed_years) == 1:
                    filing_year = next(iter(allowed_years))
        scope = Scope(
            tickers=(ticker,),
            years=(filing_year,),
            doc_type=doc_type,
            requested_groups=((ticker, filing_year),),
        )
        _ensure_scope_is_permitted(scope, permitted_scope)
        if request_scope is not None:
            _ensure_scope_is_permitted(scope, request_scope)

        retrieval_limits: dict[str, int] = {}
        if request_scope is not None:
            group_requirements = [
                item for item in request_plan.requirements
                if item.ticker == ticker
                and item.source_year == filing_year
                and _normalized_doc_type(item.doc_type) == _normalized_doc_type(doc_type)
            ]
            if (
                statement_subject is None
                and len(group_requirements) == 1
                and group_requirements[0].kind == "statement"
            ):
                statement_subject = group_requirements[0].metric
                statement_type_label = group_requirements[0].metric
                preserve_all = True
            current_doc_type = _normalized_doc_type(doc_type) or ""
            group = (ticker, filing_year, current_doc_type)
            prior_payloads, usage, completed_actions = _prior_search_state(runtime)
            prior_search = group in usage["searched_groups"]
            exact_key = (
                ticker, requested_filing_year, current_doc_type,
                requested_query.strip().casefold(),
            )
            if exact_key in prior_payloads and not unresolved_requirement_ids:
                previous = prior_payloads[exact_key]
                return _json_text({
                    "ticker": ticker,
                    "filing_year": filing_year,
                    "doc_type": doc_type,
                    "already_retrieved": True,
                    "context_budget_exhausted": True,
                    "message": "This exact search already ran. Use its earlier evidence and validated facts.",
                    "source_ids": [
                        source.get("id") for source in previous.get("sources", ())
                        if isinstance(source, dict) and source.get("id")
                    ],
                    "validated_facts": previous.get("validated_facts", []),
                })
            planned_ids = {item.requirement_id for item in group_requirements}
            unresolved_ids = set(unresolved_requirement_ids)
            if unresolved_ids:
                known_facts, known_sources = _facts_from_tool_messages(runtime.state.get("messages", ()))
                known_coverage = fact_coverage(
                    [item.model_dump() for item in group_requirements],
                    list(known_facts.values()), list(known_sources.values()),
                )
                already_covered = unresolved_ids.intersection(known_coverage["covered_requirement_ids"])
                if already_covered:
                    return _json_text({
                        "already_covered": True, "coverage": known_coverage,
                        "message": "These requirements have validated facts. Reuse them instead of retrieving again.",
                    })
            if unresolved_ids - planned_ids:
                return _json_text({
                    "request_plan_mismatch": True,
                    "message": "A refinement may reference only requirements in this source group.",
                })
            if unresolved_ids & usage["refined_requirements"]:
                return _json_text({
                    "recovery_exhausted": True,
                    "missing_requirement_ids": sorted(unresolved_ids & usage["refined_requirements"]),
                    "message": "Each missing requirement permits one focused recovery. Finalize with the available evidence and disclose remaining gaps.",
                })
            if prior_search and not unresolved_ids:
                return _json_text({
                    "refinement_requires_missing_requirements": True,
                    "unresolved_requirement_ids": sorted(planned_ids),
                    "message": "A source group was already retrieved. Refine only if a specific planned requirement is still missing, and pass its requirement_id.",
                })
            if not prior_search:
                # Before any evidence exists, all group requirements are unresolved.
                # This is the initial group search, regardless of supplied IDs.
                unresolved_ids = set()
                unresolved_requirement_ids = ()
            if unresolved_ids:
                group_requirements = [
                    item for item in group_requirements
                    if item.requirement_id in unresolved_ids
                ]
            if not group_requirements:
                return _json_text({
                    "request_plan_mismatch": True,
                    "message": "No planned evidence requirement matches this locked source group.",
                })
            query = " ".join((query, *(
                " ".join((
                    item.metric,
                    item.reported_period or "",
                    *item.qualifiers,
                ))
                for item in group_requirements
            )))
            fact_question = _latest_user_question(runtime)
            preserve_all = preserve_all or len(group_requirements) > 1
            groups = tuple(dict.fromkeys(
                (item.ticker, item.source_year, _normalized_doc_type(item.doc_type))
                for item in request_plan.requirements
            ))
            groups = groups or request_scope.retrieval_groups or scope.retrieval_groups
            group_count = max(1, len(groups))
            group_index = next((
                index for index, (group_ticker, group_year, group_doc_type)
                in enumerate(groups)
                if group_ticker == ticker and group_year == filing_year
                and (group_doc_type is None
                     or _normalized_doc_type(group_doc_type) == current_doc_type)
            ), 0)
            if completed_actions >= AGENT_MAX_ACTIONS:
                return _json_text({
                    "ticker": ticker,
                    "filing_year": filing_year,
                    "doc_type": doc_type,
                    "action_budget_exhausted": True,
                    "message": "The request tool-action limit is reached. Answer from the validated evidence already retrieved, or state which evidence remains unavailable.",
                })

            # Divide the first-pass budget by resolved groups, then subtract all
            # context already returned for this group before allowing refinement.
            total_parent_budget = max(MAX_CONTEXT_PARENTS, group_count)
            total_char_budget = MAX_CONTEXT_CHARS
            total_token_budget = MAX_CONTEXT_TOKENS
            parent_base, parent_extra = divmod(total_parent_budget, group_count)
            char_base, char_extra = divmod(total_char_budget, group_count)
            token_base, token_extra = divmod(total_token_budget, group_count)
            group = (ticker, filing_year, current_doc_type or "")
            parent_budget = parent_base + int(group_index < parent_extra)
            char_budget = char_base + int(group_index < char_extra)
            token_budget = token_base + int(group_index < token_extra)
            sibling_calls = _parallel_group_call_count(
                runtime,
                ticker=ticker,
                filing_year=filing_year,
                doc_type=current_doc_type or "",
                request_scope=request_scope,
            )
            if sibling_calls == 0 and not prior_search:
                return _json_text({
                    "shared_source_group_search": True,
                    "message": "The first parallel search in this source group retrieves every planned requirement. Reuse its result.",
                })
            sibling_calls = 1
            parent_budget = max(1, parent_budget // sibling_calls)
            char_budget = max(1, char_budget // sibling_calls)
            token_budget //= sibling_calls
            used_parents = usage["parents"].get(group, set())
            remaining_parent_ids = usage["complete_parents"].get(group, set())
            remaining_parents = parent_budget - len(used_parents)
            remaining_chars = char_budget - usage["chars"].get(group, 0)
            remaining_tokens = token_budget - usage["tokens"].get(group, 0)
            if remaining_parents <= 0 or remaining_chars <= 0 or remaining_tokens <= 0:
                return _json_text({
                    "ticker": ticker,
                    "filing_year": filing_year,
                    "doc_type": doc_type,
                    "context_budget_exhausted": True,
                    "message": "The request evidence budget for this source group is used. Reuse the earlier evidence; do not repeat retrieval.",
                    "source_ids": sorted({
                        str(source.get("id"))
                        for payload in prior_payloads.values()
                        if str(payload.get("ticker", "")).upper() == ticker
                        and str(payload.get("filing_year", "")) == filing_year
                        and _normalized_doc_type(payload.get("doc_type")) == current_doc_type
                        for source in payload.get("sources", ())
                        if isinstance(source, dict) and source.get("id")
                    }),
                })
            retrieval_limits = {
                "max_parents": remaining_parents,
                "max_chars": remaining_chars,
                "max_tokens": remaining_tokens,
                "exclude_parent_ids": frozenset(remaining_parent_ids),
            }

        direct = structured_lookup.answer(statement_subject, scope, wants_complete_table=True) if statement_subject and structured_lookup else None
        completeness = {"complete": False, "reason": "structured_table_not_found_or_ambiguous", "truncated": False}
        bundle = None
        if direct is not None:
            source = direct.sources[0]
            context = f"[SOURCE {source['id']} | TICKER: {ticker} | FISCAL YEAR: {filing_year} | TYPE: 10K]\n{source['evidence_text']}"
            if (len(context) <= retrieval_limits.get("max_chars", MAX_CONTEXT_CHARS)
                    and count_tokens(context) <= retrieval_limits.get("max_tokens", MAX_CONTEXT_TOKENS)):
                bundle = RetrievalBundle(context, direct.sources, scope, 1)
                completeness = source["table_completeness"]
            else:
                completeness = {"complete": False, "reason": "context_budget_exceeded", "truncated": True}
        if bundle is None:
            if request_scope:
                retrieval_limits["requirement_queries"] = tuple(dict.fromkeys(
                    " ".join((item.retrieval_query or item.metric, *item.qualifiers))
                    for item in group_requirements))
            if request_scope and structured_lookup and doc_type == "10K":
                retrieval_limits["preferred_parent_ids"] = tuple(dict.fromkeys(
                    row.source["parent_id"] for item in group_requirements if item.kind == "fact"
                    for row in structured_lookup.statement_rows(
                        " ".join((item.metric, *item.qualifiers)), scope,
                        value_years=tuple(re.findall(r"(?:19|20)\d{2}", item.reported_period or "")),
                        qualifiers=tuple(item.qualifiers), include_ambiguous=True, metric=item.metric)))
            bundle = retriever.retrieve(query, scope, preserve_all=preserve_all, **retrieval_limits)
        validation = pipeline.run(
            question=fact_question,
            sources=bundle.sources,
            permitted_scope=scope,
            requirements=tuple(item.model_dump() for item in group_requirements) if request_scope else (),
        ) if not request_scope or any(item.kind == "fact" for item in group_requirements) else FactValidationResult()
        valid_facts = tuple(validation.valid_facts)

        payload = _bundle_payload(
            bundle,
            ticker=ticker,
            filing_year=filing_year,
            doc_type=doc_type,
            facts=valid_facts,
            rejected_facts=validation.rejected_facts,
        )
        if request_scope:
            previous_facts, previous_sources = _facts_from_tool_messages(runtime.state.get("messages", ()))
            all_sources = [*previous_sources.values(), *payload["sources"]]
            all_facts = [*previous_facts.values(), *(ValidatedFinancialFact.model_validate(fact) for fact in payload["validated_facts"])]
            payload["coverage"] = fact_coverage(
                [item.model_dump() for item in request_plan.requirements], all_facts, all_sources,
            )
            payload["recovery_attempted"] = bool(unresolved_requirement_ids)
        if statement_subject:
            payload["statement_completeness"] = completeness
            payload["statement_type"] = statement_type_label or statement_subject
            if direct is not None and completeness["complete"]:
                source_id = payload["sources"][0]["id"]
                payload["statement_id"] = "T" + str(source_id)
                # Send the table once as evidence. The finalizer renders it
                # from stored context instead of duplicating Markdown in history.
                payload["statement_unit"] = direct.answer.split("\n\n")[1] if len(direct.answer.split("\n\n")) > 3 else ""
        if requested_filing_year != filing_year:
            payload["scope_adjustment"] = {
                "requested_source_year": requested_filing_year,
                "locked_source_year": filing_year,
            }
        return _json_text(payload)

    @tool(args_schema=SearchFilingsInput)
    def search_financial_documents(
        ticker: str,
        filing_year: str,
        doc_type: DocumentType,
        query: str,
        unresolved_requirement_ids: list[str] | None = None,
        *,
        runtime: ToolRuntime,
    ) -> str:
        """Search an in-scope 10-K or earnings transcript and validate cited facts.

        Use returned validated fact IDs with financial_calculator. The source
        context and source metadata are returned for evidence-grounded answers.
        """
        try:
            return search(
                ticker=ticker,
                filing_year=filing_year,
                doc_type=doc_type,
                query=query,
                fact_question=_latest_user_question(runtime) or query,
                unresolved_requirement_ids=tuple(unresolved_requirement_ids or ()),
                runtime=runtime,
            )
        except Exception as error:
            return _json_text(tool_error_response("Document search", error))

    @tool("financial_calculator", args_schema=CalculatorInput)
    def financial_calculator_tool(
        *,
        fact_id_1: str | None = None,
        fact_id_2: str | None = None,
        fact_ids: list[str] | None = None,
        operation: CalculationOperationName,
        periods: int | None = None,
        decimal_places: int | None = None,
        runtime: ToolRuntime,
        enforce_action_budget: bool = True,
    ) -> str:
        """Calculate from validated facts returned earlier in this conversation."""
        input_fact_ids = tuple(fact_ids) if fact_ids is not None else (
            (fact_id_1,) + ((fact_id_2,) if fact_id_2 is not None else ())
        )
        requested_fact_ids = input_fact_ids
        request_plan = None
        if _locked_scope_from_runtime(runtime) is not None:
            request_plan = _request_plan_from_runtime(runtime)
            if request_plan is None:
                return _json_text({
                    "request_plan_required": True,
                    "message": "Validate the request plan before calculating.",
                })
        if len(set(input_fact_ids)) != len(input_fact_ids):
            return _json_text({
                "calculation_skipped": True,
                "operation": operation,
                "requested_fact_ids": list(input_fact_ids),
                "instruction": (
                    "No calculation was performed because the same fact was "
                    "supplied more than once. Use distinct validated facts for "
                    "different periods, or cite a rate directly reported by the source."
                ),
            })
        _, _, completed_actions = _prior_search_state(runtime)
        if enforce_action_budget and completed_actions >= AGENT_MAX_ACTIONS:
            return _json_text({
                "action_budget_exhausted": True,
                "message": "The request tool-action limit is reached. Answer only from calculations already returned and state any unavailable result.",
            })
        available_facts, source_metadata = _facts_from_tool_messages(
            runtime.state.get("messages", ())
        )
        missing = [fact_id for fact_id in input_fact_ids if fact_id not in available_facts]
        if missing:
            return _json_text(
                {
                    "error": "Unknown or unvalidated fact IDs.",
                    "error_code": "invalid_fact_reference",
                    "error_type": "ValueError",
                    "requested_fact_ids": list(input_fact_ids),
                    "missing_fact_ids": missing,
                    "operation": operation,
                    "retryable": False,
                    "available_validated_facts": _fact_summaries(available_facts),
                }
            )

        try:
            request = CalculationRequest(
                operation=operation,
                input_fact_ids=input_fact_ids,
                periods=periods,
                decimal_places=decimal_places,
            )
        except ValueError as error:
            return _json_text(
                {
                    "error": str(error),
                    "error_code": "invalid_calculation",
                    "error_type": type(error).__name__,
                    "error_detail": " ".join(str(error).split())[:240],
                    "operation": operation,
                    "requested_fact_ids": list(input_fact_ids),
                    "retryable": False,
                    "available_validated_facts": _fact_summaries(available_facts),
                }
            )
        if request_plan is not None:
            facts = [available_facts[fact_id] for fact_id in input_fact_ids]
            bindings = fact_coverage([item.model_dump() for item in request_plan.requirements],
                                     facts, list(source_metadata.values()))["fact_ids_by_requirement"]
            matches = [
                calculation for calculation in request_plan.calculations
                if calculation.operation == operation
                and len(calculation.input_requirement_ids) == len(input_fact_ids)
                and calculation.periods == periods
                and all(len(bindings[key]) == 1 for key in calculation.input_requirement_ids)
                and {bindings[key][0] for key in calculation.input_requirement_ids} == set(input_fact_ids)
            ]
            if len(matches) != 1:
                return _json_text({
                    "error": "This operation does not match a calculation in the validated request plan.",
                    "error_code": "unplanned_calculation",
                    "retryable": False,
                    "operation": operation,
                })
            planned = matches[0]
            input_fact_ids = tuple(bindings[key][0] for key in planned.input_requirement_ids)
            request = request.model_copy(update={"input_fact_ids": input_fact_ids,
                "decimal_places": planned.decimal_places if planned.decimal_places is not None else request.decimal_places})
        inputs = {fact_id: available_facts[fact_id] for fact_id in input_fact_ids}
        try:
            result = calculator.calculate(request, inputs)
        except ValueError as error:
            return _json_text(
                {
                    "error": str(error),
                    "error_code": "calculation_rejected",
                    "error_type": type(error).__name__,
                    "error_detail": " ".join(str(error).split())[:240],
                    "operation": operation,
                    "requested_fact_ids": list(input_fact_ids),
                    "retryable": False,
                    "available_validated_facts": _fact_summaries(available_facts),
                }
            )
        except Exception as error:
            return _json_text(tool_error_response("Financial calculation", error))
        return _json_text(
            {
                "operation": result.operation.value,
                "formula": result.formula,
                "substituted_formula": result.substituted_formula,
                "result": str(result.result),
                "result_unit": result.result_unit,
                "currency": result.currency,
                "scale": result.scale,
                "input_fact_ids": list(result.input_fact_ids),
                "requested_fact_ids": list(requested_fact_ids),
                "input_order_adjusted": requested_fact_ids != input_fact_ids,
                "source_ids": list(result.source_ids),
                "calculation_id": "C" + hashlib.blake2b(
                    _json_text([operation, input_fact_ids, periods, request.decimal_places]).encode(), digest_size=8,
                ).hexdigest(),
                "periods": periods,
                "decimal_places": request.decimal_places,
                "planned_input_requirement_ids": planned.input_requirement_ids if request_plan is not None else [],
            }
        )

    @tool(args_schema=StatementLookupInput)
    def get_financial_statement(
        ticker: str,
        filing_year: str,
        statement_type: str,
        unresolved_requirement_ids: list[str] | None = None,
        *,
        runtime: ToolRuntime,
    ) -> str:
        """Retrieve an explicitly requested complete 10-K table or schedule."""
        request_plan = _request_plan_from_runtime(runtime)
        if request_plan is not None:
            group_requirements = [
                item for item in request_plan.requirements
                if item.ticker == ticker
                and item.source_year == filing_year
                and _normalized_doc_type(item.doc_type) == "10K"
            ]
            if group_requirements and not any(
                item.kind == "statement" for item in group_requirements
            ):
                return _json_text({
                    "request_plan_mismatch": True,
                    "required_tool": "search_financial_documents",
                    "message": (
                        "This source group requests scalar facts or narrative evidence, "
                        "not a complete source table. Search the planned metric instead."
                    ),
                })
        query = statement_type.replace("_", " ")
        statement_question = (
            f"Extract the reported rows and values from the {statement_type.replace('_', ' ')} "
            f"in the source {filing_year} 10-K. Preserve every presented period column "
            "and value; do not calculate."
        )
        try:
            result = search(
                ticker=ticker,
                filing_year=filing_year,
                doc_type="10K",
                query=query,
                fact_question=statement_question,
                preserve_all=True,
                statement_subject=statement_type.replace("_", " "),
                unresolved_requirement_ids=tuple(unresolved_requirement_ids or ()),
                runtime=runtime,
            )
        except Exception as error:
            return _json_text(
                {
                    **tool_error_response("Statement lookup", error),
                    "statement_type": statement_type,
                }
            )
        payload = json.loads(result)
        payload["statement_type"] = statement_type
        return _json_text(payload)

    @tool("finalize_financial_answer", args_schema=EvidenceFinalizationInput if plan_builder is not None else FinalAnswerInput)
    def finalize_financial_answer(
        answers: list[RequirementAnswer] | None = None, calculation_ids: list[str] | None = None,
        *, runtime: ToolRuntime, ready: bool = True,
    ) -> str:
        """Validate complete plan coverage and render supported values. Final tool: one repair allowed."""
        payloads = []
        for message in runtime.state.get("messages", ()):
            name = message.get("name") if isinstance(message, dict) else getattr(message, "name", None)
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            if name and isinstance(content, str):
                try:
                    payloads.append((name, json.loads(content)))
                except json.JSONDecodeError:
                    continue
        attempts = sum(
            (message.get("name") if isinstance(message, dict) else getattr(message, "name", None))
            == "finalize_financial_answer" for message in runtime.state.get("messages", ())
        )
        automatic_results = []
        try:
            if attempts >= 2:
                raise ValueError("The answer validation repair is exhausted.")
            plan = _request_plan_from_runtime(runtime)
            if plan is None or plan.clarification:
                raise ValueError("Validate a request plan before finalizing an answer.")
            facts, sources = _facts_from_tool_messages(runtime.state.get("messages", ()))
            coverage = fact_coverage([item.model_dump() for item in plan.requirements], list(facts.values()), list(sources.values()))
            _, usage, completed_actions = _prior_search_state(runtime)
            recoverable = set(coverage["missing_requirement_ids"]) - usage["refined_requirements"]
            if (attempts == 0 and recoverable and completed_actions < AGENT_MAX_ACTIONS
                    and sum(usage["tokens"].values()) < MAX_CONTEXT_TOKENS):
                return _json_text({
                    "answer_validated": False, "recovery_required": True, "final_answer_attempts": 1,
                    "missing_requirement_ids": sorted(recoverable), "coverage": coverage,
                    "instruction": "Retrieve these missing facts once per source group, using their exact metrics and reported periods as query and their IDs as unresolved_requirement_ids. Then finalize. Do not substitute another table measure or finalize before this bounded recovery.",
                })
            calculations = {}
            for name, payload in payloads:
                if isinstance(payload, dict):
                    prior_results = [payload] if name == "financial_calculator" else payload.get("automatic_calculator_results", [])
                    for prior in prior_results:
                        if prior.get("calculation_id"):
                            calculations[prior["calculation_id"]] = prior
            statements = {payload["statement_id"]: payload for _, payload in payloads
                          if isinstance(payload, dict) and payload.get("statement_id")}
            resolved = resolve_evidence_answers(
                {**plan.model_dump(), "question": _latest_user_question(runtime),
                 "verification_feedback": [payload["validation_error"] for name, payload in payloads
                                           if name == "finalize_financial_answer" and isinstance(payload, dict) and payload.get("validation_error")]},
                [RequirementAnswer.model_validate(answer) for answer in answers or []], facts, sources, statements,
            )
            selected_ids = {answer.requirement_id: answer.fact_ids for answer in resolved}
            for planned in plan.calculations:
                completed = any(result.get("planned_input_requirement_ids") == planned.input_requirement_ids
                                and result.get("operation") == planned.operation and result.get("periods") == planned.periods
                                and result.get("decimal_places") == planned.decimal_places
                                for result in calculations.values())
                if not completed and all(selected_ids[key] for key in planned.input_requirement_ids):
                    result = json.loads(financial_calculator_tool.func(
                        fact_ids=[selected_ids[key][0] for key in planned.input_requirement_ids],
                        operation=planned.operation, periods=planned.periods, decimal_places=planned.decimal_places, runtime=runtime,
                        enforce_action_budget=False,
                    ))
                    automatic_results.append(result)
                    if result.get("calculation_id"):
                        calculations[result["calculation_id"]] = result
            if not calculation_ids:
                calculation_ids = []
                for planned in plan.calculations:
                    match = next((key for key, result in reversed(tuple(calculations.items()))
                        if result.get("planned_input_requirement_ids") == planned.input_requirement_ids
                        and result.get("operation") == planned.operation and result.get("periods") == planned.periods
                        and all(fact_id in selected_ids[requirement_id]
                                for fact_id, requirement_id in zip(result["input_fact_ids"], planned.input_requirement_ids))), None)
                    if match is not None:
                        calculation_ids.append(match)
            result = render_evidence_answer(
                {**plan.model_dump(), "question": _latest_user_question(runtime)}, resolved,
                calculation_ids or [], facts, sources, calculations, statements,
            )
            result["final_answer_attempts"] = attempts + 1
            result["automatic_calculator_results"] = automatic_results
            return _json_text(result)
        except (ValueError, KeyError) as error:
            return _json_text({
                "answer_validated": False, "validation_error": str(error),
                "automatic_calculator_results": automatic_results,
                "final_answer_attempts": attempts + 1,
                "final_answer_attempt_exhausted": attempts >= 1,
                "instruction": "Numeric coverage is automatic: omit numeric summaries and extra requirement entries. Select matching fact IDs only to resolve ambiguity; supply narrative quotes or statement IDs for those requirements. Repair once; never invent evidence.",
            })
        except Exception as error:
            return _json_text(tool_error_response("Answer verification", error))

    return [
        plan_financial_request,
        search_financial_documents,
        financial_calculator_tool,
        get_financial_statement,
        finalize_financial_answer,
    ]


def _facts_from_tool_messages(
    messages: object,
) -> tuple[dict[str, ValidatedFinancialFact], dict[str, dict]]:
    """Load validated facts with the source scope that backs each fact."""
    facts: dict[str, ValidatedFinancialFact] = {}
    source_metadata: dict[str, dict] = {}
    if not isinstance(messages, (list, tuple)):
        return facts, source_metadata

    allowed_tool_names = {
        "search_financial_documents",
        "get_financial_statement",
    }
    for message in messages:
        name = message.get("name") if isinstance(message, dict) else getattr(message, "name", None)
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if name not in allowed_tool_names:
            continue
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        evidence_blocks = {}
        for block in re.split(r"(?=^\[SOURCE\s+S\d+\s*\|)", str(payload.get("context") or ""), flags=re.M):
            header = re.match(r"^\[SOURCE\s+(S\d+)\s*\|[^\]]*\]\s*\n?", block)
            if header:
                evidence_blocks[header.group(1)] = block[header.end():].strip()
        for source in payload.get("sources", ()):
            if isinstance(source, dict) and source.get("id"):
                source_id = str(source["id"]).upper()
                evidence = str(source.get("evidence_text") or evidence_blocks.get(source_id, ""))
                previous = source_metadata.get(source_id, {}).get("evidence_text", "")
                if evidence in previous:
                    evidence = previous
                elif previous and previous not in evidence:
                    evidence = previous + "\n\n" + evidence
                source_metadata[source_id] = {
                    **source,
                    "id": source_id,
                    "evidence_text": evidence,
                    "ticker": source.get("ticker", payload.get("ticker")),
                    "fiscal_year": source.get(
                        "fiscal_year", payload.get("filing_year")
                    ),
                    "doc_type": source.get("doc_type", payload.get("doc_type")),
                }
        raw_facts = payload.get("validated_facts", ())
        if not isinstance(raw_facts, list):
            continue
        for raw_fact in raw_facts:
            fact = ValidatedFinancialFact.model_validate(raw_fact)
            existing = facts.get(fact.fact_id)
            if existing is not None and (
                existing.ticker.casefold() != fact.ticker.casefold()
                or existing.metric.casefold() != fact.metric.casefold()
                or existing.period.casefold() != fact.period.casefold()
                or existing.raw_value != fact.raw_value
                or existing.numeric_value != fact.numeric_value
                or existing.base_value != fact.base_value
                or existing.value_type != fact.value_type
                or existing.normalized_unit != fact.normalized_unit
                or existing.currency != fact.currency
                or existing.scale != fact.scale
                or (existing.accounting_basis or "").casefold()
                != (fact.accounting_basis or "").casefold()
                or existing.source_id != fact.source_id
                or (existing.row_label or "").casefold()
                != (fact.row_label or "").casefold()
                or (existing.column_label or "").casefold()
                != (fact.column_label or "").casefold()
            ):
                raise ValueError("Conflicting validated facts share a fact ID")
            facts[fact.fact_id] = fact
    return facts, source_metadata
