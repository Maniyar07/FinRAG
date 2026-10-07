from __future__ import annotations

import json
from decimal import Decimal

from src.app.langchain_react_agent import (
    AgentResearchSession,
    CalculatorToolInput,
    LangChainReActAgent,
    ResearchToolInput,
)
from src.financial.calculator import CalculationOperation
from src.financial.models import ValidatedFinancialFact, ValueType
from src.orchestration.executor import MultiHopExecutionResult
from src.orchestration.models import (
    CalculationReference,
    EvidenceGroup,
    EvidenceRequirement,
    EvidenceType,
    FactReference,
    MultiHopPlan,
    PlannedCalculation,
    RequirementResult,
)
from src.schemas import Decision, RetrievalBundle, Scope
from src.tools.financial_calculator import FinancialCalculatorTool


def _fact(*, fact_id: str, ticker: str, period: str, value: str, source_id: str):
    return ValidatedFinancialFact(
        fact_id=fact_id,
        ticker=ticker,
        metric="Total revenue",
        period=period,
        raw_value=value,
        numeric_value=Decimal(value),
        base_value=Decimal(value) * Decimal("1000000"),
        value_type=ValueType.CURRENCY,
        normalized_unit="USD",
        currency="USD",
        scale="millions",
        source_id=source_id,
        evidence_excerpt=f"Total revenue {period} {value}",
        row_label="Total revenue",
        column_label=period,
        validation_checks=("source_id", "source_scope", "exact_table_row"),
    )


class FakeExecutor:
    def __init__(self) -> None:
        self.calculator = FinancialCalculatorTool()
        self.calls = []

    def execute(self, plan, *, permitted_scope, query_expansions=()):
        del query_expansions
        self.calls.append((plan, permitted_scope))
        requirement = plan.requirements[0]
        sources = []
        facts = []
        for number, group in enumerate(requirement.groups, start=1):
            source_id = f"S{number}"
            value = "100" if group.fiscal_year == "2024" else "120"
            sources.append(
                {
                    "id": source_id,
                    "ticker": group.ticker,
                    "fiscal_year": group.fiscal_year,
                    "doc_type": requirement.document_type,
                    "source": f"{group.ticker}_{group.fiscal_year}_10K.pdf",
                    "section": "Financial statements",
                    "parent_id": f"{group.ticker}-{group.fiscal_year}",
                    "source_hash": f"hash-{group.fiscal_year}",
                    "evidence_text": f"Total revenue was {value} million.",
                    "full_evidence_text": f"Total revenue was {value} million.",
                    "evidence_requirements": [
                        {
                            "requirement_id": requirement.requirement_id,
                            "evidence_type": requirement.evidence_type.value,
                            "document_type": requirement.document_type,
                        }
                    ],
                }
            )
            if requirement.evidence_type == EvidenceType.NUMERIC:
                facts.append(
                    _fact(
                        fact_id=f"F{number}",
                        ticker=group.ticker,
                        period=group.fiscal_year,
                        value=value,
                        source_id=source_id,
                    )
                )
        bundle = RetrievalBundle(
            context="\n".join(source["evidence_text"] for source in sources),
            sources=sources,
            scope=requirement.to_scope(),
            candidate_count=len(sources),
            covered_groups=tuple(group.key for group in requirement.groups),
        )
        result = RequirementResult(
            requirement_id=requirement.requirement_id,
            covered_groups=requirement.groups,
            source_ids=tuple(source["id"] for source in sources),
            fact_ids=tuple(fact.fact_id for fact in facts),
        )
        return MultiHopExecutionResult(
            bundle=bundle,
            requirement_results=(result,),
            facts=tuple(facts),
            calculations=(),
            issues=(),
            trace={
                "requirements": [
                    {
                        "requirement_id": requirement.requirement_id,
                        "retrieval_mode": "hybrid",
                        "candidate_count": len(sources),
                        "source_ids": [source["id"] for source in sources],
                        "fact_ids": [fact.fact_id for fact in facts],
                        "missing_groups": [],
                    }
                ],
                "calculations": [],
            },
        )


class NoGenerationExpected:
    def generate_with_trace(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("deterministic calculations should bypass generation")


class NumericAgentGraph:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    def invoke(self, payload, config=None):
        del payload, config
        observation = json.loads(
            self.tools["search_financial_documents"].invoke(
                {
                    "requirement_id": "revenue",
                    "query": "Find total revenue for both fiscal years.",
                    "evidence_type": "numeric",
                    "document_type": "10K",
                    "groups": [
                        {"ticker": "AAA", "fiscal_year": "2024"},
                        {"ticker": "AAA", "fiscal_year": "2025"},
                    ],
                    "requires_complete_coverage": True,
                }
            )
        )
        facts = sorted(observation["facts"], key=lambda item: item["period"])
        self.tools["financial_calculator"].invoke(
            {
                "calculation_id": "revenue_change",
                "label": "AAA revenue percentage change",
                "operation": "percentage_change",
                "input_fact_ids": [fact["fact_id"] for fact in facts],
                "periods": None,
            }
        )
        self.tools["complete_research"].invoke({})
        return {"messages": []}


class OutOfScopeGraph:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    def invoke(self, payload, config=None):
        del payload, config
        self.observation = json.loads(
            self.tools["search_financial_documents"].invoke(
                {
                    "requirement_id": "outside_scope",
                    "query": "Find revenue outside the allowed scope.",
                    "evidence_type": "numeric",
                    "document_type": "10K",
                    "groups": [{"ticker": "BBB", "fiscal_year": "2025"}],
                    "requires_complete_coverage": True,
                }
            )
        )
        return {"messages": []}


class ErrorAfterCompleteGraph(NumericAgentGraph):
    def invoke(self, payload, config=None):
        super().invoke(payload, config)
        raise RuntimeError("simulated graph failure after tool execution")


class SearchThenErrorGraph:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    def invoke(self, payload, config=None):
        del payload, config
        self.tools["search_financial_documents"].invoke(
            {
                "requirement_id": "revenue",
                "query": "Find revenue for both years.",
                "evidence_type": "numeric",
                "document_type": "10K",
                "groups": [
                    {"ticker": "AAA", "fiscal_year": "2024"},
                    {"ticker": "AAA", "fiscal_year": "2025"},
                ],
                "requires_complete_coverage": True,
            }
        )
        raise RuntimeError("simulated graph failure before calculations")


class StaticPlanner:
    def __init__(self, plan):
        self._plan = plan

    def plan(self, *, question, permitted_scope):
        del question, permitted_scope
        return self._plan


def _agent(executor, graph_type, records):
    def factory(*, model, tools, system_prompt, name):
        del model, system_prompt, name
        graph = graph_type(tools)
        factory.graph = graph
        return graph

    factory.graph = None
    return (
        LangChainReActAgent(
            executor=executor,
            generator=NoGenerationExpected(),
            record=records.append,
            retrieval_source_trace=lambda source: {
                "id": source.get("id"),
                "ticker": source.get("ticker"),
                "fiscal_year": source.get("fiscal_year"),
                "doc_type": source.get("doc_type"),
            },
            model=object(),
            agent_factory=factory,
            max_actions=4,
        ),
        factory,
    )


def test_official_agent_searches_then_calculates_with_fictional_ticker() -> None:
    executor = FakeExecutor()
    records = []
    agent, _ = _agent(executor, NumericAgentGraph, records)
    scope = Scope(
        tickers=("AAA",),
        years=("2024", "2025"),
        doc_type="10K",
        requested_groups=(("AAA", "2024"), ("AAA", "2025")),
    )

    result = agent.run(
        question="Calculate AAA total revenue percentage change from 2024 to 2025.",
        scope=scope,
        inherited_fields=(),
        history=None,
        query_expansions=(),
        wants_table=False,
        trace={"multihop": {"enabled": True, "selected": True}},
    )

    assert result.decision == Decision.ANSWERED
    assert "20%" in result.answer
    assert result.trace["agent"]["framework"] == "langchain.create_agent"
    assert result.trace["agent"]["backend"] == "langchain"
    assert [event["action"]["tool"] for event in result.trace["agent"]["events"]] == [
        "search_financial_documents",
        "financial_calculator",
        "complete_research",
    ]
    assert len(executor.calls) == 1
    assert len(records) == 1


def test_official_agent_rejects_scope_expansion_before_retrieval() -> None:
    executor = FakeExecutor()
    records = []
    agent, factory = _agent(executor, OutOfScopeGraph, records)
    scope = Scope(("AAA",), ("2025",), "10K")

    result = agent.run(
        question="Calculate the requested value.",
        scope=scope,
        inherited_fields=(),
        history=None,
        query_expansions=(),
        wants_table=False,
        trace={"multihop": {"enabled": True, "selected": True}},
    )

    assert result.decision == Decision.INSUFFICIENT_EVIDENCE
    assert factory.graph.observation["status"] == "rejected"
    assert "scope" in factory.graph.observation["error"]
    assert executor.calls == []


def test_official_agent_enforces_action_budget() -> None:
    executor = FakeExecutor()
    records = []
    agent, _ = _agent(executor, NumericAgentGraph, records)
    agent.max_actions = 1

    result = agent.run(
        question="Calculate AAA total revenue percentage change from 2024 to 2025.",
        scope=Scope(("AAA",), ("2024", "2025"), "10K"),
        inherited_fields=(),
        history=None,
        query_expansions=(),
        wants_table=False,
        trace={"multihop": {"enabled": True, "selected": True}},
    )

    assert result.decision == Decision.INSUFFICIENT_EVIDENCE
    assert result.trace["agent"]["actions_used"] == 1
    assert result.trace["agent"]["status"] == "partial"
    assert "requested_calculation_not_completed" in result.trace["multihop"]["issues"]


def test_narrative_evidence_keeps_each_required_document_type() -> None:
    executor = FakeExecutor()
    scope = Scope(
        ("AAA",),
        ("2025",),
        required_doc_types=("10K", "TRANSCRIPT"),
    )
    session = AgentResearchSession(
        question="Compare filing risks with management commentary.",
        permitted_scope=scope,
        executor=executor,
        calculator=executor.calculator,
        max_actions=4,
    )

    for requirement_id, document_type in (
        ("filing_risk", "10K"),
        ("management_commentary", "TRANSCRIPT"),
    ):
        observation = json.loads(
            session.search(
                ResearchToolInput.model_validate(
                    {
                        "requirement_id": requirement_id,
                        "query": "Find the requested narrative evidence.",
                        "evidence_type": "narrative",
                        "document_type": document_type,
                        "groups": [{"ticker": "AAA", "fiscal_year": "2025"}],
                    }
                )
            )
        )
        assert observation["status"] == "completed"

    assert {source["doc_type"] for source in session.bundle().sources} == {
        "10K",
        "TRANSCRIPT",
    }
    assert session.final_issues() == ()


def test_agent_error_fails_closed_even_after_tools_completed() -> None:
    executor = FakeExecutor()
    agent, _ = _agent(executor, ErrorAfterCompleteGraph, [])
    result = agent.run(
        question="Calculate AAA total revenue percentage change from 2024 to 2025.",
        scope=Scope(("AAA",), ("2024", "2025"), "10K"),
        inherited_fields=(),
        history=None,
        query_expansions=(),
        wants_table=False,
        trace={"multihop": {"enabled": True, "selected": True}},
    )

    assert result.decision == Decision.INSUFFICIENT_EVIDENCE
    assert result.trace["agent"]["stop_reason"] == "agent_error"
    assert result.trace["agent"]["error_type"] == "RuntimeError"


def test_typed_checklist_requires_every_exact_task() -> None:
    executor = FakeExecutor()
    requirement = EvidenceRequirement(
        requirement_id="revenue",
        question="Find total revenue.",
        evidence_type=EvidenceType.NUMERIC,
        document_type="10K",
        groups=(EvidenceGroup(ticker="AAA", fiscal_year="2025"),),
    )
    calculation = PlannedCalculation(
        calculation_id="revenue_sum",
        label="Revenue sum",
        operation=CalculationOperation.SUM,
        inputs=(
            FactReference(
                requirement_id="revenue",
                ticker="AAA",
                fiscal_year="2025",
                period="2025",
                metric_hint="total revenue",
            ),
        ),
    )
    plan = MultiHopPlan(
        original_question="Find and sum revenue.",
        requirements=(requirement,),
        calculations=(calculation,),
    )
    session = AgentResearchSession(
        question=plan.original_question,
        permitted_scope=Scope(("AAA",), ("2025",), "10K"),
        executor=executor,
        calculator=executor.calculator,
        max_actions=4,
        expected_plan=plan,
    )

    assert "checklist_requirement_not_completed:revenue" in session.final_issues()
    assert "checklist_calculation_not_completed:revenue_sum" in session.final_issues()
    rejected = json.loads(
        session.search(
            ResearchToolInput.model_validate(
                {
                    "requirement_id": "revenue",
                    "query": "Try a changed task.",
                    "evidence_type": "narrative",
                    "document_type": "10K",
                    "groups": [{"ticker": "AAA", "fiscal_year": "2025"}],
                }
            )
        )
    )
    assert rejected["status"] == "rejected"
    assert "typed checklist" in rejected["error"]


def test_calculator_accepts_prior_calculation_id_as_an_operand() -> None:
    executor = FakeExecutor()
    session = AgentResearchSession(
        question="Calculate a change and divide it by current revenue.",
        permitted_scope=Scope(("AAA",), ("2024", "2025"), "10K"),
        executor=executor,
        calculator=executor.calculator,
        max_actions=5,
    )
    search = json.loads(
        session.search(
            ResearchToolInput.model_validate(
                {
                    "requirement_id": "revenue",
                    "query": "Find revenue for both years.",
                    "evidence_type": "numeric",
                    "document_type": "10K",
                    "groups": [
                        {"ticker": "AAA", "fiscal_year": "2024"},
                        {"ticker": "AAA", "fiscal_year": "2025"},
                    ],
                }
            )
        )
    )
    facts = sorted(search["facts"], key=lambda item: item["period"])
    first = json.loads(
        session.calculate(
            CalculatorToolInput(
                calculation_id="revenue_change",
                label="Revenue change",
                operation="absolute_change",
                input_fact_ids=tuple(item["fact_id"] for item in facts),
            )
        )
    )
    second = json.loads(
        session.calculate(
            CalculatorToolInput(
                calculation_id="change_share",
                label="Change divided by current revenue",
                operation="ratio",
                input_fact_ids=("revenue_change", facts[1]["fact_id"]),
            )
        )
    )

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert Decimal(second["result"]) == Decimal("1") / Decimal("6")
    assert second["source_ids"] == ["S1", "S2"]


def test_agent_recovers_only_pending_typed_calculations_after_searches() -> None:
    executor = FakeExecutor()
    requirement = EvidenceRequirement(
        requirement_id="revenue",
        question="Find revenue for both years.",
        evidence_type="numeric",
        document_type="10K",
        groups=(
            EvidenceGroup(ticker="AAA", fiscal_year="2024"),
            EvidenceGroup(ticker="AAA", fiscal_year="2025"),
        ),
    )
    old = FactReference(
        requirement_id="revenue",
        ticker="AAA",
        fiscal_year="2024",
        metric_hint="revenue",
    )
    new = FactReference(
        requirement_id="revenue",
        ticker="AAA",
        fiscal_year="2025",
        metric_hint="revenue",
    )
    plan = MultiHopPlan(
        original_question="Sum revenue, then divide by current revenue.",
        requirements=(requirement,),
        calculations=(
            PlannedCalculation(
                calculation_id="revenue_sum",
                label="Revenue sum",
                operation="sum",
                inputs=(old, new),
            ),
            PlannedCalculation(
                calculation_id="sum_ratio",
                label="Sum divided by current revenue",
                operation="ratio",
                inputs=(
                    CalculationReference(calculation_id="revenue_sum"),
                    new,
                ),
            ),
        ),
    )

    def factory(*, model, tools, system_prompt, name):
        del model, system_prompt, name
        return SearchThenErrorGraph(tools)

    agent = LangChainReActAgent(
        executor=executor,
        planner=StaticPlanner(plan),
        generator=NoGenerationExpected(),
        record=lambda trace: None,
        retrieval_source_trace=lambda source: source,
        model=object(),
        agent_factory=factory,
        max_actions=4,
    )
    result = agent.run(
        question=plan.original_question,
        scope=Scope(("AAA",), ("2024", "2025"), "10K"),
        inherited_fields=(),
        history=None,
        query_expansions=(),
        wants_table=False,
        trace={"multihop": {"enabled": True, "selected": True}},
    )

    assert result.trace["agent"]["stop_reason"] == "typed_checklist_recovered"
    assert result.trace["agent"]["recovered_calculations"] == 2
    assert result.trace["multihop"]["issues"] == []


def test_later_typed_calculation_can_use_source_not_in_earlier_result() -> None:
    executor = FakeExecutor()
    requirements = tuple(
        EvidenceRequirement(
            requirement_id=f"revenue_{year}",
            question=f"Find revenue for {year}.",
            evidence_type="numeric",
            document_type="10K",
            groups=(EvidenceGroup(ticker="AAA", fiscal_year=year),),
        )
        for year in ("2024", "2025")
    )
    calculations = tuple(
        PlannedCalculation(
            calculation_id=f"revenue_sum_{year}",
            label=f"Revenue sum {year}",
            operation="sum",
            inputs=(
                FactReference(
                    requirement_id=f"revenue_{year}",
                    ticker="AAA",
                    fiscal_year=year,
                    metric_hint="total revenue",
                ),
            ),
        )
        for year in ("2024", "2025")
    )
    plan = MultiHopPlan(
        original_question="Calculate each year's revenue.",
        requirements=requirements,
        calculations=calculations,
    )
    session = AgentResearchSession(
        question=plan.original_question,
        permitted_scope=Scope(("AAA",), ("2024", "2025"), "10K"),
        executor=executor,
        calculator=executor.calculator,
        max_actions=6,
        expected_plan=plan,
    )
    for requirement in requirements:
        observation = json.loads(
            session.search(
                ResearchToolInput(
                    requirement_id=requirement.requirement_id,
                    query=requirement.question,
                    evidence_type=requirement.evidence_type,
                    document_type=requirement.document_type,
                    groups=requirement.groups,
                )
            )
        )
        assert observation["status"] == "completed"

    observations = []
    for calculation in calculations:
        observations.append(
            json.loads(
                session.calculate(
                    CalculatorToolInput(
                        calculation_id=calculation.calculation_id,
                        label=calculation.label,
                        operation=calculation.operation,
                        input_fact_ids=("placeholder",),
                    )
                )
            )
        )

    assert [item["status"] for item in observations] == ["completed", "completed"]
    assert observations[1]["source_ids"] == ["S2"]
