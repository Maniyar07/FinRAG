from types import SimpleNamespace

from src.app.multihop_coordinator import build_react_events
from src.financial.calculator import CalculationOperation
from src.orchestration.models import EvidenceType


def test_react_events_expose_search_and_calculator_without_private_reasoning() -> None:
    group = SimpleNamespace(ticker="AAA", fiscal_year="2025")
    requirement = SimpleNamespace(
        requirement_id="revenue",
        question="Find total revenue.",
        evidence_type=EvidenceType.NUMERIC,
        document_type="10K",
        groups=(group,),
    )
    reference = SimpleNamespace(
        requirement_id="revenue",
        ticker="AAA",
        fiscal_year="2025",
        period="2025",
        metric_hint="total revenue",
    )
    calculation = SimpleNamespace(
        calculation_id="revenue_sum",
        operation=CalculationOperation.SUM,
        inputs=(reference,),
    )
    plan = SimpleNamespace(
        requirements=(requirement,),
        calculations=(calculation,),
    )
    execution = SimpleNamespace(
        trace={
            "requirements": [
                {
                    "requirement_id": "revenue",
                    "retrieval_mode": "hybrid",
                    "candidate_count": 7,
                    "source_ids": ["S1"],
                    "fact_ids": ["F1"],
                    "missing_groups": [],
                }
            ],
            "calculations": [
                {
                    "calculation_id": "revenue_sum",
                    "status": "completed",
                    "input_fact_ids": ["F1"],
                    "source_ids": ["S1"],
                    "result": "100",
                    "result_unit": "currency",
                }
            ],
        }
    )

    events = build_react_events(plan, execution)

    assert [event["action"]["tool"] for event in events] == [
        "document_search",
        "financial_calculator",
    ]
    assert events[0]["action"]["arguments"]["groups"] == [["AAA", "2025"]]
    assert events[0]["observation"]["status"] == "completed"
    assert events[1]["observation"]["result"] == "100"
    assert all("thought" not in event for event in events)


def test_react_events_mark_missing_evidence_without_inventing_an_observation() -> None:
    requirement = SimpleNamespace(
        requirement_id="risk",
        question="Summarize disclosed risks.",
        evidence_type=EvidenceType.NARRATIVE,
        document_type="10K",
        groups=(SimpleNamespace(ticker="AAA", fiscal_year="2025"),),
    )
    plan = SimpleNamespace(requirements=(requirement,), calculations=())
    execution = SimpleNamespace(
        trace={
            "requirements": [
                {
                    "requirement_id": "risk",
                    "retrieval_mode": "hybrid",
                    "candidate_count": 0,
                    "source_ids": [],
                    "fact_ids": [],
                    "missing_groups": [["AAA", "2025"]],
                }
            ]
        }
    )

    event = build_react_events(plan, execution)[0]

    assert event["observation"]["status"] == "insufficient_evidence"
    assert event["observation"]["source_ids"] == []
    assert event["observation"]["missing_groups"] == [["AAA", "2025"]]
