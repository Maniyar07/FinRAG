"""Shared release contracts, using synthetic sources and no provider requests."""
import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from langchain_core.messages import ToolMessage
from openai import APIConnectionError, BadRequestError

from evals.evaluate_ragas import _score_row, summarize
from src.agent.graph import build_request_plan, create_finrag_agent
from src.agent.tools import RequestPlanInput
from src.financial.calculator import FinancialCalculator
from src.financial.coverage import fact_matches_requirement
from src.financial.models import ValidatedFinancialFact, ValueType
from src.financial.metric_terms import has_financial_metric_conflict
from src.generation.evidence_answer import EvidenceQuote, EvidenceSynthesis, NarrativePoint, RequirementAnswer, render_evidence_answer, resolve_evidence_answers
from src.request_runtime import RequestAborted
from src.runtime_errors import classify_runtime_error, tool_error_response
from src.schemas import RetrievalBundle, Scope

SCOPE = Scope(('ACME',), ('2028',), '10K')
REQ = dict(requirement_id='r1', ticker='ACME', source_year='2028', doc_type='10K', metric='operating income', reported_period='FY2028')
SOURCE = dict(id='S1', ticker='ACME', fiscal_year='2028', doc_type='10K')


@pytest.mark.parametrize('label', ['ACME', 'Industrial services', 'operating income'])
def test_output_subject_cannot_replace_the_requested_financial_metric(label):
    draft = dict(requested_objects=[dict(object_id='o1', quote='operating income', kind='numeric_value', subject=label)],
                 requirements=[{**REQ, 'object_id': 'o1'}])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = draft
    with patch('src.agent.graph.get_llm_engine', return_value=engine):
        plan = build_request_plan('Report ACME operating income for FY2028.', SCOPE)
    assert plan.requirements[0].metric == 'operating income'
    properties = list(engine.with_structured_output.call_args.args[0]['properties'])
    assert properties.index('subjects') < properties.index('requested_objects')
    assert properties.index('requirements') < properties.index('calculations')


def test_hyphenated_business_identity_and_narrative_period_survive_planning():
    draft = dict(subjects=[dict(subject_id='b1', name='industrial services', entity_type='business_area')],
        requested_objects=[dict(object_id='o1', quote='industrial-services outlook', kind='explanation', subject='outlook')],
        requirements=[{**REQ, 'object_id': 'o1', 'subject_id': 'b1', 'metric': 'outlook', 'reported_period': 'FY2029'}])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = draft
    with patch('src.agent.graph.get_llm_engine', return_value=engine):
        plan = build_request_plan('Describe the industrial-services outlook for FY2029 in the 2028 filing.', SCOPE)
    assert plan.requirements[0].subject == 'industrial services'
    assert plan.requirements[0].reported_period == 'FY2029'


def test_selection_receives_the_literal_request_and_only_scoped_verified_observations():
    engine = Mock()
    engine.with_structured_output.return_value.invoke.side_effect = RuntimeError('selection captured')
    sources = {'S1': {**SOURCE, 'evidence_text': 'Lower costs supported the result.'},
               'S2': {**SOURCE, 'id': 'S2', 'ticker': 'ORBIT', 'evidence_text': 'A different company.'}}
    facts = {key: SimpleNamespace(metric='operating income', period='FY2028', source_id=source,
        evidence_excerpt='Operating income was 120 million USD.') for key, source in [('F1', 'S1'), ('F2', 'S2')]}
    plan = dict(question='What was operating income and why did it improve?',
        requested_objects=[dict(object_id='o1', quote='why did it improve?')],
        requirements=[{**REQ, 'object_id': 'o1', 'kind': 'narrative', 'metric': 'drivers'}])
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine), pytest.raises(RuntimeError, match='selection captured'):
        resolve_evidence_answers(plan, [], facts, sources)
    payload = json.loads(engine.with_structured_output.return_value.invoke.call_args.args[0][-1].content)
    assert payload['requirements'][0]['requested_output'] == 'why did it improve?'
    assert len(payload['verified_observations']) == 1 and payload['verified_observations'][0]['source_id'] == 'S1'


@pytest.mark.parametrize('selected', ['P1:0', 'P999999:0'])
def test_compact_sentence_schema_keeps_exact_catalog_membership_validation(selected):
    text = 'Operating constraints will ease.'
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = dict(answers={'r1': dict(draft_points=[dict(sentence_id=selected)])})
    plan = dict(requirements=[{**REQ, 'kind': 'narrative', 'metric': 'outlook'}])
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        if selected == 'P1:0':
            answers = resolve_evidence_answers(plan, [], {}, {'S1': {**SOURCE, 'evidence_text': text}})
            assert answers[0].quotes[0].excerpt == text
        else:
            with pytest.raises(ValueError, match='unknown sentence'):
                resolve_evidence_answers(plan, [], {}, {'S1': {**SOURCE, 'evidence_text': text}})
    schema = engine.with_structured_output.call_args.args[0]
    assert 'P999999' not in json.dumps(schema)
    assert '"enum": ["P' not in json.dumps(schema)


@pytest.mark.parametrize('defect', ['self_dependency', 'reported_result', 'unknown_output'])
def test_invalid_derived_dependencies_are_rejected_before_retrieval(defect):
    draft = dict(requested_objects=[dict(object_id='o1', quote='calculate operating margin', kind='calculation', subject='operating margin')],
        requirements=[{**REQ, 'object_id': 'o1'}, {**REQ, 'requirement_id': 'r2', 'object_id': 'o1', 'metric': 'revenue'}],
        calculations=[dict(object_id='o1', operation='percentage_of', input_requirement_ids=['r1', 'r2'])])
    if defect == 'self_dependency':
        draft['requirements'][0]['metric'] = 'operating margin'
    elif defect == 'reported_result':
        draft['requested_objects'][0]['kind'] = 'fact'
    else:
        draft['calculations'][0]['object_id'] = 'o2'
    with pytest.raises(ValueError):
        RequestPlanInput.model_validate(draft)


@pytest.mark.parametrize('metric,accepted', [('operating income', True), ('operating income change', False)])
def test_requirement_id_cannot_turn_a_reported_level_into_a_change(metric, accepted):
    fact = ValidatedFinancialFact(fact_id='F1', requirement_id='r1', ticker='ACME', metric=metric,
        period='2028', raw_value='120', numeric_value=Decimal(120), base_value=Decimal(120000000),
        value_type=ValueType.CURRENCY, normalized_unit='USD', currency='USD', scale='millions',
        source_id='S1', row_label='Operating income', column_label='2028',
        evidence_excerpt='Operating income 120', validation_checks=('table_row_period_value',))
    assert fact_matches_requirement(fact, {**REQ, 'metric': metric}, SOURCE) is accepted


@pytest.mark.parametrize('label', ['up 37%', 'down 5%', 'revenue grew 12%', 'revenue increased', 'revenue decreased'])
def test_source_reported_movements_remain_valid_evidence_for_change_requests(label):
    assert not has_financial_metric_conflict(label, 'revenue growth rate')


@pytest.mark.parametrize('validated', [False, True])
def test_semantic_alias_requires_a_validated_binding_not_just_a_matching_id(validated):
    fact = ValidatedFinancialFact(fact_id='F1', requirement_id='r1', ticker='ACME', metric='RPO',
        period='2028', raw_value='120', numeric_value=Decimal(120), base_value=Decimal(120000000),
        value_type=ValueType.CURRENCY, normalized_unit='USD', currency='USD', scale='millions',
        source_id='S1', evidence_excerpt='Remaining performance obligations were 120 million USD.',
        validation_checks=('request_evidence_binding',) if validated else ())
    assert fact_matches_requirement(fact, {**REQ, 'metric': 'remaining performance obligations'}, SOURCE) is validated


@pytest.mark.parametrize('text,accepted', [('Annual operating margin was 12.5%.', True),
    ('Annual operating margin was 18.5%.', False), ('In 2027, operating margin was 18.5%.', True)])
def test_interpretation_values_are_bound_to_their_table_period(text, accepted):
    row = 'Operating margin 12.5% 18.5%'
    source = {**SOURCE, 'evidence_text': '<table><tr><th>Metric</th><th>2028</th><th>2027</th></tr><tr><td>Operating margin</td><td>12.5%</td><td>18.5%</td></tr></table>'}
    req = {**REQ, 'kind': 'narrative', 'metric': 'operating margin'}
    answer = RequirementAnswer(requirement_id='r1', quotes=[EvidenceQuote(source_id='S1', excerpt=row)],
        draft_points=[NarrativePoint(text=row, quote_indices=[0])])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = dict(interpretation=text, source_ids=['S1'], observation_ids=['r1:p0'])
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        if not accepted:
            with pytest.raises(ValueError, match='reported periods'):
                render_evidence_answer(dict(requirements=[req], synthesis='Compare operating margins.'), [answer], [], {}, {'S1': source}, {}, {})
        else:
            # Reaching the auditor proves numeric binding succeeded; no network is used.
            engine.with_structured_output.return_value.invoke.side_effect = [engine.with_structured_output.return_value.invoke.return_value, RuntimeError('audit reached')]
            with pytest.raises(RuntimeError, match='audit reached'):
                render_evidence_answer(dict(requirements=[req], synthesis='Compare operating margins.'), [answer], [], {}, {'S1': source}, {}, {})


def test_transcript_observations_cannot_route_to_the_filing_statement_tool():
    scope = Scope(('ACME',), ('2028',), 'TRANSCRIPT')
    plan = RequestPlanInput.model_validate(dict(requirements=[{**REQ, 'doc_type': 'TRANSCRIPT', 'kind': 'statement'}]))
    retriever = SimpleNamespace(retrieve=Mock(side_effect=AssertionError('An invalid plan must not retrieve.')))
    locked = 'Application-resolved scope (locked): '+json.dumps(dict(requested_groups=[['ACME', '2028']], doc_type='TRANSCRIPT'))
    graph = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=scope, plan_builder=lambda *a: plan)
    state = graph.invoke({'messages': [('user', locked+'\nUser question: Explain the recognition outlook.')]})
    tools = [m.name for m in state['messages'] if isinstance(m, ToolMessage)]
    assert 'get_financial_statement' not in tools and len(tools) <= 2


@pytest.mark.parametrize('limit,searches', [(2, 1), (8, 6)])
def test_every_source_and_recovery_route_shares_the_configured_action_budget(limit, searches):
    scope = Scope(('ACME',), ('2026', '2027', '2028'), '10K')
    plan = RequestPlanInput.model_validate(dict(requirements=[{**REQ, 'requirement_id': f'r{i}',
        'source_year': year, 'reported_period': 'FY'+year} for i, year in enumerate(scope.years)]))
    retriever = SimpleNamespace(retrieve=Mock(side_effect=lambda query, current_scope, **kw: RetrievalBundle('', [], current_scope, 0)))
    locked = 'Application-resolved scope (locked): '+json.dumps(dict(requested_groups=list(scope.groups), doc_type='10K'))
    with patch('src.agent.graph.AGENT_MAX_ACTIONS', limit), patch('src.agent.tools.AGENT_MAX_ACTIONS', limit):
        state = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=scope,
            plan_builder=lambda *a: plan).invoke({'messages': [('user', locked+'\nUser question: Report operating income for each year.')]})
    assert retriever.retrieve.call_count == searches
    final = next(json.loads(m.content) for m in reversed(state['messages']) if isinstance(m, ToolMessage) and m.name == 'finalize_financial_answer')
    assert not final['complete'] and len(final['missing_requirement_ids']) == 3


def test_failed_interpretation_reuses_source_selection_but_still_requires_audit():
    text = 'Operating income was 120 million USD.'
    plan = RequestPlanInput.model_validate(dict(requirements=[{**REQ, 'kind': 'narrative'}], synthesis='Interpret the result.'))
    retriever = SimpleNamespace(retrieve=Mock(return_value=RetrievalBundle(
        '[SOURCE S1 | TICKER: ACME | FISCAL YEAR: 2028 | TYPE: 10K]\n'+text,
        [{**SOURCE, 'parent_id': 'first'}], SCOPE, 1)))
    counts = dict(selection=0, synthesis=0, audit=0)
    def respond(schema, messages):
        payload = json.loads(messages[-1].content)
        if schema is EvidenceSynthesis:
            counts['synthesis'] += 1
            return dict(interpretation=text.replace('120', '999') if counts['synthesis'] == 1 else text, source_ids=payload['available_source_ids'])
        if 'answers' in schema['properties']:
            counts['selection'] += 1
            return dict(answers={'r1': dict(draft_points=[dict(text=text, passage_ids=['P1'])])})
        counts['audit'] += 1
        return dict(assessments={c['requirement']['requirement_id']: dict(reason='Direct supporting observation.', supported=True,
            point_support={'0': dict(evidence_spans=[text], unsupported_details=[])}) for c in payload['claims']})
    engine = SimpleNamespace(with_structured_output=lambda schema, **kw: SimpleNamespace(invoke=lambda messages: respond(schema, messages)))
    locked = 'Application-resolved scope (locked): '+json.dumps(dict(requested_groups=[['ACME', '2028']], doc_type='10K'))
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        state = create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=SCOPE,
            plan_builder=lambda *a: plan).invoke({'messages': [('user', locked+'\nUser question: Interpret the operating result.')]})
    finals = [json.loads(m.content) for m in state['messages'] if isinstance(m, ToolMessage) and m.name == 'finalize_financial_answer']
    assert finals[-1].get('complete') and len(finals) == 2, ([f.get('validation_error') for f in finals], counts)
    assert counts == dict(selection=1, synthesis=2, audit=1) and retriever.retrieve.call_count == 1
    assert 'verified_answers' not in finals[0] and finals[0]['selected_answers']


@pytest.mark.parametrize('wrapped', [False, True])
def test_local_budget_stop_keeps_its_cause_and_cannot_be_retried_as_network_failure(wrapped):
    error = RequestAborted('Approved request budget exhausted.')
    if wrapped:
        outer = APIConnectionError(request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions'))
        outer.__cause__ = error
        error = outer
    info = classify_runtime_error(error)
    assert info.code == 'request_aborted' and not info.retryable and 'budget' in info.detail


def test_provider_rejection_retains_safe_status_and_parameter_without_prompt_content():
    error = BadRequestError('private prompt content', response=httpx.Response(400,
        request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions')),
        body=dict(error=dict(code='invalid_json_schema', param='response_format', message='private prompt content')))
    payload = tool_error_response('Verification', error)
    assert payload['error_detail'] == 'HTTP 400; code=invalid_json_schema; param=response_format'
    assert 'private' not in json.dumps(payload)


@pytest.mark.parametrize('failed', [1, 5])
def test_partial_and_failed_judging_are_not_reported_as_complete_scoring(failed):
    names = ['faithfulness', 'context_precision', 'context_recall', 'response_relevancy', 'factual_correctness']
    async def grade(*args, **kwargs):
        return {name: None if i < failed else .9 for i, name in enumerate(names)}, {name: 'timeout' for name in names[:failed]}
    row = dict(id=1, question='Explain the result.', reference='Supported result.', decision='answered',
        answer='Supported result. [S1]', retrieved_contexts=['Supported result.'], generation_latency_ms=1)
    with patch('evals.evaluate_ragas.score_one', side_effect=grade):
        result = asyncio.run(_score_row(row, {}))
    assert result['score_status'] == ('scoring_failed' if failed == 5 else 'partially_scored')
    assert not result['judge_usage']['cost_complete'] and result['judge_usage']['total_token_cost_usd'] is None
    assert summarize([result])['fully_scored_answered_rows'] == 0
