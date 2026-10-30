"""Broader offline contracts: real rendering and source guards, controlled audits."""
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.financial.models import ValidatedFinancialFact, ValueType
from src.generation.evidence_answer import EvidenceQuote, NarrativePoint, RequirementAnswer, render_evidence_answer, resolve_evidence_answers


SOURCE = dict(id='S1', ticker='ACME', fiscal_year='2031', doc_type='10K', source='fixture.pdf')
REQ = dict(requirement_id='r1', ticker='ACME', source_year='2031', doc_type='10K', kind='narrative')


def literal(text, key='r1', source='S1'):
    return RequirementAnswer(requirement_id=key, quotes=[EvidenceQuote(source_id=source, excerpt=text)],
        draft_points=[NarrativePoint(text=text, quote_indices=[0])])


def audit_engine(responsive=True):
    def respond(messages):
        payload = json.loads(messages[-1].content)
        assessments = {}
        for claim in payload['claims']:
            quotes = {str(q['index']): dict(reason='Controlled audit verdict.', answers_question=responsive,
                scope_matches=True, supports_requested_relationship=True, source_is_aggregate=False)
                for q in claim['quotes']}
            if claim['requirement'].get('subject'):
                for quote in quotes.values():
                    quote.update(source_subject='storage', source_subject_type='business_area')
            assessments[claim['requirement']['requirement_id']] = dict(reason='Check coverage against stored evidence.',
                supported=True, quote_assessments=quotes,
                detail_coverage=[dict(requested_detail=claim['requirement']['metric'], point_indices=[0], quote_indices=[])])
        return dict(assessments=assessments, request_complete=True,
            unresolved_requirements={r['requirement_id']: dict(reason='Explicitly requested.', required_by_user=True)
                for r in payload['unresolved_requirements']})
    return SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=respond))


@pytest.mark.parametrize('case,value,complete', [
    ('decimal', '125.75', True), ('negative', '-25.5', True), ('zero', '0', True),
    ('missing_fact', '125.75', False), ('wrong_period', '125.75', False),
    ('wrong_company', '125.75', False), ('wrong_document', '125.75', False),
    ('wrong_source_year', '125.75', False),
])
def test_numeric_prior_period_values_and_genuine_evidence_gaps(case, value, complete):
    number = Decimal(value)
    fact = ValidatedFinancialFact(fact_id='F1', ticker='ACME', metric='Revenue',
        period='2029' if case == 'wrong_period' else '2030', raw_value=value, numeric_value=number,
        base_value=number*1000000, value_type=ValueType.CURRENCY, normalized_unit='USD', currency='USD',
        scale='millions', source_id='S1', evidence_excerpt='Revenue '+value, validation_checks=('fixture',))
    source = {**SOURCE, 'evidence_text': fact.evidence_excerpt}
    for scenario, field, replacement in [('wrong_company', 'ticker', 'ORBIT'),
        ('wrong_document', 'doc_type', 'TRANSCRIPT'), ('wrong_source_year', 'fiscal_year', '2030')]:
        if case == scenario:
            source[field] = replacement
    req = {**REQ, 'kind': 'fact', 'metric': 'total revenue', 'reported_period': 'FY2030'}
    facts = {} if case == 'missing_fact' else {'F1': fact}
    with patch('src.generation.evidence_answer.get_llm_engine', side_effect=AssertionError('Numeric rendering needs no model.')):
        answers = resolve_evidence_answers({'requirements': [req]}, [], facts, {'S1': source})
        result = render_evidence_answer({'requirements': [req]}, answers, [], facts, {'S1': source}, {}, {})
    assert result['complete'] is complete
    assert result['missing_requirement_ids'] == ([] if complete else ['r1'])
    if complete:
        assert value in result['answer'] and '2030' in result['answer'] and 'million USD' in result['answer']
    else:
        assert value not in result['answer']


@pytest.mark.parametrize('quote,context,complete', [
    ('Capital expenditures will support new facilities.', '', True),
    ('No additional investment is required this year.', '', True),
    ('Owned infrastructure includes data centers.', 'Assets are depreciated over their useful lives.', False),
    (None, 'Only operating expenses are disclosed.', False),
])
def test_narrative_positive_negative_and_missing_relationships(quote, context, complete):
    req = {**REQ, 'metric': 'capital requirements'}
    answer = literal(quote) if quote else RequirementAnswer(requirement_id='r1', unavailable_reason='Investment evidence is absent.')
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=audit_engine()):
        result = render_evidence_answer({'requirements': [req]}, [answer], [], {},
            {'S1': {**SOURCE, 'evidence_text': context+' '+(quote or '')}}, {}, {})
    assert result['complete'] is complete


DETAILS = {
    'ACME': ['We sell software subscriptions.', 'Operating costs include payroll and hosting.', 'Capital expenditures support new computing facilities.'],
    'ORBIT': ['We sell physical devices.', 'Operating costs include materials, labor and logistics.', 'Capital expenditures support new manufacturing plants.'],
}


@pytest.mark.parametrize('missing', [None, *range(6)])
def test_comparison_preserves_every_present_dimension_when_one_source_detail_is_missing(missing):
    requirements, answers, sources = [], [], {}
    for index, (ticker, texts) in enumerate(DETAILS.items()):
        source_id = f'S{index+1}'
        sources[source_id] = {**SOURCE, 'id': source_id, 'ticker': ticker,
            'evidence_text': '\n\n'.join(text for offset, text in enumerate(texts) if index*3+offset != missing)}
        for offset, (metric, text) in enumerate(zip(['revenue sources', 'cost structure', 'capital requirements'], texts)):
            number = index*3+offset
            key = f'r{number+1}'
            requirements.append({**REQ, 'requirement_id': key, 'ticker': ticker, 'metric': metric})
            answers.append(literal(text, key, source_id) if number != missing else
                RequirementAnswer(requirement_id=key, unavailable_reason='The requested relationship is absent.'))
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=audit_engine()):
        result = render_evidence_answer({'requirements': requirements}, answers, [], {}, sources, {}, {})
    assert result['complete'] is (missing is None)
    assert result['missing_requirement_ids'] == ([] if missing is None else [f'r{missing+1}'])
    assert {a['requirement_id'] for a in result['verified_answers']} == {
        f'r{i+1}' for i in range(6) if i != missing}
    for index, text in enumerate(sum(DETAILS.values(), [])):
        assert (text in result['answer']) is (index != missing)


@pytest.mark.parametrize('quote,responsive,complete', [
    ('We expect storage deployments to increase despite competitive margin pressure.', True, True),
    ('On the thermal front, revenue increased $20 million.', True, False),
    ('Thermal generation and storage revenue increased $20 million.', True, False),
    ('Our spending on products like Alpha, Beta and Gamma will remain elevated.', True, False),
    ('Last year storage deployments increased.', False, False),
    (None, True, False),
])
def test_transcript_outlook_does_not_borrow_broader_results_or_replace_forecasts_with_history(quote, responsive, complete):
    req = {**REQ, 'doc_type': 'TRANSCRIPT', 'subject': 'thermal storage business', 'metric': 'management outlook'}
    source = {**SOURCE, 'doc_type': 'TRANSCRIPT',
        'evidence_text': '## Our thermal generation and storage business\n\n'+(quote or 'Only vehicle sales are discussed.')}
    answer = literal(quote) if quote else RequirementAnswer(requirement_id='r1', unavailable_reason='No storage outlook is available.')
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=audit_engine(responsive)):
        result = render_evidence_answer({'requirements': [req]}, [answer], [], {}, {'S1': source}, {}, {})
    assert result['complete'] is complete
    assert result['decision'] == ('answered' if complete else 'insufficient_evidence')
