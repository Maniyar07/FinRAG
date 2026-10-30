"""Saved semantic false positives must fail without discarding valid evidence."""
import json
import re
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.generation.evidence_answer import (
    EvidenceQuote, NarrativePoint, PointAssessment, RequirementAnswer,
    _cost_structure_gap, _investment_policy_gap, _point_supported,
    evidence_passages, render_evidence_answer,
)
from src.retrieval.text_blocks import has_evidence


SOURCE = dict(id="S1", ticker="ACME", fiscal_year="2025", doc_type="10K", source="filing.pdf")
REQ = dict(requirement_id="r1", ticker="ACME", source_year="2025", doc_type="10K", kind="narrative")


def literal(text, key="r1", source="S1"):
    return RequirementAnswer(requirement_id=key, quotes=[EvidenceQuote(source_id=source, excerpt=text)],
        draft_points=[NarrativePoint(text=text, quote_indices=[0])])


def test_whole_paragraph_verdict_cannot_substitute_for_each_sentence_proof():
    quotes = ["Revenue increased.", "Competitive pressure may reduce margins."]
    point = dict(heading="", text="Revenue increased. Competitive pressure may reduce margins.", quote_indices=[0, 1])
    proof = dict(evidence_spans=quotes, unsupported_details=[], supports_requested_relationship=True)
    assert not _point_supported(PointAssessment(**proof), point, quotes)
    assert not _point_supported(PointAssessment(**proof, assertion_evidence={"0": [quotes[0]], "1": []}), point, quotes)
    assert _point_supported(PointAssessment(**proof, assertion_evidence={"0": [quotes[0]], "1": [quotes[1]]}), point, quotes)
    assert not _point_supported(PointAssessment(**proof, assertion_evidence={"0": [quotes[0]], "1": ["Unquoted claim."]}), point, quotes)


@pytest.mark.parametrize("claim,accepted", [
    ("Energy storage revenue increased.", False),
    ("Energy revenue increased.", True),
    ("Energy generation and storage revenue increased.", True),
    ("Hardware revenue increased.", False),
    ("Cloud and hardware revenue increased.", True),
])
def test_coordinated_reporting_scope_survives_a_positive_model_verdict(claim, accepted):
    source = ("Energy generation and storage revenue increased." if "Energy" in claim
              else "Cloud and hardware revenue increased.")
    proof = PointAssessment(evidence_spans=[source], unsupported_details=[], supports_requested_relationship=True)
    assert _point_supported(proof, dict(heading="", text=claim, quote_indices=[0]), [source]) is accepted


def test_explicit_narrower_disclosure_survives_broader_context():
    broad, narrow = "Energy generation and storage revenue increased.", "Energy storage revenue increased."
    proof = PointAssessment(evidence_spans=[broad, narrow], unsupported_details=[], supports_requested_relationship=True)
    assert _point_supported(proof, dict(heading="Results", text=narrow, quote_indices=[0, 1]), [broad, narrow])


@pytest.mark.parametrize("metric,quote,context,missing", [
    ("capital requirements", "AI infrastructure includes our owned data centers.", "Assets are depreciated over their useful lives.", True),
    ("capital requirements", "Machinery 3 to 15 years; Buildings 15 to 30 years", "Depreciation uses estimated useful lives.", True),
    ("capital requirements", "Capital expenditures primarily support factory expansion.", "Depreciation uses estimated useful lives.", False),
    ("investment requirements", "No additional factory investment is required.", "Assets are depreciated.", False),
    ("depreciation policy", "AI infrastructure includes our owned data centers.", "Assets are depreciated.", False),
    ("capital requirements", "Factories require substantial investment.", "Operations are expanding.", False),
])
def test_accounting_evidence_is_evaluated_against_the_requested_relationship(metric, quote, context, missing):
    assert bool(_investment_policy_gap({**REQ, "metric": metric}, quote, context)) is missing


def test_positive_audit_cannot_make_asset_accounting_complete_investment_evidence():
    quote = "AI infrastructure includes our owned data centers."
    text = "We depreciate assets over their useful lives. " + quote
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {"assessments": {"r1": dict(
        reason="The asset definition establishes investment needs.", supported=True,
        quote_assessments={"0": dict(reason="Productive assets are listed.", answers_question=True,
            supports_requested_relationship=True, scope_matches=True)})}}
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        result = render_evidence_answer({"requirements": [{**REQ, "metric": "capital requirements"}]},
            [literal(quote)], [], {}, {"S1": {**SOURCE, "evidence_text": text}}, {}, {})
    assert not result['complete'] and result['missing_requirement_ids'] == ['r1']
    assert "productive investment" in result['request_coverage']['assessments'][0]['missing_details'][0]


def test_cost_totals_and_allocation_policies_leave_a_driver_gap():
    req = {**REQ, "metric": "cost structure"}
    allocation = ["Cost of revenues (1) $10 $9", "Depreciation and amortization included in cost of revenues was $2.",
                  "Lease expense is recognized as cost of revenues or operating expenses."]
    assert _cost_structure_gap(req, allocation)
    assert not _cost_structure_gap(req, [*allocation, "Cost of sales includes materials, labor and logistics."])
    assert not _cost_structure_gap({**REQ, "metric": "depreciation allocation"}, allocation)


@pytest.mark.parametrize("text,present", [
    ("#### Automotive Regulatory Credits", False), ("[PAGE: 12]\n\n## Business", False),
    ("<h3>Business</h3>", False), ("**Business**", False),
    ("## Storage\n\nNo expansion is planned.", True), ("Revenue: $3 million", True),
    ("<table><tr><td>Revenue</td><td>3</td></tr></table>", True),
])
def test_heading_only_sources_are_distinguished_from_short_real_evidence(text, present):
    assert has_evidence(text) is present


def test_page_break_does_not_turn_an_investment_sentence_into_fragments():
    text = "Additions to property and equipment will\n\n[PAGE: 51]\n\ncontinue, including new facilities and datacenters."
    passages = evidence_passages([{**SOURCE, "evidence_text": text}])
    assert [p['excerpt'] for p in passages.values()] == [
        "Additions to property and equipment will continue, including new facilities and datacenters."]


@pytest.mark.parametrize("change", [None, "source", "new_source", "question", "model"])
def test_repair_reuses_only_identical_verified_requirement_audits(change):
    requirements = [{**REQ, "requirement_id": key, "metric": metric} for key, metric in [
        ("r1", "revenue sources"), ("r2", "investment requirements")]]
    first, second = "We sell software subscriptions.", "We invest in new datacenters."
    sources = {"S1": {**SOURCE, "evidence_text": first}, "S2": {**SOURCE, "id": "S2", "evidence_text": second}}
    calls = []
    def respond(messages):
        payload = json.loads(messages[-1].content)
        calls.append([claim['requirement']['requirement_id'] for claim in payload['claims']])
        return {"request_complete": True, "assessments": {claim['requirement']['requirement_id']: dict(
            reason="The literal observation answers the request.", supported=True,
            quote_assessments={str(q['index']): dict(reason="Responsive source observation.", answers_question=True,
                scope_matches=True, supports_requested_relationship=True) for q in claim['quotes']}) for claim in payload['claims']},
            "unresolved_requirements": {r['requirement_id']: dict(reason="Independently requested.", required_by_user=True)
                for r in payload['unresolved_requirements']}}
    engine = SimpleNamespace(with_structured_output=lambda *args, **kwargs: SimpleNamespace(invoke=respond))
    plan = dict(requirements=requirements, question="Compare revenue and investment needs.")
    with patch("src.generation.evidence_answer.get_llm_engine", return_value=engine):
        before = render_evidence_answer(plan, [literal(first), RequirementAnswer(requirement_id="r2", unavailable_reason="Missing investment needs.")],
            [], {}, sources, {}, {})
        if change == "source":
            sources['S1']['evidence_text'] = "Operations context changed. " + first
        if change == 'new_source':
            sources['S3'] = {**SOURCE, 'id': 'S3', 'evidence_text': 'An independent facilities investment is disclosed.'}
        if change == "question":
            plan['question'] = "Compare revenue and investment needs in detail."
        with patch("src.generation.evidence_answer.EVIDENCE_MODEL", "another-model") if change == "model" else nullcontext():
            after = render_evidence_answer({**plan, "prior_audit": before['request_coverage'],
                "audit_fingerprints": before['audit_fingerprints']}, [literal(first), literal(second, 'r2', 'S2')], [], {}, sources, {}, {})
    assert after['complete'] and before['missing_requirement_ids'] == ['r2']
    assert calls == ([['r1'], ['r2']] if change in (None, 'new_source') else [['r1'], ['r1', 'r2']])


@pytest.mark.parametrize('bad', [False, True])
def test_compact_sentence_proofs_are_checked_against_local_evidence_indices(bad):
    quotes = ["Revenue increased.", "Margins declined."]
    answer = RequirementAnswer(requirement_id='r1', quotes=[EvidenceQuote(source_id='S1', excerpt=q) for q in quotes],
        draft_points=[NarrativePoint(heading='Results', text=' '.join(quotes), quote_indices=[0, 1])])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {'assessments': {'r1': dict(
        reason='Each sentence has its own evidence.', supported=True,
        point_support={'0': dict(assertion_evidence_indices={'0': [0], '1': [2 if bad else 1]},
            unsupported_assertion_indices=[], missing_premise=None, supports_requested_relationship=True)})}}
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        args = ({'requirements': [{**REQ, 'metric': 'reported performance'}]}, [answer], [], {},
                {'S1': {**SOURCE, 'evidence_text': ' '.join(quotes)}}, {}, {})
        if bad:
            with pytest.raises(ValueError, match='outside its own point'):
                render_evidence_answer(*args)
        else:
            assert render_evidence_answer(*args)['complete']
    properties = engine.with_structured_output.call_args.args[0]['properties']['assessments']['properties']['r1']['properties']['point_support']['properties']['0']['properties']
    assert properties['assertion_evidence_indices']['required'] == ['0', '1']
    assert 'evidence_indices' not in properties


def test_table_literal_points_keep_source_columns_group_and_units():
    text = ('Amounts in millions\n\n<table><tr><th>Year</th><th>2025</th><th>2024</th></tr>'
            '<tr><td colspan="3">Hardware segment</td></tr>'
            '<tr><td>Cost of revenues</td><td>10</td><td>9</td></tr></table>')
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {'assessments': {'r1': dict(
        reason='The row provides the requested costs.', supported=True,
        quote_assessments={'0': dict(reason='Correct source row.', answers_question=True, scope_matches=True)})}}
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        result = render_evidence_answer({'requirements': [{**REQ, 'metric': 'cost of revenue by segment'}]},
            [literal('Cost of revenues 10 9')], [], {}, {'S1': {**SOURCE, 'evidence_text': text}}, {}, {})
    assert result['complete']
    assert all(value in result['answer'] for value in ['Hardware segment', '2025', '2024', 'in millions'])


def test_heading_only_parent_does_not_consume_the_evidence_budget():
    from src.retrieval.context_builder import ContextBuilder
    from src.schemas import RankedChild, Scope
    builder = ContextBuilder.__new__(ContextBuilder)
    bodies = ['#### Revenue', 'We sell software subscriptions.']
    builder.parent_store = SimpleNamespace(get_many=lambda ids: [
        SimpleNamespace(page_content=bodies[int(key)], metadata=SOURCE) for key in ids])
    children = [RankedChild(str(i), str(i), text, SOURCE, fused_score=2-i) for i, text in enumerate(bodies)]
    result = builder.build(children, Scope(('ACME',), ('2025',), '10K'), max_parents=1, max_chars=10000)
    assert len(result.sources) == 1 and result.sources[0]['parent_id'] == '1'
    assert 'We sell software subscriptions.' in result.context


def test_sentence_evidence_union_can_exceed_the_old_whole_point_limit():
    from string import ascii_lowercase
    spans = [f"Operation {letter} is supported." for letter in ascii_lowercase]
    quotes = [' '.join(spans[:13]), ' '.join(spans[13:])]
    text = 'Operations are supported. Activities continue. Resources are available.'
    answer = RequirementAnswer(requirement_id='r1', quotes=[EvidenceQuote(source_id='S1', excerpt=q) for q in quotes],
        draft_points=[NarrativePoint(text=text, quote_indices=[0, 1])])
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {'assessments': {'r1': dict(
        reason='Each sentence is supported by its selected observations.', supported=True,
        point_support={'0': dict(assertion_evidence_indices={'0': list(range(10)), '1': list(range(10, 20)),
            '2': list(range(20, 26))}, unsupported_assertion_indices=[], missing_premise=None,
            supports_requested_relationship=True)})}}
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        result = render_evidence_answer({'requirements': [{**REQ, 'metric': 'operating activities'}]}, [answer], [], {},
            {'S1': {**SOURCE, 'evidence_text': ' '.join(quotes)}}, {}, {})
    assert result['complete']
    assert len(result['request_coverage']['assessments'][0]['point_support']['0']['evidence_spans']) == 26


def test_recovery_selection_receives_gaps_without_resending_audit_proof_bodies():
    from src.generation.evidence_answer import _selection_feedback
    rows = [dict(requirement_id=key, reason='Check the actual investment relationship.',
        missing_details=['Investment uses remain absent.'], point_support={'0': {'evidence_spans': ['Repeated source. ' * 2000]}})
        for key in ['r1', 'r2']]
    result = _selection_feedback([json.dumps({'assessments': rows})], {'r2'})
    assert result == [dict(requirement_id='r2', requested_subject=None, reason=rows[1]['reason'],
                           missing_details=rows[1]['missing_details'])]


def test_two_missing_dimensions_get_two_focused_parent_slots_in_one_recovery():
    from langchain_core.messages import ToolMessage
    from src.agent.graph import create_finrag_agent
    from src.agent.tools import RequestPlanInput
    from src.financial.calculator import FinancialCalculator
    from src.schemas import RetrievalBundle, Scope
    texts = dict(r1='We sell software subscriptions.', r2='Operating costs include payroll and hosting.',
                 r3='Capital expenditures support new datacenters.')
    metrics = dict(r1='revenue sources', r2='cost structure', r3='capital requirements')
    scope = Scope(('ACME',), ('2025',), '10K')
    plan = RequestPlanInput.model_validate({'requirements': [{**REQ, 'requirement_id': key, 'metric': metric}
        for key, metric in metrics.items()]})
    calls = []
    def retrieve(query, current_scope, **kwargs):
        calls.append(kwargs)
        values = [texts['r1'], 'An unrelated accounting policy.'] if len(calls) == 1 else [texts['r2'], texts['r3']]
        sources = [{**SOURCE, 'id': f'S{i}', 'parent_id': f'{len(calls)}-{i}', 'parent_complete': True}
                   for i in range(len(values))]
        context = '\n\n'.join(f"[SOURCE {s['id']} | TICKER: ACME | FISCAL YEAR: 2025 | TYPE: 10K]\n{text}"
                               for s, text in zip(sources, values))
        return RetrievalBundle(context, sources, current_scope, len(sources))
    def respond(schema, messages):
        payload = json.loads(messages[-1].content)
        if 'answers' in schema['properties']:
            answers = {}
            for item in payload['requirements']:
                key = item['requirement_id']
                passage = next((p for p in payload['passages'] if texts[key] in p['excerpt']), None)
                answers[key] = {'draft_points': [{'sentence_id': passage['excerpt'].split(']', 1)[0][1:]}]} if passage else {
                    'unavailable_reason': 'Operating cost inputs are absent.' if key == 'r2' else 'Productive investment uses are absent.'}
            return {'answers': answers}
        return {'assessments': {c['requirement']['requirement_id']: dict(reason='Responsive source observation.', supported=True,
            quote_assessments={str(q['index']): dict(reason='Literal evidence establishes the relationship.',
                answers_question=True, scope_matches=True) for q in c['quotes']}) for c in payload['claims']}}
    engine = SimpleNamespace(with_structured_output=lambda schema, **kw: SimpleNamespace(invoke=lambda messages: respond(schema, messages)))
    locked = 'Application-resolved scope (locked): '+json.dumps(dict(requested_groups=[['ACME', '2025']], doc_type='10K'))
    with patch('src.agent.tools.MAX_CONTEXT_PARENTS', 4), patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        result = create_finrag_agent(SimpleNamespace(retrieve=retrieve), FinancialCalculator(), permitted_scope=scope,
            plan_builder=lambda *args: plan).invoke({'messages': [('user', locked+'\nUser question: Compare revenue, costs and capital needs.')]})
    final = next(json.loads(m.content) for m in reversed(result['messages']) if isinstance(m, ToolMessage) and m.name == 'finalize_financial_answer')
    assert final['complete'] and len(calls) == 2
    assert [c['max_parents'] for c in calls] == [2, 2]
    assert len(calls[1]['requirement_queries']) == 2
    assert 'Operating cost inputs' in calls[1]['requirement_queries'][0]
    assert 'Productive investment uses' in calls[1]['requirement_queries'][1]


@pytest.mark.parametrize('supported', [False, True])
def test_direct_investment_selection_skips_a_call_and_still_requires_the_semantic_audit(supported):
    from src.generation.evidence_answer import resolve_evidence_answers
    text = 'Capital expenditures primarily support new facilities. We expect investments to support cloud capacity.'
    req = {**REQ, 'metric': 'capital requirements'}
    source = {**SOURCE, 'evidence_text': text}
    calls = dict(selection=0, audit=0)
    def respond(schema, messages):
        payload = json.loads(messages[-1].content)
        if 'answers' in schema['properties']:
            calls['selection'] += 1
            return {'answers': {'r1': {'unavailable_reason': 'Investment uses are absent.', 'draft_points': []}}}
        calls['audit'] += 1
        claim = payload['claims'][0]
        return {'assessments': {'r1': dict(reason='Check explicit investment-purpose sentences.', supported=supported,
            quote_assessments={str(q['index']): dict(reason='Assess the actual requested relationship.',
                answers_question=supported, scope_matches=True) for q in claim['quotes']})}}
    engine = SimpleNamespace(with_structured_output=lambda schema, **kw: SimpleNamespace(invoke=lambda messages: respond(schema, messages)))
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        answers = resolve_evidence_answers({'requirements': [req]}, [], {}, {'S1': source})
        assert answers[0].quotes and answers[0].unavailable_reason is None
        result = render_evidence_answer({'requirements': [req]}, answers, [], {}, {'S1': source}, {}, {})
    assert result['complete'] is supported
    assert calls == dict(selection=0, audit=1)


def test_unavailable_selection_does_not_fall_back_to_funding_or_other_source_groups():
    from src.generation.evidence_answer import resolve_evidence_answers
    engine = Mock()
    engine.with_structured_output.return_value.invoke.return_value = {'answers': {'r1': {
        'unavailable_reason': 'Productive investment uses are absent.', 'draft_points': []}}}
    sources = {'S1': {**SOURCE, 'evidence_text': 'Cash and marketable securities are sufficient to fund operations.'},
               'S2': {**SOURCE, 'id': 'S2', 'ticker': 'OTHER', 'evidence_text': 'Capital expenditures primarily support factories.'}}
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        answers = resolve_evidence_answers({'requirements': [{**REQ, 'metric': 'capital requirements'}]}, [], {}, sources)
    assert not answers[0].quotes and answers[0].unavailable_reason


def test_cost_retrieval_preserves_a_definitional_query_from_the_planned_source_label():
    from src.agent.graph import create_finrag_agent
    from src.agent.tools import RequestPlanInput
    from src.financial.calculator import FinancialCalculator
    from src.schemas import RetrievalBundle, Scope
    text = 'Cost of service revenue includes hosting and payroll.'
    scope = Scope(('ACME',), ('2025',), '10K')
    plan = RequestPlanInput.model_validate({'requirements': [{**REQ, 'metric': 'cost structure',
        'retrieval_query': 'cost of service revenue, expense drivers'}]})
    retriever = SimpleNamespace(retrieve=Mock(return_value=RetrievalBundle(
        '[SOURCE S1 | TICKER: ACME | FISCAL YEAR: 2025 | TYPE: 10K]\n'+text,
        [{**SOURCE, 'parent_id': 'costs'}], scope, 1)))
    def respond(schema, messages):
        payload = json.loads(messages[-1].content)
        if 'answers' in schema['properties']:
            point = payload['passages'][0]['excerpt'].split(']', 1)[0][1:]
            return {'answers': {'r1': {'draft_points': [{'sentence_id': point}]}}}
        return {'assessments': {'r1': dict(reason='The source defines the cost components.', supported=True,
            quote_assessments={'0': dict(reason='Actual operating inputs.', answers_question=True, scope_matches=True)})}}
    engine = SimpleNamespace(with_structured_output=lambda schema, **kw: SimpleNamespace(invoke=lambda messages: respond(schema, messages)))
    locked = 'Application-resolved scope (locked): '+json.dumps(dict(requested_groups=[['ACME', '2025']], doc_type='10K'))
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        create_finrag_agent(retriever, FinancialCalculator(), permitted_scope=scope, plan_builder=lambda *args: plan).invoke(
            {'messages': [('user', locked+'\nUser question: Explain the cost structure.')]})
    queries = retriever.retrieve.call_args.kwargs['requirement_queries']
    assert queries[0] == 'cost of service revenue includes'
    assert 'cost of service revenue, expense drivers' in queries


@pytest.mark.parametrize('subject,text,excluded', [
    (None, 'Our spending on products like Alpha, Beta and Gamma will increase.', False),
    ('storage business', 'Our spending on products like Alpha, Beta and Gamma will increase.', True),
    ('storage business', 'Our storage spending on products like Alpha and Beta will increase.', False),
    ('energy storage business', 'On the energy front, revenue increased to $12 billion.', True),
    ('energy business', 'On the energy front, revenue increased to $12 billion.', False),
    ('gold mining business', 'On the gold front, revenue increased.', False),
    ('energy storage business', 'On the storage front, deployments reached a record.', False),
    ('energy storage business', 'Energy generation and storage revenue increased $3 billion.', True),
    ('energy generation and storage business', 'Energy generation and storage revenue increased $3 billion.', False),
    ('energy storage business', 'Energy storage revenue increased $3 billion.', False),
    ('energy storage business', 'Energy generation and storage revenue includes product sales and leasing.', False),
    ('energy storage business', 'Tariffs have a larger impact on our energy generation and storage business compared to our automotive business.', True),
    ('energy generation and storage business', 'Tariffs have a larger impact on our energy generation and storage business compared to our automotive business.', False),
    ('energy storage business', 'Our energy generation and storage business designs and manufactures systems.', False),
    ('energy storage business', 'We expect increasing deployments and margin pressure from competition and tariffs.', False),
])
def test_explicit_scope_mismatch_is_independent_of_a_positive_model_verdict(subject, text, excluded):
    from src.generation.evidence_answer import _scope_exclusion
    assert bool(_scope_exclusion({**REQ, 'subject': subject}, text, ['energy generation and storage'])) is excluded


def test_broader_scope_sentences_are_excluded_before_the_generation_and_audit_calls():
    from src.generation.evidence_answer import resolve_evidence_answers
    broad = 'On the energy front, revenue increased to $12 billion.'
    good = 'We expect energy storage deployments to rise despite competitive margin pressure.'
    shared = 'Our spending on products like Alpha, Beta and Gamma will increase.'
    req = {**REQ, 'subject': 'energy storage business', 'metric': 'outlook'}
    schema_ids = []
    def respond(schema, messages):
        payload = json.loads(messages[-1].content)
        if 'answers' in schema['properties']:
            rule = schema['properties']['answers']['properties']['r1']['properties']['draft_points']['items']['properties']['sentence_id']
            pattern = schema['$defs'][rule['$ref'].rsplit('/', 1)[1]]['pattern']
            ids = [key for passage in payload['passages'] for key in re.findall(r'\[(P[1-9][0-9]*:[0-9]+)\]', passage['excerpt']) if re.fullmatch(pattern, key)]
            schema_ids.extend(ids)
            return {'answers': {'r1': {'draft_points': [{'sentence_id': ids[0]}]}}}
        return {'assessments': {'r1': dict(reason='The requested product outlook is directly stated.', supported=True,
            quote_assessments={'0': dict(source_subject='storage', source_subject_type='business_area',
                reason='A direct storage outlook remains.', answers_question=True, scope_matches=True)})}}
    engine = SimpleNamespace(with_structured_output=lambda schema, **kw: SimpleNamespace(invoke=lambda messages: respond(schema, messages)))
    source = {**SOURCE, 'evidence_text': '## Our energy generation and storage business\n\n'+' '.join([broad, good, shared])}
    with patch('src.generation.evidence_answer.get_llm_engine', return_value=engine):
        answers = resolve_evidence_answers({'requirements': [req]}, [], {}, {'S1': source})
        result = render_evidence_answer({'requirements': [req]}, answers, [], {}, {'S1': source}, {}, {})
    assert len(schema_ids) == 1 and answers[0].quotes[0].excerpt == good
    assert result['complete'] and broad not in result['answer'] and shared not in result['answer']
