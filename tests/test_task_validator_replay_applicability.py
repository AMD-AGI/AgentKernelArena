"""Captured event-only WARN regression and adversarial applicability tests (CPU)."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from agents.task_validator.report_schema import (
    HARD_BENCHMARK_REVIEW_FIELDS, _normalize_benchmark_integrity,
    finalize_report, normalize_report, validation_report_is_complete,
)
from agents.task_validator.trusted_evidence import snapshot_task_evidence
from agents.task_validator.validation_prompt_v2 import build_v2_validation_prompt
from tests.test_task_validator_v2 import context, draft, normalized, replace_stdout

FIXTURE = Path(__file__).parent / 'fixtures/task_validator/event_only_moe_gemm.json'


@pytest.fixture
def captured():
    return json.loads(FIXTURE.read_text())


def review(captured, *, rebind=False):
    ctx, raw = captured['context'], captured['draft']
    if rebind:
        raw['task_evidence_sha256'] = snapshot_task_evidence(ctx, task_id=ctx['task_id']).sha256
    return normalize_report(raw, expected_task_name=ctx['task_id'], trusted_task_evidence=ctx,
                            validation_request_id=raw['validation_request_id'])


def performance_records(ctx):
    return [r for r in ctx['actions'] if r.get('result', {}).get('action') == 'performance']


def mutate_case(ctx, mutation, *, role='baseline', index=0):
    rec = next(r for r in performance_records(ctx) if r['result']['role'] == role)
    replace_stdout(rec, lambda obj: mutation(obj['cases'][index]))


def event_row(row):
    row['benchmark_method'] = 'cuda_event_fallback'
    row['metadata'] = {'timed_output_checked': True, 'device_timing': {
        'benchmark_method': 'cuda_event_fallback', 'benchmark_fallback_reason': 'backend_requires_events'}}


def test_captured_141050_event_only_warn_becomes_explicit_na(captured):
    original = deepcopy(captured)
    assert captured['original_finalized']['overall_status'] == 'WARN'
    assert captured['original_finalized']['policy_findings'] == [
        'benchmark_integrity: replay_validation_valid is undetermined']
    assert captured['draft']['checks']['benchmark_integrity']['status'] == 'PASS'
    ctx = captured['context']
    assert [len(r['result']['cases']) for r in ctx['actions']] == [108, 0, 108, 16]
    assert snapshot_task_evidence(ctx, task_id=ctx['task_id']).sha256 == captured['draft']['task_evidence_sha256']
    result = review(captured)
    assert result['overall_status'] == 'PASS', result['validation_errors']
    check = result['checks']['benchmark_integrity']
    assert check['replay_validation_applicability']['status'] == 'not_applicable'
    assert check['replay_validation_applicability']['case_count'] == 16
    assert check['replay_validation_applicability']['roles'] == ['baseline']
    assert check['replay_validation_valid'] is None
    assert check['agent_reported_status'] == 'PASS'
    assert not result['policy_findings']
    assert captured == original


@pytest.mark.parametrize('mutation', [
    lambda row: row['metadata'].pop('timed_output_checked'),
    lambda row: row['metadata'].update(timed_output_checked=False),
    lambda row: row['metadata'].update(timed_output_checked=1),
    lambda row: row['metadata'].pop('device_timing'),
    lambda row: row['metadata'].update(device_timing=None),
    lambda row: row['metadata']['device_timing'].pop('benchmark_fallback_reason'),
    lambda row: row['metadata']['device_timing'].update(benchmark_fallback_reason='   '),
    lambda row: row['metadata']['device_timing'].pop('benchmark_method'),
    lambda row: row['metadata']['device_timing'].update(benchmark_method='cuda_graph'),
    lambda row: row['metadata']['device_timing'].update(benchmark_method='mixed:cuda_event_fallback'),
    lambda row: row['metadata'].update(benchmark_method='cuda_graph'),
    lambda row: row['metadata']['device_timing'].update(benchmark_method_consistent=False),
    lambda row: row['metadata']['device_timing'].update(benchmark_method_consistent=None),
    lambda row: row['metadata']['device_timing'].update(benchmark_method_consistent=0),
    lambda row: row['metadata'].update(benchmark_method_consistent=False),
    lambda row: row.update(benchmark_method_consistent=False),
])
def test_one_incomplete_or_contradictory_event_case_cannot_grant_na(captured, mutation):
    mutate_case(captured['context'], mutation, index=15)
    result = review(captured, rebind=True)
    assert result['overall_status'] == 'WARN', result['validation_errors']
    assert result['checks']['benchmark_integrity']['replay_validation_applicability']['status'] == 'undetermined'
    assert 'benchmark_integrity: replay_validation_valid is undetermined' in result['policy_findings']


@pytest.mark.parametrize('all_graph', [False, True])
def test_graph_or_mixed_actual_methods_require_replay(captured, all_graph):
    rec = performance_records(captured['context'])[0]
    def change(obj):
        for row in obj['cases'] if all_graph else obj['cases'][:1]:
            row['benchmark_method'] = 'cuda_graph'
            row['metadata']['device_timing']['benchmark_method'] = 'cuda_graph'
    replace_stdout(rec, change)
    captured['draft']['checks']['benchmark_integrity']['replay_validation_applicability'] = {'status':'not_applicable'}
    result = review(captured, rebind=True)
    assert result['overall_status'] == 'WARN'
    assert result['checks']['benchmark_integrity']['replay_validation_applicability']['status'] == 'required'


@pytest.mark.parametrize('field', HARD_BENCHMARK_REVIEW_FIELDS)
@pytest.mark.parametrize('value,status', [(False, 'FAIL'), (None, 'WARN')])
def test_event_na_never_skips_independent_benchmark_fields(captured, field, value, status):
    captured['draft']['checks']['benchmark_integrity'][field] = value
    result = review(captured)
    assert result['overall_status'] == status
    assert any(field in finding for finding in result['policy_findings'])


@pytest.mark.parametrize('check_name', ['benchmark_integrity', 'correctness_implementation_review'])
@pytest.mark.parametrize('status', ['WARN', 'FAIL'])
def test_event_na_preserves_independent_reviewer_verdict(captured, check_name, status):
    captured['draft']['checks'][check_name]['status'] = status
    result = review(captured)
    assert result['overall_status'] == status
    if check_name == 'benchmark_integrity':
        assert result['checks'][check_name]['agent_reported_status'] == status


def test_explicit_negative_replay_field_is_not_erased(captured):
    captured['draft']['checks']['benchmark_integrity']['replay_validation_valid'] = False
    result = review(captured)
    assert result['overall_status'] == 'WARN'
    assert result['checks']['benchmark_integrity']['replay_validation_valid'] is False


def test_missing_advisory_field_is_still_malformed(captured):
    del captured['draft']['checks']['benchmark_integrity']['replay_validation_valid']
    assert review(captured)['overall_status'] == 'FAIL'


def test_missing_fallback_policy_review_still_fails(captured):
    captured['draft']['checks']['benchmark_integrity']['event_fallback_reasons'] = []
    assert review(captured)['overall_status'] == 'FAIL'


def test_model_only_metadata_cannot_grant_na(tmp_path):
    ctx = context(tmp_path)
    raw = draft(ctx)
    raw['checks']['benchmark_integrity'].update(replay_validation_valid=None,
        replay_validation_applicability={'status':'not_applicable'}, benchmark_methods=['cuda_event_fallback'])
    result = normalized(ctx, raw)
    assert result['overall_status'] == 'WARN'
    assert result['checks']['benchmark_integrity']['replay_validation_applicability']['status'] == 'required'


@pytest.mark.parametrize('candidate_event', [False, True])
def test_all_executed_roles_not_just_baseline_control_applicability(tmp_path, candidate_event):
    ctx = context(tmp_path, state='implemented', baseline_kind='provided')
    for rec in performance_records(ctx):
        if rec['result']['role'] == 'baseline' or candidate_event:
            replace_stdout(rec, lambda obj: [event_row(row) for row in obj['cases']])
    raw = draft(ctx)
    raw['checks']['benchmark_integrity'].update(replay_validation_valid=None,
        event_fallback_reasons=['backend_requires_events'])
    result = normalized(ctx, raw)
    assert result['overall_status'] == ('PASS' if candidate_event else 'WARN')
    app = result['checks']['benchmark_integrity']['replay_validation_applicability']
    assert app['roles'] == ['baseline','candidate'] and app['case_count'] == 4
    assert app['status'] == ('not_applicable' if candidate_event else 'required')


@pytest.mark.parametrize('mutation', [
    lambda ctx: ctx['actions'].pop(),
    lambda ctx: performance_records(ctx)[0]['result']['cases'].pop(),
    lambda ctx: performance_records(ctx)[0]['commands'][0].update(returncode=1),
    lambda ctx: performance_records(ctx)[0]['commands'][0].update(stdout='No recorded result'),
])
def test_invalid_or_missing_trusted_actions_never_grant_na(captured, mutation):
    mutation(captured['context'])
    result = review(captured, rebind=True)
    assert result['overall_status'] == 'FAIL'
    assert result['checks']['benchmark_integrity']['replay_validation_applicability']['status'] == 'undetermined'


def test_legacy_report_without_trusted_context_retains_warning(captured):
    check = deepcopy(captured['draft']['checks']['benchmark_integrity'])
    check.update(case_count=16, valid_case_count=16, benchmark_methods=['cuda_event_fallback'],
                 replay_validation_applicability={'status':'not_applicable'})
    errors, findings = [], []
    assert _normalize_benchmark_integrity(check,'PASS',errors,findings) == 'WARN'
    assert not errors


def test_real_finalizer_writes_na_and_valid_completion_marker_in_new_workspace(captured, tmp_path):
    ctx, raw = captured['context'], captured['draft']
    workspace = tmp_path/'candidate'; workspace.mkdir()
    ctx.update(workspace=str(workspace), baseline_workspace=str(tmp_path/'baseline'))
    raw['task_evidence_sha256'] = snapshot_task_evidence(ctx, task_id=ctx['task_id']).sha256
    (workspace/'validation_report.draft.yaml').write_text(yaml.safe_dump(raw))
    result = finalize_report(workspace, expected_task_name=ctx['task_id'], trusted_task_evidence=ctx,
                             validation_request_id=raw['validation_request_id'])
    assert result['overall_status'] == 'PASS'
    assert validation_report_is_complete(workspace)
    saved = yaml.safe_load((workspace/'validation_report.yaml').read_text())
    assert saved['checks']['benchmark_integrity']['replay_validation_valid'] is None
    assert saved['checks']['benchmark_integrity']['replay_validation_applicability']['status'] == 'not_applicable'


def test_prompt_explains_event_na_and_retains_measured_output_review(captured):
    ctx = captured['context']
    prompt = build_v2_validation_prompt(task_id=ctx['task_id'],task_config=ctx['task_config'],
        workspace=ctx['workspace'],trusted_task_evidence=snapshot_task_evidence(ctx,task_id=ctx['task_id']),
        validation_request_id=captured['draft']['validation_request_id'])
    assert 'leave replay_validation_valid\n   null' in prompt
    assert 'timed_output_checked true for every case' in prompt
    assert 'N/A does not erase it' in prompt
