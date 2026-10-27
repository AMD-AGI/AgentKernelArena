"""Source-bound review tests use a fake backend, never claim model/GPU validation."""
import copy
import json
import logging
from pathlib import Path

import pytest

from src.candidate_verification import apply_review, bind_policy, finalize, policy, verify_candidate
from tests.test_task_session_v2 import RUNNER, create


def result(session):
    return dict(pass_compilation=True, pass_correctness=True, pass_tool_gate=True,
                benchmark_method_consistent=True, workload_consistent=True,
                measurement_kind='serving', average_speedup=1.12,
                evaluated_candidate_sources=session.candidate_source_evidence()['sources'])


def draft(root, *, changed=True, status='PASS'):
    request = json.loads((root/'request.json').read_text())
    check = dict(status=status, details='Reviewed the frozen implementation and result.',
                 evidence=[dict(file='candidate/kernel.py', line=1)])
    return dict(request_id=request['request_id'], evidence_sha256=request['evidence_sha256'],
                checks={name: {**copy.deepcopy(check), **({'changed': changed} if name == 'implementation_change' else {})}
                        for name in ('allowed_scope', 'preserved_semantics', 'measurement_integrity', 'implementation_change')})


def test_policy_required_for_serving_and_fixed_on_resume(tmp_path):
    assert policy({}) is None
    config = policy({'candidate_verification': {'timeout_seconds': 30}}, serving=True)
    assert config == dict(backend='claude_code', timeout_seconds=30)
    path = tmp_path/'policy.json'
    bind_policy(path, config)
    bind_policy(path, config)
    with pytest.raises(ValueError, match='resume'):
        bind_policy(path, {**config, 'timeout_seconds': 31})
    for value in [False, {}, {'backend': 'unknown'}, {'timeout_seconds': True}]:
        if value == {}:
            continue
        with pytest.raises(ValueError):
            policy({'candidate_verification': value})


def test_review_reads_frozen_sources_and_preserves_original(tmp_path):
    session = create(tmp_path)
    (session.workspace/'kernel.py').write_text('def compute():\n    return 1 + 2\n')
    measured = result(session)
    def backend(prompt, root, settings, logger):
        assert 'UNTRUSTED' in prompt
        assert 'every compute kernel used by the model' in prompt
        assert 'separate unit test for each' in prompt
        assert 'valid general implementation' in prompt
        assert 'Missing evidence or unclear generality is\nINCONCLUSIVE' in prompt
        assert 'return 3' in (root/'baseline/kernel.py').read_text()
        assert 'return 1 + 2' in (root/'candidate/kernel.py').read_text()
        assert json.loads((root/'evaluation.json').read_text()) == measured
        assert settings['timeout_seconds'] > 0
        return draft(root)
    report = verify_candidate(session, measured, {}, logging.getLogger(__name__), backend=backend)
    assert report['status'] == 'PASS', report
    apply_review(measured, report)
    assert measured['pass_verification'] and measured['average_speedup'] == 1.12
    assert len(list(session.state_directory.glob('candidate-review-*/review.json'))) == 1


@pytest.mark.parametrize('failure', ['wrong_request', 'wrong_hash', 'bad_line', 'missing_check', 'mutated_evidence', 'mutated_candidate', 'timeout', 'inconclusive'])
def test_review_failures_block_credit(tmp_path, failure):
    session = create(tmp_path)
    measured = result(session)
    def backend(prompt, root, settings, logger):
        value = draft(root)
        if failure == 'wrong_request': value['request_id'] = 'old'
        if failure == 'wrong_hash': value['evidence_sha256'] = 'old'
        if failure == 'bad_line': value['checks']['allowed_scope']['evidence'][0]['line'] = 99999
        if failure == 'missing_check': del value['checks']['allowed_scope']
        if failure == 'mutated_evidence': (root/'candidate/kernel.py').write_text('changed')
        if failure == 'mutated_candidate': (session.workspace/'kernel.py').write_text('def compute(): return 7\n')
        if failure == 'timeout': raise TimeoutError('expired')
        if failure == 'inconclusive': value['checks']['allowed_scope']['status'] = 'INCONCLUSIVE'
        return value
    report = verify_candidate(session, measured, {}, logging.getLogger(__name__), backend=backend)
    assert report['status'] != 'PASS'
    apply_review(measured, report)
    assert measured['pass_verification'] is False and measured['average_speedup'] == 0
    assert measured['pass_correctness'] is True  # Keep independent gate identities.


def test_failed_numeric_gate_never_calls_reviewer(tmp_path):
    session = create(tmp_path)
    measured = {**result(session), 'pass_correctness': False, 'average_speedup': 0}
    def backend(*args): raise AssertionError('must not run')
    report = verify_candidate(session, measured, {}, logging.getLogger(__name__), backend=backend)
    assert report['status'] == 'NOT_RUN'
    apply_review(measured, report)
    assert not measured['pass_correctness'] and not measured['pass_verification']


def test_measured_source_identity_and_deadline_fail_closed(tmp_path, monkeypatch):
    session = create(tmp_path)
    measured = result(session)
    measured['evaluated_candidate_sources'] = {}
    report = verify_candidate(session, measured, {}, logging.getLogger(__name__))
    assert 'differs' in report['error']
    monkeypatch.setenv('ARENA_TASK_DEADLINE', '1')
    report = verify_candidate(session, result(session), {}, logging.getLogger(__name__))
    assert 'budget' in report['error']


def test_noop_review_caps_gain_without_discarding_samples():
    measured = dict(average_speedup=1.12, serving_measurement=dict(speedup_ratio=1.12,
        raw_speedup_ratio=1.12, paired_ratios={'case': [1.12]}, scored_case_ratios={'case': 1.12},
        gain_threshold_met={'case': True}))
    apply_review(measured, dict(required=True, status='PASS', implementation_changed=False))
    assert measured['pass_verification'] and measured['average_speedup'] == 1
    assert measured['serving_measurement']['raw_speedup_ratio'] == 1.12
    assert measured['serving_measurement']['scored_case_ratios'] == {'case': 1}
    assert measured['serving_measurement']['gain_threshold_met'] == {'case': False}


@pytest.mark.parametrize('status,accepted', [('PASS', True), ('FAIL', False), ('INCONCLUSIVE', False)])
def test_review_gate_controls_final_acceptance_and_exports(tmp_path, monkeypatch, status, accepted):
    import sys
    from types import SimpleNamespace
    import yaml
    from tests.test_task_run_v2 import package
    from src.task_run import run_task_v2
    monkeypatch.setattr('src.task_runtime._runtime_identity', lambda: {
        'gpu_arch': 'gfx950', 'gpu_name': 'CPU protocol fixture, no GPU', 'torch': 'fixture'})
    monkeypatch.setenv('AGENT_KERNEL_ARENA_PYTHON', sys.executable)
    monkeypatch.setattr('agents.candidate_verifier.backend.review',
        lambda prompt, root, settings, logger: draft(root, status=status))
    path = package(tmp_path, exporter='import os\nfrom pathlib import Path\nPath(os.environ["ARENA_EXPORT_PATH"]).parent.mkdir(parents=True,exist_ok=True)\nPath(os.environ["ARENA_EXPORT_PATH"]).write_text("exported")\n')
    # A synthetic 2x candidate keeps the score assertion able to tell accepted from rejected.
    (path.parent / 'evaluate.py').write_text(RUNNER.replace(
        'execution_time_ms=0.1', "execution_time_ms=(0.1 if role == 'baseline' else 0.05)"))
    _, workspace = run_task_v2(eval_config={'agent': {'template': 'codex'},
        'candidate_verification': {'backend': 'claude_code'}}, agent=SimpleNamespace(value='codex'),
        agent_launcher=lambda **_: None, task_name='suite/protocol_fixture', task_config_dir=str(path),
        run_directory=tmp_path/'run', timestamp='fixture', logger=logging.getLogger(__name__))
    saved = yaml.safe_load((workspace/'task_result.yaml').read_text())
    assert saved['pass_correctness'] and saved['pass_verification'] is accepted
    assert saved['candidate_accepted'] is accepted
    assert bool(saved['exports']) is accepted
    assert saved['speedup_ratio'] == (2 if accepted else 0)
    assert saved['score'] == (pytest.approx(0.475) if accepted else 0)


def test_backend_uses_read_only_tools_and_never_exposes_shell(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from agents.candidate_verifier import backend
    calls = []
    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, timed_out=False, error=None,
            output=json.dumps({'type':'result','subtype':'success','result':'{"ok":true}'}))
    monkeypatch.setattr(backend, '_run_backend', run)
    assert backend.review('review',tmp_path,dict(backend='claude_code',timeout_seconds=12),logging.getLogger(__name__)) == {'ok': True}
    cmd, kwargs = calls[0]
    assert '--safe-mode' in cmd and '--restricted' in cmd
    assert cmd[cmd.index('--tools')+1] == 'Read,Glob,Grep'
    assert cmd[cmd.index('--allowedTools')+1] == 'Read,Glob,Grep'
    assert '--strict-mcp-config' in cmd and '--no-session-persistence' in cmd
    assert kwargs['timeout_seconds'] == 12


def test_evidence_symlink_injection_fails(tmp_path):
    session=create(tmp_path)
    def backend(prompt, root, config, logger):
        value=draft(root)
        (root/'injected').symlink_to(tmp_path/'outside')
        return value
    report=verify_candidate(session,result(session),{},logging.getLogger(__name__),backend=backend)
    assert report['status']=='ERROR' and 'symlink' in report['error']

@pytest.mark.parametrize('payload,valid', [
    ('```json\n{"ok": true}\n```', True),
    ('```\n{"ok": true}\n```', True),
    ('Explanation\n```json\n{"ok": true}\n```', False),
    ('```json\n{"ok": true}\n```\n{"ok": false}', False),
])
def test_review_final_object_may_have_one_complete_fence(monkeypatch,tmp_path,payload,valid):
    from types import SimpleNamespace
    from agents.candidate_verifier import backend
    monkeypatch.setattr(backend,'_run_backend',lambda *a,**k:SimpleNamespace(
        returncode=0,timed_out=False,error=None,output=json.dumps(
            dict(type='result',subtype='success',result=payload))))
    args=('review',tmp_path,dict(backend='claude_code',timeout_seconds=12),logging.getLogger(__name__))
    if valid:
        assert backend.review(*args)=={'ok': True}
    else:
        with pytest.raises(ValueError): backend.review(*args)


def test_review_includes_trusted_original_runtime_source(tmp_path):
    from types import SimpleNamespace
    import hashlib
    session=create(tmp_path)
    text='def compute(x):\n    return x + 1\n'
    command=dict(original_compute_sources=[dict(module='runtime.compute',module_source=text,
        module_sha256=hashlib.sha256(text.encode()).hexdigest())])
    session.results[('candidate_evaluation','candidate','compile')]=SimpleNamespace(
        result=SimpleNamespace(to_mapping=lambda:dict(metadata={'commands':[command]})))
    def backend(prompt,root,settings,logger):
        assert (root/'runtime_originals/runtime/compute.py').read_text()==text
        assert 'execution.json' in prompt
        return draft(root)
    report=verify_candidate(session,result(session),{},logging.getLogger(__name__),backend=backend)
    assert report['status']=='PASS',report
    command['original_compute_sources'][0]['module_sha256']='0'*64
    report=verify_candidate(session,result(session),{},logging.getLogger(__name__),backend=backend)
    assert report['status']=='ERROR' and 'digest' in report['error']
