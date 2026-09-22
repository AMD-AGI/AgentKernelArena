"""V2 controller tests. Synthetic CPU protocol data is never GPU validation."""
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agents.quality_loop.config import QualityLoopConfig
from agents.quality_loop.orchestrator import QualityLoop, _source_paths, _task_slug, difficulty_is_easy
from agents.quality_loop.prompts import optimizer_prompt
from agents.quality_loop.runtime import EvaluationSession
from agents.quality_loop.state import stable_fingerprint
from agents.quality_loop.filesystem import snapshot_tree
from agents.task_validator.report_schema import (
    CHECK_NAMES, COMPLETION_MARKER_FILENAME, REPORT_SCHEMA_VERSION, finalize_report,
)
from src.task_spec import TaskConfigError, load_task_spec


LOGGER = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def cpu_protocol_runtime(monkeypatch):
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {
        "gpu_arch": "gfx950", "gpu_name": "CPU protocol fixture, no GPU"})


def make_task(root, *, language='flydsl', state='unimplemented', baseline='provided'):
    root.mkdir(parents=True)
    config = {
        'schema_version': 2,
        'candidate': {'language': language, 'initial_state': state,
                      'editable': ['source/kernel.py'],
                      'entrypoints': [{'file': 'source/kernel.py', 'kind': 'function', 'symbol': 'kernel'}]},
        'baseline': {'kind': baseline},
        'evaluation': {'runner': ['python3', 'runner.py'], 'workloads': 'cases.json'},
    }
    (root / 'source').mkdir()
    (root / 'source/kernel.py').write_text(
        'def kernel(x):\n    raise NotImplementedError\n' if state == 'unimplemented'
        else 'def kernel(x):\n    return 2*x\n')
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    (root / 'cases.json').write_text('[1, 3]')
    # A real CPU scalar correctness check; timing rows below are explicitly
    # synthetic protocol fixtures, never measured GPU performance.
    (root / 'runner.py').write_text('''
import ast,json,os,sys
from pathlib import Path
import yaml
c=yaml.safe_load(Path('config.yaml').read_text())
source=Path('source/kernel.py').read_text()
if sys.argv[1]=='validate-task': role,action='task','validate-task'
else: role,action=sys.argv[1:3]
rows=[{'test_case_id':str(x),'shape':[1],'dtype':'int32','params':{'x':x},'status':'PASS'} for x in json.loads(Path('cases.json').read_text())]
result={'protocol':'arena-eval-v1','role':role,'action':action,'status':'PASS','cases':rows}
if action=='validate-task':
 result['metadata']={'candidate_state':'unimplemented' if 'NotImplementedError' in source else 'implemented'}
 for r in rows:r['checks']=['correctness','performance']
elif action=='compile':
 ast.parse(source);result['cases']=[]
elif action=='correctness':
 namespace={};exec(source,namespace)
 for r in rows:
  x=r['params']['x']; actual=2*x if role=='baseline' and c['baseline']['kind']=='provided' else namespace['kernel'](x)
  r['metrics']={'actual':actual,'reference':2*x}
  if actual!=2*x:r['status']='FAIL';result['status']='FAIL';result['reason']='wrong output'
elif action=='performance':
 for r in rows:r.update(execution_time_ms=1.0,benchmark_method='cuda_graph',metadata={'synthetic_cpu_fixture':True})
print('ARENA_EVAL_RESULT='+json.dumps(result))
sys.exit(0 if result['status']=='PASS' else 1)
''')
    return root


def workflow(root, **kwargs):
    return QualityLoop(root, QualityLoopConfig.from_dict({'target_gpu_model': 'MI355X'}),
                       logger=LOGGER, **kwargs)


def copied_workspace(monkeypatch, loop):
    def make(task_id, task_dir, stage_dir):
        stage_dir.mkdir(parents=True, exist_ok=True)
        target = stage_dir / 'workspace'
        loop._copy_task(task_dir, target)
        return target
    monkeypatch.setattr(loop, '_make_workspace', make)


def write_report(workspace, task_id, *, settings, age=0, framework_error=None):
    """Exercise the real normalizer/marker, not a claimed overall_status alone."""
    from agents.task_validator.report_v2 import V2_REPORT_SCHEMA_VERSION, DRAFT_FILENAME
    from agents.task_validator.trusted_evidence import snapshot_task_evidence
    context = json.loads(Path(os.environ['ARENA_VALIDATION_CONTEXT']).read_text())
    checks = {name: {'status': 'PASS', 'details': 'fixture evidence',
                     'evidence': [{'path': 'runner.py', 'finding': 'CPU fixture review'}]} for name in CHECK_NAMES}
    for name in ('compilation', 'correctness', 'performance'):
        checks[name]['attempts'] = [{'command': 'python3 runner.py', 'exit_code': 0, 'timed_out': False}]
    checks['benchmark_integrity'].update({
        'case_count': 2, 'valid_case_count': 2, 'benchmark_methods': ['cuda_graph'],
        **{k: True for k in ('method_metadata_complete', 'method_policy_valid', 'case_identity_complete',
                            'baseline_policy_immutable', 'state_restore_valid', 'workload_symmetric',
                            'replay_validation_valid', 'representative_inputs_valid', 'timing_boundaries_valid')},
    })
    checks['harness_integrity'].update(guard_coverage_reviewed=True, editable_targets_preserved=True)
    raw = {'validation_schema_version': V2_REPORT_SCHEMA_VERSION, 'task_name': task_id,
           'validation_request_id': settings['_task_validation_request_id'],
           'task_evidence_sha256': snapshot_task_evidence(context, task_id=context['task_id']).sha256,
           'validation_timestamp': (datetime.now(timezone.utc)-timedelta(seconds=age)).isoformat(),
           'overall_status': 'PASS', 'checks': checks, 'summary': 'CPU report fixture only'}
    (workspace / DRAFT_FILENAME).write_text(yaml.safe_dump(raw))
    return finalize_report(workspace, expected_task_name=context['task_id'], framework_error=framework_error,
                           trusted_task_evidence=context, task_schema_version=2,
                           validation_request_id=settings['_task_validation_request_id'])


def test_plan_uses_v2_state_and_platform_not_directory(tmp_path):
    task = make_task(tmp_path/'tasks/torch2hip/deep/name')
    config = yaml.safe_load((task/'config.yaml').read_text())
    config['platform_support'] = {'required_arch':'gfx942'}
    (task/'config.yaml').write_text(yaml.safe_dump(config))
    # A bundled config is not another Arena task.
    (task/'source/config.yaml').write_text('unrelated: true')
    loop = workflow(tmp_path)
    assert list(loop.discover_tasks()) == ['torch2hip/deep/name']
    assert loop.plan()['platform_deferred'] == ['torch2hip/deep/name']
    spec = load_task_spec(task/'config.yaml', task_id='torch2hip/deep/name')
    text = optimizer_prompt('contract', spec.task_id, spec)
    assert 'flydsl' in text and 'unimplemented' in text
    assert 'runner.py candidate correctness' in text
    assert _task_slug('a/b') != _task_slug('a__b')


def test_legacy_tasks_are_rejected_instead_of_silently_adapted(tmp_path):
    task = tmp_path/'tasks/old';task.mkdir(parents=True)
    (task/'config.yaml').write_text('task_type: torch2hip\nsource_file_path: [kernel.py]\n')
    with pytest.raises(TaskConfigError): workflow(tmp_path).plan()


def test_explicit_budgets_and_validator_independence(tmp_path):
    config = QualityLoopConfig.from_dict({'target_gpu_model':'MI355X', 'quality_loop':{
        'backend':{'model':'gpt-6-astra','effort':'high'},
        'reviewer':{'model':'gpt-5.6-luna','effort':'medium'},
    }})
    loop = QualityLoop(tmp_path, config, logger=LOGGER)
    run = loop._eval_config(task_id='suite/task', validator=True)
    assert run['agent']['model'] == 'gpt-5.6-terra'
    assert run['agent']['effort'] == 'medium'
    assert run['task_id'] == 'suite/task'
    assert run['_task_id'] == 'suite/task'
    assert not any(key.endswith('_timeout') for key in run['agent'])


def test_missing_tool_evidence_cannot_satisfy_easy_gate():
    config = QualityLoopConfig.from_dict({'target_gpu_model':'MI355X',
                                         'evaluation_tools': {'enabled':['gpu_asan']}})
    result = {'pass_compilation':True, 'pass_correctness':True, 'benchmark_method_consistent':True,
              'valid_baseline_cases':2, 'valid_optimized_cases':2}
    review = {'accepted':True, 'logic_equivalent':True, 'evidence_sufficient':True}
    assert not difficulty_is_easy(speedups=[6, 6, 6], result=result, review=review, config=config)
    result.update(pass_tool_gate=True, tool_policy_satisfied=True)
    assert difficulty_is_easy(speedups=[6, 6, 6], result=result, review=review, config=config)


def test_copy_and_archive_preserve_original_reports(tmp_path):
    source = make_task(tmp_path/'source')
    (source/'validation_report.yaml').write_text('original report')
    (source/COMPLETION_MARKER_FILENAME).write_text('old marker')
    QualityLoop._copy_task(source, tmp_path/'copy')
    assert not (tmp_path/'copy/validation_report.yaml').exists()
    QualityLoop._reset_path(source/'validation_report.yaml')
    saved = list((source/'.quality_loop_history').glob('validation_report.yaml-*'))
    assert len(saved)==1 and saved[0].read_text()=='original report'
    assert (source/COMPLETION_MARKER_FILENAME).read_text()=='old marker'


def test_source_scopes_support_nested_tree_and_reject_escape(tmp_path):
    task = make_task(tmp_path/'task')
    config = yaml.safe_load((task/'config.yaml').read_text())
    config['candidate']['editable']=[{'path':'source','scope':'tree'}]
    (task/'config.yaml').write_text(yaml.safe_dump(config))
    (task/'source/helper.py').write_text('value=1')
    spec=load_task_spec(task/'config.yaml',task_id='suite/task')
    assert _source_paths(spec,task)==('source/helper.py','source/kernel.py')
    (task/'source/escape.py').symlink_to(tmp_path/'outside.py')
    with pytest.raises(TaskConfigError): _source_paths(spec,task)


@pytest.mark.parametrize('failure',['missing','raw','stale','future','wrong_id','framework','tampered','modified_task'])
def test_validator_cannot_pass_missing_stale_or_untrusted_evidence(tmp_path,monkeypatch,failure):
    task=make_task(tmp_path/'task')
    def validator(settings, config_path, workspace):
        workspace=Path(workspace)
        if failure=='missing':return
        if failure=='raw':
            (workspace/'validation_report.yaml').write_text('overall_status: PASS\n');return
        write_report(workspace,'other/task' if failure=='wrong_id' else 'suite/task',
                     settings=settings,
                     age=120 if failure=='stale' else -120 if failure=='future' else 0,
                     framework_error='failed command' if failure=='framework' else None)
        if failure=='tampered': (workspace/'validation_report.yaml').write_text('overall_status: PASS\n')
        if failure=='modified_task': (workspace/'cases.json').write_text('[1]')
    loop=workflow(tmp_path,validator_launcher=validator);copied_workspace(monkeypatch,loop)
    with pytest.raises(RuntimeError):loop._validate('suite/task',task,tmp_path/'validation')


def test_validator_uses_stable_identity_and_real_completion_gate(tmp_path,monkeypatch):
    task=make_task(tmp_path/'arbitrary-copy-name')
    seen=[]
    def validator(settings,config_path,workspace):
        assert str(config_path).endswith('/tasks/SIKL/group/operator/config.yaml')
        assert settings['agent']['template']=='task_validator'
        seen.append(settings['task_id'])
        write_report(Path(workspace),settings['task_id'],settings=settings)
    loop=workflow(tmp_path,validator_launcher=validator);copied_workspace(monkeypatch,loop)
    _,report=loop._validate('SIKL/group/operator',task,tmp_path/'validation')
    assert report['overall_status']=='PASS' and seen==['SIKL/group/operator']
    assert len(loop._validation_evidence)==1


def test_actual_initial_correctness_failure_is_returned_for_repair(tmp_path, monkeypatch):
    task = make_task(tmp_path/'broken-task', state='implemented', baseline='initial_candidate')
    (task/'source/kernel.py').write_text('def kernel(x):\n    return 2*x if x < 3 else 0\n')
    def validator(settings, config_path, workspace):
        write_report(Path(workspace), settings['task_id'], settings=settings)
    loop = workflow(tmp_path, validator_launcher=validator)
    copied_workspace(monkeypatch, loop)
    _, report = loop._validate('suite/broken', task, tmp_path/'validation')
    assert report['framework_status'] == 'PASS'
    assert report['overall_status'] == 'FAIL'
    assert report['checks']['correctness']['status'] == 'FAIL'
    assert report['checks']['performance']['status'] == 'NOT_RUN'
    assert report['task_validation_failures']


def test_provided_baseline_is_not_promoted_by_copying_a_stub(tmp_path):
    task=make_task(tmp_path/'task');optimized=make_task(tmp_path/'optimized',state='implemented')
    before=snapshot_tree(task)
    accepted,reason=workflow(tmp_path)._promote_baseline('SIKL/op',task,task,optimized,tmp_path/'artifacts')
    assert accepted is False and 'independent' in reason
    assert snapshot_tree(task)==before


def test_stale_reviewer_decision_rejected(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    (workspace/'quality_loop_review.yaml').write_text('accepted: true')
    with pytest.raises(RuntimeError,match='old reviewer'):
        workflow(tmp_path)._review('suite/task',workspace,{})


def test_session_cannot_reuse_old_score_file(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    original='task_name: suite/task\npass_correctness: true\n'
    (workspace/'task_result.yaml').write_text(original)
    loop=workflow(tmp_path)
    loop._sessions[workspace]=SimpleNamespace(evaluate_candidate=lambda:yaml.safe_load(original))
    with pytest.raises(RuntimeError,match='fresh task_result'):
        loop._evaluate_session(workspace)
    assert next((workspace/'.quality_loop_history').iterdir()).read_text()==original


def test_each_measurement_uses_same_session_and_checks_identity(tmp_path):
    workspace = tmp_path/'workspace'; workspace.mkdir()
    calls = []
    def evaluate():
        result = {'task_name': 'suite/task' if not calls else 'wrong/task',
                  'speedup_ratio': 2.0}
        calls.append(result)
        (workspace/'task_result.yaml').write_text(yaml.safe_dump(result))
        return result
    loop = workflow(tmp_path)
    loop._sessions[workspace] = SimpleNamespace(evaluate_candidate=evaluate)
    loop._session_task_ids[workspace] = 'suite/task'
    assert loop._evaluate_session(workspace)['task_name'] == 'suite/task'
    with pytest.raises(RuntimeError, match='different task identity'):
        loop._evaluate_session(workspace)
    assert len(calls) == 2


@pytest.mark.parametrize('corruption', ['none', 'source', 'report', 'marker', 'record', 'escape'])
def test_resume_requires_unchanged_task_and_finalized_evidence(tmp_path, monkeypatch, corruption):
    task = make_task(tmp_path/'task')
    def validator(settings, config, workspace):
        write_report(Path(workspace), settings['task_id'], settings=settings)
    loop = workflow(tmp_path, validator_launcher=validator)
    copied_workspace(monkeypatch, loop)
    workspace, _ = loop._validate('suite/task', task, tmp_path/'validation')
    record = {'state': 'completed', 'accepted_fingerprint': stable_fingerprint(snapshot_tree(task)),
              'validation_evidence': loop._validation_evidence}
    loop.state = SimpleNamespace(task=lambda task_id: record)
    if corruption == 'source': (task/'cases.json').write_text('[1]')
    if corruption == 'report': (workspace/'validation_report.yaml').write_text('overall_status: PASS')
    if corruption == 'marker': (workspace/COMPLETION_MARKER_FILENAME).unlink()
    if corruption == 'record': record['validation_evidence'] = [{}]
    if corruption == 'escape': record['validation_evidence'][0]['workspace'] = '../elsewhere'
    assert loop._terminal_evidence_current('suite/task', task) is (corruption == 'none')


@pytest.mark.parametrize('validation_status', ['PASS', 'WARN', 'FAIL'])
def test_cross_language_promotion_keeps_v2_and_rolls_back_failed_validation(tmp_path, monkeypatch, validation_status):
    original = make_task(tmp_path/'original', state='implemented', baseline='initial_candidate')
    config = yaml.safe_load((original/'config.yaml').read_text())
    config['candidate']['initial_language'] = 'pytorch'
    (original/'config.yaml').write_text(yaml.safe_dump(config))
    loop = workflow(tmp_path)
    candidate = tmp_path/'candidate'; loop._copy_task(original, candidate)
    optimized = tmp_path/'optimized'; loop._copy_task(original, optimized)
    (optimized/'source/kernel.py').write_text('def kernel(x):\n    return x+x\n')
    artifacts = tmp_path/'artifacts'; artifacts.mkdir()
    before = snapshot_tree(candidate)
    monkeypatch.setattr(loop, '_dual_correctness_gate', lambda *args, **kwargs: True)
    monkeypatch.setattr(loop, '_validate', lambda *args: (None, {'overall_status':validation_status}))
    accepted, _ = loop._promote_baseline('suite/task', original, candidate, optimized, artifacts)
    assert accepted is (validation_status == 'PASS')
    if accepted:
        spec = load_task_spec(candidate/'config.yaml', task_id='suite/task')
        assert spec.candidate.initial_language == spec.baseline.language == 'flydsl'
        assert spec.to_mapping()['schema_version'] == 2
        assert 'return x+x' in (candidate/'source/kernel.py').read_text()
    else:
        assert snapshot_tree(candidate) == before
        assert list((tmp_path/'.quality_loop_history').iterdir())


@pytest.fixture
def core_session():
    # Main integration owns this dependency (introduced after this worker's base).
    # These tests execute the real shared lifecycle when that commit is present.
    return pytest.importorskip('src.task_session').TaskSession


@pytest.mark.parametrize('initial_state,baseline',[('implemented','initial_candidate'),('unimplemented','provided')])
def test_real_shared_lifecycle_uses_frozen_baseline_and_final_candidate(tmp_path,core_session,initial_state,baseline):
    task=make_task(tmp_path/'workspace',state=initial_state,baseline=baseline)
    spec=load_task_spec(task/'config.yaml',task_id='misleading/torch2hip')
    session=core_session.create(spec,task,tmp_path/'state')
    adapter=EvaluationSession(session,eval_config={},logger=LOGGER)
    adapter.prepare()
    assert adapter.baseline_cases
    (task/'source/kernel.py').write_text('def kernel(x):\n    return 3*x\n')
    assert adapter.check_candidate() is False
    (task/'source/kernel.py').write_text('def kernel(x):\n    return 2*x\n')
    assert adapter.check_candidate() is True
    assert session.spec.candidate.language=='flydsl'
    assert session.spec.task_id=='misleading/torch2hip'
    assert ('task_validation','candidate','correctness') not in session.results


def test_dual_gate_checks_optimized_generation_candidate_on_new_cases(tmp_path,monkeypatch,core_session):
    original=make_task(tmp_path/'original')
    proposed=make_task(tmp_path/'proposed')
    (proposed/'cases.json').write_text('[1, 3, 5]')
    optimized=make_task(tmp_path/'optimized',state='implemented')
    (optimized/'source/kernel.py').write_text('def kernel(x):\n    return 2*x if x<5 else 0\n')
    loop=workflow(tmp_path);copied_workspace(monkeypatch,loop)
    assert loop._dual_correctness_gate('SIKL/op',original,proposed,tmp_path/'gate1',optimized_workspace=optimized) is False
    (optimized/'source/kernel.py').write_text('def kernel(x):\n    return 2*x\n')
    assert loop._dual_correctness_gate('SIKL/op',original,proposed,tmp_path/'gate2',optimized_workspace=optimized) is True
    assert 'NotImplementedError' in (proposed/'source/kernel.py').read_text()


def test_case_enhancement_uses_declared_workloads_and_preserves_empty_candidate(tmp_path, monkeypatch, core_session):
    original = make_task(tmp_path/'original')
    candidate = make_task(tmp_path/'candidate')
    optimized = make_task(tmp_path/'optimized', state='implemented')
    class CaseBackend:
        def run(self, prompt, workspace, *, role):
            assert role == 'case_enhancer'
            (workspace/'cases.json').write_text('[1, 3, 5]')
    def validator(settings, config, workspace):
        write_report(Path(workspace), settings['task_id'], settings=settings)
    loop = workflow(tmp_path, backend=CaseBackend(), validator_launcher=validator)
    copied_workspace(monkeypatch, loop)
    assert loop._enhance_cases('suite/task', original, candidate, 'missing x=5',
                               tmp_path/'artifacts', optimized_workspace=optimized)
    assert json.loads((candidate/'cases.json').read_text()) == [1, 3, 5]
    assert 'NotImplementedError' in (candidate/'source/kernel.py').read_text()


def test_rejected_baseline_on_new_cases_rolls_back_instead_of_accepting_change(tmp_path, monkeypatch, core_session):
    original = make_task(tmp_path/'original', state='implemented', baseline='initial_candidate')
    (original/'source/kernel.py').write_text('def kernel(x):\n    return 2*x if x<5 else 0\n')
    candidate = tmp_path/'candidate'; QualityLoop._copy_task(original, candidate)
    optimized = make_task(tmp_path/'optimized', state='implemented')
    class CaseBackend:
        def run(self, prompt, workspace, *, role):
            (workspace/'cases.json').write_text('[1, 3, 5]')
    loop = workflow(tmp_path, backend=CaseBackend())
    copied_workspace(monkeypatch, loop)
    before = snapshot_tree(candidate)
    assert not loop._enhance_cases('suite/task', original, candidate, 'missing x=5',
                                  tmp_path/'artifacts', optimized_workspace=optimized)
    assert snapshot_tree(candidate) == before
    assert list((tmp_path/'.quality_loop_history').iterdir())


def test_shared_prompt_uses_stable_task_id_for_scratch_package(tmp_path):
    pytest.importorskip('src.task_prompt')
    from src.prompt_builder import prompt_builder
    task = make_task(tmp_path/'renamed-scratch-copy')
    loop = workflow(tmp_path)
    prompt = prompt_builder(str(task/'config.yaml'), task,
                            loop._eval_config(task_id='SIKL/group/operator'), LOGGER)
    assert 'SIKL/group/operator' in prompt and 'source/kernel.py' in prompt


def test_optimizer_wiring_uses_shared_session_and_prompt(tmp_path, monkeypatch, core_session):
    pytest.importorskip('src.task_prompt')
    import src.evaluator
    task = make_task(tmp_path/'task')
    sessions = []
    class Optimizer:
        def run(self, prompt, workspace, *, role):
            assert role == 'optimizer'
            assert 'SIKL/group/operator' in prompt
            assert 'unimplemented' in prompt and 'flydsl' in prompt
            assert 'agent_context.json' in prompt
            (workspace/'source/kernel.py').write_text('def kernel(x):\n    return 2*x\n')
    def fixture_evaluator(session, *, eval_config, logger):
        # Contract stub for the parent's not-yet-integrated scoring entrypoint.
        # Compilation/correctness below use the real session and scalar runner;
        # the resulting timing/score remains explicitly synthetic CPU data.
        assert eval_config['_task_id'] == 'SIKL/group/operator'
        for action in ('compile', 'correctness', 'performance'):
            assert session.candidate_action(action).result.passed
        assert 'NotImplementedError' in (session.baseline_workspace/'source/kernel.py').read_text()
        sessions.append(session)
        result = {'task_name':session.spec.task_id, 'pass_compilation':True,
                  'pass_correctness':True, 'speedup_ratio':1.0,
                  'synthetic_cpu_fixture':True}
        (session.workspace/'task_result.yaml').write_text(yaml.safe_dump(result))
        return result
    monkeypatch.setattr(src.evaluator, 'evaluate_task_session', fixture_evaluator, raising=False)
    loop = workflow(tmp_path, backend=Optimizer()); copied_workspace(monkeypatch, loop)
    workspace, baseline_cases, result = loop._optimize_once('SIKL/group/operator', task, tmp_path/'optimize')
    assert baseline_cases and result['synthetic_cpu_fixture'] is True
    loop._evaluate_session(workspace)
    assert len(sessions) == 2 and sessions[0] is sessions[1]
    assert 'NotImplementedError' in (task/'source/kernel.py').read_text()
