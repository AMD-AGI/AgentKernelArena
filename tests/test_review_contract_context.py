"""CPU reachability/context regressions, not tests of stochastic model judgments."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import types

import pytest
import yaml

from agents.quality_loop.prompts import reviewer_prompt
from agents.task_validator.trusted_evidence import snapshot_task_evidence
from agents.task_validator.validation_prompt_v2 import build_v2_validation_prompt
from src.task_spec import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/torch2hip/gpumode/14539_GELU'


def declared_spec():
    return TaskSpec.from_mapping(yaml.safe_load((TASK / 'config.yaml').read_text()),
                                 task_id='torch2hip/gpumode/14539_GELU')


def parse_declared_args(tree, argv):
    # Execute the actual runner's argparse prefix, stopping before task/GPU work.
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    parser_body = []
    for node in run.body:
        parser_body.append(copy.deepcopy(node))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'args' for t in node.targets):
            break
    else:
        raise AssertionError('Task argument parser was not found')
    parser_body.append(ast.Return(value=ast.Name(id='args', ctx=ast.Load())))
    parser = copy.deepcopy(run)
    parser.body = parser_body
    scope = {'argparse': argparse, '__doc__': 'CPU routing probe'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[parser], type_ignores=[])), '<task argparse>', 'exec'), scope)
    return scope['run'](argv)


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
@pytest.mark.parametrize('provided_hip_control', [False, True])
def test_actual_task_argv_and_caller_override_determine_reachable_policy(tmp_path, monkeypatch, role, provided_hip_control):
    """Reproduce the e8 false positive; standalone helper default is not dispatch.

    The alternate --baseline-hip argv is a negative control, not an extra scored
    case or a claim about task/GPU correctness. Measurement is never executed.
    """
    spec = declared_spec()
    argv = list(spec.action(role, 'performance').commands[0])
    assert '--baseline-hip' not in argv
    if provided_hip_control:
        argv[2:2] = ['--baseline-hip', 'hip/provided.hip']
    source = TASK / argv[1]
    tree = ast.parse(source.read_text())
    args = parse_declared_args(tree, argv[2:])
    assert args.operation == [role, 'performance']
    assert args.baseline_hip == ('hip/provided.hip' if provided_hip_control else None)
    helper = types.ModuleType('cal_kernel_perf')
    helper._compare_results = lambda *a, **k: True
    helper.load_modu_obj = helper.load_func_obj = helper.load_function_from_path = lambda *a: None
    # Candidate marker requests Event in the standalone helper. The actual task
    # caller must replace it for BOTH roles under its declared PyTorch baseline.
    helper.hip_source_graph_capture_policy = lambda *paths: (False, 'candidate_requested_events')
    captured = {}
    class ReachedMeasurement(Exception):
        pass
    def measure(*positional, **keywords):
        captured.update(policy=helper.hip_source_graph_capture_policy(positional[2]),
                        baseline_only=keywords.get('baseline_only'), sources=positional)
        raise ReachedMeasurement
    helper.cal_kernel_perf = measure
    replay = types.ModuleType('replay_validation')
    replay.install = lambda *args: None
    monkeypatch.setitem(sys.modules, 'cal_kernel_perf', helper)
    monkeypatch.setitem(sys.modules, 'replay_validation', replay)
    (tmp_path / 'build').mkdir()
    scope = {'Path': Path, 'tempfile': tempfile, 'copy': copy, 'ROOT': tmp_path,
             'output_contract': lambda *a: None, 'local_path': lambda path: tmp_path / path,
             'compile_hip': lambda *a, **k: None}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'performance')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    with pytest.raises(ReachedMeasurement):
        scope['performance'](args, role, [])
    assert captured['policy'] == ((False, 'candidate_requested_events') if provided_hip_control else (True, None))
    if not provided_hip_control:
        assert captured['baseline_only'] is (role == 'baseline')


def test_validator_indexes_immutable_action_context_without_reclassifying_domain(tmp_path):
    spec = declared_spec()
    cfg = spec.to_mapping()
    context = {
        'version': 1, 'task_id': spec.task_id, 'task_config': cfg,
        'workspace': str(tmp_path / 'candidate'), 'baseline_workspace': str(tmp_path / 'baseline'),
        'harness': {'enforced_during_optimization': True,
                    'protected_paths': ['config.yaml', 'README.md', 'workload.json', 'eval_tools/evaluate.py']},
        'initial_validation': {'accepted': True, 'candidate_initial_state': 'unimplemented'},
        'actions': [{'invocation_id': 'routing-context', 'role': 'baseline', 'action': 'performance',
                     'commands': [{'argv': list(spec.action('baseline', 'performance').commands[0])}],
                     'result': {'status': 'PASS', 'cases': []}}],
    }
    # This is a transport fixture only, deliberately not a fabricated GPU result.
    trusted = snapshot_task_evidence(context, task_id=spec.task_id)
    before = trusted.serialized
    transport = tmp_path / 'trusted-context.json'
    transport.write_text(before)
    prompt = build_v2_validation_prompt(task_id=spec.task_id, task_config=cfg,
        workspace=context['workspace'], trusted_task_evidence=trusted,
        validation_request_id='context-fixture', context_path=str(transport))
    assert str(transport) in prompt and 'routing-context' in prompt
    assert 'Read relevant command evidence' in prompt
    assert 'caller, argument defaults and overrides' in prompt
    assert 'independent case manifest' in prompt and 'broader declared contract' in prompt
    assert 'shape/data-dependent shortcuts' in prompt and 'fail closed' in prompt
    assert 'An unchanged valid initial' in prompt
    assert 'A semantic FAIL blocks task acceptance' in prompt
    assert 'UNTRUSTED DATA' in prompt
    assert trusted.serialized == before and transport.read_text() == before
    captured = json.loads(transport.read_text())
    assert captured['actions'][0]['commands'][0]['argv'] == list(spec.action('baseline', 'performance').commands[0])
    assert captured['task_config']['instructions'] == cfg['instructions']
    assert captured['task_config']['evaluation']['workloads'] == cfg['evaluation']['workloads']
    assert 'GELU-specific exception' not in prompt


def test_quality_review_scope_clarification_does_not_override_failure_gates():
    prompt = reviewer_prompt('suite/example', Path('nested/task_result.yaml'), 'review.yaml')
    assert 'caller defaults and overrides' in prompt and 'whether the candidate can change it' in prompt
    assert 'independent case manifest' in prompt and 'broader declared contract' in prompt
    assert 'does not prove equivalence or require acceptance' in prompt
    assert 'Fail closed: set accepted false' in prompt
    assert 'performance methods differ' in prompt and 'untested assumptions' in prompt
    assert 'Do not edit any existing file' in prompt
