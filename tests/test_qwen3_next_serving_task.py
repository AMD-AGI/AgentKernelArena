"""CPU contract checks; GPU qualification remains a separate run."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
import yaml

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.task_protocol import parse_command_result

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/e2e/qwen3_next_80b_sglang'


@pytest.fixture
def checks():
    spec = importlib.util.spec_from_file_location('q80_checks', TASK/'scripts/checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_recipe_and_executable_manifest(tmp_path):
    recipe = ROOT/'tasks/e2e/sglang_mi355x/benchmark_sglang_qwen3_next_80b_fp8/benchmark_sglang_qwen3_next_80b_fp8.yaml'
    assert (TASK/'benchmark.yaml').read_bytes() == recipe.read_bytes()
    lock = json.loads((TASK/'runtime.lock.json').read_text())
    assert lock['workload_sha256'] == hashlib.sha256(recipe.read_bytes()).hexdigest()
    assert lock['model']['id'] == 'Qwen/Qwen3-Next-80B-A3B-Instruct-FP8'
    assert yaml.safe_load((TASK/'config.yaml').read_text())['evaluation']['measurement']['pairs'] == 3
    task = tmp_path/'task'
    shutil.copytree(TASK, task)
    materialize_perf_helpers_in_workspace(task)
    result = subprocess.run([sys.executable, 'scripts/evaluate.py', 'validate-task'], cwd=task,
                            capture_output=True, text=True)
    parsed = parse_command_result(result.stdout, role='task', action='validate-task',
                                  returncode=result.returncode, measurement='serving')
    assert parsed.passed, result.stdout + result.stderr
    params = parsed.cases[0]['params']
    assert params['num_requests'] == 320 and params['tp'] == 4
    assert all(819 <= n <= 1024 for n in params['output_lens'])


def test_per_rank_dimensions_and_numerical_negative_controls(checks):
    args = checks.make_case(2, device='cpu')
    assert args[0].shape == (2, 2048)
    assert args[6].shape[-3:] == (8, 128, 128)
    expected, state = checks.oracle(args)
    rules = json.loads((TASK/'runtime.lock.json').read_text())['correctness']
    checks.compare_tensor(expected, expected, rules)
    checks.compare_tensor(state, state, rules, state=True)
    for bad in (torch.zeros_like(expected), expected * 0.8, expected + 0.015):
        with pytest.raises((AssertionError, ValueError)):
            checks.compare_tensor(bad, expected, rules)
    with pytest.raises((AssertionError, ValueError)):
        checks.compare_tensor(state + 0.01, state, rules, state=True)


def test_tensor_math_cannot_substitute_for_kernel(checks):
    x = torch.ones((2, 8))
    for compute in (lambda: x.square(), lambda: x+x, lambda: x.copy_(x)):
        with pytest.raises(ValueError, match='delegates computation'):
            with checks.candidate_operations():
                compute()
    with checks.candidate_operations():
        assert torch.empty_like(x).shape == x.shape
        assert x.view(-1).numel() == 16


@pytest.fixture
def quality(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('q80_quality', TASK/'scripts/quality.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shutil.copy(TASK/'quality.json', tmp_path/'quality.json')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    return module


def test_instruct_answers_need_no_reasoning_tag(quality):
    assert quality.completion_answer('The answer is 159.\n#### 159', 'stop', None) == '159'
    assert quality.completion_answer('<think>draft #### 5</think>\n#### 159', 'stop', None) == '159'
    assert quality.completion_answer('#### 159', 'stop', 'separate reasoning') == '159'
    for text, finish in (
        ('#### 159', 'length'), ('<think>draft\n#### 159', 'stop'),
        ('The answer is 159', 'stop'), ('#### 159\nmore text', 'stop'),
        ('<think>#### 159</think>', 'stop'),
    ):
        assert quality.completion_answer(text, finish, None) is None


def test_instruct_quality_keeps_answer_and_regression_gates(quality):
    import copy
    lock = json.loads((TASK/'runtime.lock.json').read_text())
    data = quality.load_cases(lock)
    def request(port, endpoint, payload):
        prompt = payload['messages'][0]['content']
        question = prompt.rsplit('Question: ', 1)[1].split('\nEnd your response', 1)[0]
        expected = next(row['answer'] for row in data['samples'] if row['question'] == question)
        return {'choices': [{'message': {'content': expected}, 'finish_reason': 'stop'}],
                'usage': {'completion_tokens': 64}}
    reference = quality.run(1234, lock, request, 'reference')
    assert quality.validate(reference, reference, lock)['candidate_accuracy'] == 1
    bad = copy.deepcopy(reference)
    for row in bad['results'][:2]:
        row.update(response='#### -999999999', actual='-999999999', correct=False)
    bad.update(correct=62, score=62/64)
    assert quality.validate(bad, reference, lock)['candidate_accuracy'] == 62/64
    for row in bad['results'][2:4]:
        row.update(response='#### -999999999', actual='-999999999', correct=False)
    bad.update(correct=60, score=60/64)
    with pytest.raises(ValueError, match='regressed'):
        quality.validate(bad, reference, lock)
    bad['results'][0]['actual'] = bad['results'][0]['expected']
    with pytest.raises(ValueError, match='answer evidence'):
        quality.validate(bad, reference, lock)
