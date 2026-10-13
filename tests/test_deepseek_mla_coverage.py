"""MLA coverage and rejection checks on small CPU buffers, not GPU qualification."""
from contextlib import contextmanager
from src.tools.perf.aka_benchmark import TimedRun
import ast
import copy
import importlib
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASKS = sorted((ROOT / 'tasks/Aiter-task').glob('flash_mla_with_kvcache_dsv4_fp8_*'))


@contextmanager
def modules(task, monkeypatch):
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(task))
        saved = [name for name in sys.modules if name == 'scripts' or name.startswith('scripts.')]
        for name in saved:
            patch.delitem(sys.modules, name, raising=False)
        runner = importlib.import_module('scripts.task_runner')
        try:
            yield runner, importlib.import_module('scripts.task_inputs'), importlib.import_module('scripts.mla_coverage')
        finally:
            for name in list(sys.modules):
                if name == 'scripts' or name.startswith('scripts.'):
                    sys.modules.pop(name, None)


def small_case(task, profile_id='all_branches_empty'):
    data = json.loads((task / 'scripts/workload.json').read_text())
    row = copy.deepcopy(next(row for row in data['rows']
                             if row.get('input_profile', {}).get('id') == profile_id))
    row['workload']['axes']['pages'] = 2
    if 'extra_pages' in row['workload']['axes']:
        row['workload']['axes']['extra_pages'] = 2
    return data, row


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_boundary_cases_are_independent_and_scoreable(task):
    data = json.loads((task / 'scripts/workload.json').read_text())
    profiles = [r['input_profile'] for r in data['rows'] if 'input_profile' in r]
    cases = {c['test_case_id']: c for c in data['cases']}
    for row in data['rows']:
        case = cases[row['workload']['uuid']]
        assert set(case['checks']) == {'correctness', 'performance'}
        if 'input_profile' in row:
            assert case['params']['runtime_profile'] == row['input_profile']
            assert row['workload']['axes']['batch'] == 1
    widths = {'main': 128}
    if 'extra_topk' in data['definition']['axes']:
        widths['extra'] = data['definition']['axes']['extra_topk']['value']
    for branch, width in widths.items():
        fixed_sweeps = [p for p in profiles if p['id'].startswith(branch + '_length_')]
        seen = {p['lengths'][branch] for p in fixed_sweeps}
        assert {0, 1, 63, 64, 65, width - 1, width} <= seen
        for profile in fixed_sweeps:
            for other in widths.keys() - {branch}:
                assert profile['lengths'][other] == widths[other]
        assert {p['padding'].get(branch) for p in profiles} >= {'holes', 'all_negative'}
    assert any(all(n == 0 for n in p['lengths'].values()) for p in profiles)
    assert data['bundle_readme'] == (task / 'BUNDLE_README.md').read_text()


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_every_profile_changes_runtime_lengths_without_coupling_padding(task, monkeypatch):
    torch = pytest.importorskip('torch')
    data = json.loads((task / 'scripts/workload.json').read_text())
    with modules(task, monkeypatch) as (_, _, coverage):
        for row in data['rows']:
            profile = row.get('input_profile')
            if profile is None:
                continue
            values = {}
            for branch, length in profile['lengths'].items():
                prefix = '' if branch == 'main' else 'extra_'
                width = data['definition']['axes'][prefix + 'topk']['value']
                values[prefix + 'sparse_indices'] = torch.ones((1, 1, width), dtype=torch.int32)
                values[prefix + 'sparse_lens'] = torch.zeros(1, dtype=torch.int32)
            coverage.apply_profile(values, profile)
            for branch, length in profile['lengths'].items():
                prefix = '' if branch == 'main' else 'extra_'
                indices = values[prefix + 'sparse_indices'][0, 0]
                assert values[prefix + 'sparse_lens'].item() == length
                assert (indices[length:] >= 0).all()
                if profile['padding'].get(branch) == 'all_negative':
                    assert (indices[:length] == -1).all()
                elif profile['padding'].get(branch) == 'holes' and length:
                    assert indices[0] == -1
                else:
                    assert (indices[:length] >= 0).all()
            coverage.apply_profile(values, profile, replay=True)
            for branch, original in profile['lengths'].items():
                prefix = '' if branch == 'main' else 'extra_'
                assert values[prefix + 'sparse_lens'].item() != original


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_seeded_cache_reuse_matches_fresh_initialization_and_refills_in_place(task, monkeypatch):
    torch = pytest.importorskip('torch')
    data, row = small_case(task)
    with modules(task, monkeypatch) as (runner, inputs, _):
        templates = {}
        values = inputs.make_inputs(data['definition'], row, data['policy'], 'cpu', cache_templates=templates)
        pristine = runner.clone_inputs(values)
        fresh = inputs.make_inputs(data['definition'], row, data['policy'], 'cpu')
        runner.assert_unmodified(values, fresh)
        addresses = {name: value.data_ptr() for name, value in values.items() if isinstance(value, torch.Tensor)}
        first_templates = {key: value.clone() for key, value in templates.items()}
        inputs.refill_inputs(values, data['definition'], row, data['policy'], 'cpu', cache_templates=templates)
        for name, address in addresses.items():
            assert values[name].data_ptr() == address
            assert not torch.equal(values[name].view(torch.uint8), pristine[name].view(torch.uint8)), name
        for key, initial in first_templates.items():
            assert torch.equal(templates[key].view(torch.uint8), initial.view(torch.uint8))
        inputs.reset_inputs(values, data['definition'], row, data['policy'], 'cpu', cache_templates=templates)
        runner.assert_unmodified(values, pristine)
        assert len(templates) == 2 * len(first_templates)


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_comparator_rejects_nonzero_empty_output_and_wrong_lse(task, monkeypatch):
    torch = pytest.importorskip('torch')
    data, row = small_case(task)
    with modules(task, monkeypatch) as (runner, inputs, _):
        values = inputs.make_inputs(data['definition'], row, data['policy'], 'cpu')
        reference = runner.load_solution(task / 'scripts/reference', 'main.py::run')
        expected = reference(**values)
        assert torch.count_nonzero(expected[0]) == 0
        assert torch.isposinf(expected[1]).all()
        for output, lse in ((expected[0] + 0.001, expected[1]),
                            (expected[0], -expected[1]),
                            (expected[0], torch.zeros_like(expected[1]))):
            with pytest.raises(AssertionError):
                runner.assert_outputs((output, lse), expected, data['definition'], row, data['policy'], 'cpu')
        active = copy.deepcopy(row)
        active['input_profile']['lengths'] = dict.fromkeys(row['input_profile']['lengths'], 1)
        inputs.reset_inputs(values, data['definition'], active, data['policy'], 'cpu')
        expected = reference(**values)
        with pytest.raises(AssertionError):
            runner.assert_outputs((expected[0], expected[1] + 1), expected,
                                  data['definition'], active, data['policy'], 'cpu')


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
@pytest.mark.parametrize('stale_lengths', [False, True])
def test_exact_timed_replay_rejects_captured_lengths(task, stale_lengths, monkeypatch):
    torch = pytest.importorskip('torch')
    data, row = small_case(task)
    with modules(task, monkeypatch) as (runner, inputs, _):
        templates = {}
        values = inputs.make_inputs(data['definition'], row, data['policy'], 'cpu', cache_templates=templates)
        reference = runner.load_solution(task / 'scripts/reference', 'main.py::run')
        lengths = {name: value.clone() for name, value in values.items() if name.endswith('sparse_lens')}

        def launch(**kwargs):
            return reference(**(dict(kwargs, **lengths) if stale_lengths else kwargs))

        def simulated_graph(call, *, timed_run, **kwargs):
            captured = tuple(x.clone() for x in call())

            def replay():
                for dest, value in zip(captured, call()):
                    dest.copy_(value)
                return captured

            # Match the canonical helper's complete binding API.
            timed_run._bind(replay, captured, lambda: (replay(), 1.0))
            return 1.0, {'benchmark_method': 'cuda_graph'}

        monkeypatch.setitem(sys.modules, '_aka_benchmark', types.SimpleNamespace(
            TimedRun=TimedRun, benchmark_cuda_graph_or_events=simulated_graph))
        if stale_lengths:
            with pytest.raises(AssertionError):
                runner.measure_case(launch, reference, values, data['definition'], row,
                                    data['policy'], 'cpu', cache_templates=templates)
        else:
            result = runner.measure_case(launch, reference, values, data['definition'], row,
                                         data['policy'], 'cpu', cache_templates=templates)
            assert result['metadata']['runtime_length_replay_validated']


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_reference_ignores_unselected_tail_and_internal_negative_slots(task, monkeypatch):
    torch = pytest.importorskip('torch')
    data, row = small_case(task, 'main_length_1')
    # Isolate the main branch to make incorrect length/padding treatment observable.
    if 'extra' in row['input_profile']['lengths']:
        row['input_profile']['lengths']['extra'] = 0
    with modules(task, monkeypatch) as (runner, inputs, _):
        values = inputs.make_inputs(data['definition'], row, data['policy'], 'cpu')
        reference = runner.load_solution(task / 'scripts/reference', 'main.py::run')
        expected = reference(**values)
        values['sparse_indices'][0, 0, 1:] = 0
        if 'extra_sparse_indices' in values:
            values['extra_sparse_indices'].fill_(1)
        runner.assert_outputs(reference(**values), expected, data['definition'], row, data['policy'], 'cpu')
        values['sparse_lens'].fill_(3)
        values['sparse_indices'][0, 0, 1:3] = torch.tensor([-1, -7], dtype=torch.int32)
        runner.assert_outputs(reference(**values), expected, data['definition'], row, data['policy'], 'cpu')
        # The actual callbacks and embedded contract must tell the same story.
        for callback in ('initialize', 'compare'):
            assert data['definition'][callback] == (task / f'scripts/{callback}/main.py').read_text()


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_runner_uses_materialized_canonical_timed_run(task):
    tree = ast.parse((task / 'scripts/task_runner.py').read_text())
    assert not any(isinstance(node, ast.ClassDef) and node.name == 'TimedRun'
                   for node in ast.walk(tree))
    measure = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'measure_case')
    imports = {alias.name for node in ast.walk(measure)
               if isinstance(node, ast.ImportFrom) and node.module == '_aka_benchmark'
               for alias in node.names}
    assert {'TimedRun', 'benchmark_cuda_graph_or_events'} <= imports


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_declared_runtime_matches_actual_initial_and_baseline_imports(task):
    import yaml
    config = yaml.safe_load((task / 'config.yaml').read_text())
    run_config = yaml.safe_load((ROOT / 'example_configs/task_validator_deepseek_drafts_mi355x.yaml').read_text())
    assert config['candidate']['initial_state'] == 'implemented'
    assert config['candidate']['initial_language'] == 'python'
    assert config['candidate']['language'] == 'triton'
    readme = (task / 'BUNDLE_README.md').read_text()
    for relative in ('source/implementation/main.py', 'scripts/baseline/main.py'):
        tree = ast.parse((task / relative).read_text())
        imports = [f'{node.module}.{alias.name}' for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module.startswith('sglang.')
                   for alias in node.names]
        assert imports
        for entrypoint in imports:
            assert entrypoint in readme
            assert entrypoint in config['description']
    assert run_config['docker_image'] in readme
    assert run_config['docker_image'] in config['description']
    data = json.loads((task / 'scripts/workload.json').read_text())
    assert data['bundle_readme'] == readme
