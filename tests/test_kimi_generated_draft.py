"""CPU contracts for the isolated Kimi generated-input draft; no native GPU claim."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
from head_kernel_generated_test_utils import generated_helper

def kimi_helper(filename):
    return generated_helper('kimi', filename)
TASKS = {path.parent.name: path.parent for path in (ROOT / 'tasks/head_kernels/kimi-k3').rglob('config.yaml')}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def contract():
    return load('kimi_contract_tests', kimi_helper('generated_contract.py'))


def test_all_captured_cases_and_unscored_timing_contracts_are_retained(contract):
    controller = load('kimi_controller_tests', kimi_helper('generated_controller.py'))
    expected_counts = {'fwd_grouped_kernel_stage1': 2, 'moe_gemm1_stage1': 3, 'moe_gemm2_stage2': 4}
    for name, task in TASKS.items():
        compact = contract.load_contract(task / 'ut')
        meta = json.loads((task / 'ut/meta.json').read_text())
        assert compact['case_count'] == expected_counts[name]
        profiles = controller.expected_profiles(meta, compact)
        catalog = json.loads((task / 'SHAPES.json').read_text())
        assert set(json.loads(value)[0] for value in profiles['performance']) == set(catalog['case_inventory']['diagnostic_case_ids'])
        legacy = contract.legacy_definitions(task / 'ut', torch, 'cpu', compact)
        if compact['kind'] == 'moe':
            actual = [json.dumps([row['spec']['sig'], row['regime']], separators=(',', ':'))
                      for row in legacy['timing_cases']()]
            assert actual == profiles['performance']
        original = json.loads(subprocess.check_output(
            ['git', 'show', 'b66e373d:' + (task / 'ut/meta.json').relative_to(ROOT).as_posix()], cwd=ROOT))
        assert meta['tol'] == original['tol'] and meta['random_draws'] == original['random_draws']
        assert compact['contains_reference_output_values'] is False
        if name == 'moe_gemm1_stage1':
            assert meta['median_launches'] == 21
            assert 'single-launch' in meta['generated_inputs']['qualification_issue']


def test_committed_contract_hashes_retain_original_archive_identity():
    total = 0
    for task in TASKS.values():
        path = task / 'ut/generated_cases.json'
        meta = json.loads((task / 'ut/meta.json').read_text())
        compact = json.loads(path.read_text())
        assert compact['source_reference_sha256'] == meta['archival_capture']['reference_io_sha256']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['generated_inputs']['contract_sha256']
        total += path.stat().st_size
    assert total == 3358681


def tensor_desc(value, group, recipe, extractor):
    description = extractor.tensor_description(value, {})
    description.update(storage_group=group, recipe=recipe)
    if recipe == 'captured_integer':
        description['payload'] = extractor.packed_bytes(value, torch)
    return description


def small_record():
    extractor = load('kimi_extractor_tests', kimi_helper('extract_contract.py'))
    k = torch.zeros(3, 1, 2, dtype=torch.bfloat16)
    q = torch.zeros(1, 1, 2, dtype=torch.bfloat16)
    integers = [torch.tensor([0, 3], dtype=torch.int32), torch.arange(3, dtype=torch.int64), torch.tensor([1], dtype=torch.int32)]
    outputs = {'att_out': torch.zeros(1, 1, 1, 1), 'att_lse': torch.zeros(1, 1, 1)}
    return {'sig': 'captured-small', 'regime': 'decode', 'k': tensor_desc(k, 'k', 'normal_std_0.1', extractor),
            'kv_indices': tensor_desc(integers[1], 'indices', 'captured_integer', extractor),
            'v_is_slice': True, 'v_head_dim': 1, 'kw': {},
            'pos': [tensor_desc(q, 'q', 'normal_std_0.1', extractor), {'__slot__': 'k'}, {'__slot__': 'v'},
                    {'__slot__': 'att_out'}, {'__slot__': 'att_lse'},
                    tensor_desc(integers[0], 'indptr', 'captured_integer', extractor), {'__slot__': 'kv_indices'},
                    tensor_desc(integers[2], 'splits', 'captured_integer', extractor), 1, 1.0, 0.0, -1],
            'output_contract': extractor.output_contract(outputs, torch)}


def test_seeded_attention_keeps_aliases_exact_indices_and_views(contract):
    record = small_record()
    first = contract.attention_record(record, 42, torch, 'cpu')
    second = contract.attention_record(record, 42, torch, 'cpu')
    changed = contract.attention_record(record, 43, torch, 'cpu')
    assert torch.equal(first['pos'][0], second['pos'][0])
    assert not torch.equal(first['pos'][0], changed['pos'][0])
    k, v = first['pos'][1:3]
    assert k.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    assert v.stride() == (2, 2, 1)
    assert torch.equal(first['pos'][6], torch.arange(3, dtype=torch.int64))
    assert torch.equal(first['pos'][7], torch.tensor([1], dtype=torch.int32))


def test_structural_checksum_and_size_tampering_fail(contract):
    desc = small_record()['kv_indices']
    desc['payload']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        contract.build_tensors(torch, 'cpu', 0)(desc)


def test_original_routing_indices_are_included_but_weight_values_are_generated(contract):
    for name in ('moe_gemm1_stage1', 'moe_gemm2_stage2'):
        compact = contract.load_contract(TASKS[name] / 'ut')
        for row in compact['records']:
            build = contract.build_tensors(torch, 'cpu', 12)
            for key, desc in row['routing'].items():
                value = build(desc)
                assert list(value.shape) == desc['shape']
                assert list(value.stride()) == desc['stride']
                if key == 'sorted_weights':
                    assert desc['recipe'] == 'generated_routing_weights'
                    assert 'payload' not in desc and value.is_floating_point()
                else:
                    assert desc['recipe'] == 'captured_integer'
                    assert not value.is_floating_point()


def test_integer_boolean_outputs_are_exact_and_float_rule_is_unchanged(contract):
    encode = lambda value: contract.encode_output(value, torch)
    assert not contract.compare_output(encode(torch.tensor([2])), encode(torch.tensor([1])), 100, torch)
    assert not contract.compare_output(encode(torch.tensor([False])), encode(torch.tensor([True])), 100, torch)
    assert contract.compare_output(encode(torch.tensor([1.001])), encode(torch.tensor([1.])), .02, torch)
    assert not contract.compare_output(encode(torch.tensor([3.])), encode(torch.tensor([1.])), .02, torch)


def test_codec_uses_tensor_metadata_not_instance_method_overrides(contract):
    value = torch.ones(2, 3)
    value.stride = lambda: (99, 99)
    assert contract.encode_output(value, torch)['stride'] == [3, 1]


def test_replay_transform_removes_all_in_process_goldens(contract):
    for name in ('moe_gemm1_stage1', 'moe_gemm2_stage2'):
        tree = ast.parse((TASKS[name] / 'ut/unittest.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_replay')
        changed = contract._NoCachedReferences().visit(function)
        assert not any(isinstance(node, ast.Name) and node.id in {'BASELINE_FN', 'ref_out'} for node in ast.walk(changed))
        assert any(isinstance(node, ast.Constant) and node.value == 'state_inputs' for node in ast.walk(changed))


def test_generated_worker_has_no_private_attestation():
    worker = load('kimi_worker_tests', kimi_helper('generated_worker.py'))
    assert not hasattr(worker, 'attest')
    assert 'guard.check_module' in (kimi_helper('generated_worker.py')).read_text()


def test_worker_detects_mutating_inputs_and_reused_outputs(contract):
    worker = load('kimi_worker_tests', kimi_helper('generated_worker.py'))
    values = {'x': torch.ones(3)}
    with pytest.raises(RuntimeError, match='read-only'):
        worker.checked_call(lambda args: args['x'].add_(1), values, contract, torch, lambda: None, [])
    previous = []
    static = torch.ones(3)
    worker.checked_call(lambda _: static, {}, contract, torch, lambda: None, previous)
    with pytest.raises(RuntimeError, match='reused'):
        worker.checked_call(lambda _: static, {}, contract, torch, lambda: None, previous)


def test_parent_rejects_wrong_stale_partial_and_mislabelled_median_outputs(contract):
    controller = load('kimi_controller_tests', kimi_helper('generated_controller.py'))
    payload = {'schema_version': 1, 'profile': 'recorded', 'seed': 7, 'reference': False,
               'rows': [{'id': 'case', 'output': contract.encode_output(torch.tensor([3.]), torch)}],
               'correctness_policy': 'elementwise_median_21', 'single_launch_correctness_established': False}
    proc = SimpleNamespace(returncode=0, stderr='', stdout=controller.PREFIX + json.dumps(payload))
    assert controller.parse_worker(proc, 'recorded', 7, False, ['case'], True)
    with pytest.raises(RuntimeError, match='stale'):
        controller.parse_worker(proc, 'recorded', 8, False, ['case'], True)
    with pytest.raises(RuntimeError, match='stale'):
        controller.parse_worker(proc, 'recorded', 7, False, ['case', 'missing'], True)
    payload['single_launch_correctness_established'] = True
    proc.stdout = controller.PREFIX + json.dumps(payload)
    with pytest.raises(RuntimeError, match='median policy'):
        controller.parse_worker(proc, 'recorded', 7, False, ['case'], True)
    reference = [{'id': 'case', 'output': contract.encode_output(torch.tensor([1.]), torch)}]
    with pytest.raises(RuntimeError, match='reference mismatch'):
        controller.compare_rows(reference, payload['rows'], contract, .02, torch)


def test_default_workers_are_never_given_golden_paths(tmp_path):
    controller = load('kimi_controller_tests', kimi_helper('generated_controller.py'))
    commands = []
    def run_worker(script, arguments, overlay, timeout, candidate, **kwargs):
        commands.append(arguments)
        payload = {'schema_version': 1, 'profile': 'recorded', 'seed': 5, 'reference': not candidate,
                   'rows': [{'id': 'case', 'output': 1}]}
        return SimpleNamespace(returncode=0, stderr='', stdout=controller.PREFIX + json.dumps(payload))
    runner = SimpleNamespace(UT_DIR=tmp_path, TASK_DIR=tmp_path, run_worker=run_worker)
    import time
    controller.run_pair(runner, 'recorded', 5, ['case'], {}, 30, time.monotonic())
    assert all('--out' not in args and not any(arg.endswith('.pt') for arg in args) for args in commands)
    assert commands[0][-1] == '--reference' and '--reference' not in commands[1]


def test_source_kernels_abi_shapes_and_timing_settings_are_unchanged():
    for task in TASKS.values():
        for folder in ('source',):
            for path in (task / folder).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    relative = path.relative_to(ROOT)
                    old = subprocess.check_output(['git', 'show', f'b66e373d:{relative.as_posix()}'], cwd=ROOT)
                    assert path.read_bytes() == old
        relative = (task / 'scripts/source_abi.json').relative_to(ROOT)
        assert (ROOT / relative).read_bytes() == subprocess.check_output(['git', 'show', f'b66e373d:{relative}'], cwd=ROOT)
        worker = (task / 'scripts/generated_worker.py').read_text()
        assert 'warmup=10, repetition=100' in worker


def toy_attention_task(tmp_path):
    import shutil
    task = tmp_path / 'toy'
    for name in ('ut', 'ut/kernel_src', 'ut/baseline_ref'):
        (task / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(kimi_helper('generated_contract.py'), task / 'ut/generated_contract.py')
    shutil.copyfile(TASKS['fwd_grouped_kernel_stage1'] / 'ut/bindings.py', task / 'ut/bindings.py')
    record = small_record()
    compact = {'schema_version': 1, 'kind': 'attention', 'task': 'toy', 'case_count': 1,
               'records': [record], 'contains_numeric_input_values': False, 'contains_reference_output_values': False}
    path = task / 'ut/generated_cases.json'
    path.write_text(json.dumps(compact))
    metadata = {'geometry': {'num_q_heads': 1, 'head_dim_k': 2, 'v_head_dim': 1, 'max_kv_splits': 1,
                             'sm_scale': 1, 'logit_cap': 0, 'xai_temperature_len': -1, 'has_mla': True, 'page_size': 1},
                'regime': {}, 'cases': [], 'workload': {'cases': []}, 'random_draws': 3, 'tol': .02,
                'generated_inputs': {'contract_file': path.name, 'contract_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    (task / 'ut/meta.json').write_text(json.dumps(metadata))
    original = ast.parse((TASKS['fwd_grouped_kernel_stage1'] / 'ut/unittest.py').read_text())
    maker = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == '_make_call')
    (task / 'ut/unittest.py').write_text(ast.unparse(maker) + '\n')
    (task / 'ut/harness_lib.py').write_text('def correct(a,b,tol): return (True,0)\ndef to_device_like(x,d): return x\n')
    (task / 'ut/baseline_ref/decode_attention.py.orig').write_text(
        'def _decode_grouped_att_m_fwd(q, k, v, output, lse, *args, **kwargs):\n'
        '    output.copy_(q[..., :1].unsqueeze(-1) + k[0, :, :1].unsqueeze(-1))\n'
        '    lse.copy_(q[..., :1])\n')
    (task / 'ut/kernel_src/geak_mla_stage1.py').write_text(
        'def make_launcher(base):\n'
        '    def candidate(*args, **kwargs): return base._decode_grouped_att_m_fwd(*args, **kwargs)\n'
        '    return candidate\n')
    (task / 'scripts').mkdir()
    shutil.copyfile(TASKS['fwd_grouped_kernel_stage1'] / 'scripts/_bench.py', task / 'scripts/_bench.py')
    return task


def run_cpu_worker(task, reference):
    script = f'''import importlib.util,json,pathlib,sys
spec=importlib.util.spec_from_file_location('actual_generated_worker',{str(kimi_helper('generated_worker.py'))!r})
worker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=worker;spec.loader.exec_module(worker)
import torch
bootstrap=worker.load('_bootstrap_test',{str(ROOT / 'tasks/head_kernels/_support/_trusted_worker.py')!r})
task=pathlib.Path({str(task)!r})
contract=worker.load('generated_contract',task/'ut/generated_contract.py')
helper=worker.load('_kimi_timing_helpers',task/'scripts/_bench.py')
binding=worker.load('_kimi_generated_bindings',task/'ut/bindings.py')
h=worker.load('harness_lib',task/'ut/harness_lib.py')
benchmark=worker.load('_aka_benchmark',{str(ROOT / 'src/tools/perf/aka_benchmark.py')!r})
monitor=worker.load('runtime_integrity',{str(ROOT / 'tasks/head_kernels/_support/runtime_integrity.py')!r})
def reject_archive(*args,**kwargs): raise AssertionError('default worker attempted torch.load')
torch.load=reject_archive
guard=monitor.RuntimeIntegrity(task,torch,benchmark,h,trusted_modules={{'generated_contract':contract,'_kimi_timing_helpers':helper,'_kimi_generated_bindings':binding,'actual_generated_worker':worker}})
guard.install()
result=worker.run_profile(pathlib.Path({str(task / 'ut')!r}),'recorded',23,{reference!r},'cpu')
print(worker.PREFIX+json.dumps(result,separators=(',',':')))
'''
    return subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=30)


def test_actual_separate_workers_use_fresh_reference_and_reject_changed_candidate(tmp_path, contract):
    task = toy_attention_task(tmp_path)
    controller = load('kimi_controller_tests', kimi_helper('generated_controller.py'))
    reference = run_cpu_worker(task, True)
    candidate = run_cpu_worker(task, False)
    refs = controller.parse_worker(reference, 'recorded', 23, True, ['captured-small'], False)
    outputs = controller.parse_worker(candidate, 'recorded', 23, False, ['captured-small'], False)
    controller.compare_rows(refs, outputs, contract, .02, torch)
    path = task / 'ut/kernel_src/geak_mla_stage1.py'
    path.write_text('def make_launcher(base):\n'
                    '    def candidate(*args, **kwargs):\n'
                    '        base._decode_grouped_att_m_fwd(*args, **kwargs)\n'
                    '        args[3].add_(5)\n'
                    '    return candidate\n')
    wrong = run_cpu_worker(task, False)
    wrong_rows = controller.parse_worker(wrong, 'recorded', 23, False, ['captured-small'], False)
    with pytest.raises(RuntimeError, match='reference mismatch'):
        controller.compare_rows(refs, wrong_rows, contract, .02, torch)
    # An invalid candidate import cannot influence the reference worker.
    path.write_text("raise RuntimeError('candidate was imported')\n")
    assert run_cpu_worker(task, True).returncode == 0
    assert not list(task.rglob('*.pt'))


@pytest.mark.parametrize('attack', ['codec', 'input_check'])
def test_actual_worker_rejects_loaded_helper_function_tampering(tmp_path, attack):
    task = toy_attention_task(tmp_path)
    source = task / 'ut/kernel_src/geak_mla_stage1.py'
    prefix = ('import sys\nsys.modules["generated_contract"].encode_output = lambda *args: {}\n'
              if attack == 'codec' else
              'import sys\nsys.modules["actual_generated_worker"].unchanged = lambda *args: None\n')
    source.write_text(prefix + source.read_text())
    result = run_cpu_worker(task, False)
    assert result.returncode != 0
    assert 'IntegrityError' in result.stderr


@pytest.mark.parametrize('failure', [None, 'stale_replay', 'capture_error'])
def test_generated_performance_keeps_10_100_and_parent_validates_actual_replay(contract, monkeypatch, failure):
    worker = load('kimi_worker_bench_tests', kimi_helper('generated_worker.py'))
    helper = load('kimi_bench_helpers_tests', TASKS['fwd_grouped_kernel_stage1'] / 'scripts/_bench.py')
    scope = {'META': {}, '_make_call': lambda fn: lambda args: fn(args['q'])}
    rows = [{'sig': 'fixed-shape', 'regime': 'decode', 'm': 2, 'args': {'q': torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)}}]
    expected = worker.benchmark_rows(rows, scope, lambda x: x.square(), True, contract, torch, lambda: None, helper)
    observations = []
    def benchmark(call, *, warmup, repetition, prepare_fn, timed_run):
        observations.append((warmup, repetition))
        if failure == 'capture_error':
            raise RuntimeError('required capture failed')
        prepare_fn()
        captured = call()
        def replay():
            prepare_fn()
            return captured if failure == 'stale_replay' else call()
        timed_run._bind(replay, captured)
        return [1.0] * repetition, {'benchmark_method': 'cuda_graph', 'benchmark_samples': repetition}
    monkeypatch.setitem(sys.modules, '_aka_benchmark', SimpleNamespace(benchmark_cuda_graph_or_events_samples=benchmark))
    if failure == 'capture_error':
        with pytest.raises(RuntimeError, match='capture failed'):
            worker.benchmark_rows(rows, scope, lambda x: x.square(), False, contract, torch, lambda: None, helper)
    else:
        candidate = worker.benchmark_rows(rows, scope, lambda x: x.square(), False, contract, torch, lambda: None, helper)
        controller = load('kimi_controller_bench_tests', kimi_helper('generated_controller.py'))
        if failure:
            with pytest.raises(RuntimeError, match='reference mismatch'):
                controller.compare_rows(expected, candidate, contract, .02, torch)
        else:
            controller.compare_rows(expected, candidate, contract, .02, torch)
            assert candidate[0]['timing']['benchmark_samples'] == 100
    assert observations == [(10, 100)]


def test_single_launch_fault_and_second_hand_scenario_labels_remain_unresolved(contract):
    attention = json.loads((TASKS['fwd_grouped_kernel_stage1'] / 'ut/meta.json').read_text())['generated_inputs']
    assert attention['scenario_fidelity']['archive_pool_rows'] == [395, 524352]
    assert attention['scenario_fidelity']['timed_pool_rows'] == [8768, 557120]
    stage1 = json.loads((TASKS['moe_gemm1_stage1'] / 'ut/meta.json').read_text())
    assert 'SECOND-HAND' in stage1['provenance_note']
    assert 'assumed' in stage1['generated_inputs']['scenario_fidelity']['decode_launch']
    assert stage1['generated_inputs']['actual_hyperloom_scenario_verified'] is False
    assert stage1['median_launches'] == 21
    assert 'single-launch' in stage1['generated_inputs']['qualification_issue']
