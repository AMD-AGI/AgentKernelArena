"""CPU coverage for the two captured DS branches and owned replay snapshots."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TASKS = [REPO / 'tasks/headkernel' / name for name in (
    'deepseek-v4-pro__moe_stage1_grouped_gemm_silu_opus_a8w4',
    'deepseek-v4-pro__unified_paged_attention_prefill')]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def functions(task, names):
    tree = ast.parse((task/'scripts/task_runner.py').read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = {'Path': Path, 'strict_json': json.loads,
                 'sha256': lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<protected-runner-functions>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_only_actual_captured_dispatch_is_admitted(task):
    manifest = json.loads((task/'cases.json').read_text())
    evidence = json.loads((task/'provenance/EDITABLE-DISPATCH.json').read_text())
    contract = load(task/'ut/dispatch_contract.py', 'dispatch_'+task.name)
    assert contract.validate_dispatch(manifest)
    assert evidence['case_manifest_sha256'] == hashlib.sha256((task/'cases.json').read_bytes()).hexdigest()
    assert [row['case_id'] for row in evidence['cases']] == [row['case_id'] for row in manifest['cases']]
    assert [row['occurrences'] for row in evidence['cases']] == [row['occurrences'] for row in manifest['cases']]
    changed = copy.deepcopy(manifest)
    if manifest['seam'] == 'moe1_prefill':
        changed['cases'][0]['scalars']['arguments']['kernelName'] = 'opus_moe1_afp8_wfp4_bf16_t16x384_pair_kw7_m1_stream_fp8'
        common = load(task/'ut/native/csrc/opus_moe/opus_moe_common.py', 'observed_opus_metadata')
        for case in manifest['cases']:
            name = case['scalars']['arguments']['kernelName']
            matches = [row for row in common.STAGE1_A8W4_KERNELS.values() if name in (row.name, row.profile_name)]
            assert len(matches) == 1 and matches[0].gate_up_group_split and matches[0].kid == 1021
    else:
        assert all(row['tensors']['arg.q']['shape'][1] == 16 for row in manifest['cases'])
        changed['cases'][0]['tensors']['arg.q']['shape'][1] = 64
    with pytest.raises(ValueError, match='outside the captured'):
        contract.validate_dispatch(changed)
    for relative, expected in evidence['dispatch_sources'].items():
        assert hashlib.sha256((task/relative).read_bytes()).hexdigest() == expected


def test_unused_opus_pair_source_is_frozen_everywhere():
    task = TASKS[0]
    pair = 'source/opus_moe_stage1_a8w4_pipeline_pair_kwave_gfx950.cuh'
    config = yaml.safe_load((task/'config.yaml').read_text())
    source = json.loads((task/'provenance/SOURCE.json').read_text())
    policy = json.loads((task/'ut/source_guard_policy.json').read_text())
    assert pair not in config['source_file_path']
    assert pair not in config['trusted_evaluation']['reference_sources']
    assert pair not in source['editable_sources'] and pair not in policy['sources']
    assert policy['frozen_files'][pair] == hashlib.sha256((task/pair).read_bytes()).hexdigest()
    assert (task/pair).read_bytes() == (task/'ut/reference'/Path(pair).name).read_bytes()


def test_unused_mla_body_rejected_while_observed_body_stays_editable():
    task = TASKS[1]
    guard = load(task/'ut/cpp_body_guard.py', 'observed_cpp_guard')
    policy = json.loads((task/'ut/source_guard_policy.json').read_text())
    relative = 'source/pa_sparse_prefill_opus.h'
    active = policy['sources'][relative]['markers']
    assert active == ['__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void pa_prefill_16mx1_16nx4_kernel(']
    original = (task/relative).read_text()
    start, _ = guard.span(original, active[0])
    guard.validate_cpp(original[:start]+';'+original[start:], original, active)
    unused = '__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void pa_prefill_16mx8_32nx1_kernel('
    start, _ = guard.span(original, unused)
    with pytest.raises(ValueError, match='frozen'):
        guard.validate_cpp(original[:start]+';'+original[start:], original, active)


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_snapshot_owns_cpu_bytes_without_a_second_copy(task, monkeypatch):
    calls = []
    class Tensor:
        def detach(self): return self
        def to(self, *, device, copy):
            calls.append((device, copy))
            return SimpleNamespace(owned=True)
        def cpu(self): raise AssertionError('redundant CPU transfer path')
        def clone(self): raise AssertionError('redundant second CPU clone')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor)))
    namespace = functions(task, ['cpu_clone'])
    result = namespace['cpu_clone']({'x': (Tensor(), None)})
    assert result['x'][0].owned and result['x'][1] is None
    assert calls == [('cpu', True)]


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_owned_snapshot_preserves_full_storage_alias_and_padding_checks(task, monkeypatch):
    torch = pytest.importorskip('torch')
    abi = load(task/'ut/abi.py', 'observed_abi_'+task.name)
    def raw_storage(tensor):
        return torch.empty(0, dtype=torch.uint8).set_(tensor.untyped_storage(), 0,
            (tensor.untyped_storage().nbytes(),), (1,))
    monkeypatch.setitem(sys.modules, 'snapshots', SimpleNamespace(raw_storage=raw_storage))
    namespace = functions(task, ['cpu_clone', 'leaves', 'storage_snapshots', 'assert_immutable_inputs'])
    namespace['runtime_abi'] = abi.runtime_abi
    storage = torch.arange(16, dtype=torch.float32)
    view = storage.as_strided((2, 2), (4, 1), 1)
    inputs = {'q': view, 'alias': storage[5:8], 'out': None}
    before = namespace['storage_snapshots'](inputs)
    assert len(before) == 1 and before['arg.q'].numel() == storage.untyped_storage().nbytes()
    assert before['arg.q'].untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
    namespace['assert_immutable_inputs'](inputs, before)
    storage[-1] = 1000  # Outside either logical view, inside the original storage.
    with pytest.raises(AssertionError, match='mutated input storage'):
        namespace['assert_immutable_inputs'](inputs, before)


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_reference_fixture_skips_only_duplicate_golden_restore(task, tmp_path, monkeypatch):
    calls = []
    def restore_phase(root, record, phase, **kwargs):
        calls.append((phase, kwargs['device']))
        return {'x': SimpleNamespace(), 'side': SimpleNamespace()} if phase == 'inputs' else {'output': 'CPU golden'}
    monkeypatch.setitem(sys.modules, 'runtime_capture', SimpleNamespace(restore_phase=restore_phase))
    monkeypatch.setitem(sys.modules, 'snapshots', SimpleNamespace(restore=lambda value, device, module: value['tree']))
    namespace = functions(task, ['fixture']); namespace['ROOT'] = tmp_path
    (tmp_path/'provenance').mkdir()
    (tmp_path/'provenance/SOURCE.json').write_text(json.dumps({'signature_parameters': ['x', 'side', 'mode']}))
    record = {'provenance': {'run_id': 'CPU-only'}, 'source_sha256': 'a'*64,
              'served': {'stage': 'prefill'}, 'origin': 'served_eager', 'startup_values': False,
              'controls': {'mode': 7, 'x_attributes': {'side': {'tensor_binding': 'side'}}}}
    fixture = tmp_path/'fixture.json'; fixture.write_text(json.dumps(record))
    case = {'fixture': {'path': 'fixture.json', 'sha256': namespace['sha256'](fixture)}}
    manifest = {'run_id': 'CPU-only', 'native_source_sha256': 'a'*64, 'seam': 'mla_prefill'}
    candidate, golden = namespace['fixture'](case, manifest, object())
    reference, unused = namespace['fixture'](case, manifest, object(), include_golden=False)
    assert golden == 'CPU golden' and unused is None
    assert candidate['x'] is not reference['x']
    assert candidate['x'].side is candidate['side'] and reference['x'].side is reference['side']
    assert candidate['mode'] == reference['mode'] == 7
    assert calls == [('inputs', 'cuda'), ('outputs', 'cpu'), ('inputs', 'cuda')]


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_cpu_thread_cap_is_bounded_and_does_not_expand_existing_limit(task):
    configure = functions(task, ['configure_cpu_threads'])['configure_cpu_threads']
    for previous, expected in [(112, 8), (8, 8), (4, 4)]:
        state = [previous]
        fake = SimpleNamespace(get_num_threads=lambda: state[0], set_num_threads=lambda n: state.__setitem__(0, n))
        report = configure(fake)
        assert report['intraop_threads_before'] == previous and report['intraop_threads'] == expected


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_complete_checked_replay_policy_and_realistic_timeout_are_preserved(task):
    manifest = json.loads((task/'cases.json').read_text())
    policy = manifest['measurement']
    assert policy['warmup_iterations'] == 10 and policy['benchmark_iterations'] == 100
    assert policy['refresh_inputs'] == policy['initialize_outputs'] == policy['validate_outputs'] == 'each_replay'
    assert policy['negative_controls'] == ['no_op', 'wrong_output']
    config = yaml.safe_load((task/'config.yaml').read_text())
    assert config['performance_timeout'] == 3600
    assert config['compile_timeout'] + config['correctness_timeout'] + config['performance_timeout'] <= 7200


@pytest.mark.parametrize('task', TASKS, ids=lambda p: p.name)
def test_reference_runs_only_after_candidate_observations_and_all_checks_remain(task, monkeypatch):
    log = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: log.append('sync'))))
    namespace = functions(task, ['verify_after_snapshot'])
    namespace.update(cpu_clone=lambda value: log.append('snapshot '+value) or value,
                     assert_immutable_inputs=lambda inputs, expected: log.append('immutable '+inputs),
                     restore_storages=lambda inputs, expected: log.append('restore '+inputs),
                     invoke=lambda fn, inputs: log.append('reference launch') or 'reference output',
                     compare=lambda actual, expected, tol: log.append('compare'))
    namespace['verify_after_snapshot']('candidate output', 'candidate inputs', object(), None, 'reference inputs', 0.02)
    assert log == ['sync', 'snapshot candidate output', 'immutable candidate inputs',
                   'restore reference inputs', 'reference launch', 'sync',
                   'snapshot reference output', 'immutable reference inputs', 'compare']
