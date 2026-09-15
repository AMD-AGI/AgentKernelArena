"""CPU counterexamples for the isolated PR107/main integration.

Device literals are redirected only in temporary modules. These tests exercise
case dispatch, independent operators and failures; they are not GPU qualification.
"""
import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
PORTS = ['apply_grammar_bitmask', 'compute_identity', 'expert_kernel',
         'fla_layernorm_gated', 'layernorm_gated', 'merge_16x16_to_32x32',
         'topk_log_softmax', 'unpack_seq']


def control_module(name, monkeypatch):
    path = ROOT / f'tasks/triton2triton/vllm/triton_{name}/_upstream_controls.py'
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'cuda':
            node.value = 'cpu'
    module = ModuleType('integration_control')
    module.__file__ = str(path)
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    return module


def mathematical_candidate(name):
    def grammar(logits, indices, masks, vocab):
        for row, selected in enumerate(indices.tolist()):
            ids = torch.arange(vocab)
            allowed = ((masks[row, ids // 32].long() >> (ids % 32)) & 1).bool()
            logits[selected, ~allowed] = -torch.inf

    def grouped_norm(x, weight, bias, eps, z=None, out=None,
                     group_size=None, norm_before_gate=True, is_rms_norm=False):
        size = group_size or x.shape[-1]
        values = x.float()
        if z is not None and not norm_before_gate:
            values = values * torch.nn.functional.silu(z.float())
        groups = values.reshape(x.shape[0], -1, size)
        mean = None if is_rms_norm else groups.mean(-1, keepdim=True)
        centered = groups if mean is None else groups - mean
        rstd = (centered.square().mean(-1, keepdim=True) + eps).rsqrt()
        result = (centered * rstd).reshape_as(x) * weight.float()
        if bias is not None: result = result + bias.float()
        if z is not None and norm_before_gate:
            result = result * torch.nn.functional.silu(z.float())
        result = result.to(x.dtype)
        if out is not None:
            out.copy_(result)
            result = out
        mean = None if mean is None else mean.squeeze(-1).T.contiguous().flatten()
        return result, mean, rstd.squeeze(-1).T.contiguous().flatten()

    def gated_norm(x, g, weight=None, bias=None, activation='swish', eps=1e-5, is_rms_norm=True):
        values = x.float()
        mean = None if is_rms_norm else values.mean(-1)
        centered = values if mean is None else values - mean[:, None]
        rstd = (centered.square().mean(-1) + eps).rsqrt()
        result = centered * rstd[:, None]
        if weight is not None: result = result * weight.float()
        if bias is not None: result = result + bias.float()
        result = result * torch.sigmoid(g.float())
        if activation in ('silu', 'swish'): result = result * g.float()
        return result.to(x.dtype), mean, rstd

    def triangular(data):
        answer = torch.zeros_like(data)
        for b in range(data.shape[0]):
            for h in range(data.shape[2]):
                for start in range(0, data.shape[1], 32):
                    n = min(32, data.shape[1] - start)
                    matrix = torch.eye(n, dtype=torch.float64) + torch.tril(data[b, start:start+n, h, :n].double(), diagonal=-1)
                    answer[b, start:start+n, h, :n] = torch.linalg.solve_triangular(matrix, torch.eye(n, dtype=torch.float64), upper=False).to(data.dtype)
        return answer

    implementations = {
        'apply_grammar_bitmask': ('apply_grammar_bitmask', grammar),
        'compute_identity': ('compute_identity', lambda x, scales, top_k: (x.float() * scales[:, :top_k].sum(-1, keepdim=True)).to(x.dtype)),
        'expert_kernel': ('expert_gemm', lambda a, b: (a.float() @ b.float()).to(a.dtype)),
        'fla_layernorm_gated': ('layer_norm_gated_fwd', gated_norm),
        'layernorm_gated': ('layer_norm_fwd', grouped_norm),
        'merge_16x16_to_32x32': ('merge_16x16_to_32x32', triangular),
        'topk_log_softmax': ('compute_token_logprobs', lambda x, ids: torch.log_softmax(x.float(), -1).gather(1, ids)),
        'unpack_seq': ('unpack_seq', lambda x, lengths, **kw: torch.cat([row[:int(n)] for row, n in zip(x, lengths)], dim=0)),
    }
    return implementations[name]


@pytest.mark.parametrize('name', PORTS)
@pytest.mark.parametrize('corrupt', [False, True])
def test_every_added_control_invokes_candidate_and_rejects_wrong_result(name, corrupt, monkeypatch):
    task = control_module(name, monkeypatch)
    symbol, implementation = mathematical_candidate(name)
    calls = []

    def candidate(*args, **kwargs):
        calls.append(tuple(args[0].shape))
        result = implementation(*args, **kwargs)
        if corrupt:
            if name == 'apply_grammar_bitmask':
                args[0].fill_(12)
            elif isinstance(result, tuple):
                result = (result[0] + 50, *result[1:])
            else:
                result = result + 50
        return result

    module = SimpleNamespace(**{symbol: candidate})
    for index in range(len(task.EXTRA_CASES)):
        before = len(calls)
        ok, reason = task.run_control(index, lambda: module)
        assert len(calls) == before + 1, (name, index, reason)
        assert ok is (not corrupt), (name, index, reason)
        assert (reason is not None) is corrupt
    with pytest.raises(ValueError, match='Unknown upstream'):
        task.run_control(len(task.EXTRA_CASES), lambda: module)


@pytest.mark.parametrize('name', PORTS)
def test_new_controls_are_manifested_without_entering_performance(name, monkeypatch):
    task = control_module(name, monkeypatch)
    root = ROOT / f'tasks/triton2triton/vllm/triton_{name}'
    manifest = json.loads((root / 'workloads.json').read_text())
    rows = [r for r in manifest['cases'] if r['test_case_id'].startswith('control-upstream-')]
    assert len(rows) == len(task.EXTRA_CASES)
    assert [r['params']['configuration'] for r in rows] == json.loads(json.dumps(task.EXTRA_CASES))
    assert [r['params']['case_index'] for r in rows] == list(range(10000, 10000 + len(rows)))
    assert all(r['checks'] == ['correctness'] for r in rows)


def load_python(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rope_reference_pair_rotation_and_both_candidate_outputs(monkeypatch):
    import sys
    root = ROOT / 'tasks/triton2triton/geak_eval/L1/mla_decode'
    helper = load_python(ROOT / 'src/tools/perf/aka_benchmark.py', 'integration_benchmark')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', helper)
    timed = load_python(root / '_timed_contract.py', 'integration_timed_contract')
    monkeypatch.setitem(sys.modules, '_timed_contract', timed)
    rope = load_python(root / '_rope_controls.py', 'integration_rope')
    # Independent exact 90-degree adjacent-pair known answer.
    x = torch.tensor([[1., 2., 3., 4.]])
    cache = torch.tensor([[0., 0., 1., 1.]])
    assert torch.equal(rope.rotate(x, cache, torch.tensor([0])), torch.tensor([[-2., 1., -4., 3.]]))

    rounding = load_python(root / '_rounding_reference.py', 'integration_rounding')
    tree = ast.parse((root / 'test_kernel_harness.py').read_text())
    names = {'setup_inputs', 'run_ref', 'check_correctness_val', '_mla_contract'}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(torch=torch, KV_LORA_RANK=512, QK_ROPE_HEAD_DIM=64,
               assert_output_contract=timed.assert_output_contract,
               NumericalMismatch=rounding.NumericalMismatch,
               attention_rounding_bounds=rounding.attention_rounding_bounds,
               check_rounding_bounds=rounding.check_rounding_bounds)
    exec(compile(ast.Module(body=body, type_ignores=[]), 'protected_mla_cpu', 'exec'), env)
    harness = SimpleNamespace(**{n: env[n] for n in names})

    for corruption in (None, 'rotated_key', 'attention', 'readonly'):
        def candidate(q, k, v, out, indptr, indices, key_out, rank, dim, cache, positions,
                      logits, splits, *, sm_scale, **options):
            inputs = dict(q=q, k_input=k, v_input=v, output=out, attn_logits=logits,
                          kv_indptr=indptr, kv_indices=indices, kv_lora_rank=rank,
                          num_kv_splits=splits, sm_scale=sm_scale)
            private, expected_key = rope.transformed(harness, inputs, cache, positions)
            out.copy_(harness.run_ref(private))
            key_out.copy_(expected_key)
            if corruption == 'rotated_key': key_out.add_(10)
            if corruption == 'attention': out.add_(10)
            if corruption == 'readonly': q.add_(1)
        harness.decode_attention_fwd_grouped_rope = candidate
        if corruption:
            with pytest.raises(AssertionError):
                rope.run(harness, rope.CASES[0])
        else:
            for case in rope.CASES:
                rope.run(harness, case)


def test_native_mla_candidate_macros_cannot_change_host_harness(tmp_path):
    import shutil
    import subprocess
    compiler = shutil.which('g++')
    if not compiler: pytest.skip('C++ preprocessor unavailable')
    source = ROOT / 'tasks/hip2hip/others/mla_decode'
    task = tmp_path / 'task'
    shutil.copytree(source, task)
    include = tmp_path / 'include/hip'
    include.mkdir(parents=True)
    for name in ('hip_runtime.h', 'hip_bf16.h', 'hip_fp16.h'):
        (include / name).write_text('/* Preprocessor-only HIP header fixture. */\n')
    with (task / 'source/kernel.hpp').open('a') as handle:
        handle.write('\n#define NHEAD 1\n#undef HIP_CHECK\n#define HIP_CHECK(x)\n')
    def macros(relative):
        result = subprocess.run([compiler, '-x', 'c++', '-E', '-dM', '-I', str(include.parent),
                                 str(task / relative)], capture_output=True, text=True, check=True)
        return result.stdout
    assert '#define NHEAD 1' in macros('scripts/native/candidate_driver.hip')
    protected = macros('mla_decode.hip')
    assert '#define NHEAD ' not in protected
    assert 'fprintf(stderr,' in next(line for line in protected.splitlines() if line.startswith('#define HIP_CHECK'))

ALL_PORTS = sorted(path.parent.name.removeprefix('triton_') for path in
                   (ROOT / 'tasks/triton2triton/vllm').glob('*/_upstream_controls.py'))


@pytest.mark.parametrize('name', ALL_PORTS)
def test_every_new_manifest_case_actually_reaches_candidate(name, monkeypatch):
    """A missing branch/invalid input must not silently return success."""
    module = control_module(name, monkeypatch)
    # Only integer launch-shape arithmetic is needed before the CPU sentinel.
    import sys
    triton = ModuleType('triton')
    triton.cdiv = lambda a, b: (a+b-1)//b
    triton.next_power_of_2 = lambda x: 1 << (x-1).bit_length()
    monkeypatch.setitem(sys.modules, 'triton', triton)
    calls = []

    class Candidate:
        def _get_fp8_dtype(self):
            return torch.float8_e4m3fnuz

        def __getattr__(self, symbol):
            def invoked(*args, **kwargs):
                calls.append(symbol)
                raise RuntimeError('CONTROL_CANDIDATE_REACHED')
            return invoked

    for index in range(len(module.EXTRA_CASES)):
        before = len(calls)
        ok, reason = module.run_control(index, Candidate)
        assert not ok and 'CONTROL_CANDIDATE_REACHED' in str(reason), (name, index, reason)
        assert len(calls) == before + 1
    for invalid in (-1, len(module.EXTRA_CASES), True):
        with pytest.raises(ValueError, match='Unknown'):
            module.run_control(invalid, Candidate)


@pytest.mark.parametrize('name', ALL_PORTS)
def test_new_controls_have_public_failure_rows_and_leave_scored_bytes_unchanged(name, monkeypatch):
    import subprocess
    from src.task_protocol import parse_command_result
    root = ROOT / f'tasks/triton2triton/vllm/triton_{name}'
    before = subprocess.check_output(['git', 'show',
        'e8ec5d6b:' + str((root / 'scripts/task_runner.py').relative_to(ROOT))], cwd=ROOT)
    dispatcher = (b'    if case_index is not None and case_index >= 10000:\n'
                  b'        from _upstream_controls import run_control\n'
                  b'        return run_control(case_index - 10000, load_module)\n')
    after = (root / 'scripts/task_runner.py').read_bytes()
    assert after.count(dispatcher) == 1
    assert after.replace(dispatcher, b'', 1) == before
    original = json.loads(subprocess.check_output(['git', 'show',
        'e8ec5d6b:' + str((root / 'workloads.json').relative_to(ROOT))], cwd=ROOT))
    manifest = json.loads((root / 'workloads.json').read_text())
    assert manifest['cases'][:len(original['cases'])] == original['cases']
    added = manifest['cases'][len(original['cases']):]
    assert added and all(row['checks'] == ['correctness'] for row in added)
    spec = importlib.util.spec_from_file_location('integration_adapter', root / '_arena_eval.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    assert adapter.load_manifest() == manifest
    visited = []
    def check(*, case_index):
        visited.append(case_index)
        return (case_index < 10000, 'deliberately invalid added candidate')
    harness = SimpleNamespace(**{manifest['case_table']: manifest['input_table']},
        run_correctness=check, run_contract_correctness=lambda case: (True, None),
        run_semantic_controls=lambda: [],
        CONTRACT_CASES={r['test_case_id']: r['params'] for r in manifest['cases']
                        if 'contract_case' in r['params']})
    monkeypatch.setattr(adapter, 'load_harness', lambda: harness)
    monkeypatch.setattr(adapter, 'inspect_candidate', lambda *a, **kw: 'implemented')
    result = adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'FAIL', result
    failed = [row for row in result['cases'] if row['status'] == 'FAIL']
    assert [row['test_case_id'] for row in failed] == [row['test_case_id'] for row in added], result
    assert [i for i in visited if i >= 10000] == list(range(10000, 10000 + len(added)))
    parsed = parse_command_result('ARENA_EVAL_RESULT=' + json.dumps(result),
                                  role='candidate', action='correctness', returncode=1)
    assert parsed.status == 'FAIL'


@pytest.mark.parametrize('name', ['apply_write', 'awq_dequantize', 'ssd_chunk_cumsum', 'ssd_chunk_scan'])
@pytest.mark.parametrize('raises', [False, True])
def test_new_controls_restore_pristine_inputs_even_on_failure(name, raises, monkeypatch):
    module = control_module(name, monkeypatch)
    x = torch.arange(12).reshape(3, 4)[:, ::2]
    initial = x.clone()
    output = torch.zeros_like(x)
    def candidate(readonly, output):
        readonly.zero_()
        output.fill_(7)
        if raises:
            raise RuntimeError('deliberate launch failure')
        return output
    wrapped = module._guarded_module(SimpleNamespace(kernel=candidate), 'kernel', ('output',))
    with pytest.raises((RuntimeError, AssertionError), match='launch failure|read-only'):
        wrapped.kernel(x, output)
    assert torch.equal(x, initial)
    assert torch.all(output == 7)


@pytest.mark.parametrize('corrupt', [False, True])
@pytest.mark.parametrize('name', ['apply_write', 'pack_bitmatrix', 'ranks', 'per_token_group_quant_int8', 'ssd_chunk_cumsum'])
def test_additional_operator_controls_use_numerical_comparison(name, corrupt, monkeypatch):
    module = control_module(name, monkeypatch)
    def write(output, indices, starts, contents, cumulative):
        low = 0
        for row, start, end in zip(indices.tolist(), starts.tolist(), cumulative.tolist()):
            output[row, start:start+end-low] = contents[low:end]
            low = end
        if corrupt: output.fill_(12345)
    def bits(ids, experts):
        result = torch.zeros(ids.shape[0], (experts+31)//32, dtype=torch.int32)
        for row, selected in enumerate(ids.tolist()):
            for item in set(selected):
                word = int(result[row, item//32]) | (1 << (item % 32))
                result[row, item//32] = word if word < (1 << 31) else word - (1 << 32)
        return (result ^ 1 if corrupt else result).to(torch.uint32)
    def ranks(x, ids):
        result = (x >= x.gather(1, ids[:, None])).sum(1)
        return result + 1 if corrupt else result
    def quant(x, group_size, eps=1e-10):
        values = x.float().reshape(*x.shape[:-1], -1, group_size)
        scale = values.abs().amax(-1).clamp_min(eps)/127
        q = (values/scale[...,None]).clamp(-128,127).to(torch.int8).reshape_as(x)
        return q, scale + 1 if corrupt else scale
    def cumsum(dt, A, chunk_size, cu, dt_bias=None, dt_softplus=False, dt_limit=(0., float('inf'))):
        values = dt.float() + (dt_bias if dt_bias is not None else 0)
        if dt_softplus: values = torch.nn.functional.softplus(values)
        values = values.clamp(*dt_limit)
        padded = torch.zeros(A.numel(), cu.numel()-1, chunk_size)
        for i, (lo, hi) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
            padded[:,i,:hi-lo] = values[lo:hi].T
        decay = (padded*A[:,None,None]).cumsum(-1)
        return decay+10 if corrupt else decay, padded
    symbol, candidate = {
        'apply_write': ('apply_write', write),
        'pack_bitmatrix': ('pack_topk_to_bitmatrix', bits),
        'ranks': ('compute_ranks', ranks),
        'per_token_group_quant_int8': ('per_token_group_quant_int8', quant),
        'ssd_chunk_cumsum': ('chunk_cumsum_fwd', cumsum),
    }[name]
    for index in range(len(module.EXTRA_CASES)):
        ok, reason = module.run_control(index, lambda: SimpleNamespace(**{symbol: candidate}))
        assert ok is (not corrupt), (name, index, reason)


def selected_definitions(source, names, namespace=None):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in tree.body} == set(names)
    env = {'torch': torch, **(namespace or {})}
    exec(compile(tree, '<protected definitions>', 'exec'), env)
    return SimpleNamespace(**env)


@pytest.mark.parametrize('shape', [(3, 5, 7), (4, 129, 131)])
def test_refk_protected_inputs_preserve_generator_bytes_and_strides(shape):
    root = ROOT / 'tasks/triton2triton/geak_eval/L1/refk_fp8_blockwise_mm'
    constants = {'BLOCK_SHAPE_N': 128, 'BLOCK_SHAPE_K': 128}
    old = selected_definitions((root / 'kernel.py').read_text(),
                              ['_generate_input', 'get_inputs'], constants)
    new = selected_definitions((root / 'test_kernel_harness.py').read_text(),
                              ['get_inputs', 'fp8_blockwise_mm_pytorch'], constants)
    for seed in (42, 6543):
        before = old.get_inputs(*shape, seed=seed, device='cpu')
        after = new.get_inputs(*shape, seed=seed, device='cpu')
        for a, b in zip(before, after):
            assert a.dtype == b.dtype and a.stride() == b.stride()
            assert torch.equal(a.flatten().contiguous().view(torch.uint8), b.flatten().contiguous().view(torch.uint8))
    result = new.fp8_blockwise_mm_pytorch(
        torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[5., 6.]]),
        torch.tensor([[2.], [3.]]), torch.tensor([[0.5]]),
        torch.empty(2, 1, dtype=torch.bfloat16))
    assert torch.equal(result, torch.tensor([[17.], [58.5]], dtype=torch.bfloat16))


@pytest.mark.parametrize('family', ['instruction2triton/rocmbench', 'triton2triton/rocmbench/hard'])
def test_moe_preparation_hoists_device_scalar_without_changing_launch(family):
    import functools
    import subprocess
    path = f'tasks/{family}/moe_gemm/moe_gemm.py'
    old_source = subprocess.check_output(['git', 'show', f'e8ec5d6b:{path}'], cwd=ROOT, text=True)
    new_source = (ROOT / path).read_text()
    calls = []
    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kw):
                calls.append((grid(kw) if callable(grid) else grid, args, kw))
            return launch
    class Scalar:
        count = 0
        def item(self):
            self.count += 1
            return 64
    env = dict(functools=functools, moe_gemm_kernel=Kernel(),
               triton=SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b))
    old = selected_definitions(old_source, ['MetaData', 'moe_gemm'], env)
    names = ['MetaData', 'moe_gemm'] + (['prepare_moe_gemm'] if family.startswith('instruction') else [])
    new = selected_definitions(new_source, names, env)
    a, b, c = torch.empty(4, 32), torch.empty(2, 16, 32), torch.empty(4, 2, 16)
    weights, ids = torch.ones(8), torch.arange(8)
    scalar = Scalar()
    config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}
    args = (2, weights, ids, ids, torch.tensor([0, 1]), scalar, config)
    old_meta = old.MetaData(*args)
    old.moe_gemm(a, b, c, old_meta)
    original = calls.pop()
    scalar.count = 0
    metadata = new.MetaData(*args)
    launch = (new.prepare_moe_gemm(a, b, c, metadata) if family.startswith('instruction')
              else lambda: new.moe_gemm(a, b, c, metadata))
    assert scalar.count == 1
    for _ in range(3):
        launch()
        grid, values, options = calls.pop()
        assert grid == original[0] and options == original[2]
        assert len(values) == len(original[1])
        for x, y in zip(values, original[1]):
            assert x is y if isinstance(x, torch.Tensor) else x == y
    assert scalar.count == 1, 'No device scalar reads belong in repeated timed launches'


@pytest.mark.parametrize('bad', [False, True])
def test_position_replay_checks_inactive_zero_writes_and_restores_inputs(bad):
    root = ROOT / 'tasks/triton2triton/vllm/triton_prepare_pos_seq_lens'
    contract = load_python(root / '_arena_contract.py', 'positions_contract')
    replay = load_python(root / '_arena_replay.py', 'positions_replay')
    harness = SimpleNamespace(_TimedRun=SimpleNamespace)
    recorder = replay.Recorder(harness, contract)
    inputs = next(contract.control_inputs(harness))
    inputs[-1].zero_()  # Original timed input alone would accept the bad kernel.
    saved = replay.clone(inputs)
    def kernel(mapping, starts, computed, pos, seq):
        for row, req in enumerate(mapping.tolist()):
            lo, hi = int(starts[row]), int(starts[row+1])
            pos[lo:hi] = torch.arange(int(computed[req]), int(computed[req])+hi-lo)
            seq[row] = computed[req]+hi-lo
        if not bad:
            seq[len(mapping):] = 0
    wrapped = recorder.wrap(kernel)
    observed = []
    def benchmark(fn, *, timed_run, **options):
        observed.append(replay.clone(inputs))
        timed_run.outputs = fn()
        timed_run.rerun = fn
        return 1., {}
    if bad:
        with pytest.raises(AssertionError, match='integer reference'):
            recorder.benchmark(benchmark, lambda: wrapped(*inputs))
    else:
        _, evidence = recorder.benchmark(benchmark, lambda: wrapped(*inputs))
        assert evidence['replay_input_control_checked']
    assert len(observed) == 1
    for measured, initial, final in zip(observed[0], saved, inputs):
        assert torch.equal(measured, initial) and torch.equal(final, initial)


@pytest.mark.parametrize('raises', [False, True])
def test_lora_public_and_direct_controls_restore_nested_readonly_inputs(raises, monkeypatch):
    module = control_module('fused_moe_lora', monkeypatch)
    readonly = torch.arange(8).reshape(2, 4)[:, ::2]
    before = readonly.clone()
    output = torch.zeros(2)
    with pytest.raises((RuntimeError, AssertionError), match='launch|read-only'):
        with module._readonly_control({'weights': [readonly], 'output': output}):
            readonly.zero_()
            output.fill_(3)
            if raises:
                raise RuntimeError('launch failed')
    assert torch.equal(readonly, before) and torch.all(output == 3)


def installed_fp8_correctness(monkeypatch, fault=None):
    """Actual run_correctness dispatcher and actual installed guard; CPU devices only."""
    import sys
    root = ROOT / 'tasks/triton2triton/vllm/triton_per_token_group_quant_fp8'
    controls = control_module('per_token_group_quant_fp8', monkeypatch)
    monkeypatch.setitem(sys.modules, '_upstream_controls', controls)
    monkeypatch.chdir(ROOT)
    tree = ast.parse((root / 'scripts/task_runner.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'cuda': node.value = 'cpu'
    harness = ModuleType('installed_fp8_harness')
    harness.__file__ = str(root / 'scripts/task_runner.py')
    exec(compile(tree, harness.__file__, 'exec'), harness.__dict__)
    checks = load_python(root / '_arena_checks.py', 'installed_fp8_guard')
    calls = []
    def quantize(x, group_size, eps=1e-10, dtype=None, use_ue8m0=False):
        calls.append((tuple(x.shape), dtype, use_ue8m0))
        groups = x.float().reshape(*x.shape[:-1], -1, group_size)
        scales = groups.abs().amax(-1).clamp_min(eps) / 240.
        if use_ue8m0: scales = torch.exp2(torch.ceil(torch.log2(scales)))
        q = (groups / scales[..., None]).clamp(-240., 240.).reshape_as(x)
        q = q.to(dtype or torch.float8_e4m3fnuz)
        if fault == 'readonly': x.zero_()
        if fault == 'dtype': q = q.to(torch.float16)
        if fault == 'scale_shape': scales = scales.flatten()
        if fault == 'zero': q = torch.zeros_like(q.float()).to(q.dtype)
        return q, scales
    module = SimpleNamespace(per_token_group_quant_fp8=quantize,
                             _get_fp8_dtype=lambda: torch.float8_e4m3fnuz)
    harness.load_module = lambda: module
    checks.install(harness)
    return harness, calls


@pytest.mark.parametrize('case_index', [0, 10000, 10001, 10002, 10003, 10004])
def test_full_installed_fp8_correctness_accepts_original_and_all_added_controls(case_index, monkeypatch):
    harness, calls = installed_fp8_correctness(monkeypatch)
    ok, error = harness.run_correctness(case_index=case_index)
    assert ok, error
    assert calls, 'The full public correctness path must actually invoke its candidate'
    if case_index == 10002: assert calls[0][1] == torch.float8_e4m3fn
    if case_index == 10003: assert calls[0][0] == (2, 3, 192)


@pytest.mark.parametrize('case_index', [10002, 10003])
@pytest.mark.parametrize('fault', ['readonly', 'dtype', 'scale_shape', 'zero'])
def test_full_installed_fp8_correctness_keeps_input_metadata_and_numerical_gates(case_index, fault, monkeypatch):
    harness, calls = installed_fp8_correctness(monkeypatch, fault)
    ok, error = harness.run_correctness(case_index=case_index)
    assert not ok and error, (case_index, fault, error)
    assert len(calls) == 1, 'Reject the requested case at its real guard before diagnostics'
    if fault == 'readonly': assert 'read-only' in str(error)
    if fault in ('dtype', 'scale_shape'): assert 'shape/dtype/device' in str(error)
