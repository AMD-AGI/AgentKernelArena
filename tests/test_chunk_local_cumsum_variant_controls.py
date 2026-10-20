"""CPU checks for chunk_local_cumsum's head_first/varlen correctness-only controls.

The fixtures exercise the protected control table, its FP64 oracle, the actual
correctness and audit wiring, and a Python emulation of the retained Triton
kernel bodies. None of this qualifies a GPU kernel or replaces task_validator.
"""
import ast
import functools
import importlib.util
from itertools import accumulate, product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import pytest
import torch
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_protocol import parse_command_result
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/triton2flydsl/sglang/chunk_local_cumsum'
REL = TASK.relative_to(ROOT).as_posix()
BASE = '099e44cf6deb74131d2cfc96a3a055949a2edbdc'
HEAD_FIRST = {'control_head_first_scalar', 'control_head_first_scalar_reverse_scale',
              'control_head_first_vector', 'control_head_first_varlen_scalar',
              'control_head_first_varlen_vector'}
VARLEN = {'control_varlen_scalar', 'control_varlen_scalar_chunk_indices',
          'control_varlen_vector', 'control_varlen_vector_chunk_indices',
          'control_head_first_varlen_scalar', 'control_head_first_varlen_vector'}
MULTI_SEQUENCE = VARLEN - HEAD_FIRST
ALL = HEAD_FIRST | VARLEN
REVERSED = {'control_head_first_scalar_reverse_scale', 'control_head_first_vector',
            'control_varlen_scalar_chunk_indices', 'control_varlen_vector_chunk_indices',
            'control_head_first_varlen_vector'}
SCALED = {'control_head_first_scalar_reverse_scale', 'control_varlen_scalar_chunk_indices',
          'control_varlen_vector', 'control_head_first_varlen_vector'}
# Exact reviewed harness delta: (added text, base text).
HARNESS_EDITS = [
    ('(scalar 3D path + vector 4D path). The ten scored cases use the regular\n'
     '[B,T,H(,S)] layout without cu_seqlens. Untimed correctness-only controls in\n'
     'scripts/semantic_controls.py also call head_first=True and varlen\n'
     'cu_seqlens/chunk_indices variants against an independent FP64 reference.\n',
     '(scalar 3D path + vector 4D path). Exercises the regular (non-varlen) path;\n'
     'the per-chunk math is identical to the varlen branch, only offsets differ.\n'),
    ('    from scripts.semantic_controls import run_controls\n'
     '    try:\n'
     '        details.extend(run_controls(mod.chunk_local_cumsum, _compare_sglang_output,\n'
     '                                    require_unchanged, start_shape_id=len(details) + 1))\n'
     '    except Exception as error:\n'
     '        return False, f"Head-first/varlen semantic control: {error}", details\n',
     ''),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def base_text(relative):
    return subprocess.check_output(['git', 'show', f'{BASE}:{REL}/{relative}'], cwd=ROOT).decode()


def controls():
    return load(TASK / 'scripts/semantic_controls.py', '_cumsum_semantic_controls')


def replay():
    return load(TASK / 'scripts/replay_checks.py', '_cumsum_replay_checks')


def harness_functions(names):
    checks = replay()
    ns = {'torch': torch, 'CHUNK_SIZE': 64, 'require_unchanged': checks.require_unchanged,
          'require_tensor_contract': checks.require_tensor_contract}
    tree = ast.parse((TASK / 'test_kernel_harness.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(nodes, type_ignores=[]), '<protected cumsum harness>', 'exec'), ns)
    return ns


def single_case(monkeypatch, module, case):
    monkeypatch.setattr(module, 'CONTROL_CASES', [case])


def test_manifest_matches_control_table_and_declares_every_branch():
    c = controls()
    rows = json.loads((TASK / 'cases.json').read_text())['cases']
    actions = load(TASK / 'scripts/task_actions.py', '_cumsum_actions_manifest')
    assert rows[10:] == c.CONTROL_CASES and len(rows) == actions.CORRECTNESS_COUNT == 19
    assert {r['test_case_id'] for r in rows[10:]} == ALL
    assert all(r['checks'] == ['correctness'] for r in rows[10:])
    assert all(r['checks'] == ['correctness', 'performance'] for r in rows[:10])
    assert set(actions.PERFORMANCE_IDS.values()) == {r['test_case_id'] for r in rows[:10]}
    params = {r['test_case_id']: r['params'] for r in rows[10:]}
    assert {k for k, p in params.items() if p['head_first']} == HEAD_FIRST
    assert {k for k, p in params.items() if p['seq_lens']} == VARLEN
    assert {k for k, p in params.items() if p['reverse']} == REVERSED
    assert {k for k, p in params.items() if p['scale'] is not None} == SCALED
    varlen = [p for p in params.values() if p['seq_lens']]
    assert {p['chunk_indices'] for p in varlen} == {'omitted', 'supplied'}
    assert {p['cu_seqlens_dtype'] for p in varlen} == {'int32', 'int64'}
    assert {p['ndim'] for p in varlen} == {p['ndim'] for p in params.values() if p['head_first']} == {3, 4}
    lengths = [n for p in varlen for n in p['seq_lens']]
    assert 1 in lengths and any(n % 64 for n in lengths) and any(n > 64 for n in lengths)
    for p in params.values():
        assert c.input_shape(p)[1 if not p['head_first'] else 2] == p['T']
        if p['seq_lens']:
            assert p['B'] == 1 and sum(p['seq_lens']) == p['T']
            assert len(p['seq_lens']) == 1 or not p['head_first']


def test_task_role_rejects_missing_reordered_or_altered_controls(tmp_path, monkeypatch):
    c = controls()
    package = types.ModuleType('scripts')
    package.__path__ = [str(TASK / 'scripts')]
    monkeypatch.setitem(sys.modules, 'scripts', package)
    monkeypatch.setitem(sys.modules, 'scripts.semantic_controls', c)
    actions = load(TASK / 'scripts/task_actions.py', '_cumsum_actions_role')
    h = types.SimpleNamespace(__file__=str(TASK / 'test_kernel_harness.py'),
                              TEST_SHAPES=actions.EXPECTED_SHAPES)
    monkeypatch.delenv('ARENA_EVAL_PHASE', raising=False)
    actions.select_role(h, 'candidate', False)
    assert h.ARENA_FINAL_CANDIDATE is True
    rows = json.loads((TASK / 'cases.json').read_text())['cases']
    altered = json.loads(json.dumps(rows))
    altered[-1]['params']['seq_lens'] = [96, 1]
    for variant in (rows[:-1], rows[:10] + rows[11:] + rows[10:11], altered,
                    rows[:10] + [dict(rows[10], checks=['correctness', 'performance'])] + rows[11:]):
        (tmp_path / 'cases.json').write_text(json.dumps({'cases': variant}))
        h.__file__ = str(tmp_path / 'test_kernel_harness.py')
        with pytest.raises(ValueError, match='Semantic controls'):
            actions.select_role(h, 'candidate', False)


def test_scored_rows_seeds_gates_timing_and_initial_source_unchanged_vs_base():
    base_rows = json.loads(base_text('cases.json'))['cases']
    current = (TASK / 'cases.json').read_text()
    assert json.loads(current)['cases'][:10] == base_rows and len(base_rows) == 10
    old = base_text('cases.json')
    assert current.startswith(old[:old.rindex('    }\n  ]\n}\n')] + '    }')
    harness = (TASK / 'test_kernel_harness.py').read_text()
    for added, previous in HARNESS_EDITS:
        assert harness.count(added) == 1
        harness = harness.replace(added, previous, 1)
    assert harness == base_text('test_kernel_harness.py')
    assert (TASK / 'chunk_local_cumsum.py').read_text() == base_text('chunk_local_cumsum.py')

    def literals(text):
        found = {}
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    found[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
        return found
    now, then = literals((TASK / 'scripts/task_actions.py').read_text()), literals(base_text('scripts/task_actions.py'))
    assert now['EXPECTED_SHAPES'] == then['EXPECTED_SHAPES']
    assert now['PERFORMANCE_IDS'] == then['PERFORMANCE_IDS']
    now, then = literals((TASK / 'test_kernel_harness.py').read_text()), literals(base_text('test_kernel_harness.py'))
    for name in ('TEST_SHAPES', 'CHUNK_SIZE', 'WARMUP_ITERATIONS', 'BENCHMARK_ITERATIONS'):
        assert now[name] == then[name]


def naive(values, p, chunk_size=64):
    """Independent flat-memory Python loop over the documented layouts."""
    B, T, H, S = p['B'], p['T'], p['H'], p.get('S', 1)
    if p['head_first']:
        addr = lambda b, t, h, s: ((b * H + h) * T + t) * S + s
    else:
        addr = lambda b, t, h, s: ((b * T + t) * H + h) * S + s
    offsets = [0] + list(accumulate(p['seq_lens'])) if p['seq_lens'] else [0, T]
    out = [None] * len(values)
    factor = 1.0 if p['scale'] is None else p['scale']
    for b, h, s in product(range(B), range(H), range(S)):
        for bos, eos in zip(offsets[:-1], offsets[1:]):
            for t0 in range(bos, eos, chunk_size):
                times = list(range(t0, min(t0 + chunk_size, eos)))
                total = 0.0
                for t in reversed(times) if p['reverse'] else times:
                    total += values[addr(b, t, h, s)]
                    out[addr(b, t, h, s)] = total * factor
    assert None not in out
    return out


@pytest.mark.parametrize('index', range(9))
def test_fp64_oracle_matches_independent_naive_loop(index):
    c = controls()
    p = c.CONTROL_CASES[index]['params']
    for changed in (False, True):
        g, kwargs, offsets = c.make_draw(p, changed, device='cpu')
        assert list(g.shape) == c.input_shape(p) and g.dtype == torch.float32 and bool((g < 0).all())
        pristine = g.clone()
        expected = c.reference(g, reverse=p['reverse'], scale=p['scale'],
                               head_first=p['head_first'], offsets=offsets)
        assert torch.equal(g, pristine) and expected.dtype == torch.float64
        loop = torch.tensor(naive(g.double().flatten().tolist(), p), dtype=torch.float64)
        torch.testing.assert_close(expected.flatten(), loop, atol=1e-9, rtol=0)
        if offsets is not None:
            assert kwargs['cu_seqlens'].tolist() == offsets
            assert kwargs['cu_seqlens'].dtype == getattr(torch, p['cu_seqlens_dtype'])
            if p['chunk_indices'] == 'supplied':
                pairs = [[n, k] for n, length in enumerate(p['seq_lens'])
                         for k in range((length + 63) // 64)]
                assert kwargs['chunk_indices'].tolist() == pairs
                assert kwargs['chunk_indices'].dtype == kwargs['cu_seqlens'].dtype
            else:
                assert 'chunk_indices' not in kwargs
        assert kwargs.get('head_first', False) is p['head_first']


def wrong_operator(behavior, c):
    cache = {}

    def operator(g, chunk_size, reverse=False, scale=None, cu_seqlens=None,
                 head_first=False, chunk_indices=None, output_dtype=torch.float):
        offsets = None if cu_seqlens is None else cu_seqlens.tolist()
        kwargs = dict(chunk_size=chunk_size, reverse=reverse, scale=scale,
                      head_first=head_first, offsets=offsets)
        if behavior == 'ignore_cu_seqlens':
            kwargs['offsets'] = None  # cumulative sums run across sequence boundaries
        elif behavior == 'global_chunk_alignment' and offsets is not None:
            kwargs['offsets'] = sorted(set(offsets) | set(range(0, offsets[-1], chunk_size)))
        elif behavior == 'no_chunk_reset':
            kwargs['chunk_size'] = 1 << 30
        elif behavior == 'wrong_time_axis':
            kwargs.update(head_first=False, offsets=None if head_first else offsets)
        elif behavior == 'ignore_reverse':
            kwargs['reverse'] = False
        elif behavior == 'ignore_scale':
            kwargs['scale'] = None
        out = c.reference(g, **kwargs).float()
        if behavior == 'zeros':
            out.zero_()
        elif behavior == 'transposed_output' and head_first:
            out = out.transpose(1, 2).contiguous()
        elif behavior == 'float64_output':
            out = out.double()
        elif behavior == 'cached':
            out = cache.setdefault(tuple(g.shape), out)
        elif behavior == 'mutate_input':
            g.mul_(.5)
        elif behavior == 'mutate_cu_seqlens' and cu_seqlens is not None:
            cu_seqlens[-1] -= 1
        return out
    return operator


REJECTED = {
    'correct': set(), 'zeros': ALL, 'ignore_cu_seqlens': MULTI_SEQUENCE,
    'global_chunk_alignment': {'control_varlen_scalar', 'control_varlen_scalar_chunk_indices',
                               'control_varlen_vector'},
    'no_chunk_reset': ALL, 'wrong_time_axis': HEAD_FIRST, 'transposed_output': HEAD_FIRST,
    'ignore_reverse': REVERSED, 'ignore_scale': SCALED, 'float64_output': ALL,
    'cached': ALL, 'mutate_input': ALL, 'mutate_cu_seqlens': VARLEN,
}


@pytest.mark.parametrize('behavior', list(REJECTED))
def test_wrong_outputs_and_input_mutation_are_rejected_at_original_gate(behavior, monkeypatch):
    c = controls()
    ns = harness_functions({'_checked_sglang_output', '_compare_sglang_output'})
    unchanged = replay().require_unchanged
    for case in list(c.CONTROL_CASES):
        single_case(monkeypatch, c, case)
        run = lambda: c.run_controls(wrong_operator(behavior, c), ns['_compare_sglang_output'],
                                     unchanged, device='cpu')
        if case['test_case_id'] in REJECTED[behavior]:
            match = {'mutate_input': 'read-only', 'mutate_cu_seqlens': 'read-only',
                     'float64_output': 'dtype', 'transposed_output': 'shape'}.get(behavior, 'mismatch')
            with pytest.raises(AssertionError, match=match):
                run()
        else:
            record, = run()
            assert record['passed'] is True and record['control'] == case['test_case_id']


def test_controls_reuse_input_allocation_and_restore_base_draw():
    c = controls()
    ns = harness_functions({'_checked_sglang_output', '_compare_sglang_output'})
    calls = []

    def operator(g, **kwargs):
        calls.append((g.data_ptr(), g.clone(), sorted(kwargs)))
        return wrong_operator('correct', c)(g, **kwargs)
    records = c.run_controls(operator, ns['_compare_sglang_output'], replay().require_unchanged,
                             device='cpu')
    assert [r['shape_id'] for r in records] == list(range(11, 20))
    assert [r['control'] for r in records] == [r['test_case_id'] for r in c.CONTROL_CASES]
    assert all(r['timed'] is False and r['actual_operator_draws'] == 3 for r in records)
    assert len(calls) == 27
    for i in range(0, 27, 3):
        (p0, a, k0), (p1, b, k1), (p2, a2, k2) = calls[i:i + 3]
        assert p0 == p1 == p2 and torch.equal(a, a2) and not torch.equal(a, b) and k0 == k1 == k2
    json.dumps(records, allow_nan=False)


def emulated_initial_source(monkeypatch, launches):
    """Run the retained Triton kernel bodies with a Python `tl` on CPU.

    Out-of-range block-pointer lanes load zero (upstream-tested Triton practice);
    every in-range address is bounds-checked against its buffer.
    """
    pid = [0, 0, 0]

    class Pointer:
        def __init__(self, flat, offset=0):
            self.flat, self.offset = flat, offset

        def __add__(self, other):
            return Pointer(self.flat, self.offset + int(other))
        __radd__ = __add__

    class Scalar(int):
        def to(self, dtype):
            return Scalar(self)

    class Block:
        def __init__(self, base, shape, strides, offsets, block_shape, order):
            self.base, self.shape, self.strides = base, shape, strides
            self.offsets, self.block = offsets, block_shape
            self.dtype = types.SimpleNamespace(element_ty=base.flat.dtype)

        def locate(self, boundary_check):
            grids = torch.meshgrid(*[torch.arange(int(o), int(o) + b)
                                     for o, b in zip(self.offsets, self.block)], indexing='ij')
            index = self.base.offset + sum(m * int(s) for m, s in zip(grids, self.strides))
            valid = torch.ones(self.block, dtype=torch.bool)
            for dim in boundary_check:
                valid &= (grids[dim] >= 0) & (grids[dim] < int(self.shape[dim]))
            assert bool(((index[valid] >= 0) & (index[valid] < self.base.flat.numel())).all())
            return index, valid

    def load_value(pointer, boundary_check=()):
        if isinstance(pointer, Pointer):
            assert 0 <= pointer.offset < pointer.flat.numel()
            return Scalar(int(pointer.flat[pointer.offset]))
        index, valid = pointer.locate(boundary_check)
        values = torch.zeros(pointer.block, dtype=pointer.base.flat.dtype)
        values[valid] = pointer.base.flat[index[valid]]
        return values

    def store(pointer, value, boundary_check=()):
        index, valid = pointer.locate(boundary_check)
        pointer.base.flat[index[valid]] = value[valid].to(pointer.base.flat.dtype)

    tl = types.ModuleType('triton.language')
    tl.constexpr, tl.float32, tl.int32 = object, torch.float32, torch.int32
    tl.load, tl.store, tl.make_block_ptr = load_value, store, Block
    tl.program_id = lambda axis: pid[axis]
    tl.arange = lambda start, end: torch.arange(start, end)
    tl.cumsum = lambda x, axis=0: torch.cumsum(x, axis)
    tl.sum = lambda x, axis=0: x.sum(axis)
    tl.where = lambda c, a, b: torch.where(c, torch.tensor(float(a)), torch.tensor(float(b)))
    tl.dot = lambda a, b, allow_tf32=True: a @ b

    class Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            def launch(**kwargs):
                launches.append(dict(kwargs, kernel=self.fn.__name__))
                shape = grid(kwargs) if callable(grid) else grid
                args = {}
                for name, value in kwargs.items():
                    if name in {'num_warps', 'num_stages'}:
                        continue
                    if isinstance(value, torch.Tensor):
                        assert value.is_contiguous()
                        value = Pointer(value.view(-1))
                    args[name] = value
                for ids in product(*(range(int(n)) for n in shape)):
                    pid[:len(ids)] = ids
                    self.fn(**args)
            return launch

    triton = types.ModuleType('triton')
    triton.language = tl
    triton.jit = lambda fn=None, **options: Kernel(fn) if fn is not None else Kernel
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.language', tl)
    return load(TASK / 'chunk_local_cumsum.py', '_cumsum_emulated_initial_source')


def test_emulated_initial_source_matches_protected_oracle_on_regular_layout(monkeypatch):
    module = emulated_initial_source(monkeypatch, [])
    ns = harness_functions({'reference_cumsum', '_checked_sglang_output', '_compare_sglang_output'})
    for cfg in [dict(B=2, T=130, H=3, reverse=False, scale=None),
                dict(B=1, T=70, H=2, reverse=True, scale=.5),
                dict(B=2, T=65, H=2, S=24, reverse=True, scale=None),
                dict(B=1, T=100, H=1, S=72, reverse=False, scale=.5)]:
        shape = [cfg['B'], cfg['T'], cfg['H']] + ([cfg['S']] if 'S' in cfg else [])
        g = torch.nn.functional.logsigmoid(torch.randn(shape, generator=torch.Generator().manual_seed(3)))
        out = module.chunk_local_cumsum(g, chunk_size=64, reverse=cfg['reverse'], scale=cfg['scale'])
        ns['_compare_sglang_output'](out, ns['reference_cumsum'](g, cfg))


def test_emulated_initial_source_passes_controls_through_declared_branches(monkeypatch):
    launches = []
    module = emulated_initial_source(monkeypatch, launches)
    c = controls()
    ns = harness_functions({'_checked_sglang_output', '_compare_sglang_output'})
    seen = []

    def operator(g, **kwargs):
        seen.append(kwargs)
        return module.chunk_local_cumsum(g, **kwargs)
    records = c.run_controls(operator, ns['_compare_sglang_output'], replay().require_unchanged,
                             device='cpu')
    assert len(records) == 9 and len(launches) == len(seen) == 27
    assert all(r['passed'] and r['max_diff'] < 1e-4 for r in records)
    repeated = [case for case in c.CONTROL_CASES for _ in range(3)]
    for case, launch, kwargs in zip(repeated, launches, seen):
        p = case['params']
        assert launch['kernel'] == ('chunk_local_cumsum_scalar_kernel' if p['ndim'] == 3
                                    else 'chunk_local_cumsum_vector_kernel')
        assert launch['HEAD_FIRST'] is p['head_first']
        assert launch['IS_VARLEN'] is (p['seq_lens'] is not None)
        assert launch['REVERSE'] is p['reverse'] and launch['HAS_SCALE'] is (p['scale'] is not None)
        if p['seq_lens']:
            assert launch['cu_seqlens'] is kwargs['cu_seqlens']
            if p['chunk_indices'] == 'supplied':
                assert launch['chunk_indices'] is kwargs['chunk_indices']
            assert launch['chunk_indices'].tolist() == c.chunk_index_pairs(c.sequence_offsets(p))


def test_multi_sequence_head_first_varlen_is_packed_blocks_and_stays_excluded(monkeypatch):
    module = emulated_initial_source(monkeypatch, [])
    c = controls()
    g = -torch.arange(1., 11.).reshape(1, 2, 5)
    out = module.chunk_local_cumsum(g, chunk_size=2, cu_seqlens=torch.tensor([0, 2, 5]), head_first=True)
    flat = g.flatten()
    # Sequence n is read as its own [H, T_n] block at bos*H, not as g[:, :, bos:eos].
    packed = torch.cat([c.reference(flat[2 * a:2 * b].reshape(1, 2, b - a), chunk_size=2,
                                    head_first=True).flatten() for a, b in [(0, 2), (2, 5)]])
    torch.testing.assert_close(out.flatten().double(), packed)
    logical = torch.cat([c.reference(g[:, :, a:b], chunk_size=2, head_first=True)
                         for a, b in [(0, 2), (2, 5)]], 2)
    assert not torch.allclose(out.double(), logical)
    params = dict(c.CONTROL_CASES[-1]['params'], seq_lens=[50, 47])
    with pytest.raises(ValueError, match='one sequence only'):
        c.sequence_offsets(params)


def fake_flydsl_kernel():
    # CPU profile-event fixture standing in for a native launch; not FlyDSL.
    namespace = {'__name__': 'flydsl.compiler._cpu_profile_fixture'}
    exec('class FakeCompiledKernel:\n'
         ' calls = 0\n'
         ' def __call__(self, out, compute):\n'
         '  from torch.utils._python_dispatch import _disable_current_modes\n'
         '  type(self).calls += 1\n'
         '  with _disable_current_modes():\n'
         '   out.copy_(compute())\n', namespace)
    return namespace['FakeCompiledKernel']()


def correctness_harness(monkeypatch, candidate):
    """Actual protected correctness bodies on CPU with the real audit module."""
    checks = replay()
    c = controls()
    h = types.ModuleType('_cumsum_cpu_harness')
    h.__file__ = str(TASK / 'test_kernel_harness.py')
    h.__dict__.update(torch=torch, time=time, require_unchanged=checks.require_unchanged,
                      require_tensor_contract=checks.require_tensor_contract)
    tree = ast.parse((TASK / 'test_kernel_harness.py').read_text())
    names = {'_is_oom', '_retry_oom', 'make_g', 'reference_cumsum', '_shape_of',
             '_checked_sglang_output', '_compare_sglang_output', 'run_correctness'}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                h.__dict__[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
    exec(compile(ast.Module(nodes, type_ignores=[]), h.__file__, 'exec'), h.__dict__)
    make_g = h.make_g
    h.make_g = lambda cfg, device='cuda': make_g(cfg, 'cpu')
    h.load_module = lambda: types.SimpleNamespace(chunk_local_cumsum=candidate)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    package = types.ModuleType('scripts')
    package.__path__ = [str(TASK / 'scripts')]
    cpu_controls = types.SimpleNamespace(CONTROL_CASES=c.CONTROL_CASES,
                                         run_controls=functools.partial(c.run_controls, device='cpu'))
    monkeypatch.setitem(sys.modules, 'scripts', package)
    monkeypatch.setitem(sys.modules, 'scripts.semantic_controls', cpu_controls)
    monkeypatch.setitem(sys.modules, 'scripts.candidate_checks',
                        load(TASK / 'scripts/candidate_checks.py', '_cumsum_candidate_checks'))
    monkeypatch.delenv('ARENA_EVAL_PHASE', raising=False)
    actions = load(TASK / 'scripts/task_actions.py', '_cumsum_actions_e2e')
    return h, actions, c


@pytest.mark.parametrize('behavior', ['correct', 'torch_math_on_controls', 'no_flydsl_on_varlen',
                                      'ignore_head_first', 'baseline_role'])
def test_correctness_action_runs_controls_through_candidate_flydsl_audit(behavior, monkeypatch):
    fly = fake_flydsl_kernel()
    calls = []
    c = controls()

    def candidate(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False,
                  output_dtype=torch.float, chunk_indices=None):
        calls.append({'head_first': head_first, 'cu_seqlens': cu_seqlens, 'chunk_indices': chunk_indices})
        offsets = None if cu_seqlens is None else cu_seqlens.tolist()
        if behavior == 'torch_math_on_controls' and len(calls) > 10:
            return g.cumsum(1)
        if behavior == 'ignore_head_first':
            head_first = False
        out = torch.empty_like(g, dtype=output_dtype)
        compute = lambda: c.reference(g, chunk_size=chunk_size, reverse=reverse, scale=scale,
                                      head_first=head_first, offsets=offsets).to(out.dtype)
        if behavior == 'no_flydsl_on_varlen' and offsets is not None:
            from torch.utils._python_dispatch import _disable_current_modes
            with _disable_current_modes():
                out.copy_(compute())
            return out
        fly(out, compute)
        return out
    h, actions, _ = correctness_harness(monkeypatch, candidate)
    actions.select_role(h, 'baseline' if behavior == 'baseline_role' else 'candidate', False)
    assert h.ARENA_FINAL_CANDIDATE is (behavior != 'baseline_role')
    if behavior in {'correct', 'baseline_role'}:
        observed = actions.check(h)
        assert observed == ([] if behavior == 'baseline_role'
                            else ['flydsl.compiler._cpu_profile_fixture.FakeCompiledKernel'])
        assert len(calls) == 37 and type(fly).calls == 37
        for call, case in zip(calls[10:], [r for r in c.CONTROL_CASES for _ in range(3)]):
            p = case['params']
            assert call['head_first'] is p['head_first']
            assert (call['cu_seqlens'] is not None) is (p['seq_lens'] is not None)
            assert (call['chunk_indices'] is not None) is (p['chunk_indices'] == 'supplied')
        assert all(call['cu_seqlens'] is None and not call['head_first'] for call in calls[:10])
        return
    expected = {'torch_math_on_controls': 'non-preparation PyTorch operation aten::cumsum',
                'no_flydsl_on_varlen': 'No FlyDSL kernel runtime invocation observed',
                'ignore_head_first': 'Numerical mismatch'}[behavior]
    with pytest.raises(RuntimeError, match='Incomplete/failed correctness evidence') as error:
        actions.check(h)
    assert expected in str(error.value) and 'Head-first/varlen semantic control' in str(error.value)


def test_guard_keeps_control_module_and_manifest_protected(tmp_path):
    workspace = tmp_path / 'task'
    shutil.copytree(TASK, workspace, ignore=shutil.ignore_patterns('__pycache__', 'build'))
    spec = load_task_spec(workspace / 'config.yaml', task_id='triton2flydsl/sglang/chunk_local_cumsum')
    guard = snapshot_workspace_harness(workspace, task_spec=spec)
    source = workspace / 'chunk_local_cumsum.py'
    source.write_text(source.read_text() + '\n# allowed candidate implementation edit\n')
    verify_workspace_harness(guard, discard_added=False)
    for relative in ('scripts/semantic_controls.py', 'cases.json'):
        path = workspace / relative
        original = path.read_text()
        path.write_text(original + '\n')
        with pytest.raises(RuntimeError):
            verify_workspace_harness(guard, discard_added=False)
        path.write_text(original)


def test_real_cpu_validate_task_reports_nineteen_rows_and_control_known_answers():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    process = subprocess.run([sys.executable, 'scripts/evaluate.py', 'validate-task'], cwd=TASK,
                             text=True, capture_output=True, timeout=120, env=env)
    result = parse_command_result(process.stdout, role='task', action='validate-task',
                                  returncode=process.returncode)
    assert result.passed, result.reason
    assert len(result.cases) == 19
    rows = result.metadata['reference_controls']
    assert len(rows) == 15
    assert [r['control'] for r in rows[-9:]] == [r['test_case_id'] for r in controls().CONTROL_CASES]
    assert all(r['known_answer'] == 'PASS' and r['negative_output'] == 'rejected' for r in rows)
