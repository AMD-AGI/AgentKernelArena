"""The actual v2 guard protects the decode wrapper that invokes the kernel."""
import ast
import importlib.util
from pathlib import Path
import shutil

import pytest

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec

TASK = Path(__file__).resolve().parents[1] / 'tasks/triton2triton/vllm/triton_prepare_eagle_docode'


@pytest.fixture
def guarded(tmp_path):
    root = tmp_path / 'task'
    shutil.copytree(TASK, root)
    spec = load_task_spec(root / 'config.yaml', task_id='triton2triton/vllm/triton_prepare_eagle_docode')
    guard = snapshot_workspace_harness(root, task_spec=spec)
    return root, root / spec.candidate.editable[0].path, guard


def write_tree(path, tree):
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + '\n')


def test_real_guard_allows_declared_kernel_and_new_implementation_helper(guarded):
    _, source, guard = guarded
    tree = ast.parse(source.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_prepare_eagle_docode_kernel')
    kernel.body.append(ast.Pass())
    tree.body.extend(ast.parse('@triton.jit\ndef copy_helper(x):\n    return x\n').body)
    write_tree(source, tree)
    verify_workspace_harness(guard)


@pytest.mark.parametrize('attack', ['torch_wrapper', 'unused_kernel', 'rebind_triton', 'rebind_wrapper'])
def test_real_guard_rejects_wrapper_backend_bypass_even_when_jit_inspection_passes(guarded, attack):
    root, source, guard = guarded
    tree = ast.parse(source.read_text())
    wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_eagle_decode')
    if attack == 'torch_wrapper':
        wrapper.body = ast.parse('''n = draft_tokens.numel()
input_ids[:n].copy_(draft_tokens)
input_hidden_states[:n].copy_(output_hidden_states[last_token_indices])
positions[:n].copy_((positions[:n] + 1).clamp(max=max_model_len - 1))
seq_lens[:n].copy_((target_seq_lens - num_rejected + 1).clamp(max=max_model_len))
seq_lens[n:max_num_reqs].zero_()
query_start_loc.copy_(torch.arange(max_num_reqs + 1, device=query_start_loc.device).clamp(max=n))
''').body
    elif attack == 'unused_kernel':
        wrapper.body = [ast.Return(ast.Constant(None))]
    elif attack == 'rebind_triton':
        tree.body.insert(0, ast.Import(names=[ast.alias(name='torch', asname='triton')]))
    else:
        tree.body.extend(ast.parse('prepare_eagle_decode = lambda *args: None').body)
    write_tree(source, tree)
    loader = importlib.util.spec_from_file_location('decode_eval', root / '_arena_eval.py')
    adapter = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(adapter)
    # Merely checking that a JIT definition exists is insufficient; the shared
    # guard must independently reject changes to the wrapper/import boundary.
    assert adapter.inspect_candidate(adapter.load_manifest(), require_implemented=True) == 'implemented'
    with pytest.raises(RuntimeError, match='[Hh]arness|protected'):
        verify_workspace_harness(guard)


# This is the independently reviewed decorator counterexample. The real CPU
# full-action fixture below preserves the public wrapper and all eight cases.
HOST_LAUNCHER = '''
def host_launcher(jit_function):
    class HostLauncher:
        def __getitem__(self, grid):
            def launch(draft_tokens, output_hs, output_stride, last_ti,
                       target_sl, num_rej, input_ids, positions, input_hs,
                       input_stride, qsl, seq_lens, hidden_size, max_ml,
                       max_nr, **options):
                nr = draft_tokens.numel()
                input_ids[:nr].copy_(draft_tokens)
                input_hs[:nr].copy_(output_hs[last_ti])
                positions[:nr].copy_((positions[:nr] + 1).clamp(max=max_ml - 1))
                seq_lens[:nr].copy_((target_sl - num_rej + 1).clamp(max=max_ml))
                seq_lens[nr:max_nr].zero_()
                qsl.copy_(torch.arange(max_nr + 1, device=qsl.device).clamp(max=nr))
            return launch
    return HostLauncher()
'''


def module_at(path, name, *, cpu=False):
    import types
    tree = ast.parse(path.read_text())
    if cpu:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == 'cuda':
                node.value = 'cpu'
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(tree, str(path), 'exec'), mod.__dict__)
    return mod


def add_decorators(source, decorators, *, host_helper=False):
    tree = ast.parse(source.read_text())
    kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_prepare_eagle_docode_kernel')
    kernel.decorator_list = ast.parse('\n'.join('@' + d for d in decorators)
                                     + '\ndef example(): pass').body[0].decorator_list
    if host_helper:
        tree.body.insert(tree.body.index(kernel), ast.parse(HOST_LAUNCHER).body[0])
    write_tree(source, tree)


@pytest.fixture
def cpu_task(guarded, monkeypatch):
    """Actual protected actions with CPU tensors and explicit launcher doubles.

    This checks dispatch and rejection, not native Triton compilation or GPU
    execution. Each accepted launch double computes the operator independently
    of the task's reference and records that the protected wrapper reached it.
    """
    import sys
    import types
    import torch
    root, source, guard = guarded
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.chdir(root)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []

    class JITFunction:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return lambda *args, **kwargs: self.run(*args, **kwargs)

        def run(self, tokens, hidden, hidden_stride, indices, seq, rejected,
                out_tokens, pos, out_hidden, out_stride, qsl, lens,
                hidden_size, max_len, max_reqs, **options):
            calls.append(self.fn.__name__)
            count = tokens.numel()
            # Loop-based fixture, distinct from the archived vectorized bypass.
            for r in range(count):
                out_tokens[r] = tokens[r]
                out_hidden[r] = hidden[int(indices[r])]
                pos[r] = min(int(pos[r]) + 1, max_len - 1)
                lens[r] = min(int(seq[r]) - int(rejected[r]) + 1, max_len)
            for r in range(count, max_reqs):
                lens[r] = 0
            for r in range(max_reqs + 1):
                qsl[r] = min(r, count)

    class Autotuner:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            return lambda *args, **kwargs: self.run(*args, **kwargs)

        def run(self, *args, **kwargs):
            return self.fn.run(*args, **kwargs)

    class Heuristics(Autotuner):
        pass

    triton = types.ModuleType('triton')
    language = types.ModuleType('triton.language')
    language.constexpr = object()
    triton.language = language
    triton.jit = lambda fn=None, **options: JITFunction(fn) if fn else lambda f: JITFunction(f)
    triton.autotune = lambda **kw: lambda fn: Autotuner(fn)
    triton.heuristics = lambda values: lambda fn: Heuristics(fn)
    triton.Config = lambda *a, **kw: (a, kw)
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.language', language)
    runtime = types.ModuleType('triton.runtime')
    jit = types.ModuleType('triton.runtime.jit')
    autotuner = types.ModuleType('triton.runtime.autotuner')
    jit.JITFunction = JITFunction
    autotuner.Autotuner, autotuner.Heuristics = Autotuner, Heuristics
    for mod in (runtime, jit, autotuner):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    policy = module_at(root / '_arena_kernel_policy.py', '_arena_kernel_policy')
    monkeypatch.setitem(sys.modules, '_arena_kernel_policy', policy)
    contract = module_at(root / '_arena_contract.py', '_arena_contract')
    replay = module_at(root / '_arena_replay.py', '_arena_replay', cpu=True)
    controls = module_at(root / '_upstream_controls.py', '_upstream_controls', cpu=True)
    for mod in (contract, replay, controls):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    adapter = module_at(root / '_arena_eval.py', '_decode_cpu_adapter')

    def load_harness():
        harness = module_at(root / 'scripts/task_runner.py', '_decode_cpu_harness', cpu=True)
        replay.install(harness, contract)
        return harness

    monkeypatch.setattr(adapter, 'load_harness', load_harness)
    yield types.SimpleNamespace(root=root, source=source, guard=guard, adapter=adapter,
                                policy=policy, triton=triton, JIT=JITFunction,
                                Autotuner=Autotuner, Heuristics=Heuristics, calls=calls)
    torch.set_num_threads(old_threads)


@pytest.mark.parametrize('decorators', [
    ['triton.jit'],
    ['triton.jit(debug=True)'],
    ['triton.heuristics({})', 'triton.jit'],
    ['triton.autotune(configs=[triton.Config({}, num_warps=4)], key=[])', 'triton.jit'],
    ['triton.autotune(configs=[], key=[])', 'triton.heuristics({})', 'triton.jit'],
])
def test_native_binding_fixture_runs_full_protected_correctness(cpu_task, decorators):
    t = cpu_task
    add_decorators(t.source, decorators)
    verify_workspace_harness(t.guard)
    assert t.adapter.inspect_candidate(t.adapter.load_manifest(), require_implemented=True) == 'implemented'
    assert t.adapter.evaluate('candidate', 'compile')['status'] == 'PASS'
    result = t.adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'PASS', result
    assert len(result['cases']) == 8
    assert len(t.calls) >= 8
    assert set(t.calls) == {'_prepare_eagle_docode_kernel'}


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
@pytest.mark.parametrize('action', ['compile', 'correctness', 'performance'])
def test_reviewed_host_decorator_fails_actual_action_before_any_launch(cpu_task, role, action):
    t = cpu_task
    before = ast.parse(t.source.read_text())
    wrapper = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_eagle_decode')
    add_decorators(t.source, ['host_launcher', 'triton.jit'], host_helper=True)
    after = ast.parse(t.source.read_text())
    actual_wrapper = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_eagle_decode')
    assert ast.dump(wrapper) == ast.dump(actual_wrapper)
    with pytest.raises(ValueError, match='unsupported launcher decorator'):
        t.policy.load_checked(t.source)
    result = t.adapter.evaluate(role, action)
    assert result['status'] == 'FAIL' and 'unsupported launcher decorator' in result['reason']
    if action == 'correctness':
        assert len(result['cases']) == 8 and all(r['status'] == 'FAIL' for r in result['cases'])
    assert t.calls == []


@pytest.mark.parametrize('mode', ['host_proxy', 'named_impostor', 'subclass', 'run_override',
                                  'cycle', 'other_source', 'other_function', 'class_method_override'])
def test_actual_loader_rejects_runtime_substitution_despite_valid_decorator(cpu_task, mode):
    t = cpu_task
    original_jit = t.triton.jit
    class Proxy:
        def __init__(self, fn):
            self.fn = fn
        def __getitem__(self, grid):
            raise AssertionError('host proxy must never launch')
    class Subclass(t.JIT):
        pass

    def substitute(fn):
        if mode == 'host_proxy':
            return Proxy(original_jit(fn))
        if mode == 'named_impostor':
            return type('JITFunction', (), {'__module__': 'triton.runtime.jit'})()
        if mode == 'subclass':
            return Subclass(fn)
        kernel = original_jit(fn)
        if mode == 'run_override':
            kernel.run = lambda *a, **kw: None
        elif mode == 'cycle':
            kernel = t.Autotuner(kernel)
            kernel.fn = kernel
        elif mode == 'other_source':
            fn.__code__ = fn.__code__.replace(co_filename='unrelated.py')
        elif mode == 'other_function':
            fn.__name__ = 'unused_other_kernel'
        elif mode == 'class_method_override':
            t.JIT.run = lambda *a, **kw: None
        return kernel

    t.triton.jit = substitute
    assert t.adapter.inspect_candidate(t.adapter.load_manifest(), require_implemented=True) == 'implemented'
    result = t.adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'FAIL', result
    assert len(result['cases']) == 8 and all(r['status'] == 'FAIL' for r in result['cases'])
    assert t.calls == []


@pytest.mark.parametrize('decorators', [[], ['triton.jit', 'triton.jit'],
                                      ['triton.jit', 'triton.heuristics({})'],
                                      ['host_launcher', 'triton.jit'], ['other.jit'],
                                      ['triton.autotune', 'triton.jit']])
def test_target_decorator_contract_rejects_unsupported_chains(cpu_task, decorators):
    add_decorators(cpu_task.source, decorators, host_helper='host_launcher' in decorators)
    result = cpu_task.adapter.evaluate('candidate', 'compile')
    assert result['status'] == 'FAIL'
    assert cpu_task.calls == []


def test_binding_fix_preserves_all_original_numerics_cases_and_timing():
    import hashlib
    # Reviewed pre-fix 9450136d byte hashes; no historical worktree is required.
    originals = {
        'source/triton_prepare_eagle_docode.py': '98804483f8b91e03ea25875adb94e8a3c57b37a5d5cdcc25555755c9bcb31ed2',
        'workloads.json': '882e46b66d938aa1354504ff484c224461cd4364e91fd11b1f367862d81b048d',
        '_arena_contract.py': '55e04eee5a189413a1a02bc8c5bf196af74803e978da8e675813d50973c3a8be',
        '_arena_replay.py': '24ed21cde8976a7b7b7449b67349de7180032c29ea5220312252987507b5da7a',
        '_upstream_controls.py': 'e0dd10dcea0817bb15b0c84bfd0f8a40eb4b466951c1c73ecf03ba4ded954ec4',
    }
    for name, digest in originals.items():
        assert hashlib.sha256((TASK / name).read_bytes()).hexdigest() == digest, name
    # Exclude only the deliberately changed module-loader boundary. All other
    # runner statements retain the reviewed pre-fix syntax tree.
    tree = ast.parse((TASK / 'scripts/task_runner.py').read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'load_module')]
    digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    assert digest == '591f74d4672918cbefe2b7c90a6323ccb562ee158ba8eb10e990618e592e5ceb'


def test_runner_resolves_protected_policy_by_task_path(cpu_task, monkeypatch):
    import sys
    from types import SimpleNamespace
    def wrong_policy(*args):
        raise AssertionError('unrelated module must not shadow the task-owned policy')
    monkeypatch.setitem(sys.modules, '_arena_kernel_policy', SimpleNamespace(load_checked=wrong_policy))
    result = cpu_task.adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'PASS', result
    assert len(result['cases']) == 8 and len(cpu_task.calls) >= 8
