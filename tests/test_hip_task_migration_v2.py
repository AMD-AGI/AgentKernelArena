"""CPU regression evidence for the 89 HIP task migrations.

No test below measures a GPU or substitutes a CPU result for GPU validation.
Meta tensors enumerate protected input generators without allocating large cases.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest
import torch
import yaml

from src.task_spec import ACTIONS, load_task_spec
from src.task_protocol import CaseManifest, parse_command_result

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / 'tasks/hip2hip').rglob('config.yaml')) + sorted((ROOT / 'tasks/torch2hip').rglob('config.yaml'))
EXTENSIONS = [p for p in CONFIGS if (p.parent / 'eval_tools/evaluate.py').is_file()]
NATIVE = [p for p in CONFIGS if (p.parent / 'scripts/evaluate.py').is_file()]


def import_path(path):
    spec = importlib.util.spec_from_file_location('test_hip_' + hashlib.sha256(str(path).encode()).hexdigest()[:12], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options(config):
    command = config['evaluation']['runner']
    return types.SimpleNamespace(**{command[i][2:].replace('-', '_'): command[i+1] for i in range(2, len(command), 2)},
                                 workloads='workload.json', **({} if '--baseline-hip' in command else {'baseline_hip': None}))


def harness_namespace(task):
    """Execute only pure CPU reference functions, never loader/JIT/GPU setup."""
    path = task / 'scripts/task_runner.py'
    tree = ast.parse(path.read_text())
    ns = {'torch': torch, 'math': math}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
             (n.name.startswith('cpu_') or n.name in ('check_point_in_box', 'generate_test_data'))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    ns['TEST_SHAPES'] = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
                            any(isinstance(t, ast.Name) and t.id == 'TEST_SHAPES' for t in n.targets))
    return types.SimpleNamespace(**ns)


def test_inventory():
    assert len(CONFIGS) == 89
    assert len(EXTENSIONS) == 79 and len(NATIVE) == 10
    assert len([p for p in CONFIGS if p.is_relative_to(ROOT / 'tasks/hip2hip')]) == 32
    assert sum(len(json.loads(p.with_name('workload.json').read_text())['cases']) for p in CONFIGS) == 551


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: str(p.parent.relative_to(ROOT / 'tasks')))
def test_v2_contract_and_paths(path):
    spec = load_task_spec(path, task_id=str(path.parent.relative_to(ROOT / 'tasks')))
    raw = spec.to_mapping()
    assert raw['schema_version'] == 2
    assert not {'task_type', 'source_file_path', 'target_kernel_functions', 'prompt', 'compile_command'} & raw.keys()
    assert spec.candidate.language == 'hip'
    assert all(edit.scope == 'file' for edit in spec.candidate.editable)
    for edit in spec.candidate.editable:
        assert (path.parent / edit.path).is_file()
    for entry in spec.candidate.entrypoints:
        assert any(edit.path == entry.file for edit in spec.candidate.editable)
    for ref in spec.baseline.source_files:
        assert (path.parent / ref).is_file()
        assert ref not in {e.path for e in spec.candidate.editable}
    for role, action in ACTIONS:
        command = spec.action(role, action).commands[0]
        assert (path.parent / command[1]).is_file()
        assert tuple(command[-1:] if role == 'task' else command[-2:]) == (('validate-task',) if role == 'task' else (role, action))
    if path in NATIVE:
        assert spec.baseline.kind == 'initial_candidate' and spec.baseline.language == 'hip'
        assert spec.candidate.initial_state == 'implemented'
    else:
        target = path.parent / spec.candidate.editable[0].path
        empty = not target.read_text().strip()
        assert (spec.candidate.initial_state == 'unimplemented') == empty
        assert spec.baseline.kind == 'provided'
        assert spec.baseline.language == ('pytorch' if empty else 'hip')
    assert (path.parent / 'README.md').is_file()


@pytest.mark.parametrize('path', EXTENSIONS, ids=lambda p: p.parent.name)
def test_complete_manifest_from_protected_generator(path):
    raw = yaml.safe_load(path.read_text())
    args = options(raw)
    runner = import_path(path.parent / 'eval_tools/evaluate.py')
    module = import_path(path.parent / args.module)
    functional = import_path(path.parent / args.functional)
    assert callable(inspect.signature(getattr(functional, args.model_class).forward).parameters['fn'].default)
    with torch.device('meta'):
        cases = module.get_inputs()
        if isinstance(cases, (list, tuple)):
            cases = [cases]
        metadata = [runner.describe_inputs(case) for case in cases]
    manifest = json.loads(path.with_name('workload.json').read_text())['cases']
    assert len(manifest) == len(metadata)
    for i, (row, value) in enumerate(zip(manifest, metadata)):
        assert row == {'test_case_id': f'case_{i}', 'params': {'inputs': value, 'model_init_seed': 0, 'correctness_seed': 1337 + i}}


@pytest.mark.parametrize('path', NATIVE, ids=lambda p: p.parent.name)
def test_native_case_manifest_from_original_performance(path):
    tree = ast.parse((path.parent / 'scripts/task_runner.py').read_text())
    h = harness_namespace(path.parent)
    manifest = json.loads(path.with_name('workload.json').read_text())
    assert manifest['test_shapes'] == [list(s) for s in h.TEST_SHAPES]
    perf = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_performance')
    loop = next(n for n in ast.walk(perf) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and
                isinstance(n.target.elts[0], ast.Name) and n.target.elts[0].id == 'shape_idx')
    names = [n.id for n in loop.target.elts[1].elts]
    exprs = []
    for n in ast.walk(perf):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'append' and isinstance(n.args[0], ast.Dict):
            obj = {ast.literal_eval(k): v for k, v in zip(n.args[0].keys, n.args[0].values) if k is not None}
            if 'test_case_id' in obj:
                exprs.append((obj['test_case_id'], obj['params']))
    expected = []
    for i, shape in enumerate(h.TEST_SHAPES):
        env = dict(zip(names, shape), shape_idx=i)
        if exprs:
            for id_expr, params_expr in exprs:
                evaluate = lambda expression: eval(compile(ast.Expression(expression), '<protected case>', 'eval'), {'__builtins__': {}}, env)
                expected.append({'test_case_id': evaluate(id_expr), 'params': evaluate(params_expr)})
        else:
            params = ({'A_rows': shape[0], 'A_cols': shape[1], 'B_cols': shape[2]} if len(shape) == 3 else {'batch': shape[0], 'ctx': shape[1]})
            expected.append({'test_case_id': f'shape_{i}', 'params': params})
    assert manifest['cases'] == expected


@pytest.mark.parametrize('path', [p for p in NATIVE if (p.parent / 'kernel_loader.py').exists()], ids=lambda p: p.parent.name)
def test_independent_reference_controls_and_zero_mutation(path, monkeypatch):
    controls = import_path(path.parent / 'scripts/reference_checks.py')
    h = harness_namespace(path.parent)
    controls.self_test(h)
    calls = [n for n in ast.walk(ast.parse(inspect.getsource(controls.self_test))) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'h' and n.func.attr.startswith('cpu_')]
    symbol = calls[0].func.attr
    original = getattr(h, symbol)
    def zeros(*args, **kwargs):
        out = original(*args, **kwargs)
        return tuple(torch.zeros_like(x) for x in out) if isinstance(out, tuple) else torch.zeros_like(out)
    monkeypatch.setattr(h, symbol, zeros)
    with pytest.raises(AssertionError):
        controls.self_test(h)


@pytest.mark.parametrize('path', EXTENSIONS[:1])
def test_output_contract_rejects_broadcast_dtype_and_nonfinite(path):
    runner = import_path(path.parent / 'eval_tools/evaluate.py')
    expected = torch.ones(2, 3)
    for wrong in (torch.ones(1, 3), expected.double(), torch.full((2, 3), float('nan')), torch.full((2, 3), float('inf'))):
        with pytest.raises(ValueError):
            runner.output_contract(expected, wrong)
    with pytest.raises(ValueError):
        runner.output_contract((expected,), [expected])
    runner.output_contract({'x': (expected,)}, {'x': (expected.clone(),)})


@pytest.mark.parametrize('path', NATIVE, ids=lambda p: p.parent.name)
def test_native_all_actions_and_errors_emit_envelopes(path, tmp_path, monkeypatch, capsys):
    runner = import_path(path.parent / 'scripts/evaluate.py')
    data = json.loads(path.with_name('workload.json').read_text())
    (tmp_path / 'workload.json').write_text(json.dumps(data))
    for relative in data['candidate_files']:
        p = tmp_path / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('native candidate')
    measured = [{**row, 'execution_time_ms': .1, 'benchmark_method': 'cuda_graph'} for row in data['cases']]
    h = types.SimpleNamespace(TEST_SHAPES=[tuple(s) for s in data['test_shapes']], run_compile=lambda: (True, None),
                              run_correctness=lambda: (True, None), run_performance=lambda: measured)
    controls = types.SimpleNamespace(self_test=lambda h: None, check_additional_paths=lambda h: None)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, 'task_runner', h)
    monkeypatch.setitem(sys.modules, 'reference_checks', controls)
    manifest = None
    for role, action in ACTIONS:
        argv = ['validate-task'] if role == 'task' else [role, action]
        code = runner.main(argv)
        result = parse_command_result(capsys.readouterr().out, role=role, action=action, returncode=code)
        assert result.passed
        if role == 'task': manifest = CaseManifest.from_result(result)
        else: manifest.validate(result)
    h.run_compile = lambda: (False, 'compiler crashed')
    h.run_correctness = lambda: (False, 'reference comparison failed')
    h.run_performance = lambda: measured[:-1]
    for role in ('baseline', 'candidate'):
        for action in ('compile', 'correctness', 'performance'):
            code = runner.main([role, action])
            result = parse_command_result(capsys.readouterr().out, role=role, action=action, returncode=code)
            assert not result.passed
            assert result.failure_kind != 'numerical_mismatch'  # No guessing from arbitrary error text.
            manifest.validate(result)


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'identity', 'nan', 'zero', 'host_time'])
def test_native_performance_rejects_partial_or_invalid_results(corruption):
    runner = import_path(NATIVE[0].parent / 'scripts/evaluate.py')
    cases = json.loads(NATIVE[0].with_name('workload.json').read_text())['cases']
    measured = [{**copy.deepcopy(row), 'execution_time_ms': .1, 'benchmark_method': 'cuda_graph'} for row in cases]
    if corruption == 'missing': measured.pop()
    elif corruption == 'duplicate': measured[1] = measured[0]
    elif corruption == 'identity': measured[0]['params']['B'] = -1
    elif corruption == 'nan': measured[0]['execution_time_ms'] = float('nan')
    elif corruption == 'zero': measured[0]['execution_time_ms'] = 0
    else: measured[0]['benchmark_method'] = 'wall_clock'
    with pytest.raises(RuntimeError): runner.checked_performance(cases, measured)


@pytest.mark.parametrize('action', ['compile', 'correctness', 'performance'])
def test_empty_final_candidate_cannot_fall_back_or_use_stale_report(action, tmp_path, monkeypatch, capsys):
    path = next(p for p in EXTENSIONS if 'torch2hip' in p.parts)
    raw = yaml.safe_load(path.read_text()); args = options(raw)
    runner = import_path(path.parent / 'eval_tools/evaluate.py')
    shutil.copy(path.with_name('workload.json'), tmp_path / 'workload.json')
    target = tmp_path / args.candidate; target.parent.mkdir(parents=True); target.write_text('// TODO only\n')
    (tmp_path / 'build').mkdir()
    (tmp_path / 'build/performance_report.json').write_text('{"status":"ok","test_cases":[]}')
    monkeypatch.setattr(runner, 'ROOT', tmp_path); monkeypatch.chdir(tmp_path)
    code = runner.main(raw['evaluation']['runner'][2:] + ['candidate', action])
    result = parse_command_result(capsys.readouterr().out, role='candidate', action=action, returncode=code)
    assert not result.passed and 'unimplemented' in result.reason


def test_hip_compilation_requires_native_runtime_and_nested_source(tmp_path, monkeypatch):
    runner = import_path(EXTENSIONS[0].parent / 'eval_tools/evaluate.py')
    nested = tmp_path / 'source/nested/kernel.hip'; nested.parent.mkdir(parents=True); nested.write_text('not valid HIP')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(torch.version, 'hip', None)
    with pytest.raises(RuntimeError, match='ROCm'): runner.compile_hip(nested)
    import torch.utils.cpp_extension as extension
    monkeypatch.setattr(torch.version, 'hip', 'test-only')
    seen = []
    def compiler(**kwargs):
        seen.append(kwargs)
        raise RuntimeError('native compiler rejected invalid HIP')
    monkeypatch.setattr(extension, 'load', compiler)
    with pytest.raises(RuntimeError, match='native compiler rejected'): runner.compile_hip(nested)
    assert seen[0]['sources'] == [str(nested)] and seen[0]['with_cuda']


@pytest.mark.parametrize('path', [EXTENSIONS[0], NATIVE[0]])
def test_cli_errors_have_envelope(path, capsys):
    runner = import_path(next(path.parent.glob('*/evaluate.py')))
    assert runner.main(['candidate', 'compile', 'extra']) == 1
    assert capsys.readouterr().out.count('ARENA_EVAL_RESULT=') == 1


# Golden hashes of all original Python/native sources at migration base 5c9f8ef2.
# Includes generators, tolerances, seeds, state reset, timing, wrappers and helpers.
ORIGINAL_SOURCE_DIGESTS = {'hip2hip/gpumode/CrossEntropyLossLabelSmoothing': (10,
                                                    'fa337b2263b20acf4177677b5da9fabfe34ae67c2af0d82380cc109b71c0bc8f'),
 'hip2hip/gpumode/Feedforward': (10, '92d64c214c217b60c875f6c23ecc49a7f4b1beee8dc52e8cb711f0c2db720bb7'),
 'hip2hip/gpumode/FusedLeakyReLU': (10, '1668336a0c97514899f3abc48835239f7b0899320c8e0c865a38a3f345ff771c'),
 'hip2hip/gpumode/GELU': (10, '8e0645b21ac68246e58103339b0cb7ba52c65c3627737cebc22a64bb28106332'),
 'hip2hip/gpumode/GateGRUSelectionLayer': (10, '00b0f44cbd329e9728058ade5005880e3f0af17d3dd58e3d02a4d6d2fccb1b47'),
 'hip2hip/gpumode/InnerProd': (10, 'b8edd8ae0192044d3b808c1775b17dbeecf237f81fda8e3b0b3d1b5f7a8704d0'),
 'hip2hip/gpumode/ItemQueryAttention': (10, 'cf4e83d9950f92f15fc43ee0663b768cd6381cb78c53dffdba2cb0b1ff15f2ef'),
 'hip2hip/gpumode/KDLoss': (10, 'f1a96b62167ecffcaeaeeb551bccbf300593c1d48255e963e53f745f66cc22e0'),
 'hip2hip/gpumode/MLP_model': (10, 'e0357ad5232bfa4ea09d6b6b85de653678093411cce9dd94d8196276f0c72a34'),
 'hip2hip/gpumode/MaskedLanguageModel': (10, 'ffbcf6f0b2bdd1bf784dadb01ea4e615e1c3e621949d6b07b54838db8ae0fbc3'),
 'hip2hip/gpumode/MultiHeadAttention': (10, '6db156f70dbac14de687e911278c0f0617ae938e8cf00da7c980622c814a159a'),
 'hip2hip/gpumode/NormalAttention_dot': (10, '2d3282896325cbda9d542cc52cb93537acdb66262b7e5db77e832a436da29327'),
 'hip2hip/gpumode/NormalAttention_embedded_gaussian': (10,
                                                       '529c4e46ef540a14b7ac22330a71650b83590a08eb089cd82c48dc274bb58c57'),
 'hip2hip/gpumode/PositionWiseFeedForward': (10,
                                             '6e4bdeab4ed2eb9efabd02a7bd548358ba128ec68432560df61414b306be67be'),
 'hip2hip/gpumode/SiLU': (10, '6357de18c8d66b74b8cebdf9814c77b715e6c344d764d3da5d374be48869f0f1'),
 'hip2hip/gpumode/Sigmoid': (10, 'a62e47d90f34f38d3450d9e6a82f4e3dda675f940ae94eb884898821bd33134f'),
 'hip2hip/gpumode/SimpleMatmulModule': (10, 'e9cab26ed75fb47aca9b678018c16817cdb0b1fcf7a960ca79892ede2d3036b2'),
 'hip2hip/gpumode/SoftmaxModule': (10, 'a637beb956a816abdf207fea96faf79d3a273c1f2608747df3f38a67b9ad929f'),
 'hip2hip/gpumode/TanH': (10, '6610c817f00ac6c09137f363ef1cb848c659fb277748940c021de9718d21ff2e'),
 'hip2hip/gpumode/TransformerFFNLayer': (10, '8a43e0e4568a5507ef8c3e9abb7fc84421c46ad576ecf9ea37a43a3808c3183d'),
 'hip2hip/gpumode/Transpose': (10, 'a0ce0efb1abb8346256aa63348dd3c993b01a13305667f3b5087d259a5a99401'),
 'hip2hip/gpumode/layer_normalization': (10, 'abb3b0b5ec06ab5517d46166b220d58a192045fb1e741b8ec77bb06c913b0d9f'),
 'hip2hip/others/assign_score_withk': (7, '35f5d4498a5c78cd9aa89e278616ef1d49a3abc5fa04a77abc9f493bebc06b1a'),
 'hip2hip/others/ball_query': (9, 'bd291b4a553ca4f841a39af5b1618f1547f5c81541dc265422f14b37d10eabe7'),
 'hip2hip/others/furthest_point_sample': (9, '4e2f2042e989a3084c876f0217a07918bd11edfaadee8bdaaf4dba93210a3176'),
 'hip2hip/others/knn': (7, '0a9f3654d031d8bd2ca6454c76c77f61aed51c9dac6982e9d65df9d74de43f57'),
 'hip2hip/others/matrix_multiplication': (12, '4a44245c3c870a834334fb67e8163507fa0c76a26ead9537f91556121c06b0a7'),
 'hip2hip/others/mla_decode': (3, 'fccef49f7a97d1e06579abc065fe6b86d73021cad2e1805d9ef7d0b8abf37770'),
 'hip2hip/others/points_in_boxes': (7, '86292ceb094ed6ee0c77bc903c51c72d6db5158732cd34e18c19565a937cec47'),
 'hip2hip/others/roiaware_pool3d': (7, '5faf7054e8cabf6f5fa1401634d94ac5bed05949edb4fb749e88e285ce88ed1f'),
 'hip2hip/others/roipoint_pool3d': (7, '367df9a8950e03ade116d3c4c13baf19455e5a31207136f1336ef6ed1ee9c820'),
 'hip2hip/others/three_nn': (7, '6c679235f9e0511896fd942337004e55f3d853831ed1fc6d122daea007ff186b'),
 'torch2hip/gpumode/1001_NormalAttention_dot': (9,
                                                '164c44698ef8352e934b9131c22a67757031c3ca1ee914084da2b72d1459d8ce'),
 'torch2hip/gpumode/10024_Feedforward': (9, '68b824549008f00760c6bc788b55a2962e5dacb69090d309eefe349d599e48b8'),
 'torch2hip/gpumode/1003_NormalAttention_embedded_gaussian': (9,
                                                              '959acf417db2a246b506afb5798c203bb496e3d4faacf06916c67e372a61d072'),
 'torch2hip/gpumode/10082_SoftmaxModule': (9, '9fe5305416b9190d484ce8127382f8d5889c714b9efe244c697b77b2b45b2ea1'),
 'torch2hip/gpumode/10099_Gather': (9, '1aabaff70e2b64f5530c3a18cb478d3fe4f64b09e578a7ade6beb5009c3dec4f'),
 'torch2hip/gpumode/10190_FusedLeakyReLU': (9, '483e610a12b98911e490eb39fcaa31eeac04615dfe94e837615379a19c09f66a'),
 'torch2hip/gpumode/102_ItemQueryAttention': (9,
                                              '89d8d59d2270275a1c5193187c5646da280bed9122d4652496fb35efed74f71f'),
 'torch2hip/gpumode/10456_MultiHeadAttention': (9,
                                                'f939e60f74bf2212673f34ba1bfcb328f03f5b9ae37d70b274d3b30b0fa32de9'),
 'torch2hip/gpumode/1067_Transpose': (9, 'a7dcb5c5ddc695a312dfbffe0a65efd87a68ce41eafb320b8e0037b9e8a08c74'),
 'torch2hip/gpumode/11122_PositionEmbedder': (9,
                                              '805a31972b706724dab615dd3955e054fa0dc01c50539641f3a5ea8d9039ed78'),
 'torch2hip/gpumode/11178_TanH': (9, 'f44ab23dfa4975a80bb317a7ce3357ce867bd99b3c61e0d2e0a174e6ba60a34f'),
 'torch2hip/gpumode/11184_Sigmoid': (9, '493cf1cfc9fc1eef59bfe6033e4629013954b25210adf51a1d74fb54f78c89c9'),
 'torch2hip/gpumode/11709_InnerProd': (9, '48ef35bc6ff27e57ecfde80c2bcb44365293d924302a155568731479d8329eb6'),
 'torch2hip/gpumode/11754_layer_normalization': (9,
                                                 '022fbed1df4b4db5248ce2daa55e1fe37674832a8a9ea5c64dd96d489a6fbc34'),
 'torch2hip/gpumode/1178_MLP_model': (9, '0c16388bbbb91cdda1bc68b3b071bbaf7b47f0847721f0a5a5d3fff5dc5122e4'),
 'torch2hip/gpumode/12501_CrossEntropyLossLabelSmoothing': (9,
                                                            'c0fcd99d102ae6704bf0c6f486df5a86bf6c34f5fb8e04631b24c986860432e1'),
 'torch2hip/gpumode/14007_KDLoss': (9, '471098a4ad428b3f19d2b0dd4e0d856c7490aabd87a03f6ebf43c8fd4b0620c6'),
 'torch2hip/gpumode/14044_PositionWiseFeedForward': (9,
                                                     'b871167c54ac55a9d4e16354cc8235446054f952a44332dacd37047d6cc9f86f'),
 'torch2hip/gpumode/14069_TransformerFFNLayer': (9,
                                                 '3c1e25fbf21d024f25de5ed916462ede19ab3ef828597d1a532a3a77dde5f64e'),
 'torch2hip/gpumode/14539_GELU': (9, 'a387ba05afb78f9b734ac5fc8cfdf23ebc018a434cedf0516a59d9317c006820'),
 'torch2hip/gpumode/16636_SiLU': (9, 'cbff466482ef3795f3bc65b65c9d799e70bfd1b3141b2b2681fbbc569f10bd25'),
 'torch2hip/gpumode/3267_SimpleMatmulModule': (9,
                                               '811d16947fb16d78b06b74736641fc5ef2294084196114a8ccb534f53174ffdd'),
 'torch2hip/gpumode/5334_GateGRUSelectionLayer': (9,
                                                  'cfaf238e199417f930b1774e1620fabca021084cc1e6beeb65dc64b9dd186f77'),
 'torch2hip/gpumode/8325_MaskedLanguageModel': (9,
                                                '36b1685a9883554155652c08b35b8933c156e903a5f00b4df77661d37629692c'),
 'torch2hip/kernelbench/level1/l1n1_Square_matrix_multiplication_': (9,
                                                                     'ee27836dd4437c4b0a10982506c3f9084140f2f500a95c0609aa6cf57af6f4bb'),
 'torch2hip/kernelbench/level1/l1n23_Softmax': (9,
                                                '45635dda04b3224729fb0d412f7337b4fde3e16eedee217f4242ed36fa6f0fa4'),
 'torch2hip/kernelbench/level1/l1n26_GELU_': (9,
                                              '9087b20312c6eabd572e8ef459a329a3b2cb092d02a15d6b09bf2a358b2821a5'),
 'torch2hip/kernelbench/level1/l1n2_Standard_matrix_multiplication_': (9,
                                                                       'ef2045d9e47ff9dc3a64d53511cd492250b923dec8f73ab3f187756b32aacfeb'),
 'torch2hip/kernelbench/level1/l1n36_RMSNorm_': (9,
                                                 'b70decc75f02519aa7d0b7c7bf4766e887714f7d673bc3295bebe61ede8cb2b8'),
 'torch2hip/kernelbench/level1/l1n3_Batched_matrix_multiplication': (9,
                                                                     '8a18076c291c2d0ff14262eae199cd74ba058b21d8bf11479000d4001cc71c84'),
 'torch2hip/kernelbench/level1/l1n40_LayerNorm': (9,
                                                  '3a567a19bc9bc066bc17b69fbced59b038ec324bf62f1421b6f53f650497436a'),
 'torch2hip/kernelbench/level1/l1n42_Max_Pooling_2D': (9,
                                                       'f4287b74fcb02c899ec127e5d236b063b78870f926c23fd4942e4c1566b689a1'),
 'torch2hip/kernelbench/level1/l1n47_Sum_reduction_over_a_dimension': (9,
                                                                       'b607837f14e4f2787bb7b9b300806ce8959278216c8ee8c445ecaf67ef82dec2'),
 'torch2hip/kernelbench/level1/l1n4_Matrix_vector_multiplication_': (9,
                                                                     '3190eded584dfa690299ff428799ca409e65d4b1a9c78c6d351dc7030c002ffc'),
 'torch2hip/kernelbench/level1/l1n63_conv_standard_2D__square_input__square_kernel': (9,
                                                                                      'e6c98e3316570beeef0ad7b7103945f4640bbb5ea914624bba8ddbf830f56d93'),
 'torch2hip/kernelbench/level1/l1n82_conv_depthwise_2D_square_input_square_kernel': (9,
                                                                                     '48392441bf1867df8bb724644dc5ae0c5101a284d8c72703e898447c27737fe0'),
 'torch2hip/kernelbench/level1/l1n8_Matmul_with_irregular_shapes_': (9,
                                                                     '1993b88010a19e5c716688af0f0c19e6a9144b4964024428f4f065d87f3ce042'),
 'torch2hip/kernelbench/level1/l1n95_CrossEntropyLoss': (9,
                                                         'fae142404675b129116d669720ad4dcf785847e6d1e2e720886809558142befc'),
 'torch2hip/kernelbench/level1/l1n9_Tall_skinny_matrix_multiplication_': (9,
                                                                          '70a2af8ccc2939b7177fcecd8f3745470e6731de9d40a640a07aa2a90944b03a'),
 'torch2hip/kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide': (9,
                                                                   '88adeb4f710fc73cba15fa62d99f870ef25487284dab80cfad2dfe5d968ff52e'),
 'torch2hip/kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm': (9,
                                                                   'f61e0b705ba5df9a49edce311ba538328a7f81752f67a41733af2ab22ea0625b'),
 'torch2hip/kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd': (9,
                                                                   '35c517ef962b182c6521973454bb82ad220d3429c4b7369547d3d3333ac644c5'),
 'torch2hip/kernelbench/level2/l2n46_Conv2d_Subtract_Tanh_Subtract_AvgPool': (9,
                                                                              '03240158f44a18d98282a838f4d6d9575cd022072e3c018c2c5920dc9b8b19fc'),
 'torch2hip/kernelbench/level2/l2n52_Conv2d_Activation_BatchNorm': (9,
                                                                    '6e1bea2488bf4631aa524dafd3110023b53217b5140f65d8eda2b2842c4fb6a8'),
 'torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale': (9,
                                                                 '126c1cdd1a94f86a8ba8096477206d4602afb6949dea11581c6d2522a2fbae48'),
 'torch2hip/kernelbench/level2/l2n59_Matmul_Swish_Scaling': (9,
                                                             '2d8df73e8377254eec280f97c46fe25ba7c210919b11434cc444f5c01f2af94d'),
 'torch2hip/kernelbench/level2/l2n66_Matmul_Dropout_Softmax': (9,
                                                               '25aadc282eba43e66e56a252403baeade9305438739a2a13d6e292ac86978d57'),
 'torch2hip/kernelbench/level2/l2n6_Conv3d_Softmax_MaxPool_MaxPool': (9,
                                                                      '78e83e002ce167dd3860e17c0102b72a29786921213bf297941b7ed79a6da4b3'),
 'torch2hip/kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling': (9,
                                                                 '406432887bad5b14b1702ed47f526e57be882a9e038310d6c41b296becab6451'),
 'torch2hip/kernelbench/level2/l2n82_Conv2d_Tanh_Scaling_BiasAdd_Max': (9,
                                                                        'b9684f0b118a83307ce9ebb1768f6eb7b833a5e55f1304ad82e9a8f2b042d332'),
 'torch2hip/kernelbench/level2/l2n85_Conv2d_GroupNorm_Scale_MaxPool_Clamp': (9,
                                                                             '5ddee96c3cd2995bbaf737ba75f9c11ecbdd9066d0a422857efe6745c7d0b0bc'),
 'torch2hip/kernelbench/level2/l2n86_Matmul_Divide_GELU': (9,
                                                           'c733f88af3277517fefa056a848f5dac47b528baa51bcef163549e28bdcb60a9'),
 'torch2hip/kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max': (9,
                                                                      '04d6ddbef3b5032c30233c6155b5245d20e1204462068f4a8a9c46067e9cce54'),
 'torch2hip/kernelbench/level2/l2n99_Matmul_GELU_Softmax': (9,
                                                            '96b9d0278d1ed7d4da304fee8b7d190f68297356a9e436b0caf385245c209a2b'),
 'torch2hip/kernelbench/level3/l3n31_VisionAttention': (9,
                                                        '36cf38f1413014d58144d715cec4de8c8fe3befd57f6d955b5ea3e079b74aef2'),
 'torch2hip/kernelbench/level3/l3n43_MinGPTCausalAttention': (9,
                                                              'f80c2bc7e43985482c53e75c17033660b5bb857bba91a7b052d4b767a63fe3a7'),
 'torch2hip/kernelbench/level3/l3n44_MiniGPTBlock': (9,
                                                     '6bc06bb58ed8af3ac63e5af4a2c9f6fd4d7b96f84a7b0b1835762f7a2819bbd0')}


# Repairs justified by finalized real-GPU validator job 139005: GELU must be
# out-of-place and validate timed replay; matrix must validate every output.
# Job 139100 additionally found missing replay checks in FusedLeakyReLU and GRU.
# The original digest remains the gate for all other 84 tasks.
GPU_VALIDATOR_REPAIR_DIGESTS = {'hip2hip/gpumode/GELU': (11, '0b72fe68a7c9bb4ef696ce876f80f0de9ed3a0dd434acd8f499f679e0188975c'), 'torch2hip/gpumode/14539_GELU': (10, '6988f6cace9f3c9a1f8da275789518c8667ba249b5c9431c6b05a57dd67d9e36'), 'hip2hip/others/matrix_multiplication': (13, 'ccb2386a2eedf9b0d5a956bb656e6bae07bf738af5b84e3aa47b9e01c7bfffab'), 'hip2hip/gpumode/FusedLeakyReLU': (11, '2e76a63ae4a0f16eadc664b81c60d5d9779104ce66304d9bac85f5d5c90e70e7'), 'hip2hip/gpumode/GateGRUSelectionLayer': (11, '434697dcc5596fee2141040bbcb1b404b5614f555a58bc8d729b190da194adfd')}


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.parent.name)
def test_original_code_case_seed_tolerance_and_timing_preserved(path):
    files = sorted(p for p in path.parent.rglob('*') if p.suffix in ('.py', '.hip', '.cpp', '.hpp', '.h')
                   and p.name not in ('evaluate.py', 'reference_checks.py') and p.relative_to(path.parent).as_posix() != 'source/kernel.hpp')
    digest = hashlib.sha256()
    for p in files:
        content = p.read_bytes()
        if p.parent == path.parent and p.name in ('main.hip', 'mla_decode.hip'):
            header = (path.parent / 'source/kernel.hpp').read_bytes().split(b'#pragma once\n', 1)[1]
            if p.name == 'mla_decode.hip':
                header += b'\n'  # Original separator stays outside the extracted header.
            assert content.count(b'#include "source/kernel.hpp"\n\n') == 1
            content = content.replace(b'#include "source/kernel.hpp"\n\n', header)
            assert 'source/kernel.hpp' in path.with_name('Makefile').read_text()
        digest.update(str(p.relative_to(path.parent)).encode() + b'\0' + content + b'\0')
    relative = str(path.parent.relative_to(ROOT / 'tasks'))
    assert (len(files), digest.hexdigest()) == GPU_VALIDATOR_REPAIR_DIGESTS.get(relative, ORIGINAL_SOURCE_DIGESTS[relative])


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
@pytest.mark.parametrize('provided_hip', [False, True])
def test_extension_performance_explicit_models_sources_and_case_coverage(tmp_path, monkeypatch, role, provided_hip):
    path = next(p for p in EXTENSIONS if ('hip2hip' in p.parts) == provided_hip)
    raw = yaml.safe_load(path.read_text()); args = options(raw)
    runner = import_path(path.parent / 'eval_tools/evaluate.py')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    (tmp_path / 'build').mkdir()
    rows = [{'test_case_id': 'case_0', 'params': {'inputs': [{'shape': [2], 'dtype': 'torch.float32', 'stride': [1]}]}}]
    loaded_models = []; compiled = []
    perf = types.SimpleNamespace(_compare_results=lambda *a, **kw: True,
        load_modu_obj=lambda path, cls, init: loaded_models.append(cls),
        load_func_obj=lambda path, cls, init: loaded_models.append(cls),
        load_function_from_path=lambda *a: lambda: iter([[torch.ones(2)]]))
    def benchmark(*paths, **kwargs):
        perf.load_modu_obj('module', 'incorrect_name_derived_from_ref_filename', 'init')
        perf.load_func_obj('functional', 'incorrect_name_derived_from_ref_filename', 'init')
        list(perf.load_function_from_path('module', 'get_inputs')())
        if not kwargs.get('baseline_only'):
            if provided_hip: perf.load_hip_kernel('ref', str(tmp_path / 'hip_ref'), 'copied_ref.hip')
            perf.load_hip_kernel('selected', str(tmp_path / 'hip_opt'), 'copied_candidate.hip')
        perf._write_perf_report({'status': 'ok', 'test_cases': [{'case_idx': 0, 'correct': True,
            'ref_time': .3, 'ori_time': .2, 'opt_time': .1,
            'reference_benchmark_method': 'cuda_graph', 'benchmark_method': 'cuda_graph'}]})
    perf.cal_kernel_perf = benchmark
    monkeypatch.setitem(sys.modules, 'cal_kernel_perf', perf)
    monkeypatch.setattr(runner, 'compile_hip', lambda source, **kw: compiled.append((source, kw)))
    measured = runner.performance(args, role, rows)
    assert loaded_models == [args.model_class, args.model_class]
    expected = (.3 if provided_hip else .2) if role == 'baseline' else .1
    assert measured[0]['execution_time_ms'] == expected
    if provided_hip:
        assert compiled[0][0] == tmp_path / args.baseline_hip
        assert compiled[1][0] == tmp_path / (args.baseline_hip if role == 'baseline' else args.candidate)
        assert compiled[0][1]['slot'] != compiled[1][1]['slot']
    elif role == 'candidate': assert compiled[0][0] == tmp_path / args.candidate
    else: assert not compiled


def test_references_against_small_independent_known_answers():
    tasks = ROOT / 'tasks/hip2hip/gpumode'
    for name in ('GELU', 'SoftmaxModule', 'InnerProd'):
        root = tasks / name
        config = yaml.safe_load((root / 'config.yaml').read_text()); args = options(config)
        module = import_path(root / args.module); functional = import_path(root / args.functional)
        if name == 'GELU':
            inputs = [torch.tensor([-1., 0., 1.])]
            expected = torch.tensor([x / 2 * (1 + math.erf(x / math.sqrt(2))) for x in (-1., 0., 1.)])
            init = []
        elif name == 'SoftmaxModule':
            inputs = [torch.tensor([[0., math.log(2), math.log(3)]])]
            expected = torch.tensor([[1/6, 2/6, 3/6]])
            init = [-1]
        else:
            inputs = [torch.tensor([[2., 3.]]), torch.tensor([[[[4., 5.]], [[6., 7.]]]])]
            expected = torch.tensor([[[[26., 31.]]]])
            init = [2]
        for code in (module, functional):
            model = getattr(code, name)(*init).eval()
            actual = model(*inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            assert not torch.allclose(torch.zeros_like(actual), expected, rtol=1e-4, atol=1e-5)


def test_minigpt_uses_actual_block_and_independent_residual_answer():
    root = ROOT / 'tasks/torch2hip/kernelbench/level3/l3n44_MiniGPTBlock'
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    assert args.model_class == 'MiniGPTBlock'
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    for filename in (args.module, args.functional):
        model = getattr(import_path(root / filename), args.model_class)(4, 2, 0., 0., 4).eval()
        with torch.no_grad():
            for parameter in model.parameters(): parameter.zero_()
        torch.testing.assert_close(model(inputs), inputs, rtol=0, atol=0)
        assert not torch.equal(model(inputs), torch.zeros_like(inputs))


def extract_cpp_function(text, prefix):
    start = text.index(prefix); brace = text.index('{', start); depth = 1; end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}'); end += 1
    return text[start:end]


def test_mla_protected_host_reference_known_answers_on_cpu(tmp_path):
    compiler = shutil.which('g++')
    if not compiler: pytest.skip('CPU C++ compiler unavailable; not GPU validation')
    text = (ROOT / 'tasks/hip2hip/others/mla_decode/mla_decode.hip').read_text()
    decoder = extract_cpp_function(text, '__host__ __device__ __forceinline__ float fp8_e4m3fn_to_f32')
    reference = extract_cpp_function(text, 'static void host_reference(')
    diff = extract_cpp_function(text, 'static void max_diff(')
    # Use exact representable known values: float shim tests the original host
    # indexing, softmax and error detection, not device BF16 conversion accuracy.
    preamble = '''
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cassert>
#define __host__
#define __device__
#define __forceinline__ inline
using bf16 = float;
inline float __float2bfloat16(float x) { return x; }
inline float __bfloat162float(float x) { return x; }
constexpr int NHEAD=128, LK=576, LV=512;
'''
    control = '''
int main() {
    std::vector<bf16> q(NHEAD*LK,0), out;
    std::vector<uint8_t> kv(2*LK,0x38); // first token is exactly 1 in e4m3fn
    std::fill(kv.begin()+LK, kv.end(), 0x44); // second token is exactly 3
    std::vector<int32_t> mapping{1,0}, lengths{2};
    host_reference(out,q,kv,mapping,lengths,1,2,1.0f);
    assert(out.size()==NHEAD*LV);
    for(float x:out) assert(x==2.0f); // zero scores => uniform attention
    lengths[0]=1;
    host_reference(out,q,kv,mapping,lengths,1,2,1.0f);
    for(float x:out) assert(x==3.0f); // sequence length and token mapping
    double absolute, relative;
    std::vector<bf16> wrong(out.size(),0);
    max_diff(wrong,out,absolute,relative);
    assert(!(absolute<=5e-2 || relative<=1e-1));
    wrong[0]=NAN;
    max_diff(wrong,out,absolute,relative);
    assert(!std::isfinite(absolute) && !std::isfinite(relative));
}
'''
    source = tmp_path / 'host_reference.cpp'; source.write_text(preamble + decoder + reference + diff + control)
    result = subprocess.run([compiler, '-std=c++17', str(source), '-o', str(tmp_path / 'check')], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(tmp_path / 'check')], check=True, timeout=10)


@pytest.mark.parametrize('outcome', ['correct', 'wrong_values', 'wrong_shape'])
def test_extension_correctness_real_comparison_and_honest_failure_kind(outcome, tmp_path, monkeypatch, capsys):
    path = EXTENSIONS[0]
    raw = yaml.safe_load(path.read_text()); args = options(raw)
    runner = import_path(path.parent / 'eval_tools/evaluate.py')
    monkeypatch.setattr(runner, 'ROOT', tmp_path); monkeypatch.chdir(tmp_path)
    candidate = tmp_path / args.candidate; candidate.parent.mkdir(parents=True); candidate.write_text('HIP source')
    rows = [{'test_case_id': 'case_0', 'params': {'inputs': [{'shape': [2], 'dtype': 'torch.float32', 'stride': [1]}]}}]
    (tmp_path / 'workload.json').write_text(json.dumps({'cases': rows}))
    tensor_to = torch.Tensor.to
    def to_cpu(tensor, *args, **kwargs):
        if args and args[0] == 'cuda': args = ('cpu',) + args[1:]
        return tensor_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', to_cpu)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: None)
    def compare(expected, actual, rtol=1e-4, atol=1e-5):
        return torch.allclose(expected, actual, rtol=rtol, atol=atol)
    checks = types.SimpleNamespace(correctness_check=compare, _compare_results=compare,
        load_function_from_path=lambda *a: lambda: iter([[torch.tensor([1., 2.])]]), _normalize_get_inputs_result=lambda x: x)
    monkeypatch.setitem(sys.modules, 'correctness_check', checks)
    monkeypatch.setattr(runner, 'prepare_models', lambda *a: (lambda x: x + 1, lambda x, fn: fn(x)))
    kernel = (lambda x: x + 1) if outcome == 'correct' else ((lambda x: torch.zeros_like(x)) if outcome == 'wrong_values' else lambda x: x[:1])
    monkeypatch.setattr(runner, 'compile_hip', lambda *a: kernel)
    code = runner.main(raw['evaluation']['runner'][2:] + ['candidate', 'correctness'])
    report = parse_command_result(capsys.readouterr().out, role='candidate', action='correctness', returncode=code)
    assert report.passed == (outcome == 'correct')
    if outcome != 'correct':
        assert report.failure_kind == ('numerical_mismatch' if outcome == 'wrong_values' else 'execution_error')
    assert report.cases[0]['test_case_id'] == 'case_0'


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.parent.name)
def test_task_runner_stays_self_contained(path):
    runner = next(path.parent.glob('*/evaluate.py'))
    for source in (runner, *path.parent.glob('scripts/reference_checks.py')):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in ('src', 'agents')
            elif isinstance(node, ast.Import):
                assert all(alias.name.split('.')[0] not in ('src', 'agents') for alias in node.names)
    assert 'task_type' not in runner.read_text()


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.parent.name)
def test_canonical_helpers_materialize_with_v2_config_only(path, tmp_path):
    from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
    workspace = tmp_path / 'task'
    shutil.copytree(path.parent, workspace, ignore=shutil.ignore_patterns('__pycache__', 'build'))
    created = materialize_perf_helpers_in_workspace(workspace, root=ROOT)
    if path in EXTENSIONS:
        assert (workspace / 'eval_tools/_aka_benchmark.py').is_file()
    elif (workspace / 'kernel_loader.py').exists():
        assert (workspace / 'scripts/_aka_benchmark.py').is_file()
    else:
        assert (workspace / 'scripts/native/hip_graph_benchmark.hpp').is_file()
    assert created and not materialize_perf_helpers_in_workspace(workspace, root=ROOT)


@pytest.mark.parametrize('task', ['hip2hip/gpumode/GELU', 'torch2hip/kernelbench/level3/l3n44_MiniGPTBlock',
                                 'hip2hip/others/assign_score_withk', 'hip2hip/others/matrix_multiplication'])
def test_real_cli_cpu_initial_action_reports_dependencies_honestly(task, tmp_path):
    from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
    original = ROOT / 'tasks' / task; workspace = tmp_path / 'task'
    shutil.copytree(original, workspace, ignore=shutil.ignore_patterns('__pycache__', 'build'))
    materialize_perf_helpers_in_workspace(workspace, root=ROOT)
    command = yaml.safe_load((workspace / 'config.yaml').read_text())['evaluation']['runner']
    completed = subprocess.run([sys.executable, *command[1:], 'validate-task'], cwd=workspace,
                               capture_output=True, text=True, timeout=60)
    report = parse_command_result(completed.stdout, role='task', action='validate-task', returncode=completed.returncode)
    if 'others' in task:
        # Native validate-task performs CPU reference/structural checks only.
        # The subsequent compile/correctness/performance still require ROCm.
        assert report.passed, report.reason
    elif not torch.version.hip or not torch.cuda.is_available():
        assert not report.passed and 'ROCm' in report.reason
    else:
        assert report.passed, report.reason
    assert len(report.cases) == len(json.loads((workspace / 'workload.json').read_text())['cases'])


def test_matrix_full_cpu_reference_and_unsampled_negative_control(tmp_path):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('CPU C++ compiler unavailable; not GPU validation')
    header = ROOT / 'tasks/hip2hip/others/matrix_multiplication/scripts/native/matrix_reference.hpp'
    source = tmp_path / 'matrix_reference.cpp'
    source.write_text('#include "' + str(header) + '"\n' + r'''
#include <cassert>
int main() {
    using namespace matrix_reference;
    auto known = product({1,2,3,4}, {5,6,7,8}, 2,2,2);
    assert((known == std::vector<double>{19,22,43,50}));
    std::vector<float> a(32*16), b(16*32);
    for (int r=0;r<32;++r) for (int k=0;k<16;++k) a[r*16+k]=(r%3)-k/16.f;
    for (int k=0;k<16;++k) for (int c=0;c<32;++c) b[k*32+c]=(c%5)+k/32.f;
    auto expected=product(a,b,32,16,32);
    std::vector<float> actual(expected.begin(),expected.end());
    for (int r=0;r<32;++r) for (int c=0;c<32;++c) {
        double independent=0;
        for (int k=0;k<16;++k) independent+=double(a[r*16+k])*double(b[k*32+c]);
        assert(expected[r*32+c]==independent);
    }
    assert(validate(actual,expected,32).empty());
    actual[7*32+9]+=10; // Outside all sixteen former sampled coordinates.
    assert(!validate(actual,expected,32).empty());
    actual.assign(expected.begin(),expected.end()); actual[7*32+9]=NAN;
    assert(!validate(actual,expected,32).empty());
    assert(!validate(std::vector<float>(actual.size(),0),expected,32).empty());
}
''')
    result = subprocess.run([compiler, '-std=c++17', str(source), '-o', str(tmp_path / 'check')], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    subprocess.run([str(tmp_path / 'check')], check=True, timeout=10)


def test_matrix_additional_checks_cover_all_original_shapes(monkeypatch):
    root = ROOT / 'tasks/hip2hip/others/matrix_multiplication'
    helper = import_path(root / 'scripts/reference_checks.py')
    shapes = json.loads((root / 'workload.json').read_text())['test_shapes']
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode=0, stdout='Full reference validation passed', stderr='')
    monkeypatch.setattr(helper.subprocess, 'run', run)
    harness = types.SimpleNamespace(TEST_SHAPES=shapes, BENCH_BINARY='nested/build/benchmark')
    helper.check_additional_paths(harness)
    assert len(calls) == 5
    for (argv, kwargs), shape in zip(calls, shapes):
        assert argv == ['nested/build/benchmark', '--A_rows', str(shape[0]), '--A_cols', str(shape[1]), '--B_cols', str(shape[2]), '--check-only', '1']
        assert kwargs['timeout'] == 300
    monkeypatch.setattr(helper.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(returncode=1, stdout='', stderr='wrong C'))
    with pytest.raises(ValueError, match='wrong C'):
        helper.check_additional_paths(harness)


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/GELU', 'torch2hip/gpumode/14539_GELU',
                                         'hip2hip/gpumode/FusedLeakyReLU', 'hip2hip/gpumode/GateGRUSelectionLayer'])
@pytest.mark.parametrize('behavior', ['correct', 'wrong_replay', 'input_mutation', 'input_alias'])
def test_gelu_exact_timed_replay_and_input_contract(relative, behavior, monkeypatch):
    root = ROOT / 'tasks' / relative
    helper = import_path(root / 'eval_tools/replay_validation.py')
    runner = import_path(root / 'eval_tools/evaluate.py')
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    inputs = [torch.tensor([-1., .5, 2.])]
    observed = []
    def benchmark(invoke, **kwargs):
        observed.append(kwargs)
        output = invoke()
        def replay():
            assert torch.isnan(output).all()  # Same timed buffer was poisoned.
            output.copy_(torch.zeros_like(output) if behavior == 'wrong_replay' else torch.nn.functional.gelu(inputs[0]))
            return output
        kwargs['timed_run']._bind(replay, output)
        return .25, {'benchmark_method': 'cuda_graph'}
    def cal_kernel_perf(rtol=1e-4, atol=1e-5): pass
    perf = types.SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
        benchmark_cuda_graph_or_events=benchmark, _compare_results=torch.allclose)
    helper.install(perf, runner.output_contract)
    def module(x, fn=torch.nn.functional.gelu): return fn(x)
    def candidate(x):
        output = torch.nn.functional.gelu(x)
        if behavior == 'input_mutation': x.add_(1)
        if behavior == 'input_alias': x.copy_(output); return x
        return output
    if behavior == 'correct':
        elapsed, metadata = perf.cal_hip_latency(module, inputs, candidate)
        assert elapsed == .25 and metadata['replay_validation_valid'] is True
    else:
        with pytest.raises((ValueError, AssertionError)):
            perf.cal_hip_latency(module, inputs, candidate)
    assert observed[0]['warmup'] == 10 and observed[0]['repetition'] == 100


@pytest.mark.parametrize('name', ['FusedLeakyReLU', 'GateGRUSelectionLayer'])
def test_added_replay_tasks_reference_known_answer_and_readonly_inputs(name):
    root = ROOT / 'tasks/hip2hip/gpumode' / name
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    for filename in [args.module, args.functional]:
        cls = getattr(import_path(root / filename), name)
        if name == 'FusedLeakyReLU':
            model = cls(channel=2, negative_slope=.2, scale=2.).eval()
            with torch.no_grad(): model.bias.copy_(torch.tensor([1., -2.]))
            inputs = [torch.tensor([[[[-3., 2.]], [[1., 4.]]]])]
            expected = torch.tensor([[[[-.8, 6.]], [[-.4, 4.]]]])
        else:
            model = cls(dim_model=2, dim_ff=4, prob_dropout=.5).eval()
            with torch.no_grad():
                for p in model.parameters(): p.zero_()
            inputs = [torch.tensor([[[[2., 4.]]]]), torch.tensor([[[[9., -3.]]]])]
            # Zero reset/update/proposal linear layers => update=.5, proposal=0.
            expected = inputs[0] / 2
        before_inputs = copy.deepcopy(inputs)
        before_state = copy.deepcopy(model.state_dict())
        actual = model(*inputs)
        torch.testing.assert_close(actual, expected)
        assert not torch.allclose(actual, torch.zeros_like(actual))
        for before, after in zip(before_inputs, inputs):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            assert actual.untyped_storage().data_ptr() != after.untyped_storage().data_ptr()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(before_state[key], value, rtol=0, atol=0)
