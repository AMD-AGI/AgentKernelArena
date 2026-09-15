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
ADDITIONAL_GPUMODE_REPLAY = ['hip2hip/gpumode/InnerProd', 'hip2hip/gpumode/KDLoss', 'hip2hip/gpumode/MLP_model', 'hip2hip/gpumode/MultiHeadAttention', 'hip2hip/gpumode/NormalAttention_embedded_gaussian', 'hip2hip/gpumode/PositionWiseFeedForward', 'hip2hip/gpumode/SimpleMatmulModule', 'hip2hip/gpumode/SoftmaxModule', 'hip2hip/gpumode/TransformerFFNLayer', 'hip2hip/gpumode/Transpose', 'hip2hip/gpumode/layer_normalization', 'torch2hip/gpumode/1003_NormalAttention_embedded_gaussian', 'torch2hip/gpumode/10082_SoftmaxModule', 'torch2hip/gpumode/10099_Gather', 'torch2hip/gpumode/10456_MultiHeadAttention', 'torch2hip/gpumode/1067_Transpose', 'torch2hip/gpumode/11122_PositionEmbedder', 'torch2hip/gpumode/11709_InnerProd', 'torch2hip/gpumode/11754_layer_normalization', 'torch2hip/gpumode/1178_MLP_model', 'torch2hip/gpumode/14007_KDLoss', 'torch2hip/gpumode/14044_PositionWiseFeedForward', 'torch2hip/gpumode/14069_TransformerFFNLayer', 'torch2hip/gpumode/3267_SimpleMatmulModule']


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
        expected_params = {'inputs': value, 'model_init_seed': 0, 'correctness_seed': 1337 + i}
        if path.parent.name in {'FusedLeakyReLU', '10190_FusedLeakyReLU'}:
            expected_params['operator'] = {
                'bias': {'pattern': 'alternating_channel', 'offset': .125, 'step': 1/128},
                'negative_slope': [.1, .2, .35, .5, .75][i],
                'scale': [.5, math.sqrt(2), 1.25, 2., 3.][i]}
        if path.parent.name in {'InnerProd', '11709_InnerProd'}:
            expected_params['operator'] = {
                'scale': {'pattern': 'channel_ramp', 'offset': .5, 'span': .5},
                'bias': [.125, -.25, .5, -.75, 1.25][i]}
        if path.parent.name in {'CrossEntropyLossLabelSmoothing', '12501_CrossEntropyLossLabelSmoothing'}:
            expected_params['operator'] = {'class_axis': -1, 'smooth_eps': [0., .1, .2, .4, .6][i],
                                           'smooth_dist': 'normalized_class_ramp'}
        if path.parent.name in {'layer_normalization', '11754_layer_normalization', 'l1n40_LayerNorm'}:
            expected_params['operator'] = {'gamma': '0.5_plus_flat_index_fraction',
                'beta': 'alternating_flat_ramp', 'beta_amplitude': (i+1)/16}
        assert row == {'test_case_id': f'case_{i}', 'params': expected_params}


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
            assignments = [n.value for n in ast.walk(perf) if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                                   and t.slice.value == 'params' for t in n.targets)]
            if assignments:
                assert len(assignments) == 1
                params = eval(compile(ast.Expression(assignments[0]), '<protected case params>', 'eval'), {'__builtins__': {}}, env)
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
                              run_correctness=lambda: (True, None), run_performance=lambda: measured,
                              HIP_GRAPH_ENABLED=True, HIP_GRAPH_FALLBACK_REASON=None)
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
# Job 139100 additionally found missing replay checks in FusedLeakyReLU, GRU and item attention.
# MaskedLanguageModel also needed its transposed weight launch axes corrected.
# Original digests remain the gate for tasks without an explicit reviewed repair.
GPU_VALIDATOR_REPAIR_DIGESTS = {'hip2hip/gpumode/GELU': (11, '4f9540b61f00d8e22d65e0907db7387c96d6a90c149ed0c46dcd57389574cf95'), 'torch2hip/gpumode/14539_GELU': (10, '6c990124da84450c8da0e6135d375be2afa8b712f1d20f0b8fd4628a21b32c6f'), 'hip2hip/others/matrix_multiplication': (13, 'ccb2386a2eedf9b0d5a956bb656e6bae07bf738af5b84e3aa47b9e01c7bfffab'), 'hip2hip/gpumode/FusedLeakyReLU': (12, '774ebc2b63b75e648ff4e40dd9f5222dc334318bb8ec6ac019586542645ca639'), 'hip2hip/gpumode/GateGRUSelectionLayer': (11, 'caf22f456cba41142677bf9455f4f7330931c65ad645b14e3830c3a0500999ac'), 'hip2hip/gpumode/ItemQueryAttention': (11, '78e4567f5a065446466e0c41d6e2135ffaa7b69ccb2ed9056ef27f93232011dd'), 'hip2hip/gpumode/MaskedLanguageModel': (11, 'bb8969ec7690e5dbbc5675e4ac97db7443714d21d8c063a1574a45ea81c468ac'), 'hip2hip/gpumode/SiLU': (11, '43c5f0a8c2cb9eaf4c0a52d75f952e27304a723c9c8f7cc4ac56bb90d09fa64c'), 'hip2hip/gpumode/Sigmoid': (11, '4e7b4b92dbdee90c258f442c994038e6bba420947f510550adf601f061000d71'), 'torch2hip/gpumode/10190_FusedLeakyReLU': (11, 'c6fbff431520d64b059cc1495a3e98b7462e771e0afec52a12c9cbfb637aa364'), 'torch2hip/gpumode/5334_GateGRUSelectionLayer': (10, '3d631fc5dce6c0574920709fdffd5c3eba28405ae3d83bd935e6e916abc14412'), 'torch2hip/gpumode/102_ItemQueryAttention': (10, 'cd9d6537d805e3d175b038e539ae36a2400a1c2cb08e4858faf663ca248528c4'), 'torch2hip/gpumode/16636_SiLU': (10, '5330ab36cee1ca0704a0d45db8bb0a47bfc8b53d2bba521bdcf3f21c3da6ba94'), 'torch2hip/gpumode/11184_Sigmoid': (10, '783a2770eb6a1c464563aabe03bc7b45bf893a21a82f9f54eb565c161748e7ea'), 'hip2hip/gpumode/TanH': (11, '007897566632c1584b5913a1592362aeaca4a3038422f93642e172777848fa43'), 'torch2hip/gpumode/11178_TanH': (10, '1d61bea5024422ea638fc84d953b20128d647fbdf88e5fe418f2686a2d9c4503'), 'hip2hip/others/assign_score_withk': (8, '42e3493889a8c098bca37112f5c93c756ca0f2710ffdc29c06c79d78c0e1152f'), 'hip2hip/others/ball_query': (10, 'cd11b843e3fb5be4644e86ad082dac15f58be645e7e98417da5e2feacd97b14e'), 'hip2hip/others/furthest_point_sample': (10, 'c76c7a8cb25cec5c79852ca7d174a19fd788517642c3f819beac596f899e8544'), 'hip2hip/others/knn': (8, '3ecfb50ee4713f8b0ecb65dadd2c76fc889d3b119449582e8ff95c4cc45de558'), 'hip2hip/others/points_in_boxes': (8, '076e6ed46ccfd82e5ee0360f8ff63ddfcc3f28bc430a42a99ade4e04c47dfa21'), 'hip2hip/others/roiaware_pool3d': (8, 'da874d09478d56aaf6b7af62df2053d306cd774e7e27078a5b46c85a86b387cd'), 'hip2hip/others/roipoint_pool3d': (8, '1777ecf526530d16b3072fdba19fbb612406af9c4f40cbd4105f9ce7759790f1'), 'hip2hip/others/three_nn': (8, '40e7b452866f727f83c37e3e9306d4c9a4fddbfb89ba4cf502f89baccbb052d4'), 'torch2hip/gpumode/8325_MaskedLanguageModel': (10, '2fb3f69dcecad77a6e350430090830dc5b3af6d7a2ff03d8b8d0cb4e29678e2b'), 'torch2hip/kernelbench/level1/l1n1_Square_matrix_multiplication_': (10, '20b6ac9b8c3d37d45587e72a62c91a794ad00538217705d12c2bac1a57e81c5d'), 'torch2hip/kernelbench/level1/l1n23_Softmax': (10, '8ef11e6fa898e93af88a5f5f665085a3a0864ee2545604c79cb5b3128d9ed64e'), 'torch2hip/kernelbench/level1/l1n26_GELU_': (10, '609c40fb91167b4156a6c8261c2f971c3cf14f50f639939999ca2251bd2b8002'), 'torch2hip/kernelbench/level1/l1n2_Standard_matrix_multiplication_': (10, '3947c3d599ec40cc77ccaefee8d4b55c9bebb5186b8dd1a97e55abb494b733d5'), 'torch2hip/kernelbench/level1/l1n36_RMSNorm_': (10, '26983a696c85e3c6f9bb0df51e131e5c1fd9e646a147e158ea210a58fdd7ac7e'), 'torch2hip/kernelbench/level1/l1n3_Batched_matrix_multiplication': (10, '499811d76f7fccd1e4cd51ed172109aadbc1a72f8795db87eef66b2755ecbb7d'), 'torch2hip/kernelbench/level1/l1n40_LayerNorm': (11, '0afcd86f516c808c9a4a9641ba22abfeee74130dccff1987fd38213130d1d7af'), 'torch2hip/kernelbench/level1/l1n42_Max_Pooling_2D': (10, '6357ce29e8c56f3182c5c40f2543ce40affffbbbfe74911ed9ff9d0e1f5d7375'), 'torch2hip/kernelbench/level1/l1n47_Sum_reduction_over_a_dimension': (10, '609c722ceb99129492ec248d2332020e5072bc76a3e8f9bfc0d303253b536e76'), 'torch2hip/kernelbench/level1/l1n4_Matrix_vector_multiplication_': (10, '9a2412cb0a319bc4d3a2bc1d5eeeee7a946af1f85f37acb3dbd9f01cccee59e1'), 'torch2hip/kernelbench/level1/l1n63_conv_standard_2D__square_input__square_kernel': (10, 'aed202bc6ee1649644598e4edeb71929ac546f1a7f7f279c36cdffc597325409'), 'torch2hip/kernelbench/level1/l1n82_conv_depthwise_2D_square_input_square_kernel': (10, '9c65cb3c33f81ce01e9a7771978b1da6e1b90317547be87983df41f6ffe9d869'), 'torch2hip/kernelbench/level1/l1n8_Matmul_with_irregular_shapes_': (10, '06563b8354b58fc1a19b9530958427b58905f732f37d0c82bb8d32fbe47dc95e'), 'torch2hip/kernelbench/level1/l1n95_CrossEntropyLoss': (10, '4760199ce48b862e98d6cc1400aa17bfaddfdc1526f77c538111488325b900d4'), 'torch2hip/kernelbench/level1/l1n9_Tall_skinny_matrix_multiplication_': (10, '4181e90055b4a574d36ddbfd712e06b44f682be7b701167aab0bc6bfb15f674f'), 'hip2hip/gpumode/CrossEntropyLossLabelSmoothing': (12, 'c28f313b850a8e34983473564d00ebdbc2113c6a2db85f1541680567fa2884fa'), 'torch2hip/gpumode/12501_CrossEntropyLossLabelSmoothing': (11, '285a57b17629499fe15b1e7a3242afe5ae7ce636b2a06029ddf6bff66f1a2acf'), 'hip2hip/gpumode/NormalAttention_dot': (11, 'e0d6aaf0d805eb8869274d3f737b0cfdaf47b998036ca61ae3175dc994aa63ac'), 'torch2hip/gpumode/1001_NormalAttention_dot': (10, '044854eccc3c0db3955fcf640c2da0fc01a5d154572b2d6f2bf5f67635cf6761'), 'hip2hip/gpumode/Feedforward': (11, '40e1e191db0544dad748a983378beff6daf4cbee497fa1f965586b888ab55dc0'), 'torch2hip/gpumode/10024_Feedforward': (10, '79b2020bdc61cb4936085128c3079882e07b8aed7046c51ef6198598084a8c0d'), 'hip2hip/gpumode/InnerProd': (12, 'be30da86654f285e6ea2de1e64602b5fcf6ca511eef2d6560e388938a40144b9'), 'hip2hip/gpumode/KDLoss': (11, '519fb7118865ba9e00cdc6b9059d7add4f7d2f0a9c1e56611d1439ea6a64b6ec'), 'hip2hip/gpumode/MLP_model': (11, 'd47715de80e5bd5e2494617707e49c325cc87bd09fede7b8bdb817a8d5e7c7a9'), 'hip2hip/gpumode/MultiHeadAttention': (11, '411d29cabd95c3c6af6aa1a5c4d18644c9b7a989749dc6e96b788e248349babd'), 'hip2hip/gpumode/NormalAttention_embedded_gaussian': (11, 'af24da05b8ea606df87fe9e7e9e26186afe7d098f98de5dce873e89c541ce643'), 'hip2hip/gpumode/PositionWiseFeedForward': (11, 'df0164ed765748cc088b24f4d56cf0999f2e492a72c1b1b0e675c3f00a8d271d'), 'hip2hip/gpumode/SimpleMatmulModule': (11, 'd4dc537d8858f7aa1c14b9f4e4f24b8fb092341229e335eb3c958c613b0418db'), 'hip2hip/gpumode/SoftmaxModule': (11, '5e1fedd4a2ac5191e543bce4e5c48eb8cfe0cf1dbaa98b46f14aaafc0e8c2817'), 'hip2hip/gpumode/TransformerFFNLayer': (11, '452dd026048faf7b0c52f39222834a943b0138936ea92efb44266edcefa1c267'), 'hip2hip/gpumode/Transpose': (11, 'e37c704ef3121fe4a37a2105084a51bb9c9d37178cd2ae5722498a052d31e168'), 'hip2hip/gpumode/layer_normalization': (12, '3c0c71c1acfcc5e3fe8ed0dbd662d55c55837dcc673a6fe6b1e8d8953b688f1b'), 'torch2hip/gpumode/1003_NormalAttention_embedded_gaussian': (10, '4e6b2b5a43cf3e0220be48f3aba290b6860ed62b91c6df024020228f30252fb1'), 'torch2hip/gpumode/10082_SoftmaxModule': (10, '07dc4bbdd87b3b72740a8f60c01717e93247d3f0cff7a5a4af4f10b7d720fc36'), 'torch2hip/gpumode/10099_Gather': (10, 'b812f72c4cc17a19c11efb380f08f3b807da768fc4e2bad5afa984804c44f6c0'), 'torch2hip/gpumode/10456_MultiHeadAttention': (10, '5716dd434e3baf6ebcffae5275b3042a726f137acb0053c5785a95d235514c07'), 'torch2hip/gpumode/1067_Transpose': (10, 'f3a6ac32eebc9f7788d3f99cd587418017f41fd879ce28daf6e34629544a8071'), 'torch2hip/gpumode/11122_PositionEmbedder': (10, 'e06e7c1eee5408c79a64fcac82801c757affd21d491c69bf424645e9a4e658a7'), 'torch2hip/gpumode/11709_InnerProd': (11, '399c11b6c10e8052cbd1266a3b8e1165a71b205eac78ec02ac61b9b30506f38e'), 'torch2hip/gpumode/11754_layer_normalization': (11, '382f0710eb619281f08554cbb55a7ac2217b7d95186e2f74284eab0e94488652'), 'torch2hip/gpumode/1178_MLP_model': (10, '7856ee9beaf792f0aec8d84aeaf54579316de9b3dea782daba32f46d02a0c139'), 'torch2hip/gpumode/14007_KDLoss': (10, 'cf7f9a76d572e0b164b54df09f55225ceec70261d25e2ae6d2052e485e094b94'), 'torch2hip/gpumode/14044_PositionWiseFeedForward': (10, 'cb0f9baa199a4b87f7e67905769be0693b68731ada888696cf9ef0c1ca0f9106'), 'torch2hip/gpumode/14069_TransformerFFNLayer': (10, '8b134fecdba1fef0e1c272721d0f3eec77cb3a3d240122a0b730919d6356cf12'), 'torch2hip/gpumode/3267_SimpleMatmulModule': (10, 'a912fd2c41021a55c029d35a410a01dc00a46fc9bb34d48bbee0a131084dd6fe'), 'torch2hip/kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide': (10, '5e683ac6cacf817d6df23145a49ef858580b228d774e12363e11e91216ce62db'), 'torch2hip/kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm': (10, 'f2f4617a554667cdb94ac1f0086414161a63b9de87b00769a00beb2e290d18d1'), 'torch2hip/kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd': (10, '968de902011ebf3c7d4426e1a59aff3df8d16d60a6d643a419c71c49f583b29c'), 'torch2hip/kernelbench/level2/l2n46_Conv2d_Subtract_Tanh_Subtract_AvgPool': (10, 'abcaa40577247626b1b6b96bc1190dbcfbe68202eafc7a0eb526a39c86386892'), 'torch2hip/kernelbench/level2/l2n52_Conv2d_Activation_BatchNorm': (10, '0ebd91aa7952491a2b79c772bf19fb58b49d8c761c5b414f01247187339ee67c'), 'torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale': (10, '23db7153133fd97e5ee40ded27e147ee5c4464d5fccb65867c8d39b22dd050ea'), 'torch2hip/kernelbench/level2/l2n59_Matmul_Swish_Scaling': (10, '316416aeae848fc04b2f925aa1d11026387abbfac16613fd1f9616a20e3795c1'), 'torch2hip/kernelbench/level2/l2n66_Matmul_Dropout_Softmax': (10, '5abfcb99c3c248270aa0e35a8f455c04995ad8d5a04829230310ce3cdf268cc9'), 'torch2hip/kernelbench/level2/l2n6_Conv3d_Softmax_MaxPool_MaxPool': (10, 'cbbcc6e39cc4874fd4eff05c65b024cfa72af6090d6b7c45a985fc19f9428a6f'), 'torch2hip/kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling': (10, 'd0392538be766d65c8cb65d2c8d6241a4c9ec3c27630981f6959b23852de7d8b'), 'torch2hip/kernelbench/level2/l2n82_Conv2d_Tanh_Scaling_BiasAdd_Max': (10, '1965dcbf0a873b2dac96f0f56237dff4c5ab0aafe50ed2f236ef922bd4482f6c'), 'torch2hip/kernelbench/level2/l2n85_Conv2d_GroupNorm_Scale_MaxPool_Clamp': (10, '30018c04c529fd9f2d71ef248d304050d7ea8f13442673fdae85c76db5fcd3ad'), 'torch2hip/kernelbench/level2/l2n86_Matmul_Divide_GELU': (10, '9b0be02c49ad151540f9809841b5cecb1103f7fcd0b425e7e06529d0579ff13d'), 'torch2hip/kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max': (10, 'ab8d0508fb2d6dd8a570ac802b217e31b178738f75382a5a6f69718f4a764458'), 'torch2hip/kernelbench/level2/l2n99_Matmul_GELU_Softmax': (10, '8fabb1d609c2327c90ac16be5ed1d124661f6a1da0eb29fcf4bfa1598a581316'), 'torch2hip/kernelbench/level3/l3n31_VisionAttention': (10, 'd0af0fbc6b95cbf5602cc2a30474c663095faf401390e4d6fcd7318d07bbc40b'), 'torch2hip/kernelbench/level3/l3n43_MinGPTCausalAttention': (10, '14353949b386acbb41c7e4f02277a5b33909e1752020d127dba87cd25e7200e9'), 'torch2hip/kernelbench/level3/l3n44_MiniGPTBlock': (10, 'd1edd1bb46b907d2fb56d68a9b1237d2bbfa0ef40d415c17a98ce5ade130d5ad'), 'hip2hip/others/mla_decode': (5, 'c085a1d2388f5b1e0b231ad6bb7810f2657745b564fddffb3045ea97e8652e39')}


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
    # This fixture isolates role/source routing; dedicated tests exercise the
    # real replay adapter against measured-output and re-invocation controls.
    monkeypatch.setitem(sys.modules, 'replay_validation', types.SimpleNamespace(install=lambda *args: None))
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
#include <cstring>
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
    path = next(p for p in EXTENSIONS if p.parent.name == 'KDLoss')
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
                                         'hip2hip/gpumode/FusedLeakyReLU', 'hip2hip/gpumode/GateGRUSelectionLayer',
                                         'hip2hip/gpumode/ItemQueryAttention',
                                         'hip2hip/gpumode/SiLU',
                                         'hip2hip/gpumode/Sigmoid',
                                         'torch2hip/gpumode/10190_FusedLeakyReLU',
                                         'torch2hip/gpumode/5334_GateGRUSelectionLayer',
                                         'torch2hip/gpumode/102_ItemQueryAttention',
                                         'torch2hip/gpumode/16636_SiLU',
                                         'torch2hip/gpumode/11184_Sigmoid',
                                         'hip2hip/gpumode/TanH', 'torch2hip/gpumode/11178_TanH',
                                         'hip2hip/gpumode/MaskedLanguageModel', 'torch2hip/gpumode/8325_MaskedLanguageModel'] +
                         [p.parent.relative_to(ROOT / 'tasks').as_posix() for p in CONFIGS if 'level1' in p.parts] + ADDITIONAL_GPUMODE_REPLAY + ['torch2hip/kernelbench/level2/l2n17_Conv2d_InstanceNorm_Divide', 'torch2hip/kernelbench/level2/l2n37_Matmul_Swish_Sum_GroupNorm', 'torch2hip/kernelbench/level2/l2n40_Matmul_Scaling_ResidualAdd', 'torch2hip/kernelbench/level2/l2n46_Conv2d_Subtract_Tanh_Subtract_AvgPool', 'torch2hip/kernelbench/level2/l2n52_Conv2d_Activation_BatchNorm', 'torch2hip/kernelbench/level2/l2n55_Matmul_MaxPool_Sum_Scale', 'torch2hip/kernelbench/level2/l2n59_Matmul_Swish_Scaling', 'torch2hip/kernelbench/level2/l2n66_Matmul_Dropout_Softmax', 'torch2hip/kernelbench/level2/l2n6_Conv3d_Softmax_MaxPool_MaxPool', 'torch2hip/kernelbench/level2/l2n73_Conv2d_BatchNorm_Scaling', 'torch2hip/kernelbench/level2/l2n82_Conv2d_Tanh_Scaling_BiasAdd_Max', 'torch2hip/kernelbench/level2/l2n85_Conv2d_GroupNorm_Scale_MaxPool_Clamp', 'torch2hip/kernelbench/level2/l2n86_Matmul_Divide_GELU', 'torch2hip/kernelbench/level2/l2n98_Matmul_AvgPool_GELU_Scale_Max', 'torch2hip/kernelbench/level2/l2n99_Matmul_GELU_Softmax', 'torch2hip/kernelbench/level3/l3n31_VisionAttention', 'torch2hip/kernelbench/level3/l3n43_MinGPTCausalAttention', 'torch2hip/kernelbench/level3/l3n44_MiniGPTBlock'])
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
        return .25, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}
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


@pytest.mark.parametrize('family', ['hip2hip', 'torch2hip'])
@pytest.mark.parametrize('name', ['FusedLeakyReLU', 'GateGRUSelectionLayer'])
def test_added_replay_tasks_reference_known_answer_and_readonly_inputs(name, family):
    directory = name if family == 'hip2hip' else {'FusedLeakyReLU': '10190_FusedLeakyReLU', 'GateGRUSelectionLayer': '5334_GateGRUSelectionLayer'}[name]
    root = ROOT / 'tasks' / family / 'gpumode' / directory
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


@pytest.mark.parametrize('family', ['hip2hip', 'torch2hip'])
def test_item_attention_zero_projection_known_answer_and_readonly_inputs(family):
    directory = 'ItemQueryAttention' if family == 'hip2hip' else '102_ItemQueryAttention'
    root = ROOT / 'tasks' / family / 'gpumode' / directory
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    queries = torch.arange(12, dtype=torch.float32).reshape(2,3,2)
    support = torch.arange(18, dtype=torch.float32).reshape(3,3,2)
    expected = support.mean(dim=1)[None, :, None, :].expand(2,3,3,2)
    for filename in (args.module, args.functional):
        model = import_path(root / filename).ItemQueryAttention(2,2).eval()
        with torch.no_grad():
            for parameter in model.parameters(): parameter.zero_()
        before = [queries.clone(), support.clone()]
        actual = model(queries, support)
        # Zero projections produce uniform attention over each support sequence.
        torch.testing.assert_close(actual, expected)
        assert not torch.allclose(actual, torch.zeros_like(actual))
        for old, value in zip(before, [queries, support]):
            torch.testing.assert_close(old,value,rtol=0,atol=0)
            assert actual.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()


@pytest.mark.parametrize('suffix', ['', '_ref'])
def test_masked_language_weight_transpose_covers_nonsquare_manifest_weights(tmp_path, suffix):
    compiler = shutil.which('g++')
    if not compiler: pytest.skip('CPU C++ compiler unavailable; not GPU validation')
    root = ROOT / 'tasks/hip2hip/gpumode/MaskedLanguageModel'
    text = (root / f'hip/hip_8325_MaskedLanguageModel{suffix}.hip').read_text()
    kernel = 'template<typename scalar_t>\n' + extract_cpp_function(text, '__global__ void transpose2d_kernel')
    grid = next(line.strip() for line in text.splitlines() if 'dim3 gridT(' in line)
    preamble = '''
#include <vector>
#include <cmath>
#include <cassert>
#define __global__
struct dim3 { int x,y; dim3(int a=0,int b=0):x(a),y(b){} };
dim3 blockIdx,blockDim,threadIdx;
'''
    body = '''
int main() {
    const int V=4096,H=512,TILE_X=32,TILE_Y=8;
    GRID
    blockDim=dim3(TILE_X,TILE_Y);
    std::vector<float> input(V*H), output(V*H,NAN);
    for(int i=0;i<V*H;++i) input[i]=(i%251)/8.f;
    auto run=[&](dim3 grid) {
        for(blockIdx.y=0;blockIdx.y<grid.y;++blockIdx.y)
            for(blockIdx.x=0;blockIdx.x<grid.x;++blockIdx.x)
                for(threadIdx.y=0;threadIdx.y<TILE_Y;++threadIdx.y)
                    for(threadIdx.x=0;threadIdx.x<TILE_X;++threadIdx.x)
                        transpose2d_kernel(input.data(),output.data(),V,H);
    };
    run(gridT);
    for(int row=0;row<V;++row) for(int col=0;col<H;++col)
        assert(output[col*V+row]==input[row*H+col]);
    std::fill(output.begin(),output.end(),NAN);
    run(dim3((V+TILE_X-1)/TILE_X,(H+TILE_Y-1)/TILE_Y));
    // The original transposed launch writes only H out of V rows.
    assert(std::isnan(output[V-1]));
}
'''.replace('GRID', grid)
    source=tmp_path/'transpose.cpp'; source.write_text(preamble+kernel+body)
    result=subprocess.run([compiler,'-std=c++17','-O2',str(source),'-o',str(tmp_path/'check')],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    subprocess.run([str(tmp_path/'check')],check=True,timeout=10)


@pytest.mark.parametrize('family', ['hip2hip', 'torch2hip'])
@pytest.mark.parametrize('name', ['SiLU', 'Sigmoid', 'TanH'])
def test_priority_activation_references_have_independent_known_answers(name, family):
    directory = name if family == 'hip2hip' else {'SiLU': '16636_SiLU', 'Sigmoid': '11184_Sigmoid', 'TanH': '11178_TanH'}[name]
    root = ROOT / 'tasks' / family / 'gpumode' / directory
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    inputs = torch.tensor([-1.,0.,1.])
    for filename in (args.module,args.functional):
        cls=getattr(import_path(root / filename),name)
        if name=='SiLU':
            model=cls().eval(); expected=torch.tensor([x/(1+math.exp(-x)) for x in (-1.,0.,1.)])
        elif name=='Sigmoid':
            model=cls(a=2,max=3).eval(); expected=torch.tensor([3/(1+math.exp(-2*x)) for x in (-1.,0.,1.)])
        else:
            model=cls(a=2,max=3).eval(); expected=torch.tensor([3*math.tanh(2*x) for x in (-1.,0.,1.)])
        before=inputs.clone(); actual=model(inputs)
        torch.testing.assert_close(actual,expected)
        assert not torch.allclose(actual,torch.zeros_like(actual))
        torch.testing.assert_close(inputs,before,rtol=0,atol=0)
        assert actual.untyped_storage().data_ptr()!=inputs.untyped_storage().data_ptr()


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/FusedLeakyReLU',
                                      'torch2hip/gpumode/10190_FusedLeakyReLU'])
def test_fused_manifest_states_exercise_bias_channels_slope_and_scale(relative):
    root = ROOT / 'tasks' / relative
    helper = import_path(root / 'eval_tools/case_controls.py')
    rows = json.loads((root / 'workload.json').read_text())['cases']
    helper.validate_controls(rows)
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    module = import_path(root / args.module).FusedLeakyReLU(channel=256)
    functional = import_path(root / args.functional).FusedLeakyReLU(channel=256)
    failures = dict.fromkeys(['omit_bias', 'wrong_channel', 'fixed_slope', 'fixed_scale'], 0)
    x = torch.tensor([-1., -.1, .2, 1.]).reshape(1, 1, 1, 4).expand(1, 4, 1, 4).clone()
    independent_bias = torch.tensor([(-1)**c * (1/8 + (c+1)/128) for c in range(4)]).reshape(1,4,1,1)
    pristine = x.clone()
    for row in rows:
        rng = torch.random.get_rng_state().clone()
        # Metadata-only tensors use the real shapes without CPU/GPU allocation.
        declared = [torch.empty(row['params']['inputs'][0]['shape'], device='meta')]
        control = helper.configure_models((module, functional), declared)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        assert control == row['params']['operator']
        torch.testing.assert_close(module.bias, functional.bias, rtol=0, atol=0)
        assert module.negative_slope == functional.negative_slope == control['negative_slope']
        assert module.scale == functional.scale == control['scale']
        assert torch.count_nonzero(module.bias) == 256
        assert torch.unique(module.bias).numel() == 256
        shifted = x + independent_bias
        expected = torch.where(shifted >= 0, shifted, shifted * control['negative_slope']) * control['scale']
        torch.testing.assert_close(module(x), expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(functional(x), expected, rtol=1e-4, atol=1e-5)
        for defect in failures:
            def bad(inputs, bias, negative_slope, scale):
                b = bias.roll(1) if defect == 'wrong_channel' else bias
                v = inputs if defect == 'omit_bias' else inputs + b[:4].reshape(1,4,1,1)
                alpha = .2 if defect == 'fixed_slope' else negative_slope
                factor = math.sqrt(2) if defect == 'fixed_scale' else scale
                return torch.where(v >= 0, v, v * alpha) * factor
            actual = functional(x, fn=bad)
            failures[defect] += not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(x, pristine, rtol=0, atol=0)
    assert failures == {'omit_bias': 5, 'wrong_channel': 5, 'fixed_slope': 4, 'fixed_scale': 4}
    with pytest.raises(ValueError, match='declared workload shape'):
        helper.configure_models((module, functional), [torch.empty(1,4,1,4,device='meta')])
    collapsed = copy.deepcopy(rows)
    for row in collapsed:
        row['params']['operator']['negative_slope'] = .2
        row['params']['operator']['scale'] = 1.
    with pytest.raises(ValueError, match='multiple slopes and scales'):
        helper.validate_controls(collapsed)
    zero = copy.deepcopy(rows)
    zero[0]['params']['operator']['bias'].update(offset=0., step=0.)
    with pytest.raises(ValueError, match='nonzero bias'):
        helper.validate_controls(zero)


@pytest.mark.parametrize('task', sorted((ROOT / 'tasks/hip2hip/others').glob('*/kernel_loader.py')),
                         ids=lambda p: p.parent.name)
def test_native_hipify_writes_only_to_fresh_build_inputs(task, tmp_path, monkeypatch):
    task_root = tmp_path / 'task'
    task_root.mkdir()
    shutil.copy2(task, task_root / 'kernel_loader.py')
    shutil.copytree(task.parent / 'src', task_root / 'src')
    nested = task_root / 'src/nested/header.hpp'
    nested.parent.mkdir()
    nested.write_text('constexpr int sentinel = 7;\n')
    editable = yaml.safe_load((task.parent / 'config.yaml').read_text())['candidate']['editable'][0]
    before = {str(p.relative_to(task_root)): p.read_bytes() for p in (task_root / 'src').rglob('*') if p.is_file()}
    calls = []
    def fake_load(*, name, sources, verbose):
        assert verbose is True
        inputs = [Path(p) for p in sources]
        assert all(p.is_relative_to(task_root / 'build/native_sources') for p in inputs)
        stage = inputs[0].parent.parent
        assert (stage / 'src/nested/header.hpp').read_bytes() == nested.read_bytes()
        assert (stage / editable).read_bytes() == (task_root / editable).read_bytes()
        calls.append(stage)
        # Emulate compiler/hipify rewriting generated neighboring translation units.
        for source in inputs:
            generated = source.with_name(source.stem.replace('_cuda', '') + '_hip' + source.suffix)
            generated.write_text('compiler-generated replacement\n')
        return object()
    monkeypatch.setitem(sys.modules, 'torch.utils.cpp_extension', types.SimpleNamespace(load=fake_load))
    monkeypatch.chdir(task_root)
    import_path(task_root / 'kernel_loader.py')
    assert {str(p.relative_to(task_root)): p.read_bytes() for p in (task_root / 'src').rglob('*') if p.is_file()} == before
    candidate = task_root / editable
    candidate.write_text(candidate.read_text() + '\n// next submitted candidate\n')
    import_path(task_root / 'kernel_loader.py')
    assert len(calls) == 2 and calls[0] != calls[1]
    assert (calls[1] / editable).read_bytes() == candidate.read_bytes()


@pytest.mark.parametrize('name', ['roiaware_pool3d', 'roipoint_pool3d'])
def test_roi_full_output_rejects_conserved_sum_corruption(name):
    helper = import_path(ROOT / 'tasks/hip2hip/others' / name / 'scripts/reference_checks.py')
    expected = torch.tensor([0., 2., 4., 8.]).reshape(1,1,2,2)
    wrong = expected.roll(1, dims=-1)
    assert torch.equal(wrong.sum(), expected.sum())
    helper.full_output(expected.clone(), expected)
    with pytest.raises(AssertionError):
        helper.full_output(wrong, expected)
    with pytest.raises(ValueError, match='shape/dtype'):
        helper.full_output(expected.double(), expected)
    invalid = expected.clone(); invalid.flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='NaN/Inf'):
        helper.full_output(invalid, expected)


def test_ball_query_native_annulus_predicate_known_boundaries(tmp_path):
    import re
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    path = ROOT / 'tasks/hip2hip/others/ball_query/src/ball_query_cuda.hip'
    text = path.read_text()
    predicate = next(value for value in re.findall(r'if\s*\(([^\n]+)\)', text)
                     if 'min_radius2' in value and 'max_radius2' in value)
    source = tmp_path / 'annulus.cpp'
    source.write_text('''#include <cassert>
static bool selected(float d2, float min_radius2, float max_radius2) { return PREDICATE; }
int main() {
  assert(selected(0.f, 0.f, 1.f));
  assert(selected(0.f, 1.f, 4.f));
  assert(!selected(1e-8f, 1.f, 4.f));
  assert(selected(1.f, 1.f, 4.f));
  assert(!selected(4.f, 1.f, 4.f));
}
'''.replace('PREDICATE', predicate))
    binary = tmp_path / 'annulus'
    subprocess.run([compiler, '-std=c++17', str(source), '-o', str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize('path', [p for p in NATIVE if 'graph_policy' in json.loads(p.with_name('workload.json').read_text())], ids=lambda p: p.parent.name)
@pytest.mark.parametrize('baseline_graph,source_safe', [(True,True),(True,False),(False,True)])
def test_native_candidate_cannot_override_frozen_graph_safety(path, baseline_graph, source_safe):
    runner = import_path(path.parent / 'scripts/evaluate.py')
    harness = types.SimpleNamespace(HIP_GRAPH_ENABLED=source_safe,
        HIP_GRAPH_FALLBACK_REASON=None if source_safe else 'default stream launch escapes capture')
    manifest = {'graph_policy': {'enabled': baseline_graph,
                                'reason': None if baseline_graph else 'original event-only policy'}}
    if baseline_graph and not source_safe:
        with pytest.raises(ValueError, match='cannot honor the frozen graph'):
            runner.configure_timing_policy(harness, manifest)
        assert harness.HIP_GRAPH_ENABLED is False
    else:
        runner.configure_timing_policy(harness, manifest)
        assert harness.HIP_GRAPH_ENABLED is baseline_graph
        assert harness.HIP_GRAPH_FALLBACK_REASON == manifest['graph_policy']['reason']


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/MaskedLanguageModel',
                                      'torch2hip/gpumode/8325_MaskedLanguageModel'])
def test_masked_language_reference_nonuniform_known_answer(relative):
    root = ROOT / 'tasks' / relative
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    x = torch.tensor([[[[-1., 2.], [3., -4.]]]])
    # Zero weights with log([1,2,3]) biases imply probabilities [1,2,3]/6,
    # independent of input. A uniform/zero shortcut fails this oracle.
    expected = torch.tensor([math.log(i / 6) for i in (1, 2, 3)]).expand(1, 1, 2, 3)
    for filename in (args.module, args.functional):
        model = import_path(root / filename).MaskedLanguageModel(hidden=2, vocab_size=3).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith('bias'):
                    parameter.copy_(torch.tensor([0., math.log(2), math.log(3)]))
                else:
                    parameter.zero_()
        before = copy.deepcopy(model.state_dict())
        pristine = x.clone()
        actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        assert not torch.allclose(actual, torch.full_like(actual, -math.log(3)))
        assert not torch.allclose(actual, torch.zeros_like(actual))
        torch.testing.assert_close(x, pristine, rtol=0, atol=0)
        assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize('name', ['knn', 'three_nn'])
@pytest.mark.parametrize('behavior', ['correct', 'wrong_replay', 'input_mutation'])
def test_native_integer_and_tuple_timed_replay(name, behavior, monkeypatch):
    root = ROOT / 'tasks/hip2hip/others' / name
    controls = import_path(root / 'scripts/reference_checks.py')
    replay = import_path(root / 'scripts/replay_validation.py')
    h = harness_namespace(root)
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    source = torch.tensor([[[1., 0., 0.], [3., 0., 0.], [5., 0., 0.]]])
    target = torch.zeros(1, 1, 3)
    if name == 'knn':
        expected = h.cpu_reference(2, source, target)
        check = lambda output: controls.check_timed_output(output, expected, gpu=False)
    else:
        expected = h.cpu_reference(target, source)
        check = lambda output: controls.check_timed_output(output, expected, target, source, gpu=False)
    invoked = []

    def benchmark(invoke, **kwargs):
        invoked.append(kwargs)
        output = invoke()
        if behavior == 'input_mutation': source.add_(1)

        def captured():
            for value, original in zip(replay.tensors(output), replay.tensors(expected)):
                if value.is_floating_point(): assert torch.isnan(value).all()
                else: assert torch.all(value == -1)
                value.copy_(original)
            if behavior == 'wrong_replay':
                replay.tensors(output)[-1].zero_()  # Valid dtype, wrong neighbor indices.
            return output

        kwargs['timed_run']._bind(captured, output)
        return .5, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}

    if behavior == 'correct':
        elapsed, metadata = replay.measure(benchmark, lambda: copy.deepcopy(expected), (target, source), check,
                                           warmup=10, repetition=100, use_cuda_graph=True)
        assert elapsed == .5 and metadata['replay_validation_valid'] is True
    else:
        with pytest.raises((AssertionError, ValueError)):
            replay.measure(benchmark, lambda: copy.deepcopy(expected), (target, source), check,
                           warmup=10, repetition=100, use_cuda_graph=True)
    assert invoked[0]['warmup'] == 10 and invoked[0]['repetition'] == 100


def test_three_nn_timed_contract_preserves_ties_and_rejects_wrong_index():
    root = ROOT / 'tasks/hip2hip/others/three_nn'
    controls = import_path(root / 'scripts/reference_checks.py')
    h = harness_namespace(root)
    target = torch.zeros(1, 1, 3)
    source = torch.tensor([[[1., 0., 0.], [-1., 0., 0.], [3., 0., 0.], [9., 0., 0.]]])
    expected = h.cpu_reference(target, source)
    distance, indices = copy.deepcopy(expected)
    indices[..., :2] = indices[..., :2].flip(-1)
    controls.check_timed_output((distance, indices), expected, target, source, gpu=False)
    for invalid in (-1, 3, 4):
        bad = indices.clone(); bad[..., 0] = invalid
        with pytest.raises(ValueError):
            controls.check_timed_output((distance, bad), expected, target, source, gpu=False)
    with pytest.raises(AssertionError):
        controls.check_timed_output((distance + 1, indices), expected, target, source, gpu=False)


@pytest.mark.parametrize('name', ['ball_query', 'furthest_point_sample', 'points_in_boxes',
                                 'roiaware_pool3d', 'roipoint_pool3d'])
@pytest.mark.parametrize('behavior', ['correct', 'wrong_replay', 'input_mutation'])
def test_native_pool_and_query_replay_checks_written_buffers(name, behavior, monkeypatch):
    root = ROOT / 'tasks/hip2hip/others' / name
    controls = import_path(root / 'scripts/reference_checks.py')
    replay = import_path(root / 'scripts/replay_validation.py')
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    inputs = [torch.tensor([1., 2., 3.])]
    if name == 'roiaware_pool3d':
        expected = torch.tensor([1., 4., 2., 3.]).reshape(1, 1, 1, 1, 4)
        check = lambda actual: controls.check_timed_output(actual, expected, 'max', gpu=False)
    elif name == 'roipoint_pool3d':
        expected = (torch.tensor([1., 4., 2., 3.]).reshape(1, 1, 1, 4), torch.zeros(1, 1, dtype=torch.int32))
        check = lambda actual: controls.check_timed_output(actual, expected, gpu=False)
    else:
        # -1 is a legitimate outside-box value, so poisoning must change it too.
        expected = torch.tensor([[0, -1 if name == 'points_in_boxes' else 1, 2]], dtype=torch.int32)
        check = lambda actual: controls.close(actual, expected)
    output = copy.deepcopy(expected)
    calls = []

    def prepare():
        calls.append('prepare')
        for value in replay.tensors(output): value.zero_()

    def invoke():
        calls.append('invoke')
        for value, original in zip(replay.tensors(output), replay.tensors(expected)): value.copy_(original)
        return output

    def benchmark(fn, **kwargs):
        assert kwargs['warmup'] == 10 and kwargs['repetition'] == 100
        prep = kwargs.get('prepare_fn')
        if prep: prep()
        observed_output = fn()

        def captured():
            for value, original in zip(replay.tensors(output), replay.tensors(expected)):
                if value.is_floating_point(): assert torch.isnan(value).all()
                elif name == 'points_in_boxes': assert torch.equal(value, original.bitwise_not())
                else: assert torch.all(value == -1)
            if prep: prep()
            fn()
            if behavior == 'wrong_replay': replay.tensors(output)[0].zero_()
            if behavior == 'input_mutation': inputs[0].add_(1)
            return observed_output

        kwargs['timed_run']._bind(captured, observed_output)
        return .5, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}

    options = dict(warmup=10, repetition=100, use_cuda_graph=True)
    if name.startswith('roi'): options['prepare_fn'] = prepare
    if behavior == 'correct':
        elapsed, metadata = replay.measure(benchmark, invoke, inputs, check, **options)
        assert elapsed == .5 and metadata['replay_validation_valid'] is True
    else:
        with pytest.raises((ValueError, AssertionError)):
            replay.measure(benchmark, invoke, inputs, check, **options)
    assert calls.count('invoke') == 2
    assert calls.count('prepare') == (2 if name.startswith('roi') else 0)


@pytest.mark.parametrize('path', [p for p in CONFIGS if 'level1' in p.parts], ids=lambda p: p.parent.name)
def test_level1_reference_independent_known_answers_and_readonly_inputs(path):
    args = options(yaml.safe_load(path.read_text()))
    number = path.parent.name.split('_', 1)[0]
    for filename in (args.module, args.functional):
        cls = getattr(import_path(path.parent / filename), args.model_class)
        if number in ('l1n1', 'l1n2', 'l1n3', 'l1n4', 'l1n8', 'l1n9'):
            model = cls()
            k = 2 if number == 'l1n1' else 3
            n = 1 if number == 'l1n4' else 2
            a = torch.arange(2*k, dtype=torch.float32).reshape(2,k) - 2
            b = torch.arange(k*n, dtype=torch.float32).reshape(k,n) + 1
            if number == 'l1n3': a, b = torch.stack((a,a+1)), torch.stack((b,b-2))
            pairs = zip(a, b) if a.ndim == 3 else [(a,b)]
            results = [torch.tensor([[sum(float(left[i,j])*float(right[j,z]) for j in range(k))
                                      for z in range(n)] for i in range(2)]) for left,right in pairs]
            expected = torch.stack(results) if a.ndim == 3 else results[0]
            inputs = [a,b]
        elif number == 'l1n23':
            model = cls(); inputs = [torch.tensor([[0.,math.log(2),math.log(3)]])]
            expected = torch.tensor([[1/6,2/6,3/6]])
        elif number == 'l1n26':
            model = cls(); values = [-2.,-1.,0.,.5,1.]; inputs = [torch.tensor(values)]
            expected = torch.tensor([x*.5*(1+math.erf(x/math.sqrt(2))) for x in values])
        elif number == 'l1n36':
            model = cls(num_features=3); inputs = [torch.tensor([[-1.,2.,3.]])]
            expected = torch.tensor([[-1.,2.,3.]]) / math.sqrt(14/3+1e-5)
        elif number == 'l1n40':
            model = cls(normalized_shape=(3,)); inputs = [torch.tensor([[-1.,2.,5.]])]
            with torch.no_grad():
                model.ln.weight.copy_(torch.tensor([1.,2.,3.]))
                model.ln.bias.copy_(torch.tensor([-1.,.5,2.]))
            expected = torch.tensor([[(x-2)/math.sqrt(6+1e-5)*w+b
                                      for x,w,b in zip((-1,2,5),(1,2,3),(-1,.5,2))]])
        elif number == 'l1n42':
            model = cls(kernel_size=2, stride=2, padding=0, dilation=1)
            inputs = [torch.arange(-8,8,dtype=torch.float32).reshape(1,1,4,4)]
            expected = torch.tensor([[[[-3.,-1.],[5.,7.]]]])
        elif number == 'l1n47':
            model = cls(dim=1); inputs = [torch.tensor([[-1.,2.,4.],[3.,-5.,1.]])]
            expected = torch.tensor([[5.],[-1.]])
        elif number in ('l1n63','l1n82'):
            model = cls(in_channels=2, kernel_size=2, bias=True,
                        **({'out_channels':2} if number == 'l1n63' else {}))
            conv = model.conv2d
            with torch.no_grad():
                conv.weight.copy_(torch.arange(conv.weight.numel()).reshape_as(conv.weight)/8+.5)
                conv.bias.copy_(torch.tensor([.25,-.5]))
            x = torch.arange(-8,10,dtype=torch.float32).reshape(1,2,3,3); inputs = [x]
            expected = torch.empty(1,2,2,2)
            for out in range(2):
                for i in range(2):
                    for j in range(2):
                        channels = [out] if number == 'l1n82' else range(2)
                        expected[0,out,i,j] = float(conv.bias[out].detach()) + sum(
                            float(x[0,ch,i+dy,j+dx])*float(conv.weight[out,0 if number == 'l1n82' else ch,dy,dx].detach())
                            for ch in channels for dy in range(2) for dx in range(2))
        elif number == 'l1n95':
            model = cls(); inputs = [torch.tensor([[0.,math.log(2),math.log(3)],
                                                  [math.log(2),math.log(3),0.]]),torch.tensor([2,0])]
            expected = torch.tensor(-(math.log(.5)+math.log(1/3))/2)
        else:
            raise AssertionError('Missing independent oracle for '+number)
        model.eval(); pristine = copy.deepcopy(inputs); state = copy.deepcopy(model.state_dict())
        actual = model(*inputs)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        assert not torch.allclose(actual,torch.zeros_like(actual))
        for before,after in zip(pristine,inputs):
            torch.testing.assert_close(before,after,rtol=0,atol=0)
            assert actual.untyped_storage().data_ptr()!=after.untyped_storage().data_ptr()
        for key,value in model.state_dict().items():
            torch.testing.assert_close(state[key],value,rtol=0,atol=0)


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/CrossEntropyLossLabelSmoothing',
                                      'torch2hip/gpumode/12501_CrossEntropyLossLabelSmoothing'])
@pytest.mark.parametrize('kind', ['captured_graph', 'eager_callable'])
@pytest.mark.parametrize('defect', ['none', 'last_sample', 'reinvocation', 'input_mutation'])
def test_loss_observes_actual_event_sample_and_distinguishes_graph(relative, kind, defect, monkeypatch):
    root = ROOT / 'tasks' / relative
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    model = import_path(root / args.functional).CrossEntropyLossLabelSmoothing().eval()
    logits = torch.tensor([[0., math.log(2), math.log(3)], [math.log(2), math.log(3), 0.]])
    targets = torch.tensor([[.25,.25,.5],[.5,0.,.5]])
    controls = import_path(root / 'eval_tools/case_controls.py')
    monkeypatch.setitem(sys.modules, 'case_controls', controls)
    controls.apply_control(model, {'smooth_eps': 0.}, [logits, targets])
    expected = torch.tensor(-(.25*math.log(1/6)+.25*math.log(2/6)+.5*math.log(3/6)
                              +.5*math.log(2/6)+.5*math.log(1/6))/2)
    for filename in (args.module, args.functional):
        reference = import_path(root / filename).CrossEntropyLossLabelSmoothing().eval()
        torch.testing.assert_close(reference(logits,targets),expected,rtol=1e-4,atol=1e-5)
    replay = import_path(root / 'eval_tools/replay_validation.py')
    runner = import_path(root / 'eval_tools/evaluate.py')
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    observed_buffers = []

    def benchmark(invoke, **kwargs):
        assert kwargs['warmup'] == 10 and kwargs['repetition'] == 100
        output = invoke()
        observed_buffers.append(output)
        if defect == 'last_sample': output.zero_()
        if defect == 'input_mutation': targets.add_(1)

        def rerun():
            assert torch.isnan(output).all()
            # An explicit Event re-invocation may allocate a different buffer.
            actual = invoke() if kind == 'eager_callable' else output
            actual.copy_(expected)
            if defect == 'reinvocation': actual.zero_()
            observed_buffers.append(actual)
            return actual

        kwargs['timed_run']._bind(rerun,output)
        return .5, {'benchmark_method':'cuda_graph' if kind == 'captured_graph' else 'cuda_event_fallback',
                    'benchmark_timed_run_kind':kind}

    def cal_kernel_perf(rtol=1e-4,atol=1e-5): pass
    perf = types.SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
        benchmark_cuda_graph_or_events=benchmark,_compare_results=torch.allclose)
    replay.install(perf,runner.output_contract)
    if defect == 'none':
        elapsed,metadata = perf.cal_hip_latency(model,[logits,targets],use_cuda_graph=(kind=='captured_graph'))
        assert elapsed == .5 and metadata['validated_invocation_kind'] == kind
        assert metadata['replay_validation_valid'] is True
        assert (observed_buffers[0] is observed_buffers[1]) == (kind=='captured_graph')
    else:
        with pytest.raises((ValueError,AssertionError)):
            perf.cal_hip_latency(model,[logits,targets],use_cuda_graph=(kind=='captured_graph'))
    if defect == 'last_sample': assert len(observed_buffers) == 1


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/NormalAttention_dot',
    'torch2hip/gpumode/1001_NormalAttention_dot', 'hip2hip/gpumode/Feedforward',
    'torch2hip/gpumode/10024_Feedforward'])
@pytest.mark.parametrize('kind', ['captured_graph', 'eager_callable'])
@pytest.mark.parametrize('defect', ['none', 'last_sample', 'reinvocation'])
def test_attention_and_feedforward_measured_outputs_have_independent_oracles(relative, kind, defect, monkeypatch):
    root = ROOT / 'tasks' / relative
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    feedforward = args.model_class == 'Feedforward'
    if feedforward:
        inputs = [torch.tensor([[1.,2.],[-3.,1.]]),torch.tensor([[2.,-1.]])]
        values = []
        for x,y in [(1,2),(-3,1),(2,-1)]:
            z = .5*max(0,x-y-.5)-max(0,.5*x+2*y+.25)+.75
            values.append([1/(1+math.exp(-z))])
        expected = torch.tensor(values)
    else:
        inputs = [torch.arange(1,9,dtype=torch.float32).reshape(1,4,1,2)]
        # Query=1, key=2 => ELU(2)/2=1, value/gamma identities. Each
        # output channel is the sum of its two spatial inputs plus its bias.
        expected = torch.tensor([3.25,7.5,11.75,16.]).reshape(1,4,1,1).expand(1,4,1,2)
    models = []
    for filename in (args.module,args.functional):
        cls = getattr(import_path(root / filename),args.model_class)
        model = cls(input_size=2,hidden_size=2) if feedforward else cls(input_channel_num=4)
        with torch.no_grad():
            if feedforward:
                weights = {'fc1_weight':torch.tensor([[1.,-1.],[.5,2.]]),
                           'fc1_bias':torch.tensor([-.5,.25]),
                           'fc2_weight':torch.tensor([[.5,-1.]]),'fc2_bias':torch.tensor([.75])}
                for name,value in model.named_parameters(): value.copy_(weights[name.replace('.','_')])
            else:
                for parameter in model.parameters(): parameter.zero_()
                model.query_conv.bias.fill_(1); model.key_conv.bias.fill_(2)
                model.value_conv.weight.copy_(torch.eye(4).reshape(4,4,1,1))
                model.gamma.weight.copy_(torch.eye(4).reshape(4,4,1,1))
                model.gamma.bias.copy_(torch.tensor([.25,.5,.75,1.]))
        model.eval(); models.append(model)
        pristine = copy.deepcopy(inputs); before = copy.deepcopy(model.state_dict())
        actual = model(*inputs)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        for old,new in zip(pristine,inputs): torch.testing.assert_close(old,new,rtol=0,atol=0)
        for key,value in model.state_dict().items(): torch.testing.assert_close(before[key],value,rtol=0,atol=0)
    replay = import_path(root / 'eval_tools/replay_validation.py')
    runner = import_path(root / 'eval_tools/evaluate.py')
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules,'_aka_benchmark',timed)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)

    def benchmark(invoke,**kwargs):
        assert kwargs['warmup']==10 and kwargs['repetition']==100
        output=invoke().detach()
        if defect=='last_sample': output.zero_()
        def rerun():
            assert torch.isnan(output).all()
            actual=invoke().detach() if kind=='eager_callable' else output
            actual.copy_(expected)
            if defect=='reinvocation': actual.zero_()
            return actual
        kwargs['timed_run']._bind(rerun,output)
        return .5,{'benchmark_method':'cuda_graph' if kind=='captured_graph' else 'cuda_event_fallback',
                   'benchmark_timed_run_kind':kind}

    def cal_kernel_perf(rtol=1e-4,atol=1e-5): pass
    perf=types.SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
        benchmark_cuda_graph_or_events=benchmark,_compare_results=torch.allclose)
    replay.install(perf,runner.output_contract)
    if defect=='none':
        elapsed,metadata=perf.cal_hip_latency(models[1],inputs,use_cuda_graph=(kind=='captured_graph'))
        assert elapsed==.5 and metadata['validated_invocation_kind']==kind
    else:
        with pytest.raises(ValueError,match='protected reference'):
            perf.cal_hip_latency(models[1],inputs,use_cuda_graph=(kind=='captured_graph'))


@pytest.mark.parametrize('relative', ADDITIONAL_GPUMODE_REPLAY)
def test_remaining_gpumode_references_against_independent_answers(relative):
    root=ROOT/'tasks'/relative; args=options(yaml.safe_load((root/'config.yaml').read_text()))
    name=args.model_class
    for filename in (args.module,args.functional):
        cls=getattr(import_path(root/filename),name)
        if name=='InnerProd':
            model=cls(2); inputs=[torch.tensor([[2.,3.]]),torch.tensor([[[[4.,5.]],[[6.,7.]]]])]
            with torch.no_grad(): model.scale.copy_(torch.tensor([2.,-1.])); model.bias.fill_(.5)
            expected=torch.tensor([[[[-1.5,-.5]]]])
        elif name=='KDLoss':
            model=cls(2); inputs=[torch.tensor([[0.,2*math.log(3)]]),torch.tensor([[.5,.5]])]
            expected=torch.tensor(2*math.log(4/3))
        elif name=='MLP_model':
            model=cls(2,2); inputs=[torch.tensor([[2.,-1.],[-2.,1.]])]
            with torch.no_grad():
                for parameter in model.parameters(): parameter.zero_()
                for index in range(1,7):
                    layer=getattr(model,'linear'+str(index)); layer.weight[0,0]=1; layer.bias[0]=.1
                model.linear7.weight[:,0]=torch.tensor([2.,-1.])
                model.linear7.bias.copy_(torch.tensor([.25,-.5]))
            values=[]
            for x in (2.,-2.):
                for _ in range(6): x=max(0,x+.1)
                values.append([2*x+.25,-x-.5])
            expected=torch.tensor(values)
        elif name=='MultiHeadAttention':
            model=cls(heads=2,d_model=4)
            inputs=[torch.ones(1,2,4),torch.ones(1,2,4),torch.arange(1,9,dtype=torch.float32).reshape(1,2,4)]
            with torch.no_grad():
                for parameter in model.parameters(): parameter.zero_()
                model.v_linear1.copy_(torch.eye(4)); model.out.weight.copy_(torch.eye(4))
                model.out.bias.copy_(torch.tensor([.25,.5,.75,1.]))
            expected=torch.tensor([3.25,4.5,5.75,7.]).reshape(1,1,4).expand(1,2,4)
        elif name=='NormalAttention_embedded_gaussian':
            model=cls(input_channel_num=4); inputs=[torch.arange(1,9,dtype=torch.float32).reshape(1,4,1,2)]
            with torch.no_grad():
                for parameter in model.parameters(): parameter.zero_()
                model.value_conv.weight.copy_(torch.eye(4).reshape(4,4,1,1))
                model.gamma.weight.copy_(torch.eye(4).reshape(4,4,1,1))
                model.gamma.bias.copy_(torch.tensor([.25,.5,.75,1.]))
            expected=torch.tensor([1.75,4.,6.25,8.5]).reshape(1,4,1,1).expand(1,4,1,2)
        elif name=='PositionWiseFeedForward':
            model=cls(d_model=3,hidden_size=2); inputs=[torch.tensor([[[-1.,2.,5.]]])]
            with torch.no_grad():
                for parameter in model.parameters(): parameter.zero_()
                model.layer_norm.weight.copy_(torch.tensor([1.,2.,3.]))
                model.layer_norm.bias.copy_(torch.tensor([-1.,.5,2.]))
            expected=torch.tensor([[[(x-2)/math.sqrt(6+1e-5)*w+b
                                     for x,w,b in zip((-1,2,5),(1,2,3),(-1,.5,2))]]])
        elif name=='SimpleMatmulModule':
            model=cls(); inputs=[torch.tensor([[1.,-2.],[3.,4.]]),torch.tensor([[2.,1.],[0.,-1.]])]
            expected=torch.tensor([[4.,6.],[12.,-2.]])
        elif name=='SoftmaxModule':
            model=cls(1); inputs=[torch.tensor([[0.,math.log(2),math.log(3)]])]
            expected=torch.tensor([[1/6,2/6,3/6]])
        elif name=='TransformerFFNLayer':
            model=cls(hidden_size=2,filter_size=3); inputs=[torch.tensor([[[1.,-2.]],[[-1.,3.]]])]
            weights={'ffn1weight':torch.tensor([[1.,0.],[0.,1.],[1.,1.]]).reshape(3,2,1),
                     'ffn1bias':torch.tensor([.1,.2,.3]),'ffn2weight':torch.tensor([[1.,2.,-1.],[.5,-.5,2.]]),
                     'ffn2bias':torch.tensor([.25,-.75])}
            with torch.no_grad():
                for key,value in model.named_parameters(): value.copy_(weights[key.replace('.','').replace('_','')])
            values=[]
            for x,y in ((1.,-2.),(-1.,3.)):
                g=[z*.5*(1+math.erf(z/math.sqrt(2))) for z in (x+.1,y+.2,x+y+.3)]
                values.append([[g[0]+2*g[1]-g[2]+.25,.5*g[0]-.5*g[1]+2*g[2]-.75]])
            expected=torch.tensor(values)
        elif name=='Transpose':
            model=cls(); inputs=[torch.arange(6,dtype=torch.float32).reshape(2,3)]
            expected=torch.tensor([[0.,3.],[1.,4.],[2.,5.]])
            # Every declared transpose requires a real layout conversion; no
            # degenerate no-copy case is subjected to an invented no-alias rule.
            module=import_path(root/args.module); init_args,init_kwargs=module.get_init_inputs()
            reference=module.Transpose(*init_args,**init_kwargs)
            for row in json.loads((root/'workload.json').read_text())['cases']:
                desc=row['params']['inputs'][0]
                original=torch.empty_strided(desc['shape'],desc['stride'],device='meta')
                assert not original.transpose(reference.dim1,reference.dim2).is_contiguous()
        elif name=='layer_normalization':
            model=cls(3); inputs=[torch.tensor([[-1.,2.,5.]])]
            with torch.no_grad():
                model.gamma.copy_(torch.tensor([2.,3.,4.])); model.beta.copy_(torch.tensor([.1,.2,.3]))
            expected=torch.tensor([[(x-2)/(3+1e-8)*g+b for x,g,b in zip((-1,2,5),(2,3,4),(.1,.2,.3))]])
        elif name=='Gather':
            model=cls(dim=1); inputs=[torch.tensor([[1.,2.,3.],[4.,5.,6.]]),torch.tensor([2,0])]
            expected=torch.tensor([[3.,1.],[6.,4.]])
        elif name=='PositionEmbedder':
            model=cls(max_sequence_length=3,embedding_dim=2)
            with torch.no_grad(): model.embedding.weight.copy_(torch.tensor([[0.,0.],[2.,-3.],[4.,-6.]]))
            inputs=[torch.arange(1,7,dtype=torch.float32).reshape(1,3,2)]
            expected=torch.tensor([[[1.,2.],[5.,1.],[9.,0.]]])
        else:
            raise AssertionError('Missing independent oracle for '+name)
        model.eval(); pristine=copy.deepcopy(inputs); state=copy.deepcopy(model.state_dict())
        actual=model(*inputs)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        assert not torch.allclose(actual,torch.zeros_like(actual))
        for old,new in zip(pristine,inputs):
            torch.testing.assert_close(old,new,rtol=0,atol=0)
            assert actual.untyped_storage().data_ptr()!=new.untyped_storage().data_ptr()
        for key,value in model.state_dict().items(): torch.testing.assert_close(value,state[key],rtol=0,atol=0)


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/FusedLeakyReLU',
                                      'torch2hip/gpumode/10190_FusedLeakyReLU'])
@pytest.mark.parametrize('defect', ['none', 'omit_bias', 'wrong_channel', 'fixed_slope', 'fixed_scale'])
def test_fused_timed_replay_uses_nonzero_manifest_reference(relative, defect, monkeypatch):
    """A correct eager call cannot excuse wrong fused math on the timed replay.

    CPU control-plane test: the bound replay stands in for the captured graph;
    real graph capture and HIP execution require the separate GPU validator.
    """
    root = ROOT / 'tasks' / relative
    controls = import_path(root / 'eval_tools/case_controls.py')
    replay = import_path(root / 'eval_tools/replay_validation.py')
    runner = import_path(root / 'eval_tools/evaluate.py')
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    model = import_path(root / args.functional).FusedLeakyReLU(channel=256).eval()
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    x = torch.tensor([-1., -.1, .2, 1.]).reshape(1, 1, 1, 4).expand(1, 4, 1, 4).clone()
    pristine = x.clone()
    rows = json.loads((root / 'workload.json').read_text())['cases']
    reruns = []

    def benchmark(invoke, **kwargs):
        assert kwargs['warmup'] == 10 and kwargs['repetition'] == 100
        output = invoke().detach()  # Eager candidate is always correct.

        def captured():
            assert torch.isnan(output).all()
            reruns.append(defect)
            with torch.no_grad():
                bias = model.bias.roll(1) if defect == 'wrong_channel' else model.bias
                shifted = x if defect == 'omit_bias' else x + bias[:4].reshape(1, 4, 1, 1)
                slope = .2 if defect == 'fixed_slope' else model.negative_slope
                scale = math.sqrt(2) if defect == 'fixed_scale' else model.scale
                output.copy_(torch.where(shifted >= 0, shifted, shifted * slope) * scale)
            return output

        kwargs['timed_run']._bind(captured, output)
        return .25, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}

    def cal_kernel_perf(rtol=1e-4, atol=1e-5): pass
    perf = types.SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
        benchmark_cuda_graph_or_events=benchmark, _compare_results=torch.allclose)
    replay.install(perf, runner.output_contract)
    for index, row in enumerate(rows):
        controls.apply_control(model, row['params']['operator'])
        before_state = copy.deepcopy(model.state_dict())
        accepted = defect == 'none' or (defect in ('fixed_slope', 'fixed_scale') and index == 1)
        if accepted:
            elapsed, metadata = perf.cal_hip_latency(model, [x])
            assert elapsed == .25 and metadata['replay_validation_valid'] is True
        else:
            with pytest.raises(ValueError, match='protected reference'):
                perf.cal_hip_latency(model, [x])
        torch.testing.assert_close(x, pristine, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before_state[name], rtol=0, atol=0)
    assert len(reruns) == 5 + (5 if defect == 'none' else int(defect in ('fixed_slope', 'fixed_scale')))


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/FusedLeakyReLU',
                                      'torch2hip/gpumode/10190_FusedLeakyReLU'])
@pytest.mark.parametrize('role', ['baseline', 'candidate'])
@pytest.mark.parametrize('behavior', ['correct', 'cached_answer', 'fresh_replay_error',
                                      'fresh_oracle_error', 'fresh_input_mutation'])
def test_fused_changed_input_replay_restores_scored_workload(relative, role, behavior, monkeypatch):
    """CPU graph double: old-answer replay passes same-input checks but fails fresh inputs."""
    root = ROOT / 'tasks' / relative
    helper = import_path(root / 'eval_tools/replay_validation.py')
    controls = import_path(root / 'eval_tools/case_controls.py')
    runner = import_path(root / 'eval_tools/evaluate.py')
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    model = import_path(root / args.functional).FusedLeakyReLU(channel=256).eval()
    row = json.loads((root / 'workload.json').read_text())['cases'][0]
    controls.apply_control(model, row['params']['operator'])
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    x = torch.tensor([-1., -.1, .2, 1.]).reshape(1, 1, 1, 4).expand(1, 4, 1, 4).clone()
    pristine, rng = x.clone(), torch.random.get_rng_state().clone()
    state = copy.deepcopy(model.state_dict())
    seen, oracle_inputs, compared = [], [], []

    def oracle(value, fn=None):
        if fn is not None:
            return fn(value)
        oracle_inputs.append(value.clone())
        if behavior == 'fresh_oracle_error' and not torch.equal(value, pristine):
            raise RuntimeError('fresh oracle deliberately failed')
        return model(value)

    def candidate(value):
        # Independent fused math, never delegates to the protected module.
        shifted = value + model.bias[:4].detach().reshape(1, 4, 1, 1)
        return torch.where(shifted >= 0, shifted, shifted * model.negative_slope) * model.scale

    def benchmark(invoke, **kwargs):
        assert kwargs['warmup'] == 10 and kwargs['repetition'] == 100
        torch.testing.assert_close(x, pristine, rtol=0, atol=0)
        output = invoke().detach()
        cached = output.clone()
        def replay():
            assert torch.isnan(output).all()
            seen.append(x.clone())
            changed = not torch.equal(x, pristine)
            if changed and behavior == 'fresh_replay_error':
                raise RuntimeError('fresh replay deliberately failed')
            output.copy_(cached if behavior == 'cached_answer' else candidate(x))
            if changed and behavior == 'fresh_input_mutation':
                x.add_(1)
            return output
        kwargs['timed_run']._bind(replay, output)
        return .25, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}

    def cal_kernel_perf(rtol=1e-4, atol=1e-5): pass
    def compare(expected, actual, *, rtol, atol):
        assert rtol == 1e-4 and atol == 1e-5
        compared.append((expected.clone(), actual.clone()))
        return torch.allclose(expected, actual, rtol=rtol, atol=atol)
    perf = types.SimpleNamespace(cal_kernel_perf=cal_kernel_perf,
        cal_modu_latency=None, benchmark_cuda_graph_or_events=benchmark, _compare_results=compare)
    helper.install(perf, runner.output_contract)
    def run():
        return perf.cal_modu_latency(oracle, [x]) if role == 'baseline' else perf.cal_hip_latency(oracle, [x], candidate)
    if behavior == 'correct':
        elapsed, metadata = run()
        assert elapsed == .25 and metadata['changed_input_validation_valid'] is True
        assert metadata['changed_input_restore'].startswith('finally')
    else:
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            run()
    # The same-input replay passed before the fresh control was exercised.
    torch.testing.assert_close(seen[0], pristine, rtol=0, atol=0)
    torch.testing.assert_close(oracle_inputs[-1], .5 - pristine, rtol=0, atol=0)
    if behavior != 'fresh_oracle_error':
        torch.testing.assert_close(seen[1], .5 - pristine, rtol=0, atol=0)
    torch.testing.assert_close(x, pristine, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert model.negative_slope == .1 and model.scale == .5
    if behavior == 'cached_answer':
        # Last comparison used the new oracle and the deliberately stale answer.
        assert not torch.allclose(*compared[-1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/InnerProd', 'torch2hip/gpumode/11709_InnerProd'])
def test_innerprod_manifest_affine_state_and_negative_controls(relative):
    root = ROOT / 'tasks' / relative
    helper = import_path(root / 'eval_tools/case_controls.py')
    rows = json.loads((root / 'workload.json').read_text())['cases']
    helper.validate_controls(rows)
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    channels = rows[0]['params']['inputs'][1]['shape'][1]
    models = [import_path(root / name).InnerProd(channels).eval() for name in (args.module, args.functional)]
    image = (torch.arange(1, channels + 1, dtype=torch.float32) / channels).reshape(1, 1, channels)
    sound = torch.arange(1, channels * 2 + 1, dtype=torch.float32).reshape(1, channels, 1, 2) / (channels * 2)
    scale = torch.tensor([.5 + .5 * (c + 1) / channels for c in range(channels)])
    pure_sum = torch.stack([sum(image[0, 0, c] * sound[0, c, 0, j] for c in range(channels)) for j in range(2)]).reshape(1, 1, 1, 2)
    weighted = torch.stack([sum(image[0, 0, c] * scale[c] * sound[0, c, 0, j] for c in range(channels)) for j in range(2)]).reshape(1, 1, 1, 2)
    for row in rows:
        pristine = [image.clone(), sound.clone()]
        rng = torch.random.get_rng_state().clone()
        declared = [torch.empty(v['shape'], device='meta') for v in row['params']['inputs']]
        assert helper.configure_models(models, declared) == row['params']['operator']
        bias = row['params']['operator']['bias']
        expected = weighted + bias
        for model in models:
            actual = model(image, sound)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(model.scale, scale, rtol=0, atol=0)
            assert model.bias.item() == bias
        assert not torch.allclose(expected, pure_sum + bias, rtol=1e-4, atol=1e-4)
        assert not torch.allclose(expected, weighted, rtol=1e-4, atol=1e-4)
        assert torch.equal(torch.random.get_rng_state(), rng)
        for old, new in zip(pristine, [image, sound]):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
    invalid = copy.deepcopy(rows)
    for row in invalid:
        row['params']['operator']['bias'] = 0
    with pytest.raises(ValueError, match='nonzero bias'):
        helper.validate_controls(invalid)


@pytest.mark.parametrize('path', EXTENSIONS, ids=lambda p: p.parent.name)
@pytest.mark.parametrize('role', ['baseline', 'candidate'])
@pytest.mark.parametrize('fault', ['none', 'input', 'parameter', 'buffer', 'exception'])
def test_all_python_timed_paths_preserve_and_restore_state(path, role, fault, monkeypatch):
    helper = import_path(path.parent / 'eval_tools/replay_validation.py')
    runner = import_path(path.parent / 'eval_tools/evaluate.py')
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.))
            self.register_buffer('offset', torch.tensor(.5))
        def forward(self, x, fn=None):
            return torch.nn.functional.gelu(x * self.weight + self.offset) if fn is None else fn(x, self.weight, self.offset)
    model = Model().eval()
    if path.parent.name in {'GELU', '14539_GELU'}:
        # The shared state test uses a synthetic affine model; actual GELU's
        # independent oracle and both real call paths have separate controls.
        monkeypatch.setattr(helper, 'gelu_reference', lambda value: model(value))
    x = torch.tensor([-1., .5, 2.])
    original = x.clone()
    if 'CrossEntropyLossLabelSmoothing' in path.parent.name:
        # This test isolates shared state mechanics with its synthetic model.
        # The actual loss oracle and replay are exercised separately below.
        model.smooth_eps, model.smooth_dist = .3, None
        monkeypatch.setitem(sys.modules, 'case_controls', types.SimpleNamespace(
            reference=lambda value, epsilon, distribution: model(value)))
    def candidate(x, weight, offset):
        return torch.nn.functional.gelu(x * weight + offset)
    def benchmark(invoke, **kwargs):
        assert kwargs['warmup'] == 10 and kwargs['repetition'] == 100
        output = invoke().detach()
        def replay():
            assert torch.isnan(output).all()
            with torch.no_grad():
                output.copy_(candidate(x, model.weight, model.offset))
                if fault in ('input', 'exception'): x.add_(1)
                if fault in ('parameter', 'exception'): model.weight.add_(1)
                if fault in ('buffer', 'exception'): model.offset.add_(1)
                if fault == 'exception': raise RuntimeError('deliberate replay failure')
            return output
        kwargs['timed_run']._bind(replay, output)
        return .25, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}
    def cal_kernel_perf(rtol=1e-4, atol=1e-5): pass
    perf = types.SimpleNamespace(cal_kernel_perf=cal_kernel_perf, cal_modu_latency=None,
        benchmark_cuda_graph_or_events=benchmark, _compare_results=torch.allclose)
    helper.install(perf, runner.output_contract)
    def run():
        return perf.cal_modu_latency(model, [x]) if role == 'baseline' else perf.cal_hip_latency(model, [x], candidate)
    if fault == 'none':
        _, meta = run()
        assert meta['input_state_restored'] and meta['model_state_validation_valid']
        assert meta['model_state_tensor_count'] == 2
    else:
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            run()
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert model.weight.item() == 2 and model.offset.item() == .5


@pytest.mark.parametrize('path', [p for p in EXTENSIONS if 'level2' in p.parts or 'level3' in p.parts], ids=lambda p: p.parent.name)
def test_level23_references_against_independent_small_answers(path):
    args = options(yaml.safe_load(path.read_text()))
    name = args.model_class
    constructors = {
        'Conv2d_InstanceNorm_Divide': (1, 4, 1, 2),
        'Matmul_Swish_Sum_GroupNorm': (4, 4, 2, (4,)),
        'Matmul_Scaling_ResidualAdd': (4, 4, 2),
        'Conv2d_Subtract_Tanh_Subtract_AvgPool': (1, 4, 1, .2, .3, 2),
        'Conv2d_Activation_BatchNorm': (1, 4, 1),
        'Matmul_MaxPool_Sum_Scale': (4, 4, 2, 3),
        'Matmul_Swish_Scaling': (4, 4, 2),
        'Matmul_Dropout_Softmax': (4, 4, .7),
        'Conv3d_Softmax_MaxPool_MaxPool': (1, 4, 1, 2),
        'Conv2d_BatchNorm_Scaling': (1, 4, 1, 2),
        'Conv2d_Tanh_Scaling_BiasAdd_Max': (1, 4, 1, 2, (4, 1, 1), 2),
        'Conv2d_GroupNorm_Scale_MaxPool_Clamp': (1, 4, 1, 2, (4, 1, 1), 2, -.4, .7),
        'Matmul_Divide_GELU': (4, 4, 2),
        'Matmul_AvgPool_GELU_Scale_Max': (4, 4, 2, 3),
        'Matmul_GELU_Softmax': (4, 4),
        'VisionAttention': (4, 2),
        'MinGPTCausalAttention': (4, 2, .6, .7, 4),
        'MiniGPTBlock': (4, 2, .6, .7, 4),
    }
    def norm(v, axes):
        mean = v.mean(axes, keepdim=True)
        return (v - mean) / ((v - mean).square().mean(axes, keepdim=True) + 1e-5).sqrt()
    def softmax(v, dim):
        exp = (v - v.max(dim, keepdim=True).values).exp()
        return exp / exp.sum(dim, keepdim=True)
    def gelu(v):
        return .5 * v * (1 + (v / math.sqrt(2)).erf())
    def pool2(v, mean=False):
        rows = []
        for i in range(0, v.shape[-2], 2):
            cols = []
            for j in range(0, v.shape[-1], 2):
                block = v[..., i:i+2, j:j+2]
                cols.append(block.mean((-2, -1)) if mean else block.amax((-2, -1)))
            rows.append(torch.stack(cols, -1))
        return torch.stack(rows, -2)
    for filename in (args.module, args.functional):
        model = getattr(import_path(path.parent / filename), name)(*constructors[name]).eval()
        with torch.no_grad():
            if name.startswith('Matmul'):
                x = torch.tensor([[-1., 2., .5, 3.], [2., -1., 1., .2]])
                linear = getattr(model, 'matmul', getattr(model, 'linear', None))
                weights = torch.tensor([[1., 2., 0., -1.], [.5, 0., 1., 2.], [0., -1., 2., .5], [2., 1., -.5, 0.]])
                bias = torch.tensor([.2, -.3, .4, -.1])
                linear.weight.copy_(weights); linear.bias.copy_(bias)
                v = torch.stack([sum(x[:, c] * weights[r, c] for c in range(4)) + bias[r] for r in range(4)], 1)
                if name == 'Matmul_Swish_Sum_GroupNorm':
                    model.bias.copy_(bias)
                    z = v / (1 + (-v).exp()) + bias
                    expected = norm(z.reshape(2, 2, 2), (2,)).reshape(2, 4)
                elif name == 'Matmul_Scaling_ResidualAdd': expected = 3 * v
                elif name == 'Matmul_MaxPool_Sum_Scale': expected = 3 * v.reshape(2, 2, 2).max(-1).values.sum(-1)
                elif name == 'Matmul_Swish_Scaling': expected = 2 * v / (1 + (-v).exp())
                elif name == 'Matmul_Dropout_Softmax': expected = softmax(v, 1)
                elif name == 'Matmul_Divide_GELU': expected = gelu(v / 2)
                elif name == 'Matmul_AvgPool_GELU_Scale_Max': expected = (3 * gelu(v.reshape(2, 2, 2).mean(-1))).max(-1).values
                else: expected = softmax(gelu(v), 1)
            elif name.startswith('Conv'):
                is3d = name.startswith('Conv3d')
                x = torch.arange(512 if is3d else 16, dtype=torch.float32).reshape((1, 1, 8, 8, 8) if is3d else (1, 1, 4, 4)) / 10 - .7
                weights = torch.tensor([-.5, .75, 1.25, 2.])
                bias = torch.tensor([.1, -.2, .3, -.4])
                model.conv.weight.copy_(weights.reshape_as(model.conv.weight)); model.conv.bias.copy_(bias)
                shape = (1, 4) + (1,) * (x.ndim - 2)
                v = x * weights.reshape(shape) + bias.reshape(shape)
                if name == 'Conv2d_InstanceNorm_Divide': expected = norm(v, (-2, -1)) / 2
                elif name == 'Conv2d_Subtract_Tanh_Subtract_AvgPool': expected = pool2((v - .2).tanh() - .3, mean=True)
                elif name in ('Conv2d_Activation_BatchNorm', 'Conv2d_BatchNorm_Scaling'):
                    model.bn.running_mean.copy_(bias); model.bn.running_var.copy_(torch.tensor([.5, 1., 1.5, 2.]))
                    z = v * (1 + v.exp()).log().tanh() if name == 'Conv2d_Activation_BatchNorm' else v
                    expected = (z - bias.reshape(shape)) / (model.bn.running_var.reshape(shape) + model.bn.eps).sqrt()
                    if name == 'Conv2d_BatchNorm_Scaling': expected *= 2
                elif name == 'Conv2d_Tanh_Scaling_BiasAdd_Max':
                    model.bias.copy_(bias.reshape(4, 1, 1)); expected = pool2(2 * v.tanh() + bias.reshape(shape))
                elif name == 'Conv2d_GroupNorm_Scale_MaxPool_Clamp':
                    model.scale.copy_(torch.tensor([.5, 1.5, 2., .75]).reshape(4, 1, 1))
                    z = norm(v.reshape(1, 2, -1), (-1,)).reshape_as(v) * model.scale
                    expected = pool2(z).clamp(-.4, .7)
                else:
                    z = softmax(v, 1)
                    # Two stride-2 max pools equal max over each disjoint 4^3 cube.
                    expected = torch.stack([torch.stack([torch.stack([z[..., d:d+4, h:h+4, w:w+4].amax((-3, -2, -1)) for w in (0, 4)], -1) for h in (0, 4)], -2) for d in (0, 4)], -3)
            else:
                x = torch.tensor([[[-1., .2, 1., 2.], [2., -.5, .3, 1.], [1., 3., -.4, .5]]])
                if name == 'VisionAttention':
                    model.attn.in_proj_weight.zero_(); model.attn.in_proj_weight[8:].copy_(torch.eye(4)); model.attn.in_proj_bias.zero_()
                    model.attn.out_proj.weight.copy_(torch.eye(4)); model.attn.out_proj.bias.zero_()
                    v = norm(x + x.mean(1, keepdim=True), (-1,))
                    x = x.transpose(1, 2).reshape(1, 4, 1, 3)
                    expected = v.transpose(1, 2).reshape_as(x)
                else:
                    attn = model.attn if name == 'MiniGPTBlock' else model
                    attn.c_attn.weight.zero_(); attn.c_attn.weight[8:].copy_(torch.eye(4)); attn.c_attn.bias.zero_()
                    attn.c_proj.weight.copy_(torch.eye(4)); attn.c_proj.bias.zero_()
                    normalized = norm(x, (-1,)) if name == 'MiniGPTBlock' else x
                    causal = torch.stack([normalized[:, :i+1].mean(1) for i in range(3)], 1)
                    if name == 'MinGPTCausalAttention': expected = causal
                    else:
                        model.mlp.c_fc.weight.zero_(); model.mlp.c_fc.weight[:4].copy_(torch.eye(4)); model.mlp.c_fc.bias.zero_()
                        model.mlp.c_proj.weight.zero_(); model.mlp.c_proj.weight[:, :4].copy_(torch.eye(4)); model.mlp.c_proj.bias.fill_(.2)
                        residual = x + causal; v = norm(residual, (-1,))
                        expected = residual + .5 * v * (1 + (math.sqrt(2/math.pi) * (v + .044715 * v.pow(3))).tanh()) + .2
            before = x.clone(); state = copy.deepcopy(model.state_dict())
            actual = model(x)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            assert not torch.allclose(actual, torch.zeros_like(actual))
            torch.testing.assert_close(x, before, rtol=0, atol=0)
            assert actual.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_assign_score_scalar_and_vectorized_known_gradients():
    h = harness_namespace(ROOT / 'tasks/hip2hip/others/assign_score_withk')
    scores = torch.tensor([[[[2.], [2.]]]], requires_grad=True)
    points = torch.tensor([[[[3.]], [[5.]]]], requires_grad=True)
    centers = torch.tensor([[[[1.]], [[7.]]]], requires_grad=True)
    indices = torch.tensor([[[0, 1]]])
    for reference in (h.cpu_assign_score_withk_forward, h.cpu_assign_score_withk_forward_vectorized):
        actual = reference(scores, points, centers, indices)
        torch.testing.assert_close(actual, torch.tensor([[[[4., 8.]]]]))
        grads = torch.autograd.grad(actual.sum(), (scores, points, centers))
        for actual, expected in zip(grads, (torch.tensor([[[[2.], [4.]]]]), torch.tensor([[[[2.]], [[2.]]]]), torch.tensor([[[[-4.]], [[0.]]]]))):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('fault', ['none', 'wrong_gradient', 'missing_reset', 'input_mutation', 'exception'])
def test_assign_score_timed_backward_checks_and_restores_gradients(fault, monkeypatch):
    root = ROOT / 'tasks/hip2hip/others/assign_score_withk'
    helper = import_path(root / 'scripts/replay_validation.py')
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    inputs = [torch.tensor([2., 3.], requires_grad=True), torch.tensor([4., 5.], requires_grad=True), torch.tensor([6., 7.], requires_grad=True)]
    for value in inputs: value.grad = torch.zeros_like(value)
    pristine = [value.detach().clone() for value in inputs]
    buffers = [value.grad for value in inputs]
    expected = (torch.tensor([48., 105.]), torch.tensor([24., 35.]), torch.tensor([12., 21.]), torch.tensor([8., 15.]))
    output = tuple(torch.empty_like(value) for value in expected[:1]) + tuple(buffers)
    calls = []
    def reset():
        calls.append('reset')
        for value in inputs: value.grad.zero_()
    def invoke():
        calls.append('invoke')
        output[0].copy_(expected[0])
        for value, reference in zip(inputs, expected[1:]): value.grad.add_(reference)
        return output
    def check(actual):
        for value, reference in zip(actual, expected): torch.testing.assert_close(value, reference, rtol=1e-3, atol=1e-3)
    def benchmark(fn, **kwargs):
        assert kwargs['warmup'] == 10 and kwargs['repetition'] == 100
        kwargs['prepare_fn'](); observed = fn()
        def replay():
            assert all(torch.isnan(value).all() for value in observed)
            if fault != 'missing_reset': kwargs['prepare_fn']()
            actual = fn()
            with torch.no_grad():
                if fault == 'wrong_gradient': actual[1].zero_()
                if fault == 'input_mutation': inputs[0].add_(1)
                if fault == 'exception':
                    inputs[0].add_(1); inputs[1].grad = None
                    raise RuntimeError('deliberate captured backward failure')
            return actual
        kwargs['timed_run']._bind(replay, observed)
        return .5, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}
    def run():
        return helper.measure(benchmark, invoke, inputs, check, warmup=10, repetition=100, use_cuda_graph=True, prepare_fn=reset)
    if fault == 'none': assert run()[1]['replay_validation_valid']
    else:
        with pytest.raises((AssertionError, ValueError, RuntimeError)): run()
    for value, original, buffer in zip(inputs, pristine, buffers):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        assert value.grad is buffer and torch.equal(value.grad, torch.zeros_like(value))
    assert calls.count('invoke') == 2


def test_mla_reference_gate_rejects_consistent_wrong_replay_and_nan_under_fast_math(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None: pytest.skip('CPU C++ compiler unavailable; not GPU validation')
    root = ROOT / 'tasks/hip2hip/others/mla_decode'
    header = root / 'scripts/native/output_validation.hpp'
    program = tmp_path / 'check.cpp'
    program.write_text('''
#include <cassert>
#include <limits>
#include <vector>
#include "output_validation.hpp"
int main() {
    auto decode=[](float v){return v;};
    std::vector<float> correct{1.f,2.f}, wrong{0.f,0.f};
    assert(validate_mla_outputs(correct,correct,correct,decode).empty());
    assert(!validate_mla_outputs(wrong,wrong,correct,decode).empty());
    for(float value : {std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
        std::vector<float> bad{value,2.f};
        assert(!validate_mla_outputs(bad,bad,correct,decode).empty());
    }
    std::vector<float> boundary{1.09f,2.18f};
    assert(validate_mla_outputs(boundary,boundary,correct,decode).empty());
    assert(!validate_mla_outputs(boundary,correct,correct,decode).empty());
    std::vector<float> ones{1.f,1.f};
    assert(!validate_mla_outputs(wrong,wrong,ones,decode).empty());
}
''')
    subprocess.run([compiler, '-std=c++17', '-O3', '-ffast-math', '-I', str(header.parent), str(program), '-o', str(tmp_path / 'check')], check=True, timeout=60)
    subprocess.run([str(tmp_path / 'check')], check=True, timeout=10)


def test_mla_host_reference_nonzero_known_answer_with_remapped_tokens(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None: pytest.skip('CPU C++ compiler unavailable; not GPU validation')
    root = ROOT / 'tasks/hip2hip/others/mla_decode'
    text = (root / 'mla_decode.hip').read_text()
    decode = extract_cpp_function(text, '__host__ __device__ __forceinline__ float fp8_e4m3fn_to_f32')
    reference = extract_cpp_function(text, 'static void host_reference')
    preamble = '''
#include <vector>
#include <cstdint>
#include <cmath>
#include <cassert>
#define __host__
#define __device__
#define __forceinline__ inline
// The known answer is exactly representable; no rounding/GPU claim here.
using bf16=float;
float __float2bfloat16(float x){return x;}
float __bfloat162float(float x){return x;}
constexpr int NHEAD=128,LK=576,LV=512;
'''
    main = '''
int main(){
    std::vector<bf16> q(NHEAD*LK,0), out;
    // e4m3fn byte 0x38 is 1, 0x40 is 2, 0x48 is 4.
    std::vector<uint8_t> kv(3*LK,0x38);
    for(int d=0;d<LK;++d){kv[LK+d]=0x40;kv[2*LK+d]=0x48;}
    std::vector<int32_t> mapping{2,0},lengths{2};
    host_reference(out,q,kv,mapping,lengths,1,2,1.f/std::sqrt(float(LK)));
    for(float value:out)assert(value==2.5f);
    lengths[0]=1;
    host_reference(out,q,kv,mapping,lengths,1,2,1.f/std::sqrt(float(LK)));
    for(float value:out)assert(value==4.f);
}
'''
    source=tmp_path/'reference.cpp';source.write_text(preamble+decode+reference+main)
    subprocess.run([compiler,'-std=c++17','-O3','-ffast-math',str(source),'-o',str(tmp_path/'check')],check=True,timeout=60)
    subprocess.run([str(tmp_path/'check')],check=True,timeout=10)


@pytest.mark.parametrize('control_index', range(5))
def test_ball_query_mmcv_exact_zero_boundaries_order_and_padding(control_index):
    task = ROOT / 'tasks/hip2hip/others/ball_query'
    h = harness_namespace(task)
    controls = import_path(task / 'scripts/reference_checks.py')
    lower, upper, count, xyz, center, expected = list(controls.boundary_controls())[control_index]
    actual = h.cpu_reference(lower, upper, count, xyz, center)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Wrong-order and zero-output implementations must fail the same exact gate.
    wrong = actual.roll(1, dims=-1) if actual.unique().numel() > 1 else torch.zeros_like(actual)
    with pytest.raises(AssertionError):
        controls.close(wrong, expected)


def test_ball_query_index_validation_has_no_near_zero_or_padding_escape():
    task = ROOT / 'tasks/hip2hip/others/ball_query'
    h = harness_namespace(task)
    tree = ast.parse((task / 'scripts/task_runner.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_ball_query')
    ns = dict(vars(h))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<index contract>', 'exec'), ns)
    xyz = torch.tensor([[[1e-4, 0., 0.], [0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]])
    center = torch.zeros(1, 1, 3)
    check = ns['validate_ball_query']
    assert check(torch.tensor([[[1, 2, 1]]], dtype=torch.int32), xyz, center, 1., 2.)
    for wrong in ([[[0, 2, 1]]], [[[1, 3, 1]]], [[[2, 1, 1]]], [[[1, 2, 0]]]):
        assert not check(torch.tensor(wrong, dtype=torch.int32), xyz, center, 1., 2.)
    empty = torch.full((1, 2, 3), 10.)
    assert check(torch.zeros(1, 1, 3, dtype=torch.int32), empty, center, 1., 2.)
    assert not check(torch.ones(1, 1, 3, dtype=torch.int32), empty, center, 1., 2.)


CE_TASKS = [p for p in EXTENSIONS if 'CrossEntropyLossLabelSmoothing' in p.parent.name]


@pytest.mark.parametrize('path', CE_TASKS, ids=lambda p: p.parent.name)
def test_ce_analytic_smoothing_oracle_and_probability_axis(path):
    args = options(yaml.safe_load(path.read_text()))
    helper = import_path(path.parent / 'eval_tools/case_controls.py')
    module, functional = [import_path(path.parent / name) for name in (args.module, args.functional)]
    helper.self_test(module, functional, args.model_class)
    rows = json.loads(path.with_name('workload.json').read_text())['cases']
    helper.validate_controls(rows)
    model = getattr(module, args.model_class)()
    for inputs, row in zip(module.get_inputs(), rows):
        before = [value.clone() for value in inputs]
        rng = torch.random.get_rng_state().clone()
        assert helper.configure_models((model,), inputs) == row['params']['operator']
        torch.testing.assert_close(inputs[1].sum(-1), torch.ones_like(inputs[1][..., 0]), rtol=1e-5, atol=1e-6)
        assert not torch.allclose(inputs[1].sum(1), torch.ones_like(inputs[1][:, 0]), rtol=1e-5, atol=1e-6)
        assert 'smooth_dist' in model.state_dict()
        assert model.smooth_dist.shape == inputs[1].shape and model.smooth_dist.is_contiguous()
        actual = model(*inputs)
        expected = helper.reference(*inputs, model.smooth_eps, model.smooth_dist)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        for old, new in zip(before, inputs): torch.testing.assert_close(new, old, rtol=0, atol=0)
        assert torch.equal(torch.random.get_rng_state(), rng)
    invalid = copy.deepcopy(rows)
    for row in invalid: row['params']['operator']['smooth_eps'] = 0
    with pytest.raises(ValueError, match='varied nonzero'):
        helper.validate_controls(invalid)


@pytest.mark.parametrize('path', CE_TASKS, ids=lambda p: p.parent.name)
@pytest.mark.parametrize('role', ['baseline', 'candidate'])
@pytest.mark.parametrize('fault', ['none', 'ignore_smoothing', 'wrong_axis', 'mutate_target'])
def test_ce_actual_reference_and_timed_replay_negative_controls(path, role, fault, monkeypatch):
    args = options(yaml.safe_load(path.read_text()))
    controls = import_path(path.parent / 'eval_tools/case_controls.py')
    helper = import_path(path.parent / 'eval_tools/replay_validation.py')
    runner = import_path(path.parent / 'eval_tools/evaluate.py')
    monkeypatch.setitem(sys.modules, 'case_controls', controls)
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    module = import_path(path.parent / (args.module if role == 'baseline' else args.functional))
    model = getattr(module, args.model_class)()
    logits = torch.tensor([[[[0., math.log(2), math.log(4)], [math.log(4), 0., math.log(2)]]]])
    target = torch.tensor([[[[1., 0., 0.], [0., 1., 0.]]]])
    controls.apply_control(model, {'smooth_eps': .3}, [logits, target])
    initial = target.clone()
    def correct(input, target, **kwargs):
        return controls.reference(input, target, kwargs['smooth_eps'], kwargs['smooth_dist'])
    def benchmark(invoke, **kwargs):
        output = invoke().detach()
        def replay():
            assert torch.isnan(output).all()
            eps = 0. if fault == 'ignore_smoothing' else model.smooth_eps
            value = controls.reference(logits, target, eps, model.smooth_dist)
            if fault == 'wrong_axis':
                mixed = (1-eps)*target + eps*model.smooth_dist
                value = -(mixed * torch.log_softmax(logits, dim=-2)).sum(-2).mean()
            with torch.no_grad():
                output.copy_(value)
                if fault == 'mutate_target': target.mul_(.5)
            return output
        kwargs['timed_run']._bind(replay, output)
        return .2, {'benchmark_method':'cuda_event_fallback','benchmark_timed_run_kind':'eager_callable'}
    def policy(rtol=1e-4, atol=1e-5): pass
    perf = types.SimpleNamespace(cal_kernel_perf=policy, cal_modu_latency=None,
        benchmark_cuda_graph_or_events=benchmark, _compare_results=torch.allclose)
    helper.install(perf, runner.output_contract)
    def run():
        return perf.cal_modu_latency(model, [logits, target]) if role == 'baseline' else perf.cal_hip_latency(model, [logits, target], correct)
    if fault == 'none':
        _, meta = run()
        assert meta['model_state_validation_valid'] and meta['replay_validation_valid']
    else:
        with pytest.raises((ValueError, AssertionError)):
            run()
    torch.testing.assert_close(target, initial, rtol=0, atol=0)


def test_mla_input_bytes_and_finally_restore_on_native_validation_failures(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    header = ROOT / 'tasks/hip2hip/others/mla_decode/scripts/native/output_validation.hpp'
    source = tmp_path / 'restore.cpp'
    source.write_text('#include <vector>\n#include <cassert>\n#include <stdexcept>\n#include "' + str(header) + '"\n' + r'''
int main() {
    std::vector<float> original{1.f, -0.f}, values = original;
    assert(mla_same_buffer(values, original));
    values[1] = 0.f;
    assert(!mla_same_buffer(values, original)); // exact bytes, not FP tolerance
    values = original;
    int restored = 0;
    auto restore = [&]() { values = original; ++restored; return true; };
    auto good = [&]() { values[0] = 3.f; return std::string(); };
    assert(mla_with_restored_inputs(good, restore).empty());
    assert(restored == 1 && mla_same_buffer(values, original));
    auto failed = [&]() { values[0] = 7.f; return std::string("wrong replay"); };
    assert(mla_with_restored_inputs(failed, restore) == "wrong replay");
    assert(restored == 2 && mla_same_buffer(values, original));
    bool threw = false;
    try {
        mla_with_restored_inputs([&]() -> std::string {
            values[0] = 11.f;
            throw std::runtime_error("oracle failed");
        }, restore);
    } catch(const std::runtime_error&) { threw = true; }
    assert(threw && restored == 3 && mla_same_buffer(values, original));
    assert(!mla_with_restored_inputs(good, []() { return false; }).empty());
}
''')
    binary = tmp_path / 'restore'
    subprocess.run([compiler, '-std=c++17', '-O3', '-ffast-math', str(source), '-o', str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize('name', ['hip_11709_InnerProd.hip', 'hip_11709_InnerProd_ref.hip'])
def test_innerprod_native_reduction_preserves_cancellation_answer(name, tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    text = (ROOT / 'tasks/hip2hip/gpumode/InnerProd/hip' / name).read_text()
    def definition(marker):
        start = text.index(marker); opening = text.index('{', start)
        depth = 1; end = opening + 1
        while depth:
            depth += (text[end] == '{') - (text[end] == '}'); end += 1
        return text[start:end]
    code = '\n'.join(definition(marker) for marker in ('__device__ inline void smem_load_scale', '__global__ void k_forward(', '__global__ void k_forward_nosum(', '__global__ void k_forward_pixelwise('))
    preamble = r'''
#include <cassert>
#include <cstdint>
#include <vector>
#define __device__
#define __global__
#define __shared__
#define __syncthreads() ((void)0)
struct Index { int x; } threadIdx{0}, blockIdx{0}, blockDim{1}, gridDim{1};
float smem[512];
'''
    control = r'''
int main() {
  for(int channels : {3, 512}) {
    std::vector<float> image(channels, 1.f), sound(channels*2, 0.f), scale(channels, 1.f);
    image[0] = .5f; scale[0] = 2.f; image[1] = 2.f; scale[1] = .5f;
    sound[0] = 16777216.f; sound[2] = 1.f; sound[4] = -16777216.f;
    sound[1] = -16777216.f; sound[3] = -1.f; sound[5] = 16777216.f;
    float result[2] = {0, 0};
    float bias = .25f;
    auto replay = [&]() { k_forward(image.data(), sound.data(), scale.data(), &bias, result, 1, channels, 2, 1); };
    replay();
    assert(result[0] == 1.25f && result[1] == -.75f);
    // Fixed launch arguments retain the device pointer, not a host-read scalar.
    bias = -.5f;
    replay();
    assert(result[0] == .5f && result[1] == -1.5f);
    float pixels[2];
    k_forward_pixelwise(image.data(), sound.data(), scale.data(), &bias, pixels, 1, channels, 1, 1, 2, 1);
    assert(pixels[0] == .5f && pixels[1] == -1.5f);
    std::vector<float> nosum(channels*2);
    k_forward_nosum(image.data(), sound.data(), scale.data(), &bias, nosum.data(), 1, channels, 2, 1);
    assert(nosum[2] == .5f && nosum[3] == -1.5f);
  }
}
'''
    source = tmp_path / 'reduction.cpp';source.write_text(preamble + code + control)
    binary = tmp_path / 'reduction'
    subprocess.run([compiler, '-std=c++17', '-O3', str(source), '-o', str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=10)


NORM_AFFINE_TASKS = [p for p in EXTENSIONS if p.parent.name in {'layer_normalization', '11754_layer_normalization', 'l1n40_LayerNorm'}]


@pytest.mark.parametrize('path', NORM_AFFINE_TASKS, ids=lambda p: p.parent.name)
def test_normalization_affine_controls_match_reference_and_manifest(path):
    args = options(yaml.safe_load(path.read_text()))
    helper = import_path(path.parent / 'eval_tools/case_controls.py')
    module, functional = [import_path(path.parent / name) for name in (args.module, args.functional)]
    helper.self_test(module, functional, args.model_class)
    rows = json.loads(path.with_name('workload.json').read_text())['cases']
    helper.validate_controls(rows)
    ordinary = args.model_class == 'LayerNorm'
    for implementation in (module, functional):
        model = getattr(implementation, args.model_class)((2, 2) if ordinary else 4)
        for row in rows:
            original_rng = torch.random.get_rng_state().clone()
            helper.apply_control(model, row['params']['operator'])
            gamma, beta = helper.affine(model)
            torch.testing.assert_close(gamma.flatten(), torch.tensor([.7,.9,1.1,1.3]), rtol=0, atol=1e-7)
            amplitude = row['params']['operator']['beta_amplitude']
            torch.testing.assert_close(beta.flatten(), amplitude*torch.tensor([-1.25,1.5,-1.75,2.]), rtol=0, atol=0)
            assert torch.equal(torch.random.get_rng_state(), original_rng)
        with torch.device('meta'):
            init = module.get_init_inputs()
            meta_model = getattr(implementation, args.model_class)(init[0]) if ordinary else getattr(implementation, args.model_class)(**init[1])
            for inputs,row in zip(module.get_inputs(),rows):
                assert helper.configure_models((meta_model,),inputs) == row['params']['operator']
    invalid = copy.deepcopy(rows)
    for row in invalid: row['params']['operator']['beta_amplitude'] = 0
    with pytest.raises(ValueError, match='nonzero'):
        helper.validate_controls(invalid)


def test_mla_routed_manifest_controls_reject_ignored_mapping_and_lengths(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('CPU C++ compiler unavailable; not GPU validation')
    root = ROOT / 'tasks/hip2hip/others/mla_decode'
    source = (root / 'mla_decode.hip').read_text()
    decode = extract_cpp_function(source, '__host__ __device__ __forceinline__ float fp8_e4m3fn_to_f32')
    encode = extract_cpp_function(source, 'static uint8_t f32_to_fp8_e4m3fn')
    reference = extract_cpp_function(source, 'static void host_reference')
    main = r'''
#include "routing_controls.hpp"
#include "output_validation.hpp"
#include <cassert>
#include <set>
#include <cmath>
#define __host__
#define __device__
#define __forceinline__ inline
using bf16=float;
float __float2bfloat16(float x){return x;}
float __bfloat162float(float x){return x;}
constexpr int NHEAD=128,LK=576,LV=512;
'''+decode+encode+reference+r'''
int main(){
    const int shapes[][2]={{1,512},{4,1024},{16,2048},{1,4096},{1,8192}};
    for(const auto& shape:shapes){
        const int batch=shape[0],ctx=shape[1];
        std::vector<int32_t> mapping,lengths;
        mla_routing_inputs(mapping,lengths,batch,ctx,std::size_t(batch)*ctx+1024);
        assert(std::set<int32_t>(mapping.begin(),mapping.end()).size()==mapping.size());
        std::vector<uint8_t> kv((std::size_t(batch)*ctx+1024)*LK);
        const auto expected=mla_routing_known_answer(kv,mapping,lengths,batch,ctx,LK,f32_to_fp8_e4m3fn);
        std::vector<float> correct(batch),identity(batch),full_length(batch);
        for(int b=0;b<batch;++b){
            assert(lengths[b]>=1 && lengths[b]<=ctx);
            double good=0,wrong_map=0,wrong_length=0;
            for(int t=0;t<ctx;++t){
                const auto mapped=fp8_e4m3fn_to_f32(kv[std::size_t(mapping[std::size_t(b)*ctx+t])*LK]);
                wrong_length+=mapped;
                if(t<lengths[b]){
                    good+=mapped;
                    wrong_map+=fp8_e4m3fn_to_f32(kv[(std::size_t(b)*ctx+t)*LK]);
                }
            }
            correct[b]=good/lengths[b];identity[b]=wrong_map/lengths[b];full_length[b]=wrong_length/ctx;
            assert(std::fabs(correct[b]-expected[b])<1e-6);
        }
        auto identity_decode=[](float x){return x;};
        assert(validate_mla_outputs(correct,correct,expected,identity_decode).empty());
        assert(!validate_mla_outputs(identity,identity,expected,identity_decode).empty());
        assert(!validate_mla_outputs(full_length,full_length,expected,identity_decode).empty());
    }
    // Execute the independent actual host attention reference on a small
    // remapped case; verify the same analytical control, not only the generator.
    const int batch=4,ctx=8;
    std::vector<int32_t> mapping,lengths;
    mla_routing_inputs(mapping,lengths,batch,ctx,batch*ctx+1024);
    std::vector<uint8_t> kv((batch*ctx+1024)*LK);
    auto expected=mla_routing_known_answer(kv,mapping,lengths,batch,ctx,LK,f32_to_fp8_e4m3fn);
    std::vector<bf16> q(batch*NHEAD*LK,0),out;
    host_reference(out,q,kv,mapping,lengths,batch,ctx,1.f/std::sqrt(float(LK)));
    for(int b=0;b<batch;++b)for(int i=0;i<NHEAD*LV;++i)
        assert(std::fabs(out[b*NHEAD*LV+i]-expected[b])<1e-5);
}
'''
    program = tmp_path / 'routing.cpp'
    program.write_text(main)
    subprocess.run([compiler, '-std=c++17', '-O3', '-ffast-math', '-I', str(root / 'scripts/native'), str(program), '-o', str(tmp_path / 'check')], check=True, timeout=60)
    subprocess.run([str(tmp_path / 'check')], check=True, timeout=20)
    manifest = json.loads((root / 'workload.json').read_text())
    assert [r['test_case_id'] for r in manifest['cases']] == [f'shape_{i}' for i in range(5)]
    assert [(r['params']['batch'],r['params']['ctx']) for r in manifest['cases']] == [(1,512),(4,1024),(16,2048),(1,4096),(1,8192)]


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/GELU', 'torch2hip/gpumode/14539_GELU'])
def test_gelu_reference_independent_of_shared_production_operator(relative, monkeypatch):
    root = ROOT / 'tasks' / relative
    helper = import_path(root / 'eval_tools/replay_validation.py')
    args = options(yaml.safe_load((root / 'config.yaml').read_text()))
    module = getattr(import_path(root / args.module), args.model_class)().eval()
    functional = getattr(import_path(root / args.functional), args.model_class)().eval()
    values = torch.tensor([[-6., -3., -2.5, -1., 0.], [.5, 1., 2.5, 3., 6.]]).T
    expected = torch.tensor([[.5*v*(1+math.erf(v/math.sqrt(2))) for v in row] for row in values.tolist()])
    actual = helper.gelu_reference(values, chunk_size=3)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    helper.reference_self_test(module, functional)
    # Both old implementations share F.gelu. A defect in that common operator
    # must now fail even if the two implementations still agree with each other.
    monkeypatch.setattr(torch.nn.functional, 'gelu', lambda x, **kw: torch.zeros_like(x))
    torch.testing.assert_close(helper.gelu_reference(values), expected, rtol=1e-4, atol=1e-5)
    with pytest.raises(AssertionError):
        helper.reference_self_test(module, functional)
    approximate = .5*values*(1+torch.tanh(math.sqrt(2/math.pi)*(values+.044715*values**3)))
    assert not torch.allclose(approximate, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('relative', ['hip2hip/gpumode/GELU', 'torch2hip/gpumode/14539_GELU'])
@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_gelu_timing_rejects_shared_wrong_gelu(relative, role, monkeypatch):
    root = ROOT / 'tasks' / relative
    helper = import_path(root / 'eval_tools/replay_validation.py')
    runner = import_path(root / 'eval_tools/evaluate.py')
    timed = import_path(ROOT / 'src/tools/perf/aka_benchmark.py')
    monkeypatch.setitem(sys.modules, '_aka_benchmark', timed)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    def benchmark(invoke, **kwargs):
        output = invoke()
        kwargs['timed_run']._bind(lambda: output.zero_(), output)
        return .1, {'benchmark_method':'cuda_graph', 'benchmark_timed_run_kind':'captured_graph'}
    def cal_kernel_perf(rtol=1e-4, atol=1e-5): pass
    perf = types.SimpleNamespace(cal_kernel_perf=cal_kernel_perf, benchmark_cuda_graph_or_events=benchmark,
                                 cal_modu_latency=None, _compare_results=torch.allclose)
    helper.install(perf, runner.output_contract)
    def wrong(x, fn=None): return torch.zeros_like(x)
    with pytest.raises(ValueError, match='protected reference'):
        perf.cal_hip_latency(wrong, [torch.tensor([-1.,.5,2.])], None if role=='baseline' else wrong)
