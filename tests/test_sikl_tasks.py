"""Protected operator semantics and packaging invariants for the 26 SIKL tasks."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
SIKL_ROOT = ROOT / 'tasks' / 'Aiter-task'
TASKS = sorted(p.parent for p in SIKL_ROOT.glob('*/config.yaml'))
# These existing tasks retain their main-branch protocol while their baseline
# numerical findings and timed-output validation policy remain unresolved.
DEFERRED_TIMING_TASKS = {
    'gemm_a16w16_nt_n4096_k2048',
    'gemm_a16w16_nt_n6144_k3072',
    'gemm_a16w16_nt_n16384_k2048',
}
ROTATING_TASKS = [t for t in TASKS if t.name not in DEFERRED_TIMING_TASKS]
SPLIT_TEMPLATE_FILES = {'README.md', 'scripts/task_inputs.py', 'scripts/task_measure.py'}
VAR_AXIS = {'gemm': 'm', 'moe': 'num_tokens', 'mhc': 'tokens', 'topk': 'batch', 'mla': 'batch'}
# Families whose workload cases are exactly the bundle's 13 rows. Top-k and
# MLA also vary the valid lengths per row; their own test modules cover them.
BUNDLE_ROW_FAMILIES = {'gemm', 'moe', 'mhc'}
_AITER_SOURCE = [{'kind': 'image', 'image_path': '/sgl-workspace/aiter/aiter',
                  'destination': 'aiter_source/aiter',
                  'exclude': ['jit/build', 'jit/flydsl_cache', '__pycache__']}]
SOURCE_ACQUISITION = {
    'gemm': _AITER_SOURCE, 'moe': _AITER_SOURCE, 'mhc': _AITER_SOURCE,
    'topk': [{'kind': 'image', 'image_path': '/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4',
              'destination': 'sglang_source/kernels/ops/attention/dsv4', 'exclude': ['__pycache__']},
             {'kind': 'image', 'image_path': '/sgl-workspace/sglang/python/sglang/kernels/jit/csrc/deepseek_v4',
              'destination': 'sglang_source/kernels/jit/csrc/deepseek_v4'}],
    'mla': [{'kind': 'image', 'image_path': '/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa',
             'destination': 'sglang_source/kernels/ops/attention/dsa', 'exclude': ['__pycache__']}],
}
# The MLA definitions differ in their reference and baseline bindings, so those
# callbacks are pinned per task rather than shared by the family.
PER_TASK_CALLBACKS = {'mla': {'scripts/task_reference.py', 'scripts/task_baseline.py'}}
SHARED_TEMPLATE_FILES = (
    'kernel.py', 'test_kernel_harness.py', 'scripts/task_inputs.py',
    'scripts/task_initialize.py', 'scripts/task_compare.py', 'scripts/task_reference.py',
    'scripts/task_baseline.py', 'scripts/task_measure.py', 'scripts/task_contract.py',
    'scripts/evaluate.py', 'scripts/export_solution.py', 'scripts/task_validation.py', 'README.md',
)
# SHA256 from the original PR's bundled callbacks before schema migration.
# A migration must not quietly change input distributions or acceptance gates.
CALLBACK_SHA256 = {
    'gemm': {
        'task_baseline': 'eeaf2f83a06a298b2b7c0fe2c2cad75bd4cb4da89ddf6b583f31a6b3de6de66e',
        'task_compare': 'b3237750010516954281047cc26b7045d9a276ed5322f2104a5cecb770f89b1a',
        'task_initialize': '60688988b28b137ee8cfe328077d4c4dba6d7bf5835a002062f03a88a298b89b',
        'task_reference': '4ce9f4075562e14cd48ec6e6c20e6f80deae18f34e213dd6891432001314c663',
    },
    'moe': {
        'task_baseline': '3d27b3fe67b8d030aa1fd13287fcdef7d68e657a47a169469292c22f1dc4600e',
        'task_compare': '37f0e083eb7654da9d3f36da03d3e9648d0845de9ca1ec1282da96d3d4db708c',
        'task_initialize': '80e50d81def9ac0c6f85390155e23eec48c173e8a6b754eaa465e9ed65f2120e',
        'task_reference': 'de096e2726bb4e81e4e748748b044bcaf99576070ace1f0b1662eee763f5d072',
    },
    # Verbatim callbacks of the deepseek-v4-flash bundle definition
    # mhc_fused_post_pre_flat_rmsnorm_c4_d4096 and its baseline solution.
    'mhc': {
        'task_baseline': 'a9b382a087108a834fc8b1d0a97db8eb13ce339aafe6c3175492c4648212e7c5',
        'task_compare': '68fed5cc0e5ebd128c57faf7dff01269157c99399e99e437b40d0a7530901508',
        'task_initialize': '974d427b993949c55a983857d8b7821c52dc2070de7d63cb16b7a79945b64863',
        'task_reference': '58ba2fc0f93e65dfa2d3693a877a202733aeffea79424304c7ca718c1235ee6e',
    },
    # deepseek-v4-flash bundle topk_transform_paged_paged_k512_page_size64. The
    # baseline differs from the bundle only by its sglang entry-point lookup.
    'topk': {
        'task_baseline': 'e96d3e217a56f6acffb967926ef1c3ebc4cdcb9b57654e462351a8206ea79d6f',
        'task_compare': '3371b93c48bf588efb2862ecf0d0975217088b5a202a861bf8775a2f2fe980a2',
        'task_initialize': 'f7af24a49e9771170911d619463932ea6fb57345b1bd2e5a8cf0babc4ffa9f00',
        'task_reference': '7e1e8421dad6172e55003553076e1a73ab2aeafa41cf019d7d6877e1be8de33d',
    },
    # Verbatim callbacks and baselines of the three deepseek-v4-flash bundle
    # flash_mla_with_kvcache definitions.
    'mla': {
        'flash_mla_with_kvcache_dsv4_fp8_10011_q1_h64_d512_p256_k128': {
            'task_baseline': '926f57554a42c119979e122173474cc04dcc718304830c5fc16fce6ea02ba534',
            'task_compare': '72b817d46e7b8b43806063801c29f038bd2d0badb82789ae8cff0e2563fd9ebf',
            'task_initialize': '10c29e00fe79b60f0b95589c488c6cd784171fba826251fa7ab8357309f748a4',
            'task_reference': '5c3cd40597de69a5fc921bcaed49b08fa690e0281479764ca3b96545ece20ddb',
        },
        **{name: {
            'task_baseline': '2bad75d98748e1b559f7288bcb5ee730a04ac4e7a1abffdf53305d203dc6630b',
            'task_compare': '72b817d46e7b8b43806063801c29f038bd2d0badb82789ae8cff0e2563fd9ebf',
            'task_initialize': '10c29e00fe79b60f0b95589c488c6cd784171fba826251fa7ab8357309f748a4',
            'task_reference': '3205e074974193f2eade0410901b52e0e93cab034a10e53efa8ed2a3f44ea845',
        } for name in ('flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep2_ek8256',
                       'flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep64_ek512')},
    },
}


def _config(task):
    return yaml.safe_load((task / 'config.yaml').read_text())


def _workload(task):
    return json.loads((task / 'workload.json').read_text())


def test_suite_keeps_all_26_tasks_and_1374_cases():
    assert len(TASKS) == 26
    assert sum(_workload(t)['op_type'] == 'gemm' for t in TASKS) == 17
    assert sum(_workload(t)['op_type'] == 'moe' for t in TASKS) == 4
    assert sum(_workload(t)['op_type'] == 'mhc' for t in TASKS) == 1
    assert sum(_workload(t)['op_type'] == 'topk' for t in TASKS) == 1
    assert sum(_workload(t)['op_type'] == 'mla' for t in TASKS) == 3
    assert sum(len(_workload(t)['cases']) for t in TASKS) == 1374


@pytest.mark.parametrize('op_type', sorted(VAR_AXIS))
@pytest.mark.parametrize('relative', SHARED_TEMPLATE_FILES)
def test_family_copies_are_identical(op_type, relative):
    if relative in PER_TASK_CALLBACKS.get(op_type, ()):
        pytest.skip('Pinned per task by test_original_callbacks_and_case_sampling_are_unchanged')
    tasks = [t for t in TASKS if _workload(t)['op_type'] == op_type]
    contents = [(t / relative).read_bytes() for t in tasks]
    if relative == 'README.md':
        # Per-task baseline evidence is documented after the shared contract.
        # Executable callbacks and the common instructions remain identical.
        contents = [value.split(b'\n## Production baseline numerical evidence\n')[0].rstrip()
                    for value in contents]
    groups = {}
    for task, content in zip(tasks, contents):
        cohort = task.name in DEFERRED_TIMING_TASKS if relative in SPLIT_TEMPLATE_FILES else False
        groups.setdefault(cohort, set()).add(content)
    assert all(len(values) == 1 for values in groups.values())


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_one_v2_config_and_no_agent_dependency(task):
    spec = load_task_spec(task / 'config.yaml', task_id=str(task.relative_to(ROOT / 'tasks')))
    assert spec.candidate.language == 'flydsl'
    assert spec.candidate.initial_state == 'unimplemented'
    assert spec.baseline.kind == 'provided'
    assert len(spec.actions) == 7
    config = _config(task)
    assert 'task_type' not in config and 'rewrite_source_file' not in config
    assert not (task / 'scripts/forge_driver.py').exists()
    assert not (task / 'definition.yaml').exists()
    assert len(spec.candidate.entrypoints) == 1
    assert spec.candidate.entrypoints[0].kind == 'builder'
    assert 'builder_symbol' not in _workload(task)  # One declaration, no derivation.
    for path in task.rglob('*.py'):
        source = path.read_text()
        assert 'KERNELFORGE_' not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] not in {'src', 'agents', 'kernelforge'}
            elif isinstance(node, ast.Import):
                assert not {'src', 'agents', 'kernelforge'} & {n.name.split('.')[0] for n in node.names}


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_production_source_is_separate_and_documented(task):
    config = _config(task)
    acquisitions = config['workspace']['sources']
    assert acquisitions == SOURCE_ACQUISITION[_workload(task)['op_type']]
    owner = config['kernel_identity']['source_owner']
    assert all(a['destination'].startswith(f'{owner}_source/') for a in acquisitions)
    destinations = [acquisition['destination'] for acquisition in acquisitions]
    for source in config['baseline']['source_files']:
        assert any(source.startswith(destination + '/') for destination in destinations)
        assert source in (task / 'README.md').read_text()
        assert source not in config['candidate']['editable']


def test_image_acquisition_excludes_generated_jit_trees_but_keeps_sources(tmp_path, monkeypatch):
    from src import task_materialization as materialization

    root = tmp_path / 'image_aiter' / 'aiter'
    excluded = [root / 'jit/build', root / 'jit/flydsl_cache', root / '__pycache__']
    generated = {
        'jit/build/module/build/module.so': b'compiled shared object',
        'jit/build/module/build/kernel.cuda.o': b'compiled device object',
        'jit/flydsl_cache/case/unreadable.pkl': b'compiled runtime cache',
        '__pycache__/module.pyc': b'bytecode cache',
    }
    retained = {
        'jit/__init__.py': b'jit_source = True\n',
        'jit/core.py': b'compiler_source = True\n',
        'jit/build_helpers/helper.py': b'host_helper = True\n',
        'jit/module.so': b'adjacent installed module is not excluded',
        'tuned_gemm.py': b'production_source = True\n',
        'fused_moe.py': b'moe_production_source = True\n',
        'ops/mhc.py': b'mhc_production_source = True\n',
        'ops/flydsl/gemm_kernels.py': b'kernel_source = True\n',
        'configs/model_configs/tuned.csv': b'M,N,K\n1,32,6144\n',
    }
    for name, data in {**generated, **retained}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    aiter_task = next(t for t in TASKS if _config(t)['kernel_identity']['source_owner'] == 'aiter')
    source = dict(_config(aiter_task)['workspace']['sources'][0], image_path=str(root))
    outside = root.parent / '3rdparty/composable_kernel/include/ck.hpp'
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b'repository-only header')
    original_digest = materialization._file_digest

    def digest(path, deadline):
        assert not path.is_relative_to(outside.parent), 'Repository-only headers must not be read'
        if any(path.is_relative_to(directory) for directory in excluded):
            raise PermissionError('Cached runtime artifact must not be read')
        return original_digest(path, deadline)

    monkeypatch.setattr(materialization, '_file_digest', digest)
    _, manifest = materialization._image_input(source, materialization._Deadline(10))
    assert set(retained) <= manifest.keys()
    assert not any(any((root / name).is_relative_to(directory) for directory in excluded)
                   for name in manifest)

    # Exercise actual copying, not just declaration or manifest assertions.
    workspace, state = tmp_path / 'materialized', tmp_path / 'copy-evidence'
    destination = workspace / source['destination']
    destination.mkdir(parents=True)
    state.mkdir()
    materialization._copy_tree(root, destination, manifest, materialization._Deadline(10), state)
    for name, data in retained.items():
        assert (destination / name).read_bytes() == data
    for name, data in generated.items():
        assert not (destination / name).exists()
        assert (root / name).read_bytes() == data  # Installed image tree is untouched.
    assert not (workspace / 'aiter_source/3rdparty').exists()
    for task in TASKS:
        if _config(task)['kernel_identity']['source_owner'] != 'aiter':
            continue
        for declared in _config(task)['baseline']['source_files']:
            assert (workspace / declared).is_file(), declared
    copied = materialization._tree_manifest(destination, materialization._Deadline(10))
    materialization._verify_copy(manifest, copied, root, destination)


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_original_callbacks_and_case_sampling_are_unchanged(task):
    workload = _workload(task)
    kind = workload['op_type']
    hashes = CALLBACK_SHA256[kind][task.name] if kind in PER_TASK_CALLBACKS else CALLBACK_SHA256[kind]
    for module, sha in hashes.items():
        assert hashlib.sha256((task / 'scripts' / f'{module}.py').read_bytes()).hexdigest() == sha
    cases = workload['cases']
    if kind in BUNDLE_ROW_FAMILIES:
        assert [case[VAR_AXIS[kind]] for case in cases] == [2**i for i in range(13)]
        assert len({case['uuid'] for case in cases}) == 13
        for case in cases:
            assert set(case) == {'case_id', 'uuid', VAR_AXIS[kind]}
    else:
        assert sorted({case[VAR_AXIS[kind]] for case in cases}) == [2**i for i in range(13)]
        assert len({case['uuid'] for case in cases}) == len(cases)
    assert workload['seed'] == 0
    assert workload['bench'] == {'warmup': 20, 'repetition': 100, 'target_ms': 1.0}
    assert workload['gate_policy']
    for forbidden in ('gate_multiplier', 'gate_floor', 'atol', 'rtol', 'snr_threshold'):
        assert forbidden not in workload
    inputs = (task / 'scripts/task_inputs.py').read_text()
    if kind == 'mla':
        # One full callback run per batch shape; further draws run the callback
        # without its optional extra pool (tests/test_sikl_mla.py).
        assert 'task_initialize.run(self.inputs, seed=SEED)' in inputs
        assert '**scratch}, seed=seed)' in inputs
    elif task.name in DEFERRED_TIMING_TASKS:
        assert 'task_initialize.run(inputs, seed=SEED)' in inputs
        assert 'task_initialize.run(inputs, seed=REFILL_SEED)' in inputs
        assert 'def refill_case_inputs(inputs: dict[str, Any])' in inputs
    else:
        assert 'task_initialize.run(inputs, seed=SEED)' in inputs
        assert 'task_initialize.run(inputs, seed=seed)' in inputs
        assert 'def refill_case_inputs(inputs: dict[str, Any], seed: int)' in inputs
    assert 'task_compare.run(got, expected)' in inputs


@pytest.mark.parametrize('task', ROTATING_TASKS, ids=lambda t: t.name)
def test_timing_protocol_checks_the_timed_invocations_themselves(task):
    # A captured kernel can decide on device, per invocation, whether to compute:
    # keyed on its inputs' values (return a stored result for a draw it has
    # seen) or on its own output buffer (skip while nobody has touched it). The
    # protocol therefore rotates fresh draws through the samples, checks the
    # outputs of secretly chosen samples and of invocations over draws never read
    # before, and holds the reported time to what those unseen draws cost. The
    # draws come from seeds the code being measured cannot know in advance.
    measure = (task / 'scripts/task_measure.py').read_text()
    inputs = (task / 'scripts/task_inputs.py').read_text()
    assert 'class RotatingDraws' in measure
    assert 'prepare_fn=rotation' in measure
    assert 'timed.after_sample = checks' in measure
    assert 'seeds = fresh_draw_seeds(TIMED_DRAWS + UNSEEN_DRAWS)' in measure
    assert 'secrets.SystemRandom()' in measure
    assert 'run_unseen_draws(timed, rotation, unseen)' in measure
    assert 'verify_timed_outputs(inputs, checks.kept + unseen_kept)' in measure
    assert 'if repeats != 1:' in measure
    # No invocation is singled out for checking by state prepared for it.
    assert 'float("nan")' not in measure and '.rerun()' not in measure
    if _workload(task)['op_type'] == 'mla':
        assert 'def draws(self, case: dict[str, Any], seeds: list[int])' in inputs
    else:
        assert 'def redraw_call_varying_inputs(inputs: dict[str, Any], seed: int)' in inputs
    assert 'PERSISTENT_INPUTS' in inputs
    assert 'REFILL_SEED' not in inputs and 'SEED + ' not in inputs
    ns: dict = {}
    for name in ('TIMED_DRAWS', 'UNSEEN_DRAWS', 'UNSEEN_DRAW_MARGIN', 'CHECKED_SAMPLES'):
        line = next(l for l in measure.splitlines() if l.startswith(f'{name} = '))
        exec(line, ns)
    assert ns['TIMED_DRAWS'] >= 2 and ns['UNSEEN_DRAWS'] >= 1 and ns['CHECKED_SAMPLES'] >= 1
    assert ns['UNSEEN_DRAW_MARGIN'] > 1.0


@pytest.mark.parametrize('task', TASKS, ids=lambda t: t.name)
def test_stub_and_unfilled_solution_are_not_accepted_results(task):
    assert not any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in ast.parse((task / 'kernel.py').read_text()).body)
    template = json.loads((task / 'solution.json').read_text())
    assert template['definition'] == _workload(task)['definition']
    assert template['spec']['entry_point'] == ''
    assert template['sources'] == [{'path': '', 'content': ''}]
    assert template['spec']['target'] == [{'arch': 'gfx950', 'hardware_id': 'MI355X'}]
    config = _config(task)
    assert config['exports'][0]['output'] != 'solution.json'
    assert config['exports'][0]['command'] == ['python3', 'scripts/export_solution.py']
    for path in ('_aka_benchmark.py', 'scripts/_aka_benchmark.py'):
        assert not (task / path).exists()  # Canonical helper is materialized, not forked.
