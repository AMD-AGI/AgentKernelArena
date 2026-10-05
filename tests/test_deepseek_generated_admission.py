"""Generated DS fixtures require a separate sealed, source-bound admission path."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


TASKS = Path(__file__).resolve().parents[1] / 'tasks/headkernel'
NAMES = ['deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4',
         'deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl']


@pytest.fixture(params=NAMES)
def bundle(request):
    task = TASKS / request.param
    path = task / 'ut/fixture_admission.py'
    spec = importlib.util.spec_from_file_location('admission_' + request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((task / 'cases.json').read_text())
    config = json.loads((task / 'provenance/SOURCE.json').read_text())
    return module, task, manifest, config


def record_for(module, task, manifest, config, case):
    """Small metadata fixture with the final sealed case's real parent/source identity."""
    sealed = module.coverage(task, manifest)
    parent = next(row for row in sealed['fixture_records'].values()
                  if row['kind'] == module.CAPTURED and row['stage'] == 'decode')
    if manifest['seam'] == 'moe2':
        implementation = {'kind': 'pinned_image_native', 'sha256': config['native_source_sha256'],
                          'module': config['module'], 'function': config['function']}
    else:
        sources = {key: module.sha(task / entry['reference']) for key, entry in config['editable_sources'].items()}
        implementation = {'kind': 'separately_staged_source_bound_implementation',
                          'production_wrapper_sha256': config['native_source_sha256'],
                          'binding': {'source_sha256': sources, 'emitter_source_sha256': next(iter(sources.values())),
                                      'host_wrapper_frozen': True}}
    return {'case_key': case['case_id'], 'family': 'deepseek_' + manifest['seam'],
            'origin': module.GENERATED, 'source_sha256': config['native_source_sha256'],
            'startup_values': False, 'original_capture_gate_repaired': False,
            'execution': copy.deepcopy(case['generated_scenario']),
            'provenance': {'parent_run_id': manifest['run_id'], 'runtime_image': manifest['runtime_image'],
                           'routing_generated': True, 'missing_rank_tensor_capture': False,
                           'parent_fixture_rank': 0, 'input_plan_sha256': 'a' * 64,
                           'parent_fixture': {'file': Path(parent['path']).name, 'sha256': parent['sha256']},
                           'native_implementation': implementation},
            'metadata_projection': {'schema': 'generated-work-summary-projection-v1',
                                    'source_fixture_sha256': case['native_replay_fixture_sha256'],
                                    'source_fixture': case['case_id'] + '.json',
                                    'tensor_metadata_controls_execution_and_payload_unchanged': True,
                                    'GPU_outputs_recomputed': False}}


def test_all_six_generated_scenarios_have_explicit_admission(bundle):
    module, task, manifest, config = bundle
    module.validate_fixture_manifest(task, manifest)
    generated = [case for case in manifest['cases'] if case['provenance_kind'] == module.GENERATED]
    assert len(generated) == 6
    for case in generated:
        record = record_for(*bundle, case)
        assert 'served' not in record and 'run_id' not in record['provenance']
        module.validate_fixture_record(task, case, manifest, config, record)


@pytest.mark.parametrize('field,value', [
    ('parent_run_id', 'stale-run'), ('runtime_image', 'unreviewed-image'),
    ('routing_generated', False), ('missing_rank_tensor_capture', True), ('parent_fixture_rank', 7),
])
def test_generated_parent_and_scope_tampering_fails(bundle, field, value):
    module, task, manifest, config = bundle
    case = next(c for c in manifest['cases'] if c['provenance_kind'] == module.GENERATED)
    record = record_for(*bundle, case)
    record['provenance'][field] = value
    with pytest.raises(RuntimeError):
        module.validate_fixture_record(task, case, manifest, config, record)


@pytest.mark.parametrize('attack', ['served_label', 'source', 'parent_fixture', 'native_implementation',
                                   'execution', 'projection', 'startup', 'captured_kind'])
def test_generated_fixtures_cannot_bypass_existing_identity_evidence(bundle, attack):
    module, task, manifest, config = bundle
    case = copy.deepcopy(next(c for c in manifest['cases'] if c['provenance_kind'] == module.GENERATED))
    record = record_for(*bundle, case)
    if attack == 'served_label':
        record['served'] = {'stage': 'decode'}
    elif attack == 'source':
        record['source_sha256'] = '0' * 64
    elif attack == 'parent_fixture':
        record['provenance']['parent_fixture']['sha256'] = '0' * 64
    elif attack == 'native_implementation':
        record['provenance']['native_implementation']['kind'] = 'unbound_implementation'
    elif attack == 'execution':
        record['execution']['graph_replays'] = 1
    elif attack == 'projection':
        record['metadata_projection']['source_fixture_sha256'] = '0' * 64
    elif attack == 'startup':
        record['startup_values'] = True
    elif attack == 'captured_kind':
        case['provenance_kind'] = module.CAPTURED
    with pytest.raises(RuntimeError):
        module.validate_fixture_record(task, case, manifest, config, record)


@pytest.mark.parametrize('attack', ['unsealed', 'capture_only', 'changed_coverage', 'missing_case',
                                   'changed_kind', 'changed_stage', 'claim_served_gate'])
def test_mixed_manifest_requires_exact_sealed_case_set_and_provenance(bundle, attack):
    module, task, manifest, _ = bundle
    if attack == 'unsealed':
        manifest['status'] = 'PENDING_CURRENT_TENSOR_CAPTURE'
    elif attack == 'capture_only':
        manifest['status'] = 'FROZEN_CURRENT_CAPTURE'
    elif attack == 'changed_coverage':
        manifest['coverage_sha256'] = '0' * 64
    elif attack == 'missing_case':
        manifest['cases'].pop()
    elif attack == 'changed_kind':
        manifest['cases'][-1]['provenance_kind'] = module.CAPTURED
    elif attack == 'changed_stage':
        manifest['cases'][-1]['stage'] = 'prefill'
    elif attack == 'claim_served_gate':
        manifest['original_served_work_control_gate_passed'] = True
    with pytest.raises(RuntimeError):
        module.validate_fixture_manifest(task, manifest)


def test_captured_decode_still_requires_its_own_run_and_served_graph(bundle):
    module, task, manifest, config = bundle
    case = next(c for c in manifest['cases'] if c['provenance_kind'] == module.CAPTURED and c['stage'] == 'decode')
    record = {'source_sha256': config['native_source_sha256'], 'startup_values': False,
              'provenance': {'run_id': manifest['run_id']}, 'served': {'stage': 'decode'}, 'origin': 'served_graph'}
    module.validate_fixture_record(task, case, manifest, config, record)
    record['origin'] = 'served_eager'
    with pytest.raises(RuntimeError, match='actual served graph'):
        module.validate_fixture_record(task, case, manifest, config, record)


def test_actual_restored_work_tensor_is_checked_before_execution(bundle):
    module, _, manifest, _ = bundle
    case = next(c for c in manifest['cases'] if c['provenance_kind'] == module.GENERATED)
    values = case['generated_scenario']['num_valid_ids'].copy()
    tensor = SimpleNamespace()
    tensor.detach = lambda: tensor
    tensor.cpu = lambda: tensor
    tensor.tolist = lambda: values
    module.validate_generated_inputs(case, {'num_valid_ids': tensor})
    values[0] += 32
    with pytest.raises(RuntimeError, match='work-control tensor differs'):
        module.validate_generated_inputs(case, {'num_valid_ids': tensor})


def test_task_local_admission_helpers_remain_identical():
    files = [(TASKS / name / 'ut/fixture_admission.py').read_bytes() for name in NAMES]
    assert files[0] == files[1]


@pytest.mark.parametrize('name', NAMES)
def test_generated_admission_does_not_accept_wrong_numerical_outputs(name):
    import ast
    torch = pytest.importorskip('torch')
    tree = ast.parse((TASKS / name / 'scripts/task_runner.py').read_text())
    compare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'compare')
    namespace = {}
    exec(compile(ast.Module(body=[compare], type_ignores=[]), '<protected-compare>', 'exec'), namespace)
    expected = torch.tensor([1.0, -2.0, 3.0])
    namespace['compare'](expected.clone(), expected, 0.02)
    with pytest.raises(AssertionError, match='values differ'):
        namespace['compare'](torch.zeros_like(expected), expected, 0.02)
