"""Admit sealed captured and generated fixtures without conflating their provenance."""
import hashlib
import json
from pathlib import Path
import re


CAPTURED = 'actual_served_capture'
GENERATED = 'generated_native_graph_replay'
MIXED = 'FROZEN_CAPTURE_AND_GENERATED'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coverage(root, manifest):
    root = Path(root).resolve()
    relative = Path(manifest['coverage_manifest'])
    path = root / relative
    require(not relative.is_absolute() and '..' not in relative.parts and not path.is_symlink()
            and path.resolve().is_relative_to(root), 'Untrusted coverage path')
    require(sha(path) == manifest['coverage_sha256'], 'Sealed coverage changed')
    value = json.loads(path.read_text())
    require(value['schema'] == 'deepseek-current-captured-generated-coverage-v1'
            and value['run_id'] == manifest['run_id'] and value['seam'] == manifest['seam']
            and value['current_input_set_sealed'] is True, 'Coverage source/run/seal differs')
    return value


def validate_fixture_manifest(root, manifest):
    status = manifest.get('status')
    require(status in ('FROZEN_CURRENT_CAPTURE', MIXED), 'Current fixtures and expected cases are not sealed')
    if status == 'FROZEN_CURRENT_CAPTURE':
        require(all(c.get('provenance_kind', CAPTURED) == CAPTURED for c in manifest['cases']),
                'Generated cases require an explicit mixed fixture contract')
        return
    require(manifest['seam'] in ('moe1', 'moe2')
            and manifest.get('generated_work_control_supplement') is True
            and manifest.get('original_served_work_control_gate_passed') is False,
            'Mixed fixtures must retain their generated scope and original capture gap')
    sealed = coverage(root, manifest)
    require(sealed['original_served_work_control_gate_passed'] is False
            and sealed['generated_supplement_is_actual_other_rank_tensor_capture'] is False,
            'Generated coverage cannot claim missing-rank served capture')
    cases = manifest['cases']
    require(len({c['case_id'] for c in cases}) == len(cases)
            and {c['case_id'] for c in cases} == set(sealed['fixture_records']),
            'Case list differs from the sealed coverage set')
    for case in cases:
        ref = case['fixture']
        expected = {'kind': case['provenance_kind'], 'path': ref['path'],
                    'sha256': ref['sha256'], 'stage': case['stage']}
        require(expected['kind'] in (CAPTURED, GENERATED), 'Unknown fixture provenance kind')
        if expected['kind'] == GENERATED:
            expected['native_replay_fixture_sha256'] = case['native_replay_fixture_sha256']
        require(sealed['fixture_records'][case['case_id']] == expected,
                'Fixture identity/kind/stage differs from sealed coverage')
    generated = [c for c in cases if c['provenance_kind'] == GENERATED]
    require(len(generated) == sealed['supplemental_native_repeatability']['scenario_count']
            and len(generated) > 0, 'Generated case coverage is incomplete')


def validate_fixture_record(root, case, manifest, cfg, record):
    require(record.get('source_sha256') == manifest['native_source_sha256']
            == cfg['native_source_sha256'], 'Stale fixture native source identity')
    require(record.get('startup_values') is False, 'Startup data are not runtime fixtures')
    kind = case.get('provenance_kind', CAPTURED)
    if kind == CAPTURED:
        require(record.get('provenance', {}).get('run_id') == manifest['run_id'],
                'Stale captured fixture run identity')
        stage = record.get('served', {}).get('stage')
        require(stage in ('prefill', 'decode') and case.get('stage', stage) == stage,
                'Captured fixture stage differs')
        require(record.get('origin') != GENERATED, 'Generated fixture mislabeled as captured')
        if stage == 'decode':
            require(record.get('origin') == 'served_graph',
                    'Decode fixture must follow actual served graph replay')
        return
    require(kind == GENERATED and manifest.get('status') == MIXED,
            'Generated fixture requires explicit mixed admission')
    validate_fixture_manifest(root, manifest)
    sealed = coverage(root, manifest)
    require(record.get('origin') == GENERATED and 'served' not in record
            and case['stage'] == 'decode' and record['case_key'] == case['case_id']
            and record['family'] == 'deepseek_' + manifest['seam'], 'Generated fixture identity/stage differs')
    provenance = record['provenance']
    require(provenance['parent_run_id'] == manifest['run_id']
            and provenance['runtime_image'] == manifest['runtime_image'] == cfg['image']
            and provenance['routing_generated'] is True
            and provenance['missing_rank_tensor_capture'] is False
            and provenance['parent_fixture_rank'] == 0
            and record['original_capture_gate_repaired'] is False,
            'Generated parent/image/provenance scope differs')
    require(re.fullmatch('[0-9a-f]{64}', provenance['input_plan_sha256']) is not None,
            'Generated input plan must be hash-bound')
    parent = provenance['parent_fixture']
    require(any(row['kind'] == CAPTURED and row['stage'] == 'decode'
                and row['path'] == 'fixtures/' + parent['file'] and row['sha256'] == parent['sha256']
                for row in sealed['fixture_records'].values()), 'Generated parent capture is not sealed')
    execution = record['execution']
    require(execution == case['generated_scenario']
            and execution['mode'] == 'isolated_native_graph'
            and execution['scenario_id'] + '-' + manifest['seam'] == case['case_id']
            and execution['graph_replays'] >= 16
            and execution['graph_replays'] == sealed['supplemental_native_repeatability']['repeats']
            and execution['seed'] in sealed['supplemental_native_repeatability']['seeds']
            and execution['num_valid_ids'][0] in sealed['supplemental_observed_work_counts'],
            'Generated replay parameters differ from sealed evidence')
    require(case['occurrences'] == 1
            and case['occurrence_basis'] == 'one_generated_fixture_not_a_served_launch_count',
            'Generated fixture count is not a served launch frequency')
    projection = record['metadata_projection']
    require(projection['schema'] == 'generated-work-summary-projection-v1'
            and projection['source_fixture_sha256'] == case['native_replay_fixture_sha256']
            and projection['source_fixture'] == case['case_id'] + '.json'
            and projection['tensor_metadata_controls_execution_and_payload_unchanged'] is True
            and projection['GPU_outputs_recomputed'] is False,
            'Generated metadata projection is not bound to its native replay')
    implementation = provenance['native_implementation']
    if manifest['seam'] == 'moe2':
        require(implementation['kind'] == 'pinned_image_native'
                and implementation['sha256'] == manifest['native_source_sha256']
                and implementation['module'] == cfg['module']
                and implementation['function'] == cfg['function'], 'Generated native implementation differs')
    else:
        references = {relative: sha(Path(root) / entry['reference'])
                      for relative, entry in cfg['editable_sources'].items()}
        require(implementation['kind'] == 'separately_staged_source_bound_implementation'
                and implementation['production_wrapper_sha256'] == manifest['native_source_sha256']
                and implementation['binding']['source_sha256'] == references
                and implementation['binding']['emitter_source_sha256'] in references.values()
                and implementation['binding']['host_wrapper_frozen'] is True,
                'Generated repaired implementation differs from the frozen reference')


def validate_generated_inputs(case, inputs):
    """Check actual restored control bytes before candidate execution."""
    if case.get('provenance_kind') == GENERATED:
        actual = inputs['num_valid_ids'].detach().cpu().tolist()
        require(actual == case['generated_scenario']['num_valid_ids'],
                'Generated work-control tensor differs from sealed replay')
