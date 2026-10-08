"""CPU validation of all actual control geometries and timing selection policy."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

torch.set_num_threads(min(torch.get_num_threads(), 8))
REPO = Path(__file__).resolve().parents[1]
TASKS = REPO/'tasks/headkernel'
sys.path.insert(0, str(TASKS/'minimax-m3__decode_score_kernel/ut'))
from workload_controls import RecordedControls, flat, validate_scope_reports, paired_report_costs
from evaluation_contract import fingerprint, validate_report
from minimax_data import Inputs, work_controls
from minimax_fixtures import geometry, load_bundle

KINDS = [('decode_score', 'minimax-m3__decode_score_kernel'),
         ('sparse_decode', 'minimax-m3__gqa_share_sparse_decode_kernel'),
         ('sparse_prefill', 'minimax-m3__gqa_share_sparse_fwd_kernel')]

@pytest.fixture(scope="module")
def captured_geometry_root():
    value = os.environ.get("MINIMAX_CONTROL_FIXTURES_ROOT")
    if not value:
        pytest.skip("Set MINIMAX_CONTROL_FIXTURES_ROOT to the verified captured-geometry directories")
    return Path(value)


def first_case():
    return json.loads((TASKS/KINDS[0][1]/'cases.json').read_text())['cases'][0]


def state_hash(states):
    digest = hashlib.sha256()
    for state in states:
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize('kind,name', KINDS)
def test_every_actual_control_setting_has_legal_immutable_donor_geometry(kind, name, captured_geometry_root):
    task = TASKS/name
    manifest = json.loads((task/'cases.json').read_text())
    definition = json.loads((task/'task_definition.json').read_text())
    count = 0
    for case in manifest['cases']:
        distribution = RecordedControls(case)
        specs = {name: spec for name, spec in case['tensors'].items() if spec['role'] == 'input'}
        entries = load_bundle(captured_geometry_root/kind/'prepared/task', case, definition)
        states = [geometry(entry, specs, set()) for entry in entries]
        unchanged = state_hash(states)
        inputs = Inputs.__new__(Inputs)
        inputs.case, inputs.definition = case, definition
        inputs.scalars = {k: v for k, v in case['scalars'].items() if not k.startswith(('result', 'work.'))}
        inputs.tensors = {key: torch.empty(spec['shape'], dtype=getattr(torch, spec['dtype']), device='meta') for key, spec in specs.items()}
        for index, variant in enumerate(distribution.variant_ids):
            donor, exact = distribution.geometry(states, variant)
            assert donor in range(len(states))
            assert exact['seq_lens'].tolist() == flat(distribution.controls(variant)['inputs.seq_lens'])
            refreshed = inputs._fresh_geometry(exact, index + 719)
            inputs.validate_geometry(refreshed)
            observed = {'inputs.'+k.removeprefix('work.'): v for k, v in work_controls(refreshed).items()}
            assert observed == distribution.controls(variant)
            count += 1
        assert state_hash(states) == unchanged
    assert count == {'decode_score': 2048, 'sparse_decode': 3072, 'sparse_prefill': 46}[kind]


def test_integer_occurrence_weights_assign_every_ticket_exactly(monkeypatch):
    case = first_case()
    old = RecordedControls(case)
    ids = sorted({fingerprint(state["tensor_controls"]) for state in case["states"]})
    case['work_distribution'] = {key: deepcopy(case['work_distribution'][key]) for key in ids}
    for key, weight in zip(ids, (1, 3)):
        case['work_distribution'][key]['occurrences'] = weight
    case['occurrences'] = 4
    distribution = RecordedControls(case)
    result = []
    for ticket in range(4):
        monkeypatch.setattr('random.Random.randrange', lambda self, total, chosen=ticket: chosen)
        result.append(distribution.choose(19))
    assert Counter(result) == {ids[0]: 1, ids[1]: 3}


def test_scope_reports_reject_omitted_settings_and_incomplete_timing_draws():
    case = first_case(); distribution = RecordedControls(case)
    policy = {'correctness_seeds': [0, 1, 2], 'warmup_iterations': 10, 'benchmark_iterations': 100}
    manifest = {'cases': [case], 'measurement': policy}
    coverage = distribution.coverage(policy, distribution.targeted_variants(), distribution.original_plan(policy))
    row = {'case': case, 'workload_control_coverage': coverage}
    validate_scope_reports(manifest, [row], 'correctness')
    row['workload_control_coverage']['targeted_variant_ids'].pop()
    with pytest.raises(ValueError, match='original/targeted'):
        validate_scope_reports(manifest, [row], 'correctness')
    draws = [distribution.choose(781+i) for i in range(110)]
    sampling = distribution.sampling(781, draws, policy, [0.25]*100)
    assert len(sampling['measured_variant_ids']) == 100
    assert sampling['exhaustive_timing_claim'] is False
    row = {'case': case, 'workload_control_sampling': sampling, 'samples_ms': [0.25]*100}
    with pytest.raises(ValueError, match='private challenge'):
        validate_scope_reports(manifest, [row], 'performance', challenge_seed=782)
    validate_scope_reports(manifest, [row], 'performance', challenge_seed=781)
    row['workload_control_sampling']['measured_variant_ids'].pop()
    with pytest.raises(ValueError, match='draws'):
        validate_scope_reports(manifest, [row], 'performance', challenge_seed=781)


def test_uncaptured_geometry_changes_fail_closed():
    case = first_case()
    # Remove the maximum donor: existing exact minimum stays available, but
    # intermediate lengths must not be generated from a shorter paging donor.
    shortest = min(case['states'], key=lambda s: max(flat(s['tensor_controls']['inputs.seq_lens'])))
    case['states'] = [shortest]
    with pytest.raises(ValueError, match='donor covers'):
        RecordedControls(case)


def fake_evaluation():
    spec = importlib.util.spec_from_file_location('exact_control_runner', TASKS/KINDS[0][1]/'scripts/task_runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    evaluation = runner.CaseEvaluation.__new__(runner.CaseEvaluation)
    evaluation.case = first_case()
    distribution = RecordedControls(evaluation.case)
    events = []
    state = {'valid': False, 'variant': None}
    inputs = SimpleNamespace(external=False, control_distribution=distribution)
    def reset(seed, *, control_variant=None):
        variant = distribution.choose(seed) if control_variant is None else control_variant
        inputs.current_control_variant = state['variant'] = variant
        events.append(('reset', variant, seed))
        return (variant, seed)
    def reset_recorded(seed):
        index = seed % len(evaluation.case['states'])
        variant = fingerprint(evaluation.case['states'][index]['tensor_controls'])
        inputs.current_state = index
        inputs.current_control_variant = None
        state['variant'] = variant
        events.append(('reset_recorded', variant, seed))
        return (variant, seed)
    inputs.reset_recorded = reset_recorded
    def initialize():
        state['valid'] = False
        events.append(('poison', state['variant']))
    def replay():
        state['valid'] = True
        events.append(('replay', state['variant']))
    def verify(before, *, corrupt=False):
        events.append(('verify', state['variant'], corrupt, state['valid']))
        if corrupt or not state['valid']:
            raise AssertionError('CPU callback simulation rejects invalid output')
    inputs.reset = reset
    evaluation.inputs = inputs
    evaluation.initialize = initialize
    evaluation.graph = SimpleNamespace(replay=replay)
    evaluation.verify = verify
    evaluation.observe = lambda: evaluation.case
    evaluation.measure = lambda call: (call(), 0.25)[1]
    return evaluation, events


def test_runner_preserves_original_checks_and_adds_six_targeted_values():
    evaluation, events = fake_evaluation()
    policy = {'correctness_seeds': [0, 1, 2]}
    result = evaluation.correctness(policy)
    distribution = evaluation.inputs.control_distribution
    ids = distribution.targeted_variants()
    assert len(ids) == 6
    resets = [(v, s) for action, v, *rest in events if action == 'reset' for s in rest]
    assert Counter(resets) == Counter((v, seed) for v in ids for seed in (0, 1, 2, 1000))
    original = [(v, s) for action, v, *rest in events if action == 'reset_recorded' for s in rest]
    expected_original = [(fingerprint(evaluation.case['states'][seed % 2]['tensor_controls']), seed) for seed in (0, 1, 2, 1000)]
    assert original == expected_original
    noops = [event[1] for event in events if event[0] == 'verify' and not event[3]]
    corrupt = [event[1] for event in events if event[0] == 'verify' and event[2]]
    expected_controls = Counter(ids) + Counter([expected_original[-1][0]])
    assert Counter(noops) == expected_controls and Counter(corrupt) == expected_controls
    evidence = result['workload_control_coverage']
    assert evidence['targeted_variant_seed_pairs'] == 18
    assert evidence['targeted_negative_control_checks'] == 12
    assert evidence['represented_variant_count'] == 128
    assert len(evidence['correctness_tested_variant_ids']) == 6
    assert len(evidence['untested_correctness_variant_ids']) == 122
    assert evidence['all_recorded_values_checked_in_correctness'] is False
    assert evidence['exhaustive_per_variant_seed_control_matrix'] is False


def test_runner_retains_all_110_checked_replays_and_100_device_samples():
    from paired_cpu_backend import run_cpu_pair
    pair = run_cpu_pair(159)
    assert pair.events.count('prepare') == 110
    assert pair.events.count('candidate_initialize') == 110
    assert pair.events.count('candidate_replay') == 110
    assert pair.events.count('reference_replay') == 110
    assert len(pair.row['samples_ms']) == 100 and pair.row['oracle_checks'] == 100
    assert len(pair.row['workload_control_sampling']['measured_variant_ids']) == 100
    assert pair.row['workload_control_sampling']['exhaustive_timing_claim'] is False



@pytest.mark.parametrize('kind,name', KINDS)
def test_targeted_numeric_boundaries_and_complete_sampling_support(kind, name):
    manifest = json.loads((TASKS/name/'cases.json').read_text())
    for case in manifest['cases']:
        distribution = RecordedControls(case)
        assert set(distribution.donors) == set(case['work_distribution'])
        assert distribution.cumulative[-1] == case['occurrences']
        assert all(row['occurrences'] > 0 for row in case['work_distribution'].values())
        selected = distribution.targeted_variants()
        if kind == 'sparse_prefill':
            assert selected == []
            continue
        values = sorted(max(flat(distribution.controls(v)['inputs.seq_lens'])) for v in distribution.variant_ids)
        actual = [max(flat(distribution.controls(v)['inputs.seq_lens'])) for v in selected]
        assert len(values) == 128
        assert actual == [values[i] for i in (0, 1, 63, 64, 126, 127)]


def test_original_resets_match_baseline_storage_bytes_and_state_choice():
    # A small CPU tensor case exercises the exact original reset body, aliases
    # and random seeds without allocating the multi-GB GPU task storage.
    # The prefill adapter retains the original shared reset implementation.
    baseline = TASKS/KINDS[2][1]/'ut/minimax_data.py'
    text = baseline.read_text()
    namespace = {'__name__': 'baseline_minimax_data'}
    exec(compile(text, 'baseline_minimax_data.py', 'exec'), namespace)
    spec = importlib.util.spec_from_file_location('synthetic_fixtures', REPO/'tests/test_minimax_served_contract.py')
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    import served_contract
    import minimax_data
    args = fixtures.args_for(torch, 'sparse_decode')
    case, definition = fixtures.case_for(torch, {'served_contract': served_contract, 'minimax_data': minimax_data}, args)
    second = deepcopy(case['states'][0])
    second['geometry']['req_to_token'] = served_contract.encode_bytes(((args['req_to_token']+3)%16).contiguous().view(torch.uint8).numpy().tobytes())
    case['states'].append(second)
    old = namespace['Inputs'](case, definition, device='cpu')
    new = Inputs(case, definition, device='cpu')
    for seed in (0, 1, 2, 1000):
        first, second = old.reset(seed), new.reset_recorded(seed)
        assert old.current_state == new.current_state == seed % 2
        assert first.keys() == second.keys()
        assert all(torch.equal(first[key], second[key]) for key in first)


def performance_report(case, policy, samples, seed, request_id):
    distribution = RecordedControls(case)
    draws = [distribution.choose(seed+i) for i in range(110)]
    row = {'case': case, 'correct': True, 'samples_ms': samples, 'warmup_iterations': 10,
           'fresh_input_resets': 100, 'output_initializations': 100, 'oracle_checks': 100,
           'benchmark_method': 'cuda_graph', 'workload_control_sampling': distribution.sampling(seed, draws, policy, samples)}
    full = json.loads((TASKS/KINDS[0][1]/'cases.json').read_text())
    # The original manifest validator requires capture counts/IDs to describe
    # this reduced CPU-only test fixture precisely.
    full['cases'] = [case]
    full['capture']['required_case_ids'] = [case['case_id']]
    full['capture']['target_calls'] = full['capture']['represented_calls'] = case['occurrences']
    request = {'phase': 'performance', 'manifest_sha256': fingerprint(full), 'challenge_seed': seed,
               'request_id': request_id, 'source_sha256': {'source/kernel.py': 'a'*64}, 'package_sha256': 'b'*64}
    report = {'schema_version': 1, 'status': 'ok', 'request': request, 'cases': [row]}
    return full, request, report


def test_paired_metric_is_ratio_of_raw_means_and_rejects_tampering():
    case = first_case()
    policy = {'warmup_iterations': 10, 'benchmark_iterations': 100}
    manifest, req_a, report_a = performance_report(case, policy, [1.0, 100.0]*50, 913, 'reference')
    other, req_b, report_b = performance_report(case, policy, [1.0, 10.0]*50, 913, 'candidate')
    assert manifest == other
    expected = {'reference': req_a, 'candidate': req_b}
    result = paired_report_costs(report_a, report_b, manifest, expected)
    row = result['case_results'][0]
    assert row['reference_mean_ms'] == 50.5 and row['candidate_mean_ms'] == 5.5
    assert row['reference_over_candidate'] == 50.5/5.5
    assert row['reference_over_candidate'] != 5.5  # mean of per-draw ratios
    forged = deepcopy(report_b)
    forged['cases'][0]['workload_control_sampling']['mean_gpu_cost_ms'] = 0.01
    with pytest.raises(ValueError, match='timing control evidence'):
        paired_report_costs(report_a, forged, manifest, expected)
    forged = deepcopy(report_b)
    forged['request']['request_id'] = 'foreign'
    with pytest.raises(ValueError, match='stale or foreign'):
        paired_report_costs(report_a, forged, manifest, expected)
