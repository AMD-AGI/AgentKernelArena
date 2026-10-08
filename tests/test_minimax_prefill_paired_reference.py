"""CPU checks for the unchanged prefill sampler and paired graph lifecycle."""
from copy import deepcopy
import json

import pytest
from paired_prefill_cpu_backend import TASK, run_cpu_pair, task_helpers


def test_all_40_original_prefill_cases_keep_modulo_state_sampling():
    manifest = json.loads((TASK / 'cases.json').read_text())
    assert len(manifest['cases']) == 40
    assert sum(len(case['states']) == 2 for case in manifest['cases']) == 6
    contract, _, states = task_helpers(TASK)
    for case in manifest['cases']:
        result = run_cpu_pair(142, case=case)
        receipt = result.row['workload_state_sampling']
        assert receipt['warmup_state_indices'] + receipt['measured_state_indices'] == [
            (142 + i) % len(case['states']) for i in range(110)]
        states.validate_state_sampling(case, result.request, manifest['measurement'],
                                       result.row, result.row['realized_input_signatures'])
        assert result.events.count('candidate_replay') == result.events.count('reference_replay') == 110
        assert len(result.row['samples_ms']) == 100


def test_prefill_graph_setup_and_observation_are_symmetric():
    result = run_cpu_pair()
    for role in ('candidate', 'reference'):
        assert [item['capture'] for item in result.setup[role]] == [False, False, False, True]
        assert len({item['stream'] for item in result.setup[role]}) == 1
    assert result.setup['candidate'][0]['stream'] != result.setup['reference'][0]['stream']
    assert result.events.index('candidate_snapshot') < result.events.index('reference_setup_replay')
    starts = [i for i, event in enumerate(result.events) if event == 'prepare'] + [len(result.events)]
    for start, end in zip(starts, starts[1:]):
        events = result.events[start:end]
        assert events.index('candidate_snapshot') < events.index('reference_restore')
        assert events.index('candidate_immutable') < events.index('reference_replay')
        assert events.index('reference_snapshot') < events.index('reference_clear')
        assert events.index('reference_clear') < events.index('independent_math_and_compare')
    assert result.outputs['reference'].storage.scrubbed


@pytest.mark.parametrize('failure', ['candidate_input', 'restore', 'reference_input', 'compare', 'reference_setup'])
def test_prefill_failure_paths_scrub_reference_storage(failure):
    with pytest.raises(ValueError):
        run_cpu_pair(failure=failure)


@pytest.mark.parametrize('fail_at', [1, 2, 4])
@pytest.mark.parametrize('failure_kind', ['launch_count', 'launch_contract'])
def test_real_operator_scrubs_new_reference_output_before_failed_attestation(fail_at, failure_kind):
    from paired_prefill_cpu_backend import run_real_operator_failure
    result = run_real_operator_failure(TASK, fail_at, failure_kind)
    assert result['all_reference_storage_bytes_scrubbed']


@pytest.mark.parametrize('fail_at', [1, 2])
def test_real_operator_preserves_original_failure_if_reference_cleanup_raises(fail_at):
    from paired_prefill_cpu_backend import run_real_operator_failure
    result = run_real_operator_failure(TASK, fail_at, cleanup_failure=True)
    assert result['original_attestation_exception_preserved']


def test_reference_output_ownership_does_not_change_candidate_invocation():
    from paired_prefill_cpu_backend import run_real_operator_failure
    result = run_real_operator_failure(TASK, reference=False)
    assert not result['all_reference_storage_bytes_scrubbed']


@pytest.mark.parametrize('corruption', ['missing', 'wrong_observed_state', 'mean', '99_samples'])
def test_prefill_state_receipt_rejects_semantic_and_timing_forgery(corruption):
    manifest = json.loads((TASK / 'cases.json').read_text())
    case = next(case for case in manifest['cases'] if len(case['states']) == 2)
    contract, _, states = task_helpers(TASK)
    result = run_cpu_pair(142, case=case)
    row = deepcopy(result.row)
    receipt = row['workload_state_sampling']
    if corruption == 'missing':
        del row['workload_state_sampling']
    elif corruption == 'wrong_observed_state':
        index = 1 - receipt['measured_state_indices'][0]
        receipt['measured_state_indices'][0] = index
        variant = contract.fingerprint(case['states'][index]['tensor_controls'])
        receipt['measured_variant_ids'][0] = variant
        signature = row['realized_input_signatures'][10]
        signature['variant_id'] = signature['actual_tensor_controls_sha256'] = variant
        receipt['schedule_fingerprint'] = contract.fingerprint({
            'case_id': case['case_id'], 'seed': 142,
            'state_indices': receipt['warmup_state_indices'] + receipt['measured_state_indices'],
            'variant_ids': receipt['warmup_variant_ids'] + receipt['measured_variant_ids']})
    elif corruption == 'mean':
        receipt['mean_gpu_cost_ms'] = 0.01
    else:
        row['samples_ms'].pop()
    with pytest.raises((ValueError, KeyError)):
        states.validate_state_sampling(case, result.request, manifest['measurement'],
                                       row, row['realized_input_signatures'])
