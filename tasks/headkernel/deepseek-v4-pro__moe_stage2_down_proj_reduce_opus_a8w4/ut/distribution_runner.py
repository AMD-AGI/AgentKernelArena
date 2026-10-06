"""Additional distribution cases; original fixed-case runner paths stay intact."""
import json
import os
from pathlib import Path
import time

from work_distribution import Provider, correctness_variants, digest, require


def progress(root, case_id, event, value=None):
    record = {'schema': 'distribution-progress-v1', 'case_id': case_id, 'event': event,
              'work_value': value, 'scoreable': False, 'observed_unix': time.time()}
    with (Path(root) / 'build/distribution_progress.log').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n'); stream.flush(); os.fsync(stream.fileno())


def validate_correctness_rows(group, rows, policy, mode=None):
    _, required_rows = correctness_variants(group, mode)
    expected = {row['variant_id']: row for row in required_rows}
    require(len(rows) == len(expected) and {row['variant_id'] for row in rows} == set(expected),
            'a required correctness setting was omitted or duplicated')
    for row in rows:
        require(row['num_valid_ids'] == expected[row['variant_id']]['num_valid_ids']
                and row['correct'] is True and row['seeds'] == policy['correctness_seeds'], 'per-setting oracle/seed coverage differs')
        require(row['negative_controls'] == {name: True for name in policy['negative_controls']},
                'per-setting negative controls incomplete')


def run_case(api, root, manifest, base_manifest, registry, case, phase, request,
             module, fn, reference_module, reference_fn, identity, reference_identity,
             correctness_mode=None):
    import torch
    from snapshots import raw_storage
    group = registry['groups'][case['distribution_group']]
    selected_mode = correctness_mode or group['correctness']['default_mode']
    required_rows = []
    if phase == 'correctness':
        selected_mode, required_rows = correctness_variants(group, selected_mode)
    parent = next(row for row in base_manifest['cases'] if row['case_id'] == group['base_case_id'])
    policy, tolerance = manifest['measurement'], manifest['tolerance']
    progress(root, case['case_id'], 'start')
    inputs, unused_golden = api['fixture'](parent, base_manifest, module)
    reference_inputs, unused_reference = api['fixture'](parent, base_manifest, reference_module)
    del unused_golden, unused_reference
    pristine = api['storage_snapshots'](inputs)
    initial_out = api['cpu_clone'](inputs.get('out'))
    provider = Provider(root, group, registry, inputs, pristine, api)
    first = group['histogram'][0]['num_valid_ids'][0]
    truth = provider.refresh(policy['correctness_seeds'][0], forced=first)
    output = api['invoke'](fn, inputs)
    api['verify_after_snapshot'](output, inputs, truth, reference_fn, reference_inputs, tolerance)
    tensors, scalars = api['runtime_abi'](inputs, output); api['observe_case'](case, tensors, scalars)
    compiled = {'case_id': case['case_id'], 'candidate_binding': identity,
                'reference_binding': reference_identity, 'invoked_and_synchronized': True,
                'distribution_compile_setting': first, 'correctness_mode': selected_mode,
                'all_observed_settings_correctness': False}
    if phase == 'compile':
        progress(root, case['case_id'], 'compile_completed')
        del inputs, reference_inputs, output, pristine, initial_out, provider, truth
        torch.cuda.empty_cache()
        return None, compiled
    for _ in range(3):
        provider.refresh(policy['correctness_seeds'][0], forced=first)
        api['invoke'](fn, inputs)
    provider.refresh(policy['correctness_seeds'][0], forced=first)
    torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = api['invoke'](fn, inputs)
    forced = None

    def reset_inputs(seed):
        return provider.refresh(seed, forced=forced)

    def initialize_outputs():
        if initial_out is not None:
            inputs['out'].copy_(initial_out)
        mutable = inputs['out'].untyped_storage().data_ptr() if initial_out is not None else None
        for value in api['leaves'](output):
            if value.untyped_storage().data_ptr() != mutable:
                raw_storage(value).fill_(0xAA)

    def observe():
        tensors, scalars = api['runtime_abi'](inputs, output)
        return api['observe_case'](case, tensors, scalars)

    def verify(expected):
        api['verify_after_snapshot'](output, inputs, expected, reference_fn, reference_inputs, tolerance)

    def measure(call):
        start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
        start.record(); call(); stop.record(); stop.synchronize(); return float(start.elapsed_time(stop))

    producer_start = provider.producer_calls
    provider.draws.clear()
    if phase == 'correctness':
        rows = []
        for observed in required_rows:
            forced = observed['num_valid_ids'][0]
            for seed in policy['correctness_seeds']:
                expected = reset_inputs(seed); initialize_outputs(); observe(); graph.replay(); verify(expected)
            controls = {}
            for control in policy['negative_controls']:
                expected = reset_inputs(request['challenge_seed']); initialize_outputs()
                if control == 'wrong_output':
                    graph.replay(); torch.cuda.synchronize()
                    for value in api['leaves'](output):
                        raw_storage(value).zero_()
                elif control != 'no_op':
                    raise RuntimeError('Unknown required negative control')
                try:
                    verify(expected)
                except AssertionError:
                    controls[control] = True
                else:
                    raise RuntimeError('Required negative control escaped: ' + control)
            rows.append({'variant_id': observed['variant_id'], 'num_valid_ids': observed['num_valid_ids'],
                         'correct': True, 'seeds': policy['correctness_seeds'], 'negative_controls': controls})
            progress(root, case['case_id'], 'correctness_setting_completed', forced)
        validate_correctness_rows(group, rows, policy, selected_mode)
        expected_preparations = len(required_rows) * (len(policy['correctness_seeds']) + len(policy['negative_controls']))
        require(len(provider.draws) == expected_preparations, 'setting preparation count differs')
        result = {'case': observe(), 'correct': True, 'seeds': policy['correctness_seeds'],
                  'negative_controls': {name: True for name in policy['negative_controls']},
                  'correctness_variants': rows,
                  'observed_work_distribution': {'histogram_sha256': group['histogram_sha256'],
                      'correctness_mode': selected_mode, 'correctness_settings_checked': len(rows),
                      'observed_setting_count': len(group['histogram']),
                      'all_observed_settings_correctness': len(rows) == len(group['histogram']),
                      'all_observed_settings_timed': False,
                      'actual_other_rank_routing_recovered': False,
                      'reference_generated_outputs_not_parent_goldens': True}}
    else:
        forced = None
        result = api['checked_replays'](case, policy, reset_inputs=reset_inputs, initialize_outputs=initialize_outputs,
            replay=graph.replay, verify=verify, measure=measure, observe=observe, seed=request['challenge_seed'])
        expected_preparations = policy['warmup_iterations'] + policy['benchmark_iterations']
        require(len(provider.draws) == expected_preparations, 'weighted draw count differs')
        warmups = provider.draws[:policy['warmup_iterations']]
        measured = provider.draws[policy['warmup_iterations']:]
        result['observed_work_distribution'] = {
            'histogram_sha256': group['histogram_sha256'], 'frequency_basis': 'actual eight-rank call counts',
            'sampling': 'integer histogram draws from the protected request challenge seed',
            'schedule_sha256': digest([row['variant_id'] for row in provider.draws]),
            'shared_private_request_seed_required_for_paired_comparison': True,
            'warmup_variants': warmups, 'performance_samples': [dict(draw, device_time_ms=sample)
                for draw, sample in zip(measured, result['samples_ms'])],
            'distinct_measured_settings': len({row['variant_id'] for row in measured}),
            'observed_setting_count': len(group['histogram']), 'all_observed_settings_timed': False,
            'actual_other_rank_routing_recovered': False,
            'scope': 'sampled observed work-count distribution with representative generated routes'}
    if group['seam'] == 'moe2':
        require(provider.producer_calls - producer_start == expected_preparations,
                'stage2 did not receive a fresh stage1 reference output for every preparation')
        result['stage1_input_generation'] = {'reference_binding': provider.producer_identity,
            'fresh_calls': expected_preparations, 'candidate_stage1_used': False,
            'fresh_call_scope': 'checked correctness or warmup/measured preparations',
            'total_fresh_calls_including_compile_and_capture': provider.producer_calls,
            'captured_parent_output_reused': False}
    progress(root, case['case_id'], phase + '_completed')
    del graph, inputs, reference_inputs, output, pristine, initial_out, provider, truth
    torch.cuda.empty_cache()
    return result, compiled
