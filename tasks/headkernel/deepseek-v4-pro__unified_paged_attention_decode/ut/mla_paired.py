"""Paired MLA timing receipts from the actual CPU-owned prepared inputs."""
from collections import Counter
import hashlib
import json
import math

from evaluation_contract import canonical, fingerprint
from mla_decode_distribution import KIND
from paired_reference import paired_performance, require


def byte_digest(value):
    require(value.device.type == 'cpu', 'receipt inputs must be CPU-owned')
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def signature_from_snapshot(truth, seed, length):
    return {'input_seed': seed, 'length': length,
            'kv_indices_sha256': byte_digest(truth['arg.kv_indices']),
            'kv_indptr_sha256': byte_digest(truth['arg.kv_indptr']),
            'private_cpu_storage_bytes': sum(value.numel() for value in truth.values())}


def paired_decode_performance(api, case, manifest, request, inputs, reference_inputs, fn, reference_fn,
                              pristine, initial_out, identity, reference_identity, recipe):
    import torch
    truth_receipts = []
    prepared = {'candidate': [], 'reference': []}
    last = {}

    def reset(seed):
        api['restore_storages'](inputs, pristine)
        if recipe is not None:
            recipe.apply(inputs, seed)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        values = torch.randn(inputs['q'].shape, dtype=torch.float32, device='cuda', generator=generator).to(inputs['q'].dtype)
        inputs['q'].copy_(values)
        truth = api['storage_snapshots'](inputs)
        # Both graphs start from the same CPU-owned storage token, followed by
        # a complete immutable-input observation and the same output reset.
        api['restore_storages'](inputs, truth)
        api['assert_immutable_inputs'](inputs, truth)
        signature = (signature_from_snapshot(truth, seed, recipe.current_length) if recipe is not None
                     else {'input_seed': seed, 'fixture': case['fixture']})
        last.clear(); last.update(signature)
        truth_receipts.append(json.loads(canonical(signature)))
        return truth

    def runtime_abi(values, output):
        if recipe is not None:
            recipe.verify_controls(values, output)
            if last:
                # These observations are independent of the schedule producer.
                actual = {'input_seed': last['input_seed'], 'length': recipe.current_length,
                    'kv_indices_sha256': byte_digest(values['kv_indices'].detach().cpu().view(torch.uint8)),
                    'kv_indptr_sha256': byte_digest(values['kv_indptr'].detach().cpu().view(torch.uint8)),
                    'private_cpu_storage_bytes': sum({v.untyped_storage().data_ptr(): v.untyped_storage().nbytes()
                        for v in api['leaves'](values)}.values())}
                require(actual == last, 'actual prepared controls differ from the CPU token')
                leg = 'candidate' if values is inputs else 'reference'
                require(values is inputs or values is reference_inputs, 'unrecognized paired input state')
                prepared[leg].append(actual)
        return api['runtime_abi'](values, output)

    callbacks = {**api, 'runtime_abi': runtime_abi}
    row = paired_performance(callbacks, case, manifest, request, inputs, reference_inputs, fn, reference_fn,
        pristine, initial_out, identity, reference_identity, reset, lambda: dict(last))
    row['realized_input_signatures'] = truth_receipts
    if recipe is not None:
        count = manifest['measurement']['benchmark_iterations']; warm = manifest['measurement']['warmup_iterations']
        values = [receipt['length'] for receipt in truth_receipts]
        require(len(values) == warm + count and prepared['candidate'] == prepared['reference'] == truth_receipts,
                'paired prepared-state observations are incomplete or unequal')
        require([{'length': value, 'forced': False} for value in values] == recipe.draws,
                'actual recipe draws differ from the CPU receipts')
        kernels = {item['kernel'] for item in recipe.policy['source_supported_launches']}
        for leg in ('candidate', 'reference'):
            require(all(recipe.dispatch_counts[(leg, 200, kernel)] >= 4 for kernel in kernels),
                    'both warmed graphs must capture the protected split/reduce controls')
        indices, indptr = recipe.controls[200]
        row['paired_control_registry'] = {'schema': 'mla-parent-control-registry-v1',
            'kv_indices': indices.tolist(), 'kv_indptr': indptr.tolist()}
        reference_samples = row['paired_reference']['legs']['protected_reference']['samples_ms']
        row['paired_control_timing'] = {'schema': 'mla-paired-control-timing-v1',
            'measured_pairs': [{'input_seed': item['input_seed'], 'length': item['length'],
                'candidate_ms': candidate_ms, 'reference_ms': reference_ms}
                for item, candidate_ms, reference_ms in zip(truth_receipts[warm:], row['samples_ms'], reference_samples)],
            'candidate_mean_ms': math.fsum(row['samples_ms']) / count,
            'reference_mean_ms': math.fsum(reference_samples) / count}
        row['control_distribution'] = {'kind': KIND, 'parent_fixture': recipe.policy['parent_fixture'],
            'actual_missing_captures_recovered': False, 'observed_histogram': recipe.policy['histogram'],
            'launch_controls_sha256': fingerprint(recipe.policy['source_supported_launches']),
            'selection_trace_sha256': fingerprint(values),
            'runtime_dispatch_mode': 'captured_graph_controls_then_per_replay_tensor_checks',
            'runtime_dispatch_calls': [{'leg': leg, 'length': length, 'kernel': kernel, 'calls': calls}
                for (leg, length, kernel), calls in sorted(recipe.dispatch_counts.items())],
            'prepared_state_checks': prepared,
            'warmup_draws': warm, 'measured_draws': count,
            'warmup_histogram': dict(Counter(values[:warm])), 'measured_histogram': dict(Counter(values[warm:])),
            'warmup_values': values[:warm], 'measured_values': values[warm:],
            'private_challenge_sampling': True, 'same_challenge_produces_same_baseline_candidate_states': True}
    return row
