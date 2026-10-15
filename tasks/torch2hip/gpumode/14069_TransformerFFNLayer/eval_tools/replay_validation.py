# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Validate returned values and caller state from the actual timed invocation.

Observe either the captured graph or the last actual explicit Event sample.
All numerical checks run outside the reported samples. The canonical helper
owns capture, warmups, repetitions, stream ordering and device timing.
"""
import copy
import hashlib
import json
import inspect

import torch


def unchanged_inputs(before, after):
    for expected, actual in zip(before, after):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        elif actual != expected:
            raise ValueError("operator changed a caller-owned input")


def unchanged_model_state(before, module):
    if not before:
        return
    after = module.state_dict()
    if before.keys() != after.keys():
        raise ValueError('operator changed model state entries')
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected, rtol=0, atol=0, equal_nan=True)


def check_result(actual, expected, inputs, output_contract, compare, rtol, atol):
    output_contract(expected, actual)
    if not compare(expected, actual, rtol=rtol, atol=atol):
        raise ValueError("Timed operator output disagrees with the protected reference")
    separate_output(actual, inputs)


def separate_output(actual, inputs):
    # The task reference returns an independent output: preserving values alone cannot establish its
    # caller-visible storage contract.
    for value in inputs:
        if isinstance(value, torch.Tensor) and actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
            raise ValueError("operator output aliases a caller-owned input")


def changed_input_reference(inputs, index, expected, reference, compare, rtol, atol):
    """Prepare a finite, independently distinguishable operand change, untimed."""
    value = inputs[index]
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not value.numel():
        raise ValueError(f'Operand {index} has no floating data to replay')
    for flat_index in dict.fromkeys((0, value.numel() // 2, value.numel() - 1)):
        remainder = flat_index
        coordinates = []
        for size in reversed(value.shape):
            remainder, coordinate = divmod(remainder, size)
            coordinates.append(coordinate)
        coordinates = tuple(reversed(coordinates))
        original = value[coordinates].detach().clone()
        for trial in (original + 2, original - 2, torch.zeros_like(original)):
            with torch.no_grad():
                value[coordinates].copy_(trial)
                changed = reference(*copy.deepcopy(inputs))
            if not compare(expected, changed, rtol=rtol, atol=atol):
                evidence = {'operand': index, 'coordinates': list(coordinates),
                            'original_value': original.item(), 'replay_value': trial.item(),
                            'reference_max_abs_delta': (changed.float() - expected.float()).abs().max().item()}
                return changed, copy.deepcopy(inputs), evidence
            with torch.no_grad():
                value[coordinates].copy_(original)
    raise ValueError(f'Operand {index} did not produce a distinguishable reference output')


class MeasuredModelInputs:
    """Independent draws before events; observers only read tensors and count."""
    def __init__(self, module, reference, inputs, pristine, state, samples,
                 output_contract, compare, rtol, atol, original_prepare):
        self.module, self.reference = module, reference
        self.inputs, self.pristine, self.state = inputs, pristine, state
        self.samples, self.output_contract = samples, output_contract
        self.compare, self.rtol, self.atol = compare, rtol, atol
        self.original_prepare = original_prepare
        identity = [type(module).__name__, [
            [list(value.shape), str(value.dtype), list(value.stride())]
            if isinstance(value, torch.Tensor) else [type(value).__name__, value]
            for value in inputs]]
        self.seed_base = int.from_bytes(hashlib.sha256(
            json.dumps(identity, separators=(',', ':')).encode()).digest()[:8], 'little') % (2**62)
        self.preparations = 0
        self.checked = 0
        self.pending = False
        self.measuring = True
        self.prepared_inputs = None
        self.prepared_record = None
        self.expected = None
        self.draws = []

    def prepare(self):
        if self.original_prepare is not None:
            self.original_prepare()
        if not self.measuring:
            return
        if self.pending or self.checked >= self.samples:
            raise AssertionError('Unexpected preparation around an incomplete measured sample')
        if self.prepared_inputs is not None:
            unchanged_inputs(self.prepared_inputs, self.inputs)
        unchanged_model_state(self.state, self.module)
        index = self.preparations
        expected_inputs, seeds = [], []
        digest = hashlib.sha256()
        with torch.no_grad():
            for operand, (value, original) in enumerate(zip(self.inputs, self.pristine)):
                if isinstance(value, torch.Tensor):
                    if not value.is_floating_point() or value.dtype != torch.float32:
                        raise ValueError('Declared measured input stream requires FP32 floating tensors')
                    seed = None if index == 0 else self.seed_base + 4 * index + operand
                    if seed is None:
                        host = original.detach().to('cpu', copy=True)
                    else:
                        generator = torch.Generator(device='cpu')
                        generator.manual_seed(seed)
                        # All five protected get_inputs generators draw FP32
                        # standard normals on CPU. Never consume their global RNG.
                        host = torch.randn(value.shape, dtype=value.dtype, device='cpu', generator=generator)
                    expected_input = host.to(value.device, copy=True)
                    value.copy_(expected_input)
                    torch.testing.assert_close(value, expected_input, rtol=0, atol=0)
                    expected_inputs.append(expected_input)
                    digest.update(json.dumps([operand, list(host.shape), str(host.dtype)]).encode())
                    digest.update(host.contiguous().numpy().tobytes())
                    seeds.append(seed)
                else:
                    expected_inputs.append(copy.deepcopy(original))
                    seeds.append(None)
                    digest.update(json.dumps([operand, original]).encode())
            self.expected = self.reference(*copy.deepcopy(expected_inputs))
        fingerprint = digest.hexdigest()
        if self.prepared_record is not None and fingerprint == self.prepared_record['input_sha256']:
            raise AssertionError('Consecutive invocations received identical measured inputs')
        self.prepared_inputs = expected_inputs
        self.prepared_record = {'prepare_index': index, 'seeds': seeds, 'input_sha256': fingerprint}
        self.preparations += 1

    def before_sample(self, calls):
        if calls != 1 or self.pending or self.prepared_record is None:
            raise AssertionError('Each reported sample must contain one freshly prepared invocation')
        unchanged_inputs(self.prepared_inputs, self.inputs)
        unchanged_model_state(self.state, self.module)
        self.pending = True

    def observe(self, output):
        if not self.pending or self.checked >= self.samples:
            raise AssertionError('Measured output has no pending fresh-input invocation')
        check_result(output, self.expected, self.inputs, self.output_contract,
                     self.compare, self.rtol, self.atol)
        unchanged_inputs(self.prepared_inputs, self.inputs)
        unchanged_model_state(self.state, self.module)
        self.draws.append(copy.deepcopy(self.prepared_record))
        self.pending = False
        self.checked += 1

    def finish(self, metadata, warmup):
        self.measuring = False
        if self.pending or self.checked != self.samples or metadata.get('benchmark_samples') != self.checked:
            raise AssertionError('Not every reported sample was checked with its own input draw')
        kind = metadata.get('benchmark_timed_run_kind')
        # This is the existing protected canonical preparation schedule:
        # estimate capture + two estimate replays + final capture + prime replay.
        extra = {'captured_graph': 5, 'eager_callable': 0}.get(kind)
        if extra is None:
            raise ValueError('Unknown actual timed invocation')
        first = max(0, int(warmup)) + extra
        if [record['prepare_index'] for record in self.draws] != list(range(first, first + self.samples)):
            raise AssertionError('Measured draw schedule differs from the protected canonical schedule')
        if kind == 'captured_graph' and metadata.get('benchmark_effective_repeats') != 1:
            raise AssertionError('Fresh-input graph samples cannot amortize repeated identical invocations')
        if len({record['input_sha256'] for record in self.draws}) != self.samples:
            raise AssertionError('Reported samples did not use independent input draws')
        unchanged_inputs(self.prepared_inputs, self.inputs)
        unchanged_model_state(self.state, self.module)

    def evidence(self):
        return {'validated_sample_count': self.checked,
                'measured_calls_per_sample': 1,
                'measured_input_stream': 'independent_cpu_fp32_randn_per_prepared_invocation',
                'measured_input_seed_base': self.seed_base,
                'measured_input_seed_formula': 'seed_base + 4 * prepare_index + operand_index; preparation0 is original input',
                'measured_input_draw_count': len(self.draws),
                'measured_input_distinct_draws': len({record['input_sha256'] for record in self.draws}),
                'measured_input_draws': self.draws}


def install(perf, output_contract):
    from _aka_benchmark import TimedRun
    signature = inspect.signature(perf.cal_kernel_perf).parameters
    rtol, atol = signature['rtol'].default, signature['atol'].default

    def latency(module, inputs, hip_fn=None, n_iter=100, n_warmup=10,
                use_cuda_graph=True, fallback_reason=None, prepare_fn=None):
        pristine = copy.deepcopy(inputs)
        state = {name: value.detach().clone() for name, value in module.state_dict().items()} if hasattr(module, 'state_dict') else {}
        reference = copy.deepcopy(module)
        try:
            with torch.no_grad():
                original_expected = reference(*copy.deepcopy(pristine))
            measured = MeasuredModelInputs(module, reference, inputs, pristine, state, n_iter,
                                           output_contract, perf._compare_results, rtol, atol, prepare_fn)
            observed = TimedRun()
            observed.before_sample = measured.before_sample
            observed.after_sample = measured.observe
            invoke = (lambda: module(*inputs)) if hip_fn is None else (lambda: module(*inputs, fn=hip_fn))
            elapsed, metadata = perf.benchmark_cuda_graph_or_events(
                invoke, warmup=n_warmup, repetition=n_iter,
                use_cuda_graph=use_cuda_graph, fallback_reason=fallback_reason,
                prepare_fn=measured.prepare, timed_run=observed)
            kind = metadata.get('benchmark_timed_run_kind')
            expected_method = {'captured_graph': 'cuda_graph', 'eager_callable': 'cuda_event_fallback'}.get(kind)
            if expected_method is None or metadata.get('benchmark_method') != expected_method:
                raise ValueError('Benchmark did not identify its actual observed invocation')
            measured.finish(metadata, n_warmup)
            expected = measured.expected
            measured_inputs = copy.deepcopy(measured.prepared_inputs)
            torch.cuda.synchronize()
            check_result(observed.outputs, expected, inputs, output_contract, perf._compare_results, rtol, atol)
            unchanged_inputs(measured_inputs, inputs)
            unchanged_model_state(state, module)
            with torch.no_grad():
                observed.outputs.fill_(float('nan'))
            actual = observed.rerun()
            check_result(actual, expected, inputs, output_contract, perf._compare_results, rtol, atol)
            unchanged_inputs(measured_inputs, inputs)
            unchanged_model_state(state, module)
            changed_replays = []
            for index, value in enumerate(inputs):
                if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not value.numel():
                    continue
                changed_expected, changed_inputs, detail = changed_input_reference(
                    inputs, index, expected, reference, perf._compare_results, rtol, atol)
                with torch.no_grad():
                    observed.outputs.fill_(float('nan'))
                changed_actual = observed.rerun()
                check_result(changed_actual, changed_expected, inputs, output_contract,
                             perf._compare_results, rtol, atol)
                unchanged_inputs(changed_inputs, inputs)
                unchanged_model_state(state, module)
                changed_replays.append(detail)
                with torch.no_grad():
                    for original, current in zip(measured_inputs, inputs):
                        if isinstance(current, torch.Tensor):
                            current.copy_(original)
            if not changed_replays:
                raise ValueError('No changed-input replay was validated')
            # Original case data remains an additional untimed regression, and
            # must also work through this exact captured graph/Event callable.
            with torch.no_grad():
                for original, current in zip(pristine, inputs):
                    if isinstance(current, torch.Tensor):
                        current.copy_(original)
                observed.outputs.fill_(float('nan'))
            restored_actual = observed.rerun()
            check_result(restored_actual, original_expected, inputs, output_contract,
                         perf._compare_results, rtol, atol)
            unchanged_inputs(pristine, inputs)
            unchanged_model_state(state, module)
            return elapsed, {**metadata, **measured.evidence(),
                             'replay_validation_valid': True,
                             'timed_output_checked': True,
                             'changed_input_replay_valid': True,
                             'changed_input_replay_count': len(changed_replays),
                             'changed_input_replays': changed_replays,
                             'restored_input_replay_valid': True,
                             'input_state_restored': True,
                             'model_state_validation_valid': True,
                             'model_state_tensor_count': len(state),
                             'validated_invocation_kind': kind,
                             'replay_validation': 'full_reference_output_and_unchanged_inputs'}
        finally:
            with torch.no_grad():
                for original, value in zip(pristine, inputs):
                    if isinstance(value, torch.Tensor):
                        value.copy_(original)
                if state:
                    current = module.state_dict()
                    for name, original in state.items():
                        current[name].copy_(original)
            torch.cuda.synchronize()

    perf.cal_hip_latency = latency
    if hasattr(perf, 'cal_modu_latency'):
        perf.cal_modu_latency = lambda module, inputs, **kwargs: latency(module, inputs, **kwargs)
