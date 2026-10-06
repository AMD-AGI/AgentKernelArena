"""Protected fresh replay callbacks for Kimi's fixture-specific runtime adapter.

The shared checked_replays API is unchanged: reset_inputs returns an opaque
CPU input token, and verify computes the reference only after synchronized
candidate observations have been captured. The fixture adapter owns native
bindings, legal input generation, ABI/alias checks, tolerances and inout rules.
"""
from dataclasses import dataclass


@dataclass
class InputTruth:
    value: object


@dataclass
class OwnedCPUOutputs:
    """Trusted callback owns these CPU bytes and has erased GPU reference outputs."""
    value: object


def cpu_copy(value, torch):
    """Recursively own snapshot bytes; never retain a GPU tensor as truth."""
    if torch.is_tensor(value):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, tuple):
        return tuple(cpu_copy(item, torch) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item, torch) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, (str, int)) for key in value):
            raise TypeError('Snapshot mapping keys must be strings or integers')
        return {key: cpu_copy(item, torch) for key, item in value.items()}
    if value is None or type(value) in (str, int, float, bool, bytes):
        return value
    raise TypeError('Snapshot must contain only tensors and immutable scalar data')


def require_cpu(value, torch):
    if torch.is_tensor(value):
        if value.device.type != 'cpu':
            raise AssertionError('Trusted snapshot must reside on CPU')
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_cpu(item, torch)
    elif isinstance(value, dict):
        for item in value.values():
            require_cpu(item, torch)


def clear_device_reference(value, torch, seen=None):
    """Erase returned GPU golden storage after its independent CPU capture."""
    seen = set() if seen is None else seen
    if torch.is_tensor(value):
        if value.device.type != 'cpu':
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            if key not in seen:
                seen.add(key)
                raw = torch.empty(0, dtype=torch.uint8, device=value.device)
                raw.set_(storage, 0, (storage.nbytes(),), (1,)).zero_()
    elif isinstance(value, (tuple, list)):
        for item in value:
            clear_device_reference(item, torch, seen)
    elif isinstance(value, dict):
        for item in value.values():
            clear_device_reference(item, torch, seen)
    return bool(seen)


class FreshCallbacks:
    """Bind fixture operations to protected replay, snapshot and timing order.

    snapshot_inputs() must describe complete input-storage bytes, including
    padding and alias groups where the ABI requires them. snapshot_outputs()
    returns every defined output. Both may return GPU tensors: this adapter
    takes independent CPU copies. validate_metadata() must check current
    shapes, strides, dtypes, storage offsets and expected alias relationships.

    assert_immutable(after_cpu, before_cpu) enforces the fixture's immutable
    inputs while respecting declared inout storage. reference(before_cpu)
    reconstructs independent inputs from host truth and returns expected
    outputs; compare(actual_cpu, expected_cpu) preserves task tolerances.
    None of those callbacks may substitute a candidate-produced reference.
    """
    def __init__(self, *, refresh_inputs, initialize_outputs, snapshot_inputs,
                 snapshot_outputs, validate_metadata, assert_immutable,
                 reference, compare, replay, torch_module=None, snapshot_candidate=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self._refresh = refresh_inputs
        self._initialize = initialize_outputs
        self._inputs = snapshot_inputs
        self._outputs = snapshot_outputs
        self._metadata = validate_metadata
        self._immutable = assert_immutable
        self._reference = reference
        self._compare = compare
        self.replay = replay
        self._truth = None
        self._candidate_snapshot = snapshot_candidate

    @staticmethod
    def _accepted(result, message):
        if result is not None and result is not True:
            raise AssertionError(message)

    def reset_inputs(self, seed):
        self._refresh(seed)
        self._truth = InputTruth(cpu_copy(self._inputs(), self.torch))
        require_cpu(self._truth.value, self.torch)
        return self._truth

    def initialize_outputs(self):
        if self._truth is None:
            raise RuntimeError('Input reset must precede output initialization')
        self._initialize()
        # Inout accumulators must enter the reference with the same initialized
        # state as the candidate. Preserve immutable inputs across this step.
        initialized = cpu_copy(self._inputs(), self.torch)
        self._accepted(self._immutable(initialized, self._truth.value), 'Output initialization changed immutable inputs')
        self._truth.value = initialized

    def verify(self, truth):
        if truth is not self._truth or not isinstance(truth, InputTruth):
            raise AssertionError('Stale input truth token')
        require_cpu(truth.value, self.torch)
        self.torch.cuda.synchronize()
        self._accepted(self._metadata(), 'Native output metadata/aliases are invalid')
        if self._candidate_snapshot is None:
            actual_cpu = cpu_copy(self._outputs(), self.torch)
            after_cpu = cpu_copy(self._inputs(), self.torch)
        else:
            # Every output aliases a supplied native buffer. A single owned
            # storage snapshot can represent both observations without a
            # second transfer of the same multi-GB bank allocation.
            actual_cpu, after_cpu = self._candidate_snapshot()
            require_cpu(actual_cpu, self.torch)
            require_cpu(after_cpu, self.torch)
        self._accepted(self._immutable(after_cpu, truth.value), 'Candidate changed immutable inputs')
        # Both candidate observations are immutable host snapshots BEFORE the
        # reference creates any GPU buffers or executes any device code.
        expected = self._reference(cpu_copy(truth.value, self.torch))
        if isinstance(expected, OwnedCPUOutputs):
            expected_cpu = expected.value
            require_cpu(expected_cpu, self.torch)
        else:
            expected_cpu = cpu_copy(expected, self.torch)
            if clear_device_reference(expected, self.torch):
                self.torch.cuda.synchronize()
        del expected
        require_cpu(actual_cpu, self.torch)
        require_cpu(expected_cpu, self.torch)
        accepted = self._compare(actual_cpu, expected_cpu)
        if accepted is not None and accepted is not True:
            raise AssertionError('Independent oracle rejected candidate output')
        self._truth = None
        return True

    def measure(self, call):
        start = self.torch.cuda.Event(enable_timing=True)
        stop = self.torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        stop.record()
        stop.synchronize()
        return float(start.elapsed_time(stop))

    def check_once(self, seed):
        truth = self.reset_inputs(seed)
        self.initialize_outputs()
        self._accepted(self._metadata(), 'Native output metadata/aliases are invalid')
        self.replay()
        return self.verify(truth)

    def correctness_row(self, case, policy, *, observe, corrupt_outputs, challenge_seed):
        # The fixture owner supplies corruption for all defined outputs,
        # including integer/index outputs whose sentinel differs from floats.
        from evaluation_contract import canonical, require
        for seed in policy['correctness_seeds']:
            truth = self.reset_inputs(seed)
            self.initialize_outputs()
            require(canonical(observe()) == canonical(case), 'Runtime ABI changed')
            self.replay()
            self.verify(truth)
        controls = {}
        for control in policy['negative_controls']:
            truth = self.reset_inputs(challenge_seed)
            self.initialize_outputs()
            require(canonical(observe()) == canonical(case), 'Runtime ABI changed')
            if control == 'wrong_output':
                self.replay()
                self.torch.cuda.synchronize()
                corrupt_outputs()
            elif control != 'no_op':
                raise ValueError('Unknown negative control: ' + control)
            try:
                self.verify(truth)
            except AssertionError:
                controls[control] = True
            else:
                raise AssertionError('Invalid-work control passed: ' + control)
        return {'case': observe(), 'correct': True,
                'seeds': policy['correctness_seeds'], 'negative_controls': controls}

    def performance_row(self, case, policy, *, observe, challenge_seed):
        from evaluation_contract import checked_replays
        return checked_replays(case, policy, reset_inputs=self.reset_inputs,
                               initialize_outputs=self.initialize_outputs,
                               replay=self.replay, verify=self.verify,
                               measure=self.measure, observe=observe,
                               seed=challenge_seed)
