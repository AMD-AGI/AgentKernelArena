"""Protected runtime controls derived from an actual 200-KV parent.

The distribution is observed; intermediate operand states are generated, not
recovered captures. No expected output is cached or exposed to the candidate.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

KIND = 'generated_observed_control_distribution'
STATUS = 'FROZEN_CAPTURE_AND_GENERATED_DISTRIBUTION'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError('MLA distribution: ' + message)


def controls_for_length(indices, indptr, length):
    require(type(length) is int and 192 <= length <= 200, 'length outside observed domain')
    require(len(indices) == 16384 and indptr == [200*i for i in range(65)], 'parent CSR differs')
    require(all(0 <= slot < 136856 for slot in indices[:12800]) and all(slot == -1 for slot in indices[12800:]),
            'parent live IDs or padding differ')
    selected = [slot for row in range(64) for slot in indices[200*row:200*row+length]]
    return selected + [-1]*(len(indices)-len(selected)), [length*i for i in range(65)]


def choose_length(seed, histogram, case_id):
    require(type(seed) is int and seed >= 0, 'invalid private challenge seed')
    require([row['length'] for row in histogram] == list(range(192, 201)), 'histogram domain changed')
    require(all(type(row['weight']) is int and row['weight'] > 0 for row in histogram), 'invalid histogram weight')
    total = sum(row['weight'] for row in histogram)
    ceiling = (1 << 256) // total * total
    counter = 0
    while True:
        payload = canonical(['mla-observed-kv-distribution-v1', case_id, seed, counter]).encode()
        ticket = int.from_bytes(hashlib.sha256(payload).digest(), 'big')
        if ticket < ceiling:
            ticket %= total
            break
        counter += 1
    for row in histogram:
        if ticket < row['weight']:
            return row['length']
        ticket -= row['weight']
    raise AssertionError('histogram selection escaped its domain')


def load_policy(root, manifest):
    root = Path(root)
    distributions = [case for case in manifest['cases'] if case.get('provenance_kind') == KIND]
    if not distributions:
        require(manifest.get('status') == 'FROZEN_CURRENT_CAPTURE', 'unexpected base status')
        return None
    require(manifest.get('status') == STATUS and len(distributions) == 1, 'explicit single distribution contract required')
    case = distributions[0]
    ref = case['distribution_recipe']
    relative = Path(ref['path'])
    path = root / relative
    require(not relative.is_absolute() and '..' not in relative.parts and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()), 'unsafe recipe path')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'recipe digest changed')
    policy = json.loads(raw)
    base = [row for row in manifest['cases'] if row['case_id'] != case['case_id']]
    require(fingerprint(base) == policy['base_cases_sha256'], 'original three cases changed')
    require(manifest['measurement'] == policy['measurement'] and manifest['tolerance'] == 0.02,
            'measurement policy or tolerance changed')
    require(manifest['required_case_ids'] == [row['case_id'] for row in manifest['cases']], 'required case set differs')
    parent = next(row for row in base if row['case_id'] == policy['parent_case_id'])
    require(case['fixture'] == parent['fixture'] == policy['parent_fixture'], 'parent capture binding differs')
    require(case['tensors'] == parent['tensors'] and case['scalars'] == parent['scalars'], 'distribution ABI differs')
    require(case['calls_per_sample'] == 1 and case['fixture_role'] == 'parent_inputs_and_200_calibration_only',
            'parent output must not serve as a generated-state oracle')
    require(policy['lengths'] == list(range(192, 201)) and policy['missing_actual_lengths'] == list(range(193, 200)),
            'observed or missing-actual domain changed')
    require(sum(row['weight'] for row in policy['histogram']) == 253952, 'observed histogram total differs')
    require(manifest['native_source_sha256'] == policy['native_source_sha256'], 'native source changed')
    return policy


class KernelProbe:
    def __init__(self, kernel, name, expected, owner, leg):
        self.kernel, self.name, self.expected, self.owner, self.leg = kernel, name, expected, owner, leg

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        invoke = self.kernel[grid]
        def checked(*args, **kwargs):
            require(list(grid) == self.expected['grid'] and kwargs == self.expected['kwargs'],
                    'native launch grid/constexpr/scalar controls changed: ' + self.name)
            bound = {**dict(zip(self.expected['argument_names'], args)), **kwargs}
            require(all(bound.get(name) == value for name, value in self.expected['scalar_arguments'].items()),
                    'native positional scalar/stride controls changed: ' + self.name)
            self.owner.dispatch_counts[(self.leg, self.owner.current_length, self.name)] += 1
            return invoke(*args, **kwargs)
        return checked


class RuntimeRecipe:
    def __init__(self, case, policy, inputs, candidate_module, reference_module):
        import torch
        self.case, self.policy = case, policy
        self.current_length = 200
        self.draws, self.dispatch_counts = [], Counter()
        self.inputs_metadata = policy['native_inputs']
        self.check_geometry(inputs)
        indices = inputs['kv_indices'].detach().cpu().tolist()
        indptr = inputs['kv_indptr'].detach().cpu().tolist()
        for name, values in [('kv_indices', indices), ('kv_indptr', indptr)]:
            data = struct.pack('<' + 'i'*len(values), *values)
            require(hashlib.sha256(data).hexdigest() == policy['parent_controls_sha256'][name], 'parent control bytes differ')
        self.controls = {length: tuple(torch.tensor(values, dtype=torch.int32)
                          for values in controls_for_length(indices, indptr, length)) for length in policy['lengths']}
        self.probes = []
        for leg, module in [('candidate', candidate_module), ('reference', reference_module)]:
            for expected in policy['source_supported_launches']:
                name = expected['kernel']
                original = getattr(module, name)
                probe = KernelProbe(original, name, expected, self, leg)
                setattr(module, name, probe)
                self.probes.append((module, name, original))

    def check_geometry(self, inputs):
        aliases, pointers = {}, {}
        for name, expected in self.inputs_metadata.items():
            value = inputs[name]
            require(list(value.shape) == expected['shape'] and list(value.stride()) == expected['stride']
                    and value.storage_offset() == expected['storage_offset']
                    and value.untyped_storage().nbytes() == expected['storage_nbytes']
                    and str(value.dtype) == expected['dtype'], 'complete native ABI/storage changed: ' + name)
            pointer, alias = value.untyped_storage().data_ptr(), expected['alias']
            require(aliases.get(alias, pointer) == pointer and pointers.get(pointer, alias) == alias,
                    'native input alias relationship changed')
            aliases[alias], pointers[pointer] = pointer, alias
        require(inputs['kv_scales'] is None and inputs['softmax_scale'] == self.policy['softmax_scale'],
                'captured scalar controls changed')

    def apply(self, inputs, seed, forced_length=None):
        length = choose_length(seed, self.policy['histogram'], self.case['case_id']) if forced_length is None else forced_length
        require(type(length) is int and length in self.controls, 'unobserved forced length')
        self.current_length = length
        for name, value in zip(('kv_indices', 'kv_indptr'), self.controls[length]):
            inputs[name].copy_(value)
        self.check_geometry(inputs)
        self.draws.append({'length': length, 'forced': forced_length is not None})
        return length

    def verify_controls(self, inputs, output=None):
        import torch
        self.check_geometry(inputs)
        for name, value in zip(('kv_indices', 'kv_indptr'), self.controls[self.current_length]):
            require(torch.equal(inputs[name].detach().cpu(), value), 'selected control values changed: ' + name)
        if output is not None:
            expected = self.policy['native_output']
            require(output.untyped_storage().nbytes() == expected['storage_nbytes']
                    and output.untyped_storage().data_ptr() not in {v.untyped_storage().data_ptr() for v in inputs.values() if torch.is_tensor(v)},
                    'output backing storage or input alias differs')

    def proof(self, phase, measurement):
        values = [row['length'] for row in self.draws]
        kernels = {row['kernel'] for row in self.policy['source_supported_launches']}
        for leg in ('candidate', 'reference'):
            require(kernels <= {kernel for (seen_leg, _, kernel), count in self.dispatch_counts.items() if seen_leg == leg and count},
                    'native launch controls were not checked for both source legs')
        for length in set(values):
            require(all(self.dispatch_counts[('reference', length, kernel)] > 0 for kernel in kernels),
                    'a selected state lacks checked native reference launch controls')
        proof = {'kind': KIND, 'parent_fixture': self.policy['parent_fixture'],
                 'actual_missing_captures_recovered': False, 'observed_histogram': self.policy['histogram'],
                 'launch_controls_sha256': fingerprint(self.policy['source_supported_launches']),
                 'selection_trace_sha256': fingerprint(values),
                 'runtime_dispatch_calls': [{'leg': leg, 'length': length, 'kernel': kernel, 'calls': count}
                    for (leg, length, kernel), count in sorted(self.dispatch_counts.items())]}
        if phase == 'correctness':
            expected = len(measurement['correctness_seeds']) + len(measurement['negative_controls'])
            require(Counter(values) == Counter({length: expected for length in self.policy['lengths']})
                    and all(row['forced'] for row in self.draws), 'exhaustive correctness domain changed')
            proof.update(exhaustive_lengths=self.policy['lengths'], seeds_per_length=measurement['correctness_seeds'],
                         negative_controls_per_length=measurement['negative_controls'])
        if phase == 'performance':
            warmups = measurement['warmup_iterations']
            require(len(values) == warmups + measurement['benchmark_iterations'] and not any(row['forced'] for row in self.draws),
                    'weighted replay count or sampler changed')
            proof.update(warmup_draws=warmups, measured_draws=len(values)-warmups,
                         warmup_histogram=dict(Counter(values[:warmups])), measured_histogram=dict(Counter(values[warmups:])),
                         warmup_values=values[:warmups], measured_values=values[warmups:],
                         private_challenge_sampling=True, same_challenge_produces_same_baseline_candidate_states=True)
        return proof

    def close(self):
        for module, name, original in self.probes:
            setattr(module, name, original)
