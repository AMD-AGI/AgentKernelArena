"""CPU callback and shared-consumer tests; native GPU qualification is separate."""
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest
import torch
import yaml

from src.native_baseline import load_native_measurements, as_test_cases, metric_summary
from src.task_contract import checked_replays, finalize_report, fingerprint

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'tasks/headkernel/deepseek-v4-pro__unified_paged_attention_decode'
sys.path.insert(0, str(TASK / 'ut'))
import paired_reference as pair
import mla_paired as mla
import mla_decode_distribution as recipe
from snapshots import raw_storage

MANIFEST = json.loads((TASK / 'cases.json').read_text())
POLICY = json.loads((TASK / 'ut/mla_control_distribution.json').read_text())
CONFIG = yaml.safe_load((TASK / 'config.yaml').read_text())
SOURCE = {'source/native.py': hashlib.sha256((TASK / 'source/native.py').read_bytes()).hexdigest()}


def captured_parent_controls():
    # Exact compact reconstruction of the captured parent int32 blob. The
    # immutable recipe digest below checks every byte, including full padding.
    indices = []
    for row in range(64):
        indices.extend(range(8192 - 128 * row, 8320 - 128 * row))
        indices.extend(range(8322 + 64 * row, 8386 + 64 * row))
        for group in range(4):
            indices.extend(range(12418 + 128 * group + 2 * row, 12420 + 128 * group + 2 * row))
    indices += [-1] * 3584
    indptr = [200 * i for i in range(65)]
    for name, values in [('kv_indices', indices), ('kv_indptr', indptr)]:
        assert hashlib.sha256(struct.pack('<' + 'i' * len(values), *values)).hexdigest() == POLICY['parent_controls_sha256'][name]
    return indices, indptr


class OpaqueBytes:
    """Storage-size stand-in for unexamined KV bytes in a CPU flow simulation."""
    def __init__(self, size): self.size = size
    def numel(self): return self.size
    def untyped_storage(self): return self
    def nbytes(self): return self.size
    def data_ptr(self): return id(self)


class SimulatedWriteScope:
    """Tiny predeclared buffers for control/timing-flow tests only.

    Real wrapper allocation ownership is exercised separately with all native
    out/m/l/acc buffers in test_deepseek_mla_write_ownership.py.
    """
    def __init__(self, output): self.tensors = (output,)
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def clear(self):
        for tensor in self.tensors: raw_storage(tensor).fill_(0xAA)
    def release(self): self.tensors = ()


def simulate_case(monkeypatch, case, seed, *, attack=None, evidence=None):
    evidence = evidence if evidence is not None else {}
    events = []; state = {'capture': None, 'stream': 0, 'checked': False, 'replay': False, 'last_leg': None}
    streams = []
    class Stream:
        def __init__(self): self.ident = len(streams) + 1; streams.append(self)
        def wait_stream(self, other): events.append(('wait', self.ident, other.ident))
        def synchronize(self): pass
    default = type('DefaultStream', (), {'ident': 0, 'wait_stream': lambda self, other: None})()
    @contextmanager
    def stream_context(stream):
        old = state['stream']; state['stream'] = stream.ident
        try: yield
        finally: state['stream'] = old
    class Graph:
        def replay(self):
            events.append(('replay', self.leg)); state['replay'] = True
            try: self.call()
            finally: state['replay'] = False
    @contextmanager
    def capture(graph, stream):
        with stream_context(stream):
            state['capture'] = graph
            try:
                yield
                if attack == 'reference_capture_exit' and graph.leg == 'reference': raise RuntimeError(attack)
            finally: state['capture'] = None
    class Event:
        def __init__(self, **kwargs): pass
        def record(self): events.append(('timer', state['last_leg']))
        def synchronize(self): pass
        def elapsed_time(self, other):
            return (current.current_length / 100 if current is not None else 2.0) * (1.5 if state['last_leg'] == 'reference' else 1)
    generator = torch.Generator
    randn = torch.randn
    def cpu_randn(*args, **kwargs):
        state['checked'] = True; kwargs['device'] = 'cpu'
        return randn(*args, **kwargs)
    monkeypatch.setattr(torch, 'Generator', lambda **kwargs: generator(device='cpu'))
    monkeypatch.setattr(torch, 'randn', cpu_randn)
    for name, value in {'Stream': Stream, 'current_stream': lambda: default, 'stream': stream_context,
                        'graph': capture, 'CUDAGraph': Graph, 'Event': Event, 'synchronize': lambda: None}.items():
        monkeypatch.setattr(torch.cuda, name, value)
    indices, indptr = captured_parent_controls()
    byte_count = sum(item['storage_nbytes'] for item in POLICY['native_inputs'].values())
    def inputs():
        return {'q': torch.ones(1), 'kv_indices': torch.tensor(indices, dtype=torch.int32),
                'kv_indptr': torch.tensor(indptr, dtype=torch.int32), 'opaque': OpaqueBytes(byte_count - 4 - 65536 - 260)}
    candidate_inputs, reference_inputs = inputs(), inputs()
    output, reference_output = torch.zeros(1), torch.zeros(1)
    evidence.update(reference_output=reference_output, candidate_output=output, events=events)
    def leaves(value):
        if torch.is_tensor(value) or isinstance(value, OpaqueBytes): return [value]
        if isinstance(value, dict): return sum((leaves(v) for v in value.values()), [])
        if isinstance(value, (list, tuple)): return sum((leaves(v) for v in value), [])
        return []
    names = {'arg.q': 'q', 'arg.kv_indices': 'kv_indices', 'arg.kv_indptr': 'kv_indptr', 'opaque': 'opaque'}
    def snapshots(values):
        return {name: raw_storage(values[key]).clone() if key != 'opaque' else values[key] for name, key in names.items()}
    def restore(values, truth):
        events.append(('restore', 'candidate' if values is candidate_inputs else 'reference'))
        for name, key in names.items():
            if key != 'opaque': raw_storage(values[key]).copy_(truth[name])
    def check(values, truth):
        events.append(('input_check', 'candidate' if values is candidate_inputs else 'reference'))
        for name, key in names.items():
            if key != 'opaque' and not torch.equal(raw_storage(values[key]), truth[name]): raise AssertionError('input mutation')
    def clone(value):
        leg = 'candidate' if value is output else 'reference'; events.append(('snapshot', leg))
        if (state['checked'] and attack == leg + '_snapshot') or (not state['checked'] and attack == 'reference_setup_snapshot' and leg == 'reference'):
            raise RuntimeError(attack)
        return value.clone()
    class CPURecipe:
        def __init__(self):
            self.policy = POLICY; self.current_length = 200; self.draws = []
            self.controls = {length: tuple(torch.tensor(v, dtype=torch.int32) for v in recipe.controls_for_length(indices, indptr, length)) for length in range(192, 201)}
            self.dispatch_counts = Counter({(leg, 200, item['kernel']): 1 for leg in ('candidate', 'reference') for item in POLICY['source_supported_launches']})
        def apply(self, values, input_seed):
            self.current_length = recipe.choose_length(input_seed, POLICY['histogram'], case['case_id'])
            for name, value in zip(('kv_indices', 'kv_indptr'), self.controls[self.current_length]): values[name].copy_(value)
            self.draws.append({'length': self.current_length, 'forced': False})
        def verify_controls(self, values, ignored_output):
            for name, expected in zip(('kv_indices', 'kv_indptr'), self.controls[self.current_length]):
                assert torch.equal(values[name], expected), 'actual CSR changed'
    current = CPURecipe() if case.get('provenance_kind') == recipe.KIND else None
    def compute(leg, values):
        state['last_leg'] = leg; events.append(('invoke', leg, state['checked'], state['stream']))
        if current is not None and not state['replay']:
            for item in POLICY['source_supported_launches']: current.dispatch_counts[(leg, 200, item['kernel'])] += 1
        target = output if leg == 'candidate' else reference_output
        if state['replay'] and leg == 'reference' and attack == 'reference_replay': raise RuntimeError(attack)
        if leg == 'candidate' and attack == 'mutated_input' and state['checked']: values['q'].add_(1)
        if not (leg == 'candidate' and attack == 'no_op'):
            target.copy_(values['q'] * (0 if leg == 'candidate' and attack == 'wrong_output' else 2))
        if leg == 'reference' and attack == 'reference_input': values['q'].add_(1)
        return target
    candidate = lambda **kw: compute('candidate', kw)
    reference = lambda **kw: compute('reference', kw)
    def invoke(fn, values):
        if state['capture'] is not None:
            state['capture'].call = lambda: fn(**values)
            state['capture'].leg = 'candidate' if fn is candidate else 'reference'
        return fn(**values)
    def compare(actual, expected, tol):
        assert bool((raw_storage(reference_output) == 0xAA).all()), 'reference output not scrubbed before comparison'
        if attack == 'comparison': raise RuntimeError(attack)
        if not torch.equal(actual, expected): raise AssertionError('wrong output')
    api = {'leaves': leaves, 'storage_snapshots': snapshots, 'restore_storages': restore,
           'assert_immutable_inputs': check, 'cpu_clone': clone, 'invoke': invoke,
           'runtime_abi': lambda *args: ({}, {}), 'observe_case': lambda expected, *args: expected,
           'checked_replays': checked_replays, 'compare': compare,
           'owned_outputs': lambda fn: SimulatedWriteScope(output if fn is candidate else reference_output)}
    identity = {'leg': 'candidate', 'module': 'private_candidate', 'source_sha256': SOURCE, 'gpu_binding': 'triton_module'}
    reference_identity = {**identity, 'leg': 'reference', 'module': 'private_reference'}
    request = {'schema_version': 1, 'request_id': 'cpu-' + str(seed), 'phase': 'performance', 'challenge_seed': seed,
               'manifest_sha256': fingerprint(MANIFEST), 'package_sha256': 'c' * 64, 'source_sha256': SOURCE}
    row = mla.paired_decode_performance(api, case, MANIFEST, request, candidate_inputs, reference_inputs,
        candidate, reference, snapshots(candidate_inputs), None, identity, reference_identity, current)
    compiled = {'case_id': case['case_id'], 'candidate_binding': identity, 'reference_binding': reference_identity,
                'invoked_and_synchronized': True, 'dispatch': {'kv_splits': 4}}
    return row, compiled, request, events


def complete_report(monkeypatch, seed):
    rows = []; compiled = []; events = []
    for case in MANIFEST['cases']:
        with monkeypatch.context() as patch:
            row, binding, request, trace = simulate_case(patch, case, seed)
        rows.append(row); compiled.append(binding); events.append(trace)
    report = {'schema_version': 1, 'status': 'ok', 'request': request, 'cases': rows,
              'compiled': True, 'compiled_specializations': compiled}
    pair.attach_comparison(TASK, report, MANIFEST, request)
    return finalize_report(report, MANIFEST, request), request, events


def test_actual_recipe_bytes_unchanged_and_source_protocol_admitted():
    assert hashlib.sha256((TASK / 'cases.json').read_bytes()).hexdigest() == '755efecb441865248d6e7b360a437266374cb9af381fc014b1bcae8e3f08d758'
    assert hashlib.sha256((TASK / 'ut/mla_control_distribution.json').read_bytes()).hexdigest() == 'ef2822025f2215f39752bfc4897899fd456f7115570f659576bab919df3cd4b1'
    from src.tools.trusted_task_eval import package_contract
    from src.tools.custom_perf_protocols import custom_protocol_family
    assert len(package_contract(TASK)['manifest']['cases']) == 4
    assert custom_protocol_family(TASK, {TASK / 'scripts/task_runner.py'}) == 'portable_case_contract'


def test_all_four_real_case_receipts_pass_shared_consumer_and_pair_private_draws(monkeypatch):
    derived_id = MANIFEST['cases'][-1]['case_id']
    rare_seed = next(seed for seed in range(10000) if recipe.choose_length(seed, POLICY['histogram'], derived_id) == 200)
    selected_seed = max(0, rare_seed - 50)
    report, request, events = complete_report(monkeypatch, selected_seed)
    result = load_native_measurements(TASK, CONFIG, report=report, request=request)
    assert len(result['native']) == len(result['candidate']) == 4
    for row, trace in zip(report['cases'], events):
        paired = row['paired_reference']; assert paired['checked_pair_count'] == paired['reference_output_clear_calls'] == 110
        assert len(row['samples_ms']) == len(paired['legs']['protected_reference']['samples_ms']) == 100
        for leg in ('candidate', 'reference'):
            assert len([event for event in trace if event[:2] == ('replay', leg)]) == 110
            setup = [event for event in trace if event[:2] == ('invoke', leg) and not event[2]]
            assert len(setup) == 4 and len({event[3] for event in setup}) == 1 and setup[0][3] != 0
        for i, event in enumerate(trace):
            if event == ('replay', 'reference'):
                last_candidate = max(j for j in range(i) if trace[j] == ('replay', 'candidate'))
                assert ('snapshot', 'candidate') in trace[last_candidate:i]
                assert ('input_check', 'candidate') in trace[last_candidate:i]
        assert row['realized_input_signatures'] == paired['input_schedule']['warmup_inputs'] + paired['input_schedule']['measured_inputs']
    derived = report['cases'][-1]
    assert derived['control_distribution']['prepared_state_checks']['candidate'] == derived['control_distribution']['prepared_state_checks']['reference']
    assert sum(derived['control_distribution']['measured_histogram'].values()) == 100
    assert set(derived['control_distribution']['warmup_values'] + derived['control_distribution']['measured_values']) == set(range(192, 201))
    second, second_request, _ = complete_report(monkeypatch, 725935)
    second_result = load_native_measurements(TASK, CONFIG, report=second, request=second_request)
    summary = metric_summary(as_test_cases(result, is_baseline=True), as_test_cases(second_result))
    assert summary['native_speedup_ratio'] == pytest.approx(1.5)
    assert summary['port_to_port_speedup_ratio'] is None and summary['secondary_comparison_status'] == 'unpaired_workload'


@pytest.mark.parametrize('attack', ['no_op', 'wrong_output', 'mutated_input', 'reference_input',
    'candidate_snapshot', 'reference_snapshot', 'reference_setup_snapshot', 'reference_capture_exit', 'reference_replay', 'comparison'])
def test_reference_outputs_scrubbed_on_every_observable_failure(monkeypatch, attack):
    evidence = {}
    with pytest.raises((AssertionError, RuntimeError)):
        simulate_case(monkeypatch, MANIFEST['cases'][-1], 123, attack=attack, evidence=evidence)
    assert bool((raw_storage(evidence['reference_output']) == 0xAA).all())


def rewrite_schedule_hashes(report):
    outer = report['cases'][-1]; paired = report['paired_reference_comparison']['cases'][-1]
    digest = fingerprint(paired['input_schedule'])
    outer['paired_schedule_sha256'] = digest
    for leg in paired['legs'].values(): leg['paired_schedule_sha256'] = digest


@pytest.mark.parametrize('attack', ['wrong_length', 'wrong_csr', 'wrong_private_bytes', 'parent_bytes',
    'missing_state_check', 'different_reference_state', 'missing_graph_dispatch', 'short_samples', 'no_scrub', 'snapshot_order'])
def test_shared_consumer_rejects_forged_actual_schedule_even_if_both_leg_hashes_are_updated(monkeypatch, attack):
    report, request, _ = complete_report(monkeypatch, 4567)
    row = report['cases'][-1]; paired = report['paired_reference_comparison']['cases'][-1]
    if attack in ('wrong_length', 'wrong_csr', 'wrong_private_bytes'):
        fields = {'wrong_length': ('length', 999), 'wrong_csr': ('kv_indices_sha256', '0' * 64), 'wrong_private_bytes': ('private_cpu_storage_bytes', 1)}
        key, value = fields[attack]
        paired['input_schedule']['measured_inputs'][0][key] = value
        row['realized_input_signatures'][10][key] = value
        for leg in ('candidate', 'reference'): row['control_distribution']['prepared_state_checks'][leg][10][key] = value
        rewrite_schedule_hashes(report)
    elif attack == 'parent_bytes': row['paired_control_registry']['kv_indices'][0] += 1
    elif attack == 'missing_state_check': row['control_distribution']['prepared_state_checks']['reference'].pop()
    elif attack == 'different_reference_state':
        record = row['control_distribution']['prepared_state_checks']['reference'][10]
        record['length'] = 193 if record['length'] == 192 else 192
    elif attack == 'missing_graph_dispatch': row['control_distribution']['runtime_dispatch_calls'].pop()
    elif attack == 'short_samples': paired['legs']['protected_reference']['samples_ms'].pop()
    elif attack == 'no_scrub': paired['reference_output_clear_calls'] = 109
    elif attack == 'snapshot_order': paired['candidate_snapshot_before_reference'] = False
    with pytest.raises(ValueError): load_native_measurements(TASK, CONFIG, report=report, request=request)


def test_mla_case_cannot_fall_back_to_fixed_fixture_receipt():
    from src.paired_workload import validate_realized_work
    case = MANIFEST['cases'][-1]
    schedule = {'warmup_inputs': [], 'measured_inputs': []}
    with pytest.raises(ValueError, match='Variable work'):
        validate_realized_work(case, MANIFEST, {}, {}, schedule, {'kind': 'fixed_fixture_v1'})


def test_consumer_checks_actual_manifest_bound_recipe_bytes(monkeypatch, tmp_path):
    report, request, _ = complete_report(monkeypatch, 912)
    names = ['cases.json', 'source/native.py', 'ut/baseline/native.py',
             'ut/paired_input_validation.py', 'ut/mla_control_distribution.json',
             'provenance/PAIRED-REFERENCE.json']
    for name in names:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((TASK / name).read_bytes())
    with (tmp_path / 'ut/mla_control_distribution.json').open('ab') as handle:
        handle.write(b' ')
    with pytest.raises(ValueError, match='MLA recipe digest'):
        load_native_measurements(tmp_path, CONFIG, report=report, request=request)


@pytest.mark.parametrize('name', ['ut/paired_reference.py', 'ut/mla_paired.py', 'ut/paired_input_validation.py', 'ut/native.py', 'ut/native_write_ownership.py'])
def test_new_paired_helpers_are_pinned_by_the_protected_protocol(monkeypatch, name):
    from src.tools import custom_perf_protocols
    original = custom_perf_protocols._read_file
    def altered(task, relative):
        data = original(task, relative)
        return data + b'\n# changed\n' if relative == name else data
    monkeypatch.setattr(custom_perf_protocols, '_read_file', altered)
    with pytest.raises(ValueError, match='reviewed implementation'):
        custom_perf_protocols.custom_protocol_family(TASK, {TASK / 'scripts/task_runner.py'})


@pytest.mark.parametrize('failure', ['calibration_compare', 'reference_snapshot', 'reference_input_check'])
def test_actual_runner_reference_buffers_are_scrubbed_on_errors(monkeypatch, failure):
    import ast
    source = ast.parse((TASK / 'scripts/task_runner.py').read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef)
             and node.name in ('engage_specialization', 'verify_after_snapshot')]
    reference_output = torch.tensor([3.0]); actual_output = torch.tensor([2.0]); events = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    def clone(value):
        events.append('reference_snapshot' if value is reference_output else 'candidate_snapshot')
        if failure == 'reference_snapshot' and value is reference_output: raise RuntimeError(failure)
        return value.clone()
    def invoke(fn, values): events.append('reference_invoke'); return reference_output
    def check(values, truth):
        if failure == 'reference_input_check' and values == 'reference': raise RuntimeError(failure)
    def compare(*args): raise AssertionError('calibration mismatch')
    namespace = {'storage_snapshots': lambda values: {}, 'invoke': invoke, 'cpu_clone': clone,
                 'assert_immutable_inputs': check, 'compare': compare,
                 'restore_storages': lambda *args: None, 'clear_owned': pair.clear_owned,
                 'owned_outputs': lambda fn: SimulatedWriteScope(reference_output),
                 'leaves': lambda value: [value]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual-protected-runner>', 'exec'), namespace)
    with pytest.raises((RuntimeError, AssertionError)):
        if failure == 'calibration_compare':
            namespace['engage_specialization'](None, 'reference', None, 0.02, 'calibration', scrub_output=True)
        else:
            namespace['verify_after_snapshot'](actual_output, 'candidate', {}, None, 'reference', 0.02)
    assert bool((raw_storage(reference_output) == 0xAA).all())
    if failure != 'calibration_compare': assert events.index('candidate_snapshot') < events.index('reference_invoke')


def test_fully_rehashed_wrong_but_observed_length_is_rejected(monkeypatch):
    report, request, _ = complete_report(monkeypatch, 2871)
    row = report['cases'][-1]
    paired = report['paired_reference_comparison']['cases'][-1]
    old = paired['input_schedule']['measured_inputs'][0]
    length = 193 if old['length'] == 192 else 192
    indices, indptr = captured_parent_controls()
    controls = recipe.controls_for_length(indices, indptr, length)
    forged = {**old, 'length': length,
        'kv_indices_sha256': hashlib.sha256(struct.pack('<16384i', *controls[0])).hexdigest(),
        'kv_indptr_sha256': hashlib.sha256(struct.pack('<65i', *controls[1])).hexdigest()}
    paired['input_schedule']['measured_inputs'][0] = dict(forged)
    row['realized_input_signatures'][10] = dict(forged)
    proof = row['control_distribution']
    for leg in ('candidate', 'reference'): proof['prepared_state_checks'][leg][10] = dict(forged)
    proof['measured_values'][0] = length
    proof['measured_histogram'] = dict(Counter(proof['measured_values']))
    proof['selection_trace_sha256'] = fingerprint(proof['warmup_values'] + proof['measured_values'])
    row['paired_control_timing']['measured_pairs'][0]['length'] = length
    rewrite_schedule_hashes(report)
    with pytest.raises(ValueError, match='recipe/request draw'):
        load_native_measurements(TASK, CONFIG, report=report, request=request)


@pytest.mark.parametrize('field', ['paired_control_registry', 'control_distribution', 'realized_input_signatures', 'paired_control_timing'])
def test_missing_realized_receipt_is_rejected(monkeypatch, field):
    report, request, _ = complete_report(monkeypatch, 781)
    report['cases'][-1].pop(field)
    with pytest.raises(ValueError): load_native_measurements(TASK, CONFIG, report=report, request=request)


@pytest.mark.parametrize('field', ['raw_pair', 'mean'])
def test_control_timing_receipt_cannot_contradict_raw_samples(monkeypatch, field):
    report, request, _ = complete_report(monkeypatch, 781)
    timing = report['cases'][-1]['paired_control_timing']
    if field == 'raw_pair': timing['measured_pairs'][0]['reference_ms'] += 1.0
    else: timing['candidate_mean_ms'] += 1.0
    with pytest.raises(ValueError, match='raw paired samples'):
        load_native_measurements(TASK, CONFIG, report=report, request=request)


def test_fixed_case_schedule_cannot_swap_its_captured_fixture(monkeypatch):
    report, request, _ = complete_report(monkeypatch, 781)
    outer = report['cases'][0]; paired = report['paired_reference_comparison']['cases'][0]
    for entry in paired['input_schedule']['warmup_inputs'] + paired['input_schedule']['measured_inputs']:
        entry['fixture'] = MANIFEST['cases'][1]['fixture']
    digest = fingerprint(paired['input_schedule']); outer['paired_schedule_sha256'] = digest
    for leg in paired['legs'].values(): leg['paired_schedule_sha256'] = digest
    with pytest.raises(ValueError, match='Fixed realized fixture'):
        load_native_measurements(TASK, CONFIG, report=report, request=request)


def test_cached_hook_bytecode_cannot_accept_a_forged_full_MLA_report(monkeypatch, tmp_path):
    import importlib.util
    import py_compile
    report, request, _ = complete_report(monkeypatch, 319)
    outer = report['cases'][-1]; paired = report['paired_reference_comparison']['cases'][-1]
    paired['input_schedule']['measured_inputs'][0]['kv_indices_sha256'] = '0' * 64
    outer['realized_input_signatures'][10]['kv_indices_sha256'] = '0' * 64
    for leg in ('candidate', 'reference'):
        outer['control_distribution']['prepared_state_checks'][leg][10]['kv_indices_sha256'] = '0' * 64
    rewrite_schedule_hashes(report)
    names = ['cases.json', 'source/native.py', 'ut/baseline/native.py',
             'ut/paired_input_validation.py', 'ut/mla_control_distribution.json',
             'provenance/PAIRED-REFERENCE.json']
    for name in names:
        target = tmp_path / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((TASK / name).read_bytes())
    validator = tmp_path / 'ut/paired_input_validation.py'
    verified_source = validator.read_bytes()
    validator.write_text('def validate_paired_input_receipts(*args):\n    return True\n')
    py_compile.compile(str(validator), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    validator.write_bytes(verified_source)
    assert Path(importlib.util.cache_from_source(str(validator))).is_file()
    with pytest.raises(ValueError, match='recipe/request draw'):
        load_native_measurements(tmp_path, CONFIG, report=report, request=request)
