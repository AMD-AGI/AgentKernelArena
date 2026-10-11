"""Protected image binding, captured ABI and CPU-owned MoE observations."""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import sys

from _routing_contract import case_params, file_sha, require, specs, strict_json


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensor_view(tensor):
    return tensor.view(importlib.import_module('torch').uint8) if str(tensor.dtype).startswith('torch.float4_') else tensor


def cpu_snapshot(tensor):
    return tensor_view(tensor).detach().cpu().clone()


def to_device(torch, tensor):
    # FP4 copy kernels are unavailable; transfer its packed bytes then re-view.
    value = tensor_view(tensor).to('cuda').view(tensor.dtype)
    require(tuple(value.shape) == tuple(tensor.shape) and tuple(value.stride()) == tuple(tensor.stride()),
            'captured tensor shape/stride changed during transfer')
    if getattr(tensor, 'is_shuffled', False):
        value.is_shuffled = True
    return value


class NativeRuntime:
    def __init__(self, task, torch, meta):
        self.task, self.torch, self.meta = Path(task), torch, meta
        self.native = importlib.import_module('aiter.fused_moe')
        path = Path(self.native.__file__).resolve()
        require(not path.is_relative_to(self.task) and file_sha(path) == meta['source_provenance']['source_sha256'],
                'image fused_moe source differs from the protected SG520 source')
        require(self.native.fused_moe.__module__ == 'aiter.fused_moe', 'native fused_moe was rebound')
        require(importlib.metadata.version('sglang') == '0.5.20', 'SGLang 0.5.20 runtime required')
        require(torch.cuda.is_available(), 'a ROCm GPU is required')
        require(str(torch.cuda.get_device_properties(0).gcnArchName).split(':')[0] == 'gfx950',
                'captured native MoE task requires gfx950')
        self.fn = self.native.fused_moe
        self.collector = None
        self.sorting = self.native.moe_sorting
        self.dispatcher = self.native.get_2stage_cfgs
        self.native.moe_sorting = self._sorting
        self.native.get_2stage_cfgs = self._dispatch
        self.cases = load_module('_qwen_moe_captured_cases', self.task / 'ut/cases.py')
        self.harness = load_module('_qwen_moe_correctness', self.task / 'ut/harness_lib.py')
        fixture = self.task / 'ut' / meta['reference_io']
        require(fixture.is_file() and not fixture.is_symlink(), 'regular captured fixture required')
        require(fixture.stat().st_size == 2083145699 and file_sha(fixture) == meta['reference_io_sha256'],
                'captured fixture size/SHA256 changed')
        capture = strict_json((self.task / 'ut/capture_meta.json').read_text())
        telemetry = strict_json((self.task / 'ut/capture_telemetry.json').read_text())
        require(capture['reference_io_sha256'] == telemetry['reference_io_sha256'] == meta['reference_io_sha256']
                and capture['oracle_complete'] and not capture['budget_exceeded']
                and telemetry['checksum_verified_once_after_promotion']
                and set(capture['regimes_covered']) == {'decode', 'prefill'}, 'capture provenance incomplete')
        self.cpu_cases = self.cases.load_live_cases('cpu')
        self.device_shared = {}

    def _sorting(self, *args, **kwargs):
        result = self.sorting(*args, **kwargs)
        if self.collector is not None:
            self.collector.scratch = list(result)
        return result

    def _dispatch(self, *args, **kwargs):
        result = self.dispatcher(*args, **kwargs)
        if self.collector is not None:
            self.collector.dispatch = {
                'native_block_m': int(result.block_m), 'run_1stage': bool(result.run_1stage),
                'stage1': repr(result.stage1), 'stage2': repr(result.stage2),
            }
        return result

    def call(self, kwargs, route='captured', collector=None):
        tuned = dict(kwargs)
        if route != 'captured':
            require(route in ('block32', 'block64', 'block128'), 'unsupported native route')
            tuned['block_size_M'] = int(route[5:])
        self.collector = collector
        try:
            return self.fn(**tuned)
        finally:
            self.collector = None

    def case(self, spec):
        case = self.cpu_cases[(spec['regime'], spec['m'])]
        cpu = case['args']['kwargs']
        geo = self.meta['geometry']
        expected = {
            'hidden_states': (spec['hidden_shape'], 'torch.bfloat16'),
            'w1': (geo['w1_shape'], 'torch.float4_e2m1fn_x2'),
            'w2': (geo['w2_shape'], 'torch.float4_e2m1fn_x2'),
            'w1_scale': (geo['w1_scale_shape'], 'torch.uint8'),
            'w2_scale': (geo['w2_scale_shape'], 'torch.uint8'),
            'topk_ids': ([spec['m'], geo['topk']], 'torch.int32'),
            'topk_weight': ([spec['m'], geo['topk']], 'torch.float32'),
            'expert_mask': ([geo['expert_mask_size']], 'torch.int32'),
        }
        for name, (shape, dtype) in expected.items():
            require(name in cpu and list(cpu[name].shape) == shape and str(cpu[name].dtype) == dtype,
                    'captured MoE ABI changed: ' + name)
        require(list(cpu['hidden_states'].stride()) == spec['hidden_stride'], 'captured hidden stride changed')
        require(all(getattr(cpu[name], 'is_shuffled', False) for name in ('w1', 'w2')), 'packed weight layout changed')
        require(tuple(case['ref'].shape) == tuple(cpu['hidden_states'].shape)
                and case['ref'].dtype == cpu['hidden_states'].dtype, 'captured output ABI changed')
        gpu = {}
        for name, value in cpu.items():
            if self.torch.is_tensor(value):
                key = id(value)
                if key not in self.device_shared:
                    self.device_shared[key] = to_device(self.torch, value)
                gpu[name] = self.device_shared[key]
            else:
                gpu[name] = value
        return cpu, gpu, case['ref']


class ReplayState:
    def __init__(self, runtime, spec, route, seed):
        self.runtime, self.spec, self.route = runtime, spec, route
        self.torch = runtime.torch
        self.cpu, self.args, self.captured_ref = runtime.case(spec)
        self.params = case_params(runtime.meta, spec)
        self.rng = self.torch.Generator(device='cpu').manual_seed(seed)
        self.hidden_cpu = None
        self.outputs = None
        self.scratch = []
        self.dispatch = None
        self.capture_pending = False
        self.needs_validation = False
        self.checked_invocations = 0
        self.generated = 0
        self.worst_error = 0.0
        self.tol = float(runtime.meta['tol'])
        self.use_captured = False
        self.check_captured = False

    def prepare(self):
        self.validate()
        self.capture_pending = False
        template = self.cpu['hidden_states']
        self.hidden_cpu = self.torch.empty_strided(template.shape, template.stride(), dtype=template.dtype, device='cpu')
        if self.use_captured:
            self.hidden_cpu.copy_(template)
        else:
            self.hidden_cpu.uniform_(-0.75, 0.875, generator=self.rng)
            self.hidden_cpu.reshape(-1)[0] = 0.5
        self.args['hidden_states'].copy_(self.hidden_cpu)
        self.generated += 1
        # Sorting must rewrite all exposed scratch and initialize atomic outputs.
        # Poison is outside the graph/event interval for both baseline and candidate.
        seen = set()
        for value in [self.outputs, *self.scratch]:
            if self.torch.is_tensor(value) and value.data_ptr() not in seen:
                seen.add(value.data_ptr())
                value.fill_(float('nan') if value.is_floating_point() else -1)
        self.needs_validation = True

    def run(self):
        self.outputs = self.runtime.call(self.args, self.route, self)
        self.capture_pending = self.torch.cuda.is_current_stream_capturing()
        return self.outputs

    def compare(self, observed, expected):
        require(expected.device.type == observed.device.type == 'cpu', 'expected and observed outputs must be CPU-owned')
        ok, error = self.runtime.harness.correct(observed, expected, self.tol)
        require(ok and math.isfinite(error), f'MoE output failed tolerance {self.tol}: {error}')
        self.worst_error = max(self.worst_error, error)

    def validate(self):
        if self.outputs is None or self.capture_pending or not self.needs_validation:
            return
        require(tuple(self.outputs.shape) == tuple(self.cpu['hidden_states'].shape)
                and self.outputs.dtype == self.cpu['hidden_states'].dtype
                and self.outputs.device.type == 'cuda', 'native output ABI changed')
        input_ptrs = {value.untyped_storage().data_ptr() for value in self.args.values() if self.torch.is_tensor(value)}
        require(self.outputs.untyped_storage().data_ptr() not in input_ptrs, 'output aliases an input')
        require(set(self.args) == set(self.cpu), 'captured argument names changed')
        for name, value in self.args.items():
            expected = self.cpu[name]
            if self.torch.is_tensor(value):
                require(value.dtype == expected.dtype and tuple(value.shape) == tuple(expected.shape)
                        and tuple(value.stride()) == tuple(expected.stride()) and value.device.type == 'cuda',
                        'captured tensor ABI changed: ' + name)
                if name in ('w1', 'w2'):
                    require(getattr(value, 'is_shuffled', False), 'packed weight layout changed')
            else:
                require(type(value) is type(expected) and value == expected, 'captured scalar changed: ' + name)
        # Snapshot candidate outputs AND all immutable operands before native reference work.
        observed = cpu_snapshot(self.outputs)
        observed_inputs = {name: cpu_snapshot(value) for name, value in self.args.items() if self.torch.is_tensor(value)}
        for name, actual in observed_inputs.items():
            expected = self.hidden_cpu if name == 'hidden_states' else tensor_view(self.cpu[name])
            require(self.torch.equal(actual, expected), 'native callable mutated ' + name)
        del observed_inputs
        # Frozen captured output stays on CPU; fresh references are evaluated only
        # after the candidate observations above have become independent CPU copies.
        if self.check_captured:
            self.compare(observed, self.captured_ref)
        reference = self.runtime.call(self.args)
        expected = cpu_snapshot(reference)
        del reference
        self.compare(observed, expected)
        self.checked_invocations += 1
        self.needs_validation = False
