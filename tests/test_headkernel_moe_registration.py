"""CPU checks that MoE routing engages the protected native operation safely."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


TASK = Path(__file__).resolve().parents[1] / (
    'tasks/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4'
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = _load('_qwen_moe_registration_contract', TASK / 'scripts/_routing_contract.py')
with patch.dict('sys.modules', {'_routing_contract': CONTRACT}):
    RUNTIME = _load('_qwen_moe_registration_runtime', TASK / 'scripts/_moe_runtime.py')
META = json.loads((TASK / 'ut/meta.json').read_text())
SOURCE = TASK / CONTRACT.POLICY_FILE


class MoENativeRoutingTests(unittest.TestCase):
    def test_policy_edit_reaches_native_operation_and_preserves_baseline_arguments(self):
        original = SOURCE.read_bytes()
        # Submit an admitted route change through the real source parser.
        changed = original.replace(b'"captured"', b'"block64"', 1)
        original_routes = CONTRACT.parse_policy(original, META)
        candidate_routes = CONTRACT.parse_policy(changed, META)
        calls = []
        native = object.__new__(RUNTIME.NativeRuntime)
        sentinel = object()

        def production(**kwargs):
            calls.append(kwargs)
            return sentinel

        native.fn = production
        kwargs = {
            'hidden_states': object(), 'w1': object(), 'w2': object(),
            'w1_scale': object(), 'w2_scale': object(), 'expert_mask': object(),
            'topk_ids': object(), 'topk_weight': object(), 'activation': object(),
            'quant_type': object(), 'doweight_stage1': False, 'hidden_pad': 0,
            'intermediate_pad': 0, 'a1_scale': None, 'a2_scale': None,
            'bias1': None, 'bias2': None,
        }
        for case in CONTRACT.specs(META):
            signature = case['sig']
            self.assertIs(native.call(kwargs, candidate_routes[signature]), sentinel)
            self.assertIs(native.call(kwargs, original_routes[signature]), sentinel)
        self.assertEqual(calls[0], dict(kwargs, block_size_M=64))
        self.assertEqual(calls[1:], [kwargs, kwargs, kwargs])
        self.assertNotIn('block_size_M', kwargs)
        self.assertIs(native.fn, production)

    def test_native_dispatch_failure_propagates_without_fallback(self):
        native = object.__new__(RUNTIME.NativeRuntime)
        calls = []

        def unsupported(**kwargs):
            calls.append(kwargs)
            raise RuntimeError('unsupported native block geometry')

        native.fn = unsupported
        with self.assertRaisesRegex(RuntimeError, 'unsupported native block geometry'):
            native.call({'hidden_states': 'input'}, 'block128', collector=object())
        self.assertEqual(calls, [{'hidden_states': 'input', 'block_size_M': 128}])
        self.assertIsNone(native.collector)
        self.assertIs(native.fn, unsupported)

    def test_candidate_cannot_register_a_custom_operation_or_supply_the_baseline(self):
        source = SOURCE.read_bytes()
        programs = [
            b'import torch\n' + source,
            b'from candidate import baseline_callable\n' + source,
            source.replace(b'def select_routes():', b'@torch.library.custom_op("aiter::fused_moe_")\ndef select_routes():'),
            source + b'\ndef baseline_callable():\n    return lambda **kwargs: None\n',
            source.replace(b'"captured"', b'baseline_callable()', 1),
        ]
        for program in programs:
            with self.subTest(program=program), self.assertRaises((ValueError, SyntaxError)):
                CONTRACT.parse_policy(program, META)

    def test_changed_or_task_owned_native_source_is_rejected_before_gpu_access(self):
        # NativeRuntime must reject image substitution before Torch or AITER can
        # launch anything. The stubs intentionally expose no GPU methods.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / 'task'
            task.mkdir()
            image_file = root / 'image_fused_moe.py'
            image_file.write_text('changed image source')
            module = SimpleNamespace(__file__=str(image_file))
            with patch.object(RUNTIME.importlib, 'import_module', return_value=module):
                with self.assertRaisesRegex(ValueError, 'protected SG520 source'):
                    RUNTIME.NativeRuntime(task, object(), META)
            image_file = task / 'fused_moe.py'
            image_file.write_bytes((TASK / 'ut' / META['source_provenance']['baseline_ref']).read_bytes())
            module.__file__ = str(image_file)
            with patch.object(RUNTIME.importlib, 'import_module', return_value=module):
                with self.assertRaisesRegex(ValueError, 'protected SG520 source'):
                    RUNTIME.NativeRuntime(task, object(), META)


if __name__ == '__main__':
    unittest.main()
