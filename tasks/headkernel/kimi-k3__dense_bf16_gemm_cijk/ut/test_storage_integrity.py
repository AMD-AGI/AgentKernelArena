"""CPU regression for complete backing storage, including the actual padded ABIs."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ut'))
import storage_guard as guard
from source_guard import validate_sources


class Tests(unittest.TestCase):
    def cases(self):
        return [case for case in json.loads((ROOT/'cases.json').read_text())['cases'] if case['tensors']['A']['strides'][0]>case['scalars']['K']]
    def test_guard_admitted_padding_store_changes_no_logical_values_but_is_rejected(self):
        import torch
        source=(ROOT/'source/kernels.py').read_text()
        source+='\n' if not source.endswith('\n') else ''
        source+='    tl.store(A + K, 0, (A_ROW_STRIDE > K) & (M > 1))\n'
        with tempfile.TemporaryDirectory() as temporary:
            stage=Path(temporary);(stage/'source').mkdir();(stage/'source/kernels.py').write_text(source)
            validate_sources(stage,ROOT)
        self.assertEqual(len(self.cases()),3)
        for case in self.cases():
            with self.subTest(case=case['case_id']):
                spec=case['tensors']['A'];size=guard.extents(case)['A']
                before=guard.allocate_view(spec,size,'cpu',fill=173);before.fill_(1)
                actual=guard.allocate_view(spec,size,'cpu');guard.copy_complete(actual,before)
                logical=actual.clone()
                # Exact GPU payload tl.store(A + K, 0, ...) at its actual address.
                torch.empty(0,dtype=actual.dtype).set_(actual.untyped_storage(),actual.storage_offset()+case['scalars']['K'],(1,),(1,)).zero_()
                self.assertTrue(torch.equal(actual,logical))
                with self.assertRaisesRegex(AssertionError,'Input backing storage mutation'):
                    guard.assert_inputs_unchanged({'A':actual},{'A':before})
    def test_prefix_tail_and_full_reset(self):
        import torch
        case=next(c for c in self.cases() if c['tensors']['A']['storage_offset'])
        spec=case['tensors']['A'];size=guard.extents(case)['A']
        before=guard.allocate_view(spec,size,'cpu',fill=219);before.fill_(2)
        actual=guard.allocate_view(spec,size,'cpu')
        for byte in (0,size-1):
            guard.copy_complete(actual,before);guard.raw_storage(actual)[byte]=0
            with self.assertRaisesRegex(AssertionError,'backing storage mutation'):guard.assert_inputs_unchanged({'A':actual},{'A':before})
            guard.copy_complete(actual,before);guard.assert_inputs_unchanged({'A':actual},{'A':before})
        self.assertEqual(actual.storage_offset(),6144)
    def test_reset_reinitializes_every_padding_byte(self):
        import torch
        case=next(c for c in self.cases() if c['tensors']['A']['storage_offset'])
        values={'A':torch.ones(case['tensors']['A']['shape'],dtype=torch.bfloat16),'B':torch.ones(case['tensors']['B']['shape'],dtype=torch.bfloat16).t().contiguous().t()}
        one=guard.fresh_input_storage(values,case,1);two=guard.fresh_input_storage(values,case,2)
        self.assertTrue(torch.equal(one['A'],two['A']))
        self.assertFalse(torch.equal(guard.raw_storage(one['A']),guard.raw_storage(two['A'])))
    def test_actual_verify_rejects_padding_before_mathematical_reference(self):
        import torch
        tree=ast.parse((ROOT/'scripts/task_runner.py').read_text());build=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='build_state');verify=next(x for x in build.body if isinstance(x,ast.FunctionDef) and x.name=='verify')
        case=next(c for c in self.cases() if c['tensors']['A']['storage_offset']);sizes=guard.extents(case)
        before={name:guard.allocate_view(case['tensors'][name],sizes[name],'cpu',fill=173) for name in ('A','B')}
        for value in before.values():value.fill_(1)
        tensors={name:guard.allocate_view(spec,sizes[name],'cpu') for name,spec in case['tensors'].items()}
        for name,value in before.items():guard.copy_complete(tensors[name],value)
        tensors['C'].fill_(128)
        guard.raw_storage(tensors['A'])[2*(6144+128)]=0
        env={'torch':torch,'storage_guard':guard,'case':case,'tensors':tensors,'native_events':[]}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[verify],type_ignores=[])),'actual_protected_verify','exec'),env)
        with patch.object(torch.cuda,'synchronize',return_value=None):
            with self.assertRaisesRegex(AssertionError,'Input backing storage mutation'):env['verify']((None,before))
if __name__=='__main__':unittest.main(verbosity=2)
