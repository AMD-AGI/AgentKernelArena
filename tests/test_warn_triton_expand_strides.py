"""The expansion wrapper and controls must honor legal 1-D input views."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


TASK = Path(__file__).resolve().parents[1] / 'tasks/triton2triton/vllm/triton_expand'


def checks_module():
    spec = importlib.util.spec_from_file_location('expand_stride_checks', TASK / '_arena_checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_wrapper(kernel):
    source = TASK / 'source/triton_expand.py'
    fn = next(node for node in ast.parse(source.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == 'expand_batch_to_tokens')
    scope = {'torch': torch, 'expand_kernel': kernel, 'MAX_SPEC_LEN': 128}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    return scope['expand_batch_to_tokens']


@pytest.mark.parametrize('strided_operand', ['neither', 'source', 'counts', 'both'])
def test_public_wrapper_materializes_only_noncontiguous_inputs(strided_operand):
    checks = checks_module()
    source_backing = torch.tensor([7, 99, 3, 99, 9, 99], dtype=torch.int32)
    count_backing = torch.tensor([1, 2, 4, 4, 4, 4], dtype=torch.int64)
    x = source_backing[::2] if strided_operand in {'source', 'both'} else torch.tensor([7, 3, 9])
    cu = count_backing[::2] if strided_operand in {'counts', 'both'} else torch.tensor([1, 4, 4])
    pristine = source_backing.clone(), count_backing.clone(), x.clone(), cu.clone()
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            assert grid == (3,)

            def launch(out, actual_x, actual_cu, old, new, *, MAX_NUM_TOKENS):
                assert MAX_NUM_TOKENS == 128
                assert actual_x.is_contiguous() and actual_cu.is_contiguous()
                launches.append((actual_x, actual_cu))
                out.copy_(checks.reference(actual_x, actual_cu, len(out), old, new))

            return launch

    result = public_wrapper(Kernel())(x, cu, 4)
    checks.check_output(result, checks.reference(x, cu, 4))
    assert len(launches) == 1
    assert (launches[0][0] is x) == (strided_operand not in {'source', 'both'})
    assert (launches[0][1] is cu) == (strided_operand not in {'counts', 'both'})
    assert all(torch.equal(now, saved) for now, saved in zip(
        (source_backing, count_backing, x, cu), pristine))


@pytest.mark.parametrize('broken', [None, 'ignore_source_stride', 'ignore_count_stride'])
def test_protected_controls_reject_each_unit_stride_assumption(broken):
    checks = checks_module()
    seen = []

    def expand(x, cu, num_tokens, old=0, new=0):
        seen.append((x.dtype, cu.dtype, x.stride(0), cu.stride(0)))
        if broken == 'ignore_source_stride' and x.stride(0) != 1:
            x = x.as_strided(x.shape, (1,))
        if broken == 'ignore_count_stride' and cu.stride(0) != 1:
            cu = cu.as_strided(cu.shape, (1,))
        return checks.reference(x, cu, num_tokens, old, new)

    module = SimpleNamespace(expand_batch_to_tokens=expand)
    harness = SimpleNamespace(load_module=lambda: module)

    def run():
        with checks.checked_modules(harness):
            actual = harness.load_module().expand_batch_to_tokens(
                torch.tensor([3, 9], dtype=torch.int32),
                torch.tensor([2, 4], dtype=torch.int32), 4)
            assert actual.tolist() == [3, 3, 9, 9]

    if broken is None:
        run()
        assert len(seen) == 37
        assert any(source_stride == 2 and count_stride == 1
                   for _, _, source_stride, count_stride in seen)
        assert {count_dtype for _, count_dtype, source_stride, count_stride in seen
                if source_stride == 1 and count_stride == 2} == {torch.int32, torch.int64}
    else:
        with pytest.raises(AssertionError, match='Expansion output differs'):
            run()
    assert module.expand_batch_to_tokens is expand
