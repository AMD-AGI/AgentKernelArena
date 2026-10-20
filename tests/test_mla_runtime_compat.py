"""Exercise MLA output/routing contracts and the gfx950 pipeline regression."""
import ast
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace


def _harness_namespace():
    task = Path(__file__).resolve().parents[1] / "tasks/triton2flydsl/aiter/mla"
    spec = importlib.util.spec_from_file_location("mla_replay_controls", task / "scripts/replay_checks.py")
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    selected = {"_ref_masked_attention", "torch_mla_extend", "_call_kernel", "_timed_mla_call",
                "_validate_mla_paged_variants", "_checked_mla_output", "_compare_mla_output", "_compare"}
    tree = ast.parse((task / "test_kernel_harness.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    namespace = dict(NORM_ERR_TOL=.01, ALLCLOSE_TOL=.01,
                     require_tensor_contract=checks.require_tensor_contract,
                     require_unchanged=checks.require_unchanged)
    exec(compile(ast.Module(body=functions, type_ignores=[]), "mla_contract_functions", "exec"), namespace)
    return namespace


def test_measured_invocations_cannot_reuse_warmup_output():
    import types
    torch = pytest.importorskip("torch")
    h = _harness_namespace()
    out = torch.ones((1, 2, 2), dtype=torch.bfloat16)
    computations = []
    def cached_kernel(*args):
        destination = args[2]
        if not bool(torch.isfinite(destination).all()):
            computations.append(True)
            destination.fill_(1)
        return destination
    module = types.SimpleNamespace(mla_decode_fwd=cached_kernel)
    args = (module, out, out, out, None, None, 1, None, 1., 2, 1)
    for _ in range(3):
        assert h['_timed_mla_call'](*args) is out
    assert len(computations) == 3


@pytest.mark.parametrize("fault", [None, "linear_pages", "uniform_lengths"])
def test_gpu_routing_variants_reject_address_and_length_shortcuts(fault, monkeypatch):
    import types
    torch = pytest.importorskip("torch")
    h = _harness_namespace()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    q = torch.zeros((2, 2, 3), dtype=torch.bfloat16)
    kv = torch.arange(36, dtype=q.dtype).reshape(6, 2, 1, 3)
    out = torch.empty((2, 2, 2), dtype=q.dtype)
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    used = torch.tensor([4, 4], dtype=torch.int32)
    def compute(*args):
        aq, akv, ao, acu, au, _, at, scale, lora = args[:9]
        if fault == "linear_pages":
            at = torch.arange(at.numel(), dtype=at.dtype).reshape_as(at)
        if fault == "uniform_lengths":
            au = torch.full_like(au, 4)
        ao.copy_(h['torch_mla_extend'](aq, akv, acu, au, at, lora, scale, aq.dtype))
        return ao
    module = types.SimpleNamespace(mla_decode_fwd=compute)
    invoke = lambda: h['_validate_mla_paged_variants'](module, q, kv, out, cu, used, 4, table, 1., 2, 1)
    if fault is None:
        invoke()
    else:
        with pytest.raises(AssertionError, match="Numerical mismatch"):
            invoke()


def test_gfx950_mla_repeated_decode_matches_reference(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    source = Path(__file__).resolve().parents[1] / "tasks/triton2flydsl/aiter/mla"
    task = tmp_path / "mla"
    shutil.copytree(source, task)
    materialize_perf_helpers_in_workspace(task)
    # The harness changes cwd and owns task-local imports; isolate its module state.
    subprocess.run([sys.executable, "-c", """
import torch
import test_kernel_harness as h
m = h.load_module()
cases = [(case, 42 + case) for case in range(len(h.TEST_SHAPES))]
cases += [(3, seed) for seed in range(32)] + [(3, 46), (3, 123)]
for case, seed in cases:
    shape = h.TEST_SHAPES[case]
    torch.manual_seed(seed)
    q, kv, out, table, cu, lengths, scale = h.make_test_data(*shape)
    reference = h.torch_mla_extend(q, kv, cu, lengths, table, shape[4], scale,
                                  o_dtype=q.dtype)
    for repeat in range(3):
        out.fill_(float('nan'))
        actual = h._call_kernel(m, q, kv, out, cu, lengths, shape[-1], table,
                                scale, shape[4], shape[5])
        torch.cuda.synchronize()
        h._checked_mla_output(actual, out, q, shape[4])
        h._compare_mla_output(actual, reference)
"""], cwd=task, check=True, timeout=180)
