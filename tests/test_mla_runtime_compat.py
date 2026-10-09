"""Exercise the gfx950 MLA pipeline regression on actual HIP hardware."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.perf_helper_materialization import materialize_perf_helpers_in_workspace


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
shape = h.TEST_SHAPES[3]
for seed in (7, 45, 46, 123):
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
