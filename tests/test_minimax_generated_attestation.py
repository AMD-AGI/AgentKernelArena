"""Actual trusted-worker regressions for the independent MiniMax audit findings."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
from head_kernel_generated_test_utils import generated_helper, generated_task
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "tasks/head_kernels/_support"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def actual_worker(tmp_path, monkeypatch):
    task = tmp_path / "task"
    for name in ("scripts", "ut", "source", "build"):
        (task / name).mkdir(parents=True)
    for name in ("_trusted_worker.py", "runtime_integrity.py", "_bench.py"):
        shutil.copyfile(SUPPORT / name, task / "scripts" / name)
    shutil.copyfile(ROOT / "src/tools/perf/aka_benchmark.py", task / "scripts/_aka_benchmark.py")
    shutil.copyfile(generated_helper("minimax", "generated_worker.py"), task / "scripts/generated_worker.py")
    shutil.copyfile(generated_helper("minimax", "generated_contract.py"), task / "ut/generated_contract.py")
    (task / "scripts/runtime_preflight.py").write_text("def require_runtime(cfg, **kwargs): pass\n")
    (task / "ut/harness_lib.py").write_text(
        "import torch\ndef correct(out, ref, tol): return bool(torch.equal(out, ref)), 0.0\n")
    (task / "scripts/probe.py").write_text('''import importlib.util, json
from pathlib import Path
import sys
import torch
import generated_contract
import generated_worker


def main():
    task = Path(__file__).resolve().parents[1]
    if sys.modules["probe_worker"].main is not main:
        raise RuntimeError("worker executed an unattested copy of its entrypoint")
    if sys.modules["generated_contract_alias"] is not generated_contract:
        raise RuntimeError("trusted aliases do not share the actual helper object")
    spec = importlib.util.spec_from_file_location("candidate", task / "source/kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    q = torch.tensor([1., 2., 3.])
    out = generated_worker.invoke_checked(module.kernel, (), {"q": q}, generated_contract, torch, [])
    print(generated_worker.PREFIX + json.dumps({"schema_version": 1, "profile": "recorded",
          "seed": 731, "reference": False, "rows": [{"id": "case", "output": out}]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')
    (task / "config.yaml").write_text(yaml.safe_dump({"headkernel": {"generated_input_revision": "cpu-test-v1", "trusted_worker_modules": {
        "generated_contract": "ut/generated_contract.py",
        "generated_contract_alias": "ut/generated_contract.py",
        "generated_worker": "scripts/generated_worker.py",
        "_bench": "scripts/_bench.py", "probe_worker": "scripts/probe.py",
    }}}))
    runner = load(SUPPORT / "task_runner.py", "attestation_runner")
    runner.TASK_DIR, runner.UT_DIR, runner.BUILD_DIR = task, task / "ut", task / "build"
    original_command = runner.worker_command
    # Mock only availability of the CPU test environment, not the trusted worker,
    # module loader, monitor, completion nonce, or parent result parser.
    package_paths = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    bootstrap = ("import json,runpy,sys\n"
                 "sys.path[:0]=json.loads(sys.argv.pop(1))\n"
                 "sys.argv=sys.argv[1:]\n"
                 "runpy.run_path(sys.argv[0],run_name='__main__')\n")

    def command(*args, **kwargs):
        actual = original_command(*args, **kwargs)
        return [sys.executable, "-B", "-c", bootstrap, json.dumps(package_paths), *actual[2:]]

    monkeypatch.setattr(runner, "worker_command", command)
    controller = load(generated_helper("minimax", "generated_correctness.py"), "attestation_controller")
    contract = load(generated_helper("minimax", "generated_contract.py"), "attestation_contract")
    expected = [{"id": "case", "output": contract.encode_output(torch.tensor([2., 4., 6.]), torch)}]

    def execute(source):
        (task / "source/kernel.py").write_text(source)
        proc = runner.run_worker(task / "scripts/probe.py", [], None, 30, True)
        accepted = False
        error = proc.stderr
        if proc.returncode == 0:
            try:
                rows = controller.parse_worker(proc, "recorded", 731, False, ["case"])
                controller.compare_workers(expected, rows, contract, 0.02, torch)
                accepted = True
            except RuntimeError as exc:
                error = str(exc)
        return accepted, proc, error

    return task, execute


ATTACKS = {
    "input_checker": '''import sys

def kernel(q):
    result = q * 2
    sys.modules['generated_contract'].require_inputs_unchanged = lambda *args: None
    q.add_(100)
    return result
''',
    "encoder_stride": '''import sys, torch

def kernel(q):
    contract = sys.modules['generated_contract']
    original = contract.encode_output
    def forge(value, torch):
        wire = original(value, torch)
        wire['stride'] = [1]
        return wire
    contract.encode_output = forge
    result = torch.empty(6)[::2]
    result.copy_(q * 2)
    return result
''',
    "module_alias": '''import sys, types

def kernel(q):
    sys.modules['generated_contract'] = types.ModuleType('generated_contract')
    return q * 2
''',
    "secondary_alias": "import sys, types\n\ndef kernel(q):\n    sys.modules['generated_contract_alias'] = types.ModuleType('replacement')\n    return q * 2\n",
    "worker_function": '''import sys

def kernel(q):
    sys.modules['generated_worker'].invoke_checked = lambda *args: None
    return q * 2
''',
    "entrypoint_function": '''import sys

def kernel(q):
    sys.modules['probe_worker'].main = lambda: 0
    return q * 2
''',
    "helper_code": '''import sys

def kernel(q):
    sys.modules['generated_contract'].require_inputs_unchanged.__code__ = (lambda *a: None).__code__
    return q * 2
''',
    "benchmark_class_method": '''import sys

def kernel(q):
    sys.modules['_bench'].InputState.restore = lambda self: None
    return q * 2
''',
    "tensor_stride": '''import torch

def kernel(q):
    torch.Tensor.stride = lambda self: (1,)
    return q * 2
''',
    "tensor_class": "import torch\n\ndef kernel(q):\n    result = q * 2\n    torch.Tensor = object\n    return result\n",
    "tensor_numpy": "import torch\n\ndef kernel(q):\n    result = q * 2\n    torch.Tensor.numpy = lambda self: None\n    return result\n",
    "storage_pointer": "import torch\n\ndef kernel(q):\n    result = q * 2\n    torch.UntypedStorage.data_ptr = lambda self: 123\n    return result\n",
    "base64_encoder": '''import base64

def kernel(q):
    base64.b64encode = lambda *args: b''
    return q * 2
''',
}


@pytest.mark.parametrize("attack", ATTACKS)
def test_actual_trusted_completion_rejects_mutated_helpers(actual_worker, attack):
    task, execute = actual_worker
    accepted, proc, error = execute(ATTACKS[attack])
    assert not accepted, f"trusted worker plus parent comparison accepted {attack}"
    assert proc.returncode != 0, f"tampering reached trusted completion: {attack}: {error}"
    assert "IntegrityError" in error
    assert "imported protected evaluation module" not in error
    assert not list((task / "build").glob("_worker_completion_*"))


def test_actual_trusted_worker_accepts_real_unchanged_source(actual_worker):
    _, execute = actual_worker
    accepted, proc, error = execute("def kernel(q): return q * 2\n")
    assert accepted, (proc.stdout, error)


def test_parent_rejects_unforged_wrong_stride(actual_worker):
    _, execute = actual_worker
    accepted, proc, error = execute('''import torch

def kernel(q):
    result = torch.empty(6)[::2]
    result.copy_(q * 2)
    return result
''')
    assert proc.returncode == 0, error
    assert not accepted
    assert "correctness mismatch" in error


@pytest.mark.parametrize("dtype, expected, observed", [
    (torch.int32, [0, 32, 64], [0, 33, 64]),
    (torch.int64, [2**40, 2**40 + 1], [2**40 + 1, 2**40 + 1]),
    (torch.bool, [False, True], [True, True]),
])
def test_integer_and_bool_outputs_are_exact(dtype, expected, observed):
    contract = load(generated_helper("minimax", "generated_contract.py"), "exact_integer_contract")
    good = contract.encode_output(torch.tensor(expected, dtype=dtype), torch)
    bad = contract.encode_output(torch.tensor(observed, dtype=dtype), torch)
    assert contract.compare_output(good, good, 0.02, torch)
    assert not contract.compare_output(bad, good, 0.02, torch)
    for path in (ROOT / "tasks/head_kernels/minimax-m3-mxfp4").rglob("ut/harness_lib.py"):
        harness = load(path, "exact_integer_harness")
        assert harness.correct(torch.tensor(expected, dtype=dtype), torch.tensor(expected, dtype=dtype), 0.02)[0]
        assert not harness.correct(torch.tensor(observed, dtype=dtype), torch.tensor(expected, dtype=dtype), 0.02)[0]


def test_preload_cannot_run_candidate_before_attestation(actual_worker):
    task, execute = actual_worker
    helper = task / "ut/bad_preload.py"
    helper.write_text('''import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parents[1]/"source/kernel.py"
spec=importlib.util.spec_from_file_location("too_early",path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
''')
    config = yaml.safe_load((task / "config.yaml").read_text())
    config["headkernel"]["trusted_worker_modules"] = {
        "bad_preload": "ut/bad_preload.py", **config["headkernel"]["trusted_worker_modules"]}
    (task / "config.yaml").write_text(yaml.safe_dump(config))
    accepted, proc, error = execute("def kernel(q): return q * 2\n")
    assert not accepted and proc.returncode != 0
    assert "preload attempted candidate access before attestation" in error


def test_performance_comparator_rejects_non_tensor_proxy_outputs():
    reference = torch.tensor([1., 2., 3.])
    class FakeOutput:
        shape = reference.shape
        dtype = reference.dtype
        device = reference.device
        def float(self):
            return reference  # A forged comparator-time computation must not count as a timed output.
    for path in (ROOT / "tasks/head_kernels/minimax-m3-mxfp4").rglob("ut/harness_lib.py"):
        harness = load(path, "plain_tensor_output_harness")
        assert not harness.correct(FakeOutput(), reference, 0.02)[0]


def test_worker_entrypoint_cannot_fall_back_to_an_unattested_copy(actual_worker):
    task, execute = actual_worker
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    del cfg["headkernel"]["trusted_worker_modules"]["probe_worker"]
    (task / "config.yaml").write_text(yaml.safe_dump(cfg))
    accepted, proc, error = execute("def kernel(q): return q * 2\n")
    assert not accepted and proc.returncode != 0
    assert "entrypoint is missing from headkernel.trusted_worker_modules" in error


def test_generated_revision_requires_module_attestation(actual_worker):
    task, execute = actual_worker
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    del cfg["headkernel"]["trusted_worker_modules"]
    (task / "config.yaml").write_text(yaml.safe_dump(cfg))
    accepted, proc, error = execute("def kernel(q): return q * 2\n")
    assert not accepted and proc.returncode != 0
    assert "generated_input_revision requires headkernel.trusted_worker_modules" in error
