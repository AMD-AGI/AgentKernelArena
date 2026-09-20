"""Exercise the shared bootstrap, real DeepSeek helpers, and completion nonce."""

import hashlib
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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def actual_worker(tmp_path, monkeypatch):
    task = tmp_path / "task"
    for name in (
        "scripts",
        "ut/baseline_ref",
        "source",
        "build",
        "scripts/fake_native",
    ):
        (task / name).mkdir(parents=True, exist_ok=True)
    for name in ("_trusted_worker.py", "runtime_integrity.py", "_bench.py"):
        shutil.copyfile(SUPPORT / name, task / "scripts" / name)
    shutil.copyfile(
        ROOT / "src/tools/perf/aka_benchmark.py", task / "scripts/_aka_benchmark.py"
    )
    for name in ("generated_worker.py",):
        shutil.copyfile(generated_helper("deepseek", name), task / "scripts" / name)
    for name in ("generated_contract.py", "cases.py"):
        shutil.copyfile(generated_helper("deepseek", name), task / "ut" / name)
    (task / "scripts/runtime_preflight.py").write_text(
        "def require_runtime(cfg, **kwargs): pass\n"
    )
    (task / "ut/harness_lib.py").write_text(
        "import torch\ndef correct(out, ref, tol): return bool(torch.equal(out, ref)), 0.0\n"
    )
    (task / "scripts/fake_native/__init__.py").write_text(
        "def kernel(q): return q * 2\n"
    )
    baseline = task / "ut/baseline_ref/kernel.py.orig"
    baseline.write_text("def kernel(q): return q * 2\n")
    compact = {
        "schema_version": 1,
        "task_kind": "moe2",
        "source_reference_sha256": "0" * 64,
        "records": [{"sig": "case"}],
    }
    raw = json.dumps(compact).encode()
    (task / "ut/generated_cases.json").write_bytes(raw)
    meta = {
        "num_cases": 1,
        "target_callable": "fake_native:kernel",
        "archival_capture": {"reference_io_sha256": "0" * 64},
        "generated_inputs": {
            "contract_file": "generated_cases.json",
            "contract_sha256": hashlib.sha256(raw).hexdigest(),
            "record_signatures": ["case"],
            "source_package": "fake_native",
            "candidate_source": "source/kernel.py",
            "baseline_source": "baseline_ref/kernel.py.orig",
            "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        },
    }
    (task / "ut/meta.json").write_text(json.dumps(meta))
    (task / "scripts/probe.py").write_text(
        """import json, sys, torch
import generated_contract, generated_worker
import _deepseek_generated_cases as cases
import runtime_integrity

def main():
    if sys.modules['probe_worker'].main is not main:
        raise RuntimeError('an unattested entrypoint copy executed')
    if sys.modules['_headkernel_cases'] is not cases:
        raise RuntimeError('case aliases are not the preloaded object')
    generated_contract.set_seed(731)
    q = torch.tensor([1., 2., 3.])
    output = generated_worker.invoke(cases, {'args': [q], 'kwargs': {}}, generated_contract, torch, runtime_integrity.ACTIVE_GUARD, [])
    print(generated_worker.PREFIX + json.dumps({'schema_version': 1, 'profile': 'eager', 'index': 0, 'seed': 731, 'reference': False, 'rows': [{'id': 'case', 'output': output}], 'objects': {}}))
    return 0
"""
    )
    mapping = {
        "generated_contract": "ut/generated_contract.py",
        "generated_worker": "scripts/generated_worker.py",
        "_deepseek_generated_cases": "ut/cases.py",
        "_headkernel_cases": "ut/cases.py",
        "_bench": "scripts/_bench.py",
        "probe_worker": "scripts/probe.py",
    }
    (task / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "headkernel": {
                    "generated_input_revision": "cpu-fixture",
                    "trusted_worker_modules": mapping,
                }
            },
            sort_keys=False,
        )
    )
    runner = load("_attested_runner", SUPPORT / "task_runner.py")
    runner.TASK_DIR, runner.UT_DIR, runner.BUILD_DIR = task, task / "ut", task / "build"
    original_command = runner.worker_command
    package_paths = [
        value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value
    ]
    bootstrap = 'import json,runpy,sys\nsys.path[:0]=json.loads(sys.argv.pop(1))\nsys.argv=sys.argv[1:]\nrunpy.run_path(sys.argv[0],run_name="__main__")\n'

    def command(*args, **kwargs):
        actual = original_command(*args, **kwargs)
        return [
            sys.executable,
            "-B",
            "-c",
            bootstrap,
            json.dumps(package_paths),
            *actual[2:],
        ]

    monkeypatch.setattr(runner, "worker_command", command)
    contract = load("_attestation_parent_contract", generated_helper("deepseek", "generated_contract.py"))
    controller = load(
        "_attestation_parent_controller", generated_helper("deepseek", "generated_correctness.py")
    )
    expected = contract.encode_output(torch.tensor([2.0, 4.0, 6.0]), torch)

    def execute(source):
        (task / "source/kernel.py").write_text(source)
        proc = runner.run_worker(
            task / "scripts/probe.py", [], None, 40, True, cwd=task
        )
        accepted = False
        if proc.returncode == 0:
            rows = controller.parse_worker(proc, "eager", 0, 731, False, ["case"])
            accepted = contract.compare_output(rows[0]["output"], expected, 0.02, torch)
        return accepted, proc

    return task, execute


ATTACKS = {
    "tensor_shape": "import torch\ndef kernel(q):\n torch.Tensor.shape = property(lambda self: (3,))\n return q*2\n",
    "binascii": "import binascii\ndef kernel(q):\n binascii.b2a_base64 = lambda *a, **kw: b''\n return q*2\n",
    "checker": "import sys\ndef kernel(q):\n sys.modules['generated_contract'].require_unchanged = lambda *a: None\n q.add_(1)\n return q*2\n",
    "encoder": "import sys\ndef kernel(q):\n sys.modules['generated_contract'].encode_output = lambda *a: {}\n return q*2\n",
    "worker": "import sys\ndef kernel(q):\n sys.modules['generated_worker'].invoke = lambda *a: {}\n return q*2\n",
    "cases": "import sys\ndef kernel(q):\n sys.modules['_headkernel_cases'].comparison_output = lambda *a: None\n return q*2\n",
    "tensor_numpy": "import torch\ndef kernel(q):\n torch.Tensor.numpy = lambda self: None\n return q*2\n",
    "tensor_tolist": "import torch\ndef kernel(q):\n torch.Tensor.tolist = lambda self: [0]*12\n return q*2\n",
    "tensor_stride": "import torch\ndef kernel(q):\n torch.Tensor.stride = lambda self: (1,)\n return q*2\n",
    "base64": "import base64\ndef kernel(q):\n base64.b64encode = lambda *a: b''\n return q*2\n",
    "zlib": "import zlib\ndef kernel(q):\n zlib.compressobj = lambda *a: None\n return q*2\n",
    "exp2": "import torch\ndef kernel(q):\n torch.Tensor.exp2 = lambda self: self\n return q*2\n",
    "code": "import sys\ndef kernel(q):\n sys.modules['generated_contract'].require_unchanged.__code__ = (lambda *a: None).__code__\n return q*2\n",
}


@pytest.mark.parametrize("attack", ATTACKS)
def test_shared_completion_rejects_helper_and_serialization_tampering(
    actual_worker, attack
):
    task, execute = actual_worker
    accepted, proc = execute(ATTACKS[attack])
    assert not accepted
    assert proc.returncode != 0, proc.stdout
    assert "IntegrityError" in proc.stderr, proc.stderr
    assert not list((task / "build").glob("_worker_completion_*"))


def test_unchanged_source_passes_real_shared_bootstrap(actual_worker):
    _, execute = actual_worker
    accepted, proc = execute("def kernel(q): return q*2\n")
    assert accepted, (proc.stdout, proc.stderr)


def test_wrong_stride_fails_parent_comparison_without_forgery(actual_worker):
    _, execute = actual_worker
    accepted, proc = execute(
        "import torch\ndef kernel(q):\n out=torch.empty(6)[::2]\n out.copy_(q*2)\n return out\n"
    )
    assert proc.returncode == 0, proc.stderr
    assert not accepted


def test_candidate_access_during_preload_is_rejected(actual_worker):
    task, execute = actual_worker
    (task / "ut/bad_preload.py").write_text(
        "from pathlib import Path\n(Path(__file__).resolve().parents[1]/'source/kernel.py').read_text()\n"
    )
    cfg = yaml.safe_load((task / "config.yaml").read_text())
    cfg["headkernel"]["trusted_worker_modules"] = {
        "bad_preload": "ut/bad_preload.py",
        **cfg["headkernel"]["trusted_worker_modules"],
    }
    (task / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    accepted, proc = execute("def kernel(q): return q*2\n")
    assert not accepted and "preload attempted candidate access" in proc.stderr
