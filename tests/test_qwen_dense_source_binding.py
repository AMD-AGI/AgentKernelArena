"""Exercise the real pinned AITER custom-op decorator and corrected source loader."""
import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = next((ROOT / "tasks/head_kernels/qwen3.8-2.4t-a95b-mxfp4").glob("*/*/dense_bf16_gemm_cluster"))


def test_correctness_runner_explicitly_uses_candidate_overlay(monkeypatch):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("dense_runner_direction_test", TASK / "scripts/task_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []
    monkeypatch.setattr(runner, "overlays", lambda: ("baseline-overlay", "candidate-overlay"))
    monkeypatch.setattr(runner, "run_worker", lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(returncode=0))
    runner.run_ut(30)
    args, kwargs = calls[0]
    assert args[0] == runner.UT_DIR / "unittest.py"
    assert args[2] == "candidate-overlay" and args[4] is True
    assert kwargs["cwd"] == runner.UT_DIR


@pytest.mark.parametrize("wrong", [False, True])
def test_registered_custom_op_cannot_replace_candidate_source(tmp_path, wrong):
    code = f'''import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
import runpy
import shutil
import torch
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
guard=load("guard_api",{str(ROOT / 'tests/fixtures/qwen_dense_dispatch/torch_guard_api_fixture.py')!r})
calls={{"native":0}}
def gemm_a16w16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    calls["native"]+=1
    return torch.nn.functional.linear(A,B)
registered=guard.torch_compile_guard()(gemm_a16w16)
native=SimpleNamespace(gemm_a16w16=registered)
folder=Path({str(tmp_path)!r});(folder/"ut").mkdir(parents=True);(folder/"source").mkdir()
candidate=folder/"source/tuned_gemm_candidate.py"
candidate.write_text("import torch\\nfrom guard_api import torch_compile_guard\\n@torch_compile_guard()\\ndef gemm_a16w16(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:\\n    return torch.nn.functional.linear(A,B){' + 1' if wrong else ''}\\n")
A=torch.tensor([[0.25,-0.5],[0.875,0.125]])
B=torch.tensor([[0.5,0.75],[-0.125,1.0]])
expected=torch.nn.functional.linear(A,B)
# Reproduce the bug with the real decorator: importing later returns the first
# registered op even when the later source is intentionally wrong.
broken=load("broken_candidate",candidate)
assert torch.equal(broken.gemm_a16w16(A,B),expected)
assert calls["native"]==1
dispatch=load("dense_dispatch_contract",{str(TASK / 'ut/dispatch_contract.py')!r})
dispatch.HERE=folder/"ut"
sys.modules["aiter.tuned_gemm"]=native
# Use the same runner -> harness -> overlay_setup subprocess chain as GPU runs.
original_ut=Path({str(TASK / 'ut')!r})
shutil.copytree(original_ut/"baseline_overlay",folder/"ut/baseline_overlay")
for filename in ("overlay_setup.py","harness_lib.py","meta.json"):
    shutil.copyfile(original_ut/filename,folder/"ut"/filename)
(folder/"ut/kernel_src").symlink_to("../source",target_is_directory=True)
runner=load("actual_dense_task_runner",{str(TASK / 'scripts/task_runner.py')!r})
runner.TASK_DIR=folder;runner.UT_DIR=folder/"ut";runner.BUILD_DIR=folder/"build"
base,generated=runner.overlays()
overlay=Path(generated)/"sitecustomize.py"
assert overlay.read_bytes()==(original_ut/"baseline_overlay/sitecustomize.py").read_bytes()
sys.path.insert(0,generated)
runpy.run_path(str(overlay))
fixed=sys.modules["tuned_gemm_candidate"]
assert native.gemm_a16w16 is fixed.gemm_a16w16
dispatch.prepare_runtime=lambda:native
observed=dispatch.call_candidate(A,B)
assert dispatch.candidate_call_count()==1
assert calls["native"]==1
assert Path(fixed.gemm_a16w16.__code__.co_filename).resolve()==(Path(generated)/candidate.name).resolve()
numeric_ok=torch.allclose(observed,expected,atol=1e-6,rtol=1e-6)
assert numeric_ok is {not wrong!r}
assert torch.equal(dispatch.baseline_callable()(A,B),expected)
print(json.dumps({{"candidate_source_calls":1,"numeric_ok":numeric_ok,"native_called_by_candidate":False}}))
'''
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["numeric_ok"] is not wrong


def test_candidate_probe_allows_alternative_backend_without_old_symbol(tmp_path):
    code = f'''import ast, json
from types import SimpleNamespace
import torch
source=__import__('pathlib').Path({str(TASK / 'ut/validate_selection.py')!r}).read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="validate_candidate")
calls={{"count":0}}
def candidate(args):
    calls["count"]+=1
    return torch.nn.functional.linear(args["A"],args["B"])
fake_torch=SimpleNamespace(bfloat16=torch.bfloat16,cuda=SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None))
values={{"importlib":SimpleNamespace(import_module=lambda name:fake_torch),
         "dispatch":SimpleNamespace(candidate_call_count=lambda:calls["count"]),
         "cases":SimpleNamespace(make_args=lambda case,seed:{{"A":torch.ones(2,3,dtype=torch.bfloat16),"B":torch.ones(4,3,dtype=torch.bfloat16)}},candidate_call=candidate),
         "_profile_names":lambda torch,fn:(fn(),["new_faster_backend_kernel"])}}
exec(compile(ast.Module(body=[node],type_ignores=[]),"candidate_probe","exec"),values)
result=values["validate_candidate"]({{"ledger_id":"case","m":2,"n":4,"expected_backend":"old","profile_match":"old_symbol"}})
assert result["ok"] is True and result["candidate_source_calls"]==2
assert result["device_events"]==["new_faster_backend_kernel"]
print(json.dumps(result))
'''
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
