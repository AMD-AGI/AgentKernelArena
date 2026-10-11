"""CPU regression for an already registered AITER op bypassing source edits."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster"


@pytest.mark.parametrize("wrong", [False, True])
def test_actual_overlay_builder_binds_source_instead_of_registered_op(tmp_path, wrong):
    code = f'''import importlib.util, json, runpy, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import torch
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod
guard=load("guard_api",{str(ROOT / 'tests/fixtures/qwen_dense_dispatch/torch_guard_api_fixture.py')!r})
native_calls=[]
def gemm_a16w16(A:torch.Tensor,B:torch.Tensor)->torch.Tensor:
    native_calls.append(True)
    return torch.nn.functional.linear(A,B)
registered=guard.torch_compile_guard()(gemm_a16w16)
native=SimpleNamespace(gemm_a16w16=registered,solMap={{"native":object()}})
folder=Path({str(tmp_path)!r});ut=folder/"ut";src=folder/"source"
ut.mkdir();src.mkdir()
source=src/"tuned_gemm_candidate.py"
source.write_text("import torch\\nfrom guard_api import torch_compile_guard\\nsolMap = {{'candidate': object()}}\\n@torch_compile_guard()\\ndef gemm_a16w16(A:torch.Tensor,B:torch.Tensor)->torch.Tensor:\\n    return torch.nn.functional.linear(A,B){' + 1' if wrong else ''}\\n")
before=source.read_bytes()
A=torch.tensor([[0.25,-0.5],[0.875,0.125]])
B=torch.tensor([[0.5,0.75],[-0.125,1.0]])
expected=torch.nn.functional.linear(A,B)
broken=load("unprotected_candidate",source)
assert torch.equal(broken.gemm_a16w16(A,B),expected)
assert len(native_calls)==1
original_ut=Path({str(TASK / 'ut')!r})
shutil.copytree(original_ut/"baseline_overlay",ut/"baseline_overlay")
for name in ("overlay_setup.py","harness_lib.py","meta.json","dispatch_contract.py"):
    shutil.copyfile(original_ut/name,ut/name)
(ut/"kernel_src").symlink_to("../source",target_is_directory=True)
sys.modules["aiter.tuned_gemm"]=native
runner=load("tested_dense_runner",{str(TASK / 'scripts/task_runner.py')!r})
runner.TASK_DIR=str(folder);runner.UT_DIR=str(ut);runner.BUILD_DIR=str(folder/"build")
generated=runner.candidate_overlay()
assert (Path(generated)/"sitecustomize.py").read_bytes()==(ut/"baseline_overlay/sitecustomize.py").read_bytes()
sys.path.insert(0,generated)
runpy.run_path(str(Path(generated)/"sitecustomize.py"))
dispatch=sys.modules["dense_dispatch_contract"]
candidate=sys.modules["tuned_gemm_candidate"]
assert native.gemm_a16w16 is candidate.gemm_a16w16
assert dispatch.candidate_dispatch_module().solMap is candidate.solMap
assert candidate.solMap is not native.solMap
out=dispatch.call_candidate(A,B)
assert dispatch.candidate_call_count()==1 and len(native_calls)==1
assert torch.allclose(out,expected,rtol=1e-6,atol=1e-6) is {not wrong!r}
assert source.read_bytes()==before
assert dispatch.candidate_identity()["source_sha256"]==__import__('hashlib').sha256(before).hexdigest()
assert torch.equal(dispatch.native_baseline_callable()(A,B),expected)
native.gemm_a16w16=registered
try:dispatch.call_candidate(A,B)
except RuntimeError as e:assert "binding changed" in str(e)
else:raise AssertionError("stale native rebind accepted")
print(json.dumps({{"wrong_candidate_reaches_comparison":{wrong!r},"source_calls":1,"native_calls_by_candidate":0}}))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                            cwd=ROOT, timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["source_calls"] == 1


@pytest.mark.parametrize("change", ["different_copy", "duplicate_dispatcher", "extra_decorator"])
def test_loader_rejects_changed_or_ambiguous_source_before_execution(tmp_path, change):
    code = f'''import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
root=Path({str(tmp_path)!r});ut=root/"ut";source=root/"source";overlay=ut/"_cand_overlay"
overlay.mkdir(parents=True);source.mkdir()
declared=source/"tuned_gemm_candidate.py";copy=overlay/declared.name
text="@torch_compile_guard()\\ndef gemm_a16w16(A,B):\\n    raise AssertionError('candidate executed')\\n"
if {change!r}=="duplicate_dispatcher":text+=text
if {change!r}=="extra_decorator":text="@another_decorator\\n"+text
declared.write_text(text);copy.write_text(text+"#different" if {change!r}=="different_copy" else text)
spec=importlib.util.spec_from_file_location("dense_rejection_test",{str(TASK / 'ut/dispatch_contract.py')!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.HERE=ut
module.META={{"candidate_bind":{{"impl_module":"tuned_gemm_candidate","impl_attr":"gemm_a16w16","target":"aiter.tuned_gemm:gemm_a16w16","file":"../source/tuned_gemm_candidate.py"}}}}
try:module.load_candidate_module("tuned_gemm_candidate",copy,SimpleNamespace(gemm_a16w16=object()))
except RuntimeError as error:print(json.dumps({{"rejected":str(error)}}))
else:raise AssertionError("invalid source accepted")
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                            cwd=ROOT, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "rejected" in json.loads(result.stdout)
