"""CPU evidence for task-owned v2 actions; does not claim GPU kernel validation."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml
from src.task_protocol import CaseManifest, parse_command_result
from src.task_spec import load_task_spec

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=("flydsl2flydsl",)
TASKS=sorted(p.parent for family in FAMILIES for p in (ROOT/"tasks"/family).rglob("config.yaml"))


def module(path, name="task_runtime_test"):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def invoke(task,*args):
    p=subprocess.run([sys.executable,"scripts/evaluate.py",*args],cwd=task,text=True,capture_output=True)
    role,action=("task","validate-task") if args==("validate-task",) else args
    return parse_command_result(p.stdout,role=role,action=action,returncode=p.returncode)


@pytest.mark.parametrize("task",TASKS,ids=lambda t:str(t.relative_to(ROOT/"tasks")))
def test_schema_and_observed_starting_state(task):
    spec=load_task_spec(task/"config.yaml",task_id=str(task.relative_to(ROOT/"tasks")))
    runtime=module(task/"task_runtime.py")
    cfg=runtime.config()
    state,defined=runtime.source_state(cfg)
    assert state==cfg["candidate"]["initial_state"]
    assert cfg["candidate"]["language"]=="flydsl"
    assert bool(defined)
    assert spec.to_mapping()["evaluation"]["runner"]==["python3","scripts/evaluate.py"]


@pytest.mark.parametrize("task",TASKS,ids=lambda t:t.name)
def test_real_cpu_known_answers_and_complete_manifest(task):
    result=invoke(task,"validate-task")
    assert result.passed,result.reason
    expected=json.loads((task/"cases.json").read_text())["cases"]
    assert [{k:v for k,v in row.items() if k!="status"} for row in result.cases]==expected
    manifest=CaseManifest.from_result(result)
    assert manifest.cases
    controls=result.metadata["reference_controls"]
    assert controls and all(c["known_answer"]=="PASS" and c["negative_output"]=="rejected" for c in controls)


def test_missing_or_stub_candidate_never_falls_back(tmp_path):
    src=ROOT/"tasks/flydsl2flydsl/softmax_kernel"
    task=tmp_path/"task";shutil.copytree(src,task)
    (task/"kernel.py").write_text('def build_softmax_module(*args,**kwargs):\n    raise NotImplementedError("stub")\n')
    for action in ("compile","correctness","performance"):
        result=invoke(task,"candidate",action)
        assert not result.passed
        assert "no baseline fallback" in result.reason
    result=invoke(task,"validate-task")
    assert not result.passed
    assert result.metadata["candidate_state"]=="unimplemented"


def test_known_answer_rejects_reference_bug(tmp_path):
    src=ROOT/"tasks/flydsl2flydsl/softmax_kernel"
    task=tmp_path/"task";shutil.copytree(src,task)
    p=task/"test_kernel_harness.py"
    s=p.read_text(); tree=ast.parse(s)
    n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="reference_softmax")
    lines=s.splitlines();lines[n.lineno-1:n.end_lineno]=['def reference_softmax(x):','    return x * 0']
    p.write_text('\n'.join(lines)+'\n')
    result=invoke(task,"validate-task")
    assert not result.passed
    assert "known answer" in result.reason


def test_nested_target_and_path_containment(tmp_path):
    src=ROOT/"tasks/flydsl2flydsl/softmax_kernel"
    task=tmp_path/"task";shutil.copytree(src,task)
    (task/"source").mkdir();(task/"kernel.py").rename(task/"source/implementation.py")
    cfg=yaml.safe_load((task/"config.yaml").read_text())
    cfg["candidate"]["editable"]=["source/implementation.py"]
    cfg["candidate"]["entrypoints"][0]["file"]="source/implementation.py"
    (task/"config.yaml").write_text(yaml.safe_dump(cfg))
    runtime=module(task/"task_runtime.py")
    assert runtime.candidate_relative_path()=="source/implementation.py"
    assert invoke(task,"candidate","compile").passed
    for path in ("../outside.py","/tmp/outside.py","source/../kernel.py"):
        with pytest.raises(ValueError):runtime.local_path(path)
    (task/"escape").symlink_to(tmp_path)
    with pytest.raises(ValueError):runtime.local_path("escape/any.py")


@pytest.mark.parametrize("source",[
    'import flydsl\nfrom scripts import task_reference\n',
    'import flydsl\nfrom scripts import task_baseline\n',
    'import flydsl\nimport model\n',
    'import flydsl\nfrom torch import matmul as mm\nx = mm(a,b)\n',
    'import triton\n',
    'def build(*a): return None\n',
])
def test_candidate_dependency_and_backend_policy(source,tmp_path):
    runtime=module(TASKS[0]/"task_runtime.py")
    p=tmp_path/"kernel.py";p.write_text(source)
    with pytest.raises(ValueError):runtime.check_dependencies([p],True)


def test_timing_report_cannot_drop_cases_or_accept_nonfinite():
    runtime=module(TASKS[0]/"task_runtime.py")
    cases=[{"test_case_id":"a","checks":["correctness","performance"],"params":{"M":2}}]
    mapping={"old":"a"}
    valid=[{"test_case_id":"old","execution_time_ms":.2,"benchmark_method":"cuda_graph"}]
    result=runtime.require_result_rows(valid,cases,mapping)
    assert result[0]["params"]=={"M":2}
    for rows in ([],valid*2,[dict(valid[0],execution_time_ms=float("nan"))],[dict(valid[0],execution_time_ms=float("inf"))],[dict(valid[0],benchmark_method="host_clock")]):
        with pytest.raises(ValueError):runtime.require_result_rows(rows,cases,mapping)


def test_declaring_flydsl_without_executing_it_is_rejected():
    runtime=module(TASKS[0]/"task_runtime.py")
    with pytest.raises(RuntimeError,match="No FlyDSL"):
        runtime.check_flydsl_execution(lambda:None)


def test_f2f_preserves_all_original_case_counts_and_adds_missing_correctness():
    expected={"blockscale_preshuffle_gemm_kernel":(4,4),"flash_attn_func_kernel":(10,10),"fp8_gemm_4wave_kernel":(5,5),"fp8_gemm_8wave_kernel":(5,5),"fused_rope_cache_kernel":(6,6),"hgemm_splitk_kernel":(14,14),"layernorm_kernel":(10,10),"moe_sorting_kernel":(4,2),"pa_decode_fp8_kernel":(8,8),"pa_decode_swa_kernel":(5,5),"preshuffle_gemm_v2_kernel":(4,4),"rmsnorm_kernel":(10,10),"silu_and_mul_fq_kernel":(5,5),"softmax_kernel":(10,10),"topk_gating_softmax_kernel":(5,2)}
    for name,(correctness,performance) in expected.items():
        task=ROOT/"tasks/flydsl2flydsl"/name
        rows=json.loads((task/"cases.json").read_text())["cases"]
        assert sum("correctness" in r["checks"] for r in rows)==correctness
        assert sum("performance" in r["checks"] for r in rows)==performance
        actions=module(task/"scripts/task_actions.py")
        assert len(actions.CORRECTNESS_CASES)==correctness
        assert len(actions.PERFORMANCE_CASES)==performance
        assert set(actions.PERFORMANCE_IDS.values())=={r["test_case_id"] for r in rows if "performance" in r["checks"]}


# AST fingerprints captured from parent 5c9f8ef2, excluding only the moved
# TopK reference block and resolving the Flash Attention builder alias.
F2F_CORRECTNESS_SHA256 = {'blockscale_preshuffle_gemm_kernel': 'c0f2b8a6184a4a37cdd07c6c533263551aa8d1a313010a106050ffc7d1a0f309',
 'flash_attn_func_kernel': '2b021df0a44507c9aa94d5b74e632f1abdb59e97002f4a7750c27b361f37ffab',
 'fp8_gemm_4wave_kernel': '6b1e07bdc9613907fee91cc1ced6842ddecb5dc71e7a4159632db453255247c0',
 'fp8_gemm_8wave_kernel': 'f3cf6e48a38a1ab3a1b0a03070b5b9cd0a4c98e93e71d112ecf13752e49e222e',
 'fused_rope_cache_kernel': 'a7fc94e172288d53e283fa8eec2f5b6218d9e9ef10a51917d32e83f12deff5c7',
 'hgemm_splitk_kernel': '57fb63b89a3f0a4737bb3d979da83f95743f423739b9aa5a835f632d0c95e3d1',
 'layernorm_kernel': '3ef03523bec4ed773721eb04b023aaec30c7a48cb9d15f9a82d383e0730fc3c7',
 'moe_sorting_kernel': '35be7e277adad7e2b5e60b25ab3ffe483d8c43f19818a53e36a8ac9fcaab82b9',
 'pa_decode_fp8_kernel': '556ad8ece24e443b4dfb81385d817e9b64f7ac12bb019aa48d8c18aa04413d8b',
 'pa_decode_swa_kernel': '7ce08115b7389020788405982968352db6edbf0a3fa9d35b6ef1763378c1e0d4',
 'preshuffle_gemm_v2_kernel': '91eef9437be190a815497bc7205657a2c5104f601e5ba82439e22de0621d99ae',
 'rmsnorm_kernel': 'f34ed9c0eb5912d9ffa8ccc961856a10b4c85d68ad8683956419fae1a3515d59',
 'silu_and_mul_fq_kernel': '9f7c1edbb94ec02d4babb621f7fce05935e34c5c6a34602227fd8cc10788b56a',
 'softmax_kernel': '0dbec2e3349e14cfd67aeb6682feb2cce3fb6b2fcd736a81bf7444a98b417689',
 'topk_gating_softmax_kernel': 'd8ad389109b73109c6d5f7c6155473f3a924ebbeb59cdb0b2b437b75750bd119'}


def test_original_f2f_numerical_gates_and_output_contracts_unchanged():
    for name,expected in F2F_CORRECTNESS_SHA256.items():
        task=ROOT/"tasks/flydsl2flydsl"/name
        tree=ast.parse((task/"test_kernel_harness.py").read_text())
        fn=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_correctness"),None)
        if fn is None:fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_test")
        if name=="topk_gating_softmax_kernel":
            start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="atol_weight" for t in n.targets))
            fn=ast.Module(fn.body[start:],type_ignores=[])
        actual=ast.dump(fn,include_attributes=False).replace("build_flash_attn_func_module_primary","build_flash_attn_func_module")
        assert hashlib.sha256(actual.encode()).hexdigest()==expected,name
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("run_benchmark","run_geak_benchmark"))
        direct=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="arena_benchmark")
        def calls(fn):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(fn) if isinstance(n,ast.Call) and (getattr(n.func,"id","") in {"benchmark_cuda_graph_or_events","_time_mean_ms","_mean_ms"})]
        assert calls(original)==calls(direct),name
