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
FAMILIES=("flydsl2flydsl","torch2flydsl","triton2flydsl")
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


CANDIDATE_AUDIT_TASKS = [ROOT / "tasks/torch2flydsl" / name for name in (
    "rope_2d_fwd_kernel",
    "quant_mxfp4_kernel",
    "moe_biased_grouped_topk_kernel",
    'per_tensor_fp8_quant_kernel', 'per_token_fp8_quant_kernel', 'per_1x128_fp8_quant_kernel', 'per_token_i8_quant_kernel',
    "layernorm2d_kernel", "layernorm2d_with_add_kernel",
    'gemm_a8w8_kernel', 'gemm_a8w8_per_token_scale_kernel', 'gemm_a8wfp4_kernel', 'gemm_afp4wfp4_kernel', 'gemm_afp8wfp8_kernel',
    'gemm_a16w8_blockscale_kernel', 'gemm_a16wfp4_kernel', 'gemm_a4w4_kernel', 'gemm_a8w8_blockscale_kernel',
    "fused_add_rmsnorm_kernel", "fmoe_fp8_blockscale_g1u1_kernel", "fmoe_g1u1_tkw1_kernel", "silu_and_mul_kernel", "dynamic_mxfp8_quant_kernel", "batched_gemm_a8w8_kernel", "rmsnorm2d_kernel", "moe_topk_softmax_kernel", "moe_topk_sigmoid_kernel",
    "moe_topk_softplus_kernel", "gelu_and_mul_kernel", "gelu_fast_kernel",
    "gelu_tanh_and_mul_kernel", "swiglu_and_mul_kernel",
)]


@pytest.mark.parametrize("task", CANDIDATE_AUDIT_TASKS, ids=lambda p:p.name)
@pytest.mark.parametrize("source", [
    "import aiter as baseline\nbaseline.rms_norm(x, w, eps)",
    "from aiter import rms_norm as op\nop(x, w, eps)",
    "from aiter import topk_gating\ntopk_gating(x)",
    "from aiter.ops import activation",
    "import subprocess\nsubprocess.run(['kernel'])",
    "from ctypes import CDLL\nCDLL('kernel.so')",
])
def test_final_flydsl_candidate_rejects_operator_backends(task, source, tmp_path):
    runtime = module(task / "task_runtime.py")
    path = tmp_path / "candidate.py"
    path.write_text("import flydsl\n" + source + "\n")
    with pytest.raises(ValueError, match="Final operator must execute FlyDSL"):
        runtime.check_dependencies([path], final_language=True)


@pytest.mark.parametrize("task", CANDIDATE_AUDIT_TASKS, ids=lambda p:p.name)
def test_candidate_dispatch_rejects_correct_torch_rms_with_unrelated_call(task):
    torch = pytest.importorskip("torch")
    audit = module(task / "scripts/candidate_checks.py")
    x = torch.tensor([[3., 4.]])
    weights = torch.ones(2)
    calls = []
    def operator(x, weights):
        calls.append("unrelated backend invocation")
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True)) * weights
    # It is a mathematically correct answer. Rejection is for operator compute
    # in PyTorch, regardless of a preceding unrelated backend invocation.
    expected = torch.tensor([[3 / (12.5 ** .5), 4 / (12.5 ** .5)]])
    torch.testing.assert_close(operator(x, weights), expected)
    calls.clear()
    with pytest.raises(RuntimeError, match="non-preparation PyTorch operation"):
        with audit.candidate_preparation_only():
            operator(x, weights)
    assert calls == ["unrelated backend invocation"]


@pytest.mark.parametrize("task", CANDIDATE_AUDIT_TASKS, ids=lambda p:p.name)
def test_candidate_dispatch_allows_preparation_but_not_routing(task):
    torch = pytest.importorskip("torch")
    audit = module(task / "scripts/candidate_checks.py")
    x = torch.tensor([[1., 3., 2.]])
    with audit.candidate_preparation_only():
        out = torch.empty_like(x)
        out.copy_(x)
        viewed = out.view(3).unsqueeze(0)
        scratch = torch.zeros_like(x)
        scratch.fill_(1)
    torch.testing.assert_close(viewed, x)
    for op in (lambda: torch.softmax(x, -1), lambda: x.topk(2), lambda: x + 1):
        with pytest.raises(RuntimeError, match="non-preparation PyTorch operation"):
            with audit.candidate_preparation_only():
                op()


@pytest.mark.parametrize("task", CANDIDATE_AUDIT_TASKS, ids=lambda p:p.name)
def test_candidate_audit_scopes_import_operator_and_restores_timing_loader(task):
    import types
    torch = pytest.importorskip("torch")
    audit = module(task / "scripts/candidate_checks.py")
    x = torch.tensor([2., 3.])
    def load(directory, filename, alias):
        return types.SimpleNamespace(flydsl_operator=lambda x: x.square())
    h = types.SimpleNamespace(ARENA_PROVIDED_BASELINE=False, KERNEL_FILE="kernel.py", _load_module=load)
    with pytest.raises(RuntimeError, match="non-preparation PyTorch operation"):
        with audit.audit_candidate_calls(h):
            # A protected baseline/reference module is not a candidate.
            torch.testing.assert_close(h._load_module(".", "model.py", "model").flydsl_operator(x), x.square())
            h._load_module(".", "kernel.py", "candidate").flydsl_operator(x)
    assert h._load_module is load
    # The ordinary timing load after correctness is not wrapped in the audit.
    torch.testing.assert_close(h._load_module(".", "kernel.py", "timing").flydsl_operator(x), x.square())
    h.ARENA_PROVIDED_BASELINE = True
    with audit.audit_candidate_calls(h):
        assert h._load_module is load
    h.ARENA_PROVIDED_BASELINE = False
    def computing_import(*args):
        x.square()
        return types.SimpleNamespace()
    h._load_module = computing_import
    with pytest.raises(RuntimeError, match="non-preparation PyTorch operation"):
        with audit.audit_candidate_calls(h):
            h._load_module(".", "kernel.py", "candidate")
    assert h._load_module is computing_import


@pytest.mark.parametrize("task", CANDIDATE_AUDIT_TASKS, ids=lambda p:p.name)
def test_flydsl_launch_evidence_must_come_from_candidate_call(task):
    torch = pytest.importorskip("torch")
    audit = module(task / "scripts/candidate_checks.py")
    # CPU-only profile fixture represents a backend runtime call, not GPU work.
    scope = {"__name__": "flydsl.test_fixture"}
    exec("class CompiledKernel:\n def __call__(self, x): return x\n", scope)
    launch = scope["CompiledKernel"]()
    x = torch.tensor([3., 4.])
    observed = set()
    launch(x)  # An oracle's earlier backend call must not count for candidate.
    with pytest.raises(RuntimeError, match="No FlyDSL kernel runtime invocation"):
        audit.checked_candidate_invocation(lambda x: x, observed, x)
    assert not observed
    previous = sys.getprofile()
    assert audit.checked_candidate_invocation(launch, observed, x) is x
    assert observed == {"flydsl.test_fixture.CompiledKernel"}
    assert sys.getprofile() is previous
    with pytest.raises(RuntimeError, match="non-preparation PyTorch operation"):
        audit.checked_candidate_invocation(lambda x: launch(x).square(), set(), x)
    assert sys.getprofile() is previous


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
        if name in {"fp8_gemm_4wave_kernel", "fp8_gemm_8wave_kernel", "blockscale_preshuffle_gemm_kernel", "preshuffle_gemm_v2_kernel", "pa_decode_fp8_kernel"}:
            fn = _RemoveAddedReplayChecks().visit(fn)
        if name in {"hgemm_splitk_kernel", "moe_sorting_kernel"}:
            added_check = ("compare_output(c, ref, RTOL, torch_dtype)" if name == "hgemm_splitk_kernel"
                           else "compare_outputs((gpu_ids, gpu_w, gpu_eids, gpu_nvalid, gpu_moe_buf), (ref_ids, ref_w, ref_eids, ref_nvalid), token_count=T, topk=topk, unit_size=unit_size)")
            class RemovePortOutputCheck(ast.NodeTransformer):
                def visit_Expr(self, node):
                    return None if ast.unparse(node.value) == added_check else self.generic_visit(node)
            assert sum(isinstance(n, ast.Expr) and ast.unparse(n.value) == added_check for n in ast.walk(fn)) == 1
            fn = RemovePortOutputCheck().visit(fn)
        actual=ast.dump(fn,include_attributes=False).replace("build_flash_attn_func_module_primary","build_flash_attn_func_module")
        assert hashlib.sha256(actual.encode()).hexdigest()==expected,name
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("run_benchmark","run_geak_benchmark"))
        direct=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="arena_benchmark")
        def calls(fn):
            import copy
            collected=[]
            for node in ast.walk(fn):
                if not isinstance(node,ast.Call) or getattr(node.func,"id","") not in {"benchmark_cuda_graph_or_events","_time_mean_ms","_mean_ms"}:
                    continue
                call=copy.deepcopy(node)
                if name in {"layernorm_kernel", "rmsnorm_kernel", "softmax_kernel", "topk_gating_softmax_kernel",
                            "flash_attn_func_kernel", "hgemm_splitk_kernel", "fused_rope_cache_kernel",
                            "silu_and_mul_fq_kernel", "moe_sorting_kernel", "pa_decode_swa_kernel"} and fn is direct:
                    # These reviewed ports expose the original launch's outputs
                    # to TimedRun. Preserve comparison of the launch expression
                    # and every sampling/timing argument, ignoring only the
                    # additional observer and Python return of output buffers.
                    for kw in call.keywords:
                        if kw.arg=="timed_run":
                            assert isinstance(kw.value,ast.Name) and kw.value.id=="timed"
                    call.keywords=[kw for kw in call.keywords if kw.arg!="timed_run"]
                    if name != "moe_sorting_kernel" and isinstance(call.args[0],ast.Name) and call.args[0].id=="launch":
                        launch=next(n for n in ast.walk(fn) if isinstance(n,ast.FunctionDef) and n.name=="launch")
                        assert not launch.args.args and len(launch.body)==2
                        assert isinstance(launch.body[0],ast.Expr) and isinstance(launch.body[0].value,ast.Call)
                        assert isinstance(launch.body[1],ast.Return)
                        expected_output = {"flash_attn_func_kernel": "o_flat", "hgemm_splitk_kernel": "c",
                                           "silu_and_mul_fq_kernel": "(out_buf, out_scale)"}.get(name, "output")
                        assert ast.unparse(launch.body[1].value)==expected_output
                        if name == "pa_decode_swa_kernel":
                            assert ast.unparse(launch.body[0].value) == "run_fn()"
                            call.args[0] = ast.Name(id="run_fn", ctx=ast.Load())
                        else:
                            call.args[0]=ast.Lambda(args=copy.deepcopy(launch.args),body=copy.deepcopy(launch.body[0].value))
                collected.append(ast.dump(call,include_attributes=False))
            return collected
        assert calls(original)==calls(direct),name


@pytest.mark.parametrize("name",["silu_and_mul_kernel","rmsnorm2d_kernel","batched_gemm_bf16_kernel"])
def test_provided_baseline_does_not_import_working_candidate(name,tmp_path):
    task=tmp_path/name
    shutil.copytree(ROOT/"tasks/torch2flydsl"/name,task)
    (task/"kernel.py").write_text('raise RuntimeError("candidate must not load")\n')
    result=invoke(task,"baseline","compile")
    assert result.passed,result.reason
    assert result.metadata["sources"]==["model.py"]
    # Exercise the real protected harness module loader, without pretending that
    # the CPU process executed an AITER operator or a GPU benchmark.
    script='''
import types,sys
bench=types.ModuleType('_aka_benchmark')
bench.benchmark_cuda_graph_or_events=lambda *a,**k: (_ for _ in ()).throw(AssertionError('GPU timing not allowed in CPU test'))
bench.TimedRun=bench.benchmark_cuda_graph_or_events
sys.modules['_aka_benchmark']=bench
import test_kernel_harness as h
h.ARENA_PROVIDED_BASELINE=True
assert h._load_module(h._KERNEL_DIR,h.KERNEL_FILE,'candidate') is None
h.ARENA_PROVIDED_BASELINE=False
try:
    h._load_module(h._KERNEL_DIR,h.KERNEL_FILE,'candidate')
except RuntimeError as e:
    assert str(e)=='candidate must not load'
else:
    raise AssertionError('candidate import silently skipped')
'''
    p=subprocess.run([sys.executable,"-c",script],cwd=task,text=True,capture_output=True)
    assert p.returncode==0,p.stderr


def test_none_and_conditional_stub_outputs_are_real_failures(tmp_path):
    task=tmp_path/"task"
    shutil.copytree(ROOT/"tasks/torch2flydsl/silu_and_mul_kernel",task)
    (task/"kernel.py").write_text('def flydsl_silu_and_mul(x, *args):\n    if x: raise NotImplementedError("case unavailable")\n    return None\n')
    script='''
import types,sys
bench=types.ModuleType('_aka_benchmark')
bench.benchmark_cuda_graph_or_events=lambda *a,**k: (_ for _ in ()).throw(AssertionError('GPU timing not allowed in CPU test'))
bench.TimedRun=bench.benchmark_cuda_graph_or_events
sys.modules['_aka_benchmark']=bench
import test_kernel_harness as h
m=h._load_module(h._KERNEL_DIR,h.KERNEL_FILE,'candidate')
for arg in (False,True):
    try:m.flydsl_silu_and_mul(arg)
    except RuntimeError:pass
    else:raise AssertionError('incomplete output accepted')
'''
    p=subprocess.run([sys.executable,"-c",script],cwd=task,text=True,capture_output=True)
    assert p.returncode==0,p.stderr


def test_actual_torch_starters_and_frozen_implementations():
    implemented={"gemm_a8w8_bpreshuffle_kernel","hgemm_kernel","jagged_dense_bmm_kernel","moe_sorting_kernel","qk_norm_rope_quant_kernel"}
    configs=sorted((ROOT/"tasks/torch2flydsl").glob("*/config.yaml"))
    assert len(configs)==45
    for cp in configs:
        cfg=yaml.safe_load(cp.read_text())
        expected=cp.parent.name in implemented
        assert cfg["candidate"]["initial_state"]==("implemented" if expected else "unimplemented")
        assert cfg["baseline"]["kind"]==("initial_candidate" if expected else "provided")
        actions=module(cp.parent/"scripts/task_actions.py")
        cases=json.loads((cp.parent/"cases.json").read_text())["cases"]
        assert len(cases)==len(actions.EXPECTED_CASES)
        assert set(actions.PERFORMANCE_IDS.values())=={r["test_case_id"] for r in cases}


TORCH_ORIGINAL_CORRECTNESS_AND_MODEL_SHA256 = {'batched_gemm_a8w8_kernel': ('e8a5c8f56a669851e551ffd469d1a69d63976ea7274a23b44e82545862367eb2',
                              'ab356ed7ac83ec132538b586050492bda9da8a0b7f867295ba5581ae2fdb7201'),
 'batched_gemm_bf16_kernel': ('115d516277b1f70b94c0cb8254ba853043fd6bcea64438aa7da018d2cc1ec4ad',
                              '6b4a8ddc5126f583cd6eeaaca07e0a4a22fa24ef42ecb954ab5575bf78c49dfa'),
 'dynamic_mxfp8_quant_kernel': ('33239fead909cb4bbfe81a565356be1a145286ebde010b249493e4b76595324f',
                                '9e5b1e289eee05aba727b71e28a98e6a7611d9fd6737d5e87b83fe9469eed39d'),
 'fmoe_fp8_blockscale_g1u1_kernel': ('c8c84b6c4a15a95ab78bd60b9d8a9ae4d4db99a2852e42b7afaacf8df84390f4',
                                     '26a76514d4e58e9ec2e4dd5fe5e50b4e11407d1013d09d110b7c246fe786cecb'),
 'fmoe_g1u1_tkw1_kernel': ('f665a728f797bbafafcffc2238403fcb6f085efdbddf0d9403b51af79d406e45',
                           '0e4edfe2a30cb964153fae028536559123374734e2d53c17145e5f4cd8ee6b13'),
 'fused_add_rmsnorm_kernel': ('70bf0b8f5c08b726579825cc5ed3f859c62e75a05a268f8b32b49362dfc34c84',
                              '264ea626dead2c516d8c9b4847a5c7a2ade668740434634330fee4c7e50a38ca'),
 'gelu_and_mul_kernel': ('3b71034fa9800e77261a1d214f83ff9eee63e11a992326f52a9510c2a07a6720',
                         'c171ab0b489b1cb87a3f551c3ba8ecd820e3147a9becb6810154040f4027f7dd'),
 'gelu_fast_kernel': ('481b5acbc9b4c2be79711ab515c5f78994f3ff3d1a6ed5b9bf490d9f18968fa6',
                      '807bee93fab96cb1cad26b0ed775329243da1e36460edba2f73e4eb26d4fef18'),
 'gelu_tanh_and_mul_kernel': ('4fd7acbeabffa979ae22df3410be4853dcda23304bec562ae4b1745f5f3b3ae7',
                              '95988833405bac9d10624c4ca4e78ee0251a5a60c1dd457d6b915904b9b2dadf'),
 'gemm_a16w8_blockscale_kernel': ('23b11cf99cae3024e64a3d5366cc9920cac455c500716d16ed0ad38a9758b712',
                                  'cd3965839d8e34961eaadbfbd8f1138c40fbb32eed1ec549949e99fa6db4b01d'),
 'gemm_a16wfp4_kernel': ('60d3577015f8037ef41a67c1fe2113e64ad38363b8b3ede5ca6c05d11b7dae2e',
                         'a606c99594cfd9dc86431d40fd5018ea2b05fa214d1e9de6177f9c26734968e1'),
 'gemm_a4w4_kernel': ('b7771260339dc1566b8a873e01a4ca8f1b874efe9cf2b960290644f51177749f',
                      'ac9142c61247dfe2b374299277fbcc3a02ed87047013a4ee147deb09756b520e'),
 'gemm_a8w8_blockscale_kernel': ('ff7ec99f94ad4c7bc42384c564546240a2046b1d45a2dcd5dbc95c3ecee03571',
                                 '9f4921d182bb6dd05b97abd49735262301e068a0163457e1f36626334123aca1'),
 'gemm_a8w8_bpreshuffle_kernel': ('8a508376c948226a90fb1fdfda698f68415b64050741a626e87790ea7306d01c',
                                  'e4278a3637b56eab94baec3b712ff1fb5ac44206a0ac21b1925a7497ca0b9643'),
 'gemm_a8w8_kernel': ('2d6158f0cc752c3a8146288e1adf12e987048739da04b16d15bbb8b21745313d',
                      '4721f66a9c655b75e8d814a7e0deae406b8d11b25421b3fee0a60c99314331ae'),
 'gemm_a8w8_per_token_scale_kernel': ('943d9bc97ef710fa7434522107472eb74d09e91fe7c1722d0cf1ec09d4d70118',
                                      '2d26bb5545a53365308d980b908ebdd877f1e0cce1698de8635a633065080544'),
 'gemm_a8wfp4_kernel': ('f7e04ed989f9d56c7b1f7e1af88dd80a78c0ada6d9ab323e10f11cc99dd7d8bf',
                        '0d85c50a5ff6aa72ff02edfe008c4c8902cf062592c85cbbfdb3dde91f8c2b4d'),
 'gemm_afp4wfp4_kernel': ('01eb151afb088368410913a203e9655b0b05aaef842504560f0638416f63deca',
                          'c2284219eb9736fc4490dd16907ee829f96857d6ba79ad734c162deb7afffc92'),
 'gemm_afp8wfp8_kernel': ('2867d94442ea900accc29c6eb44e3996dedbf93a80f1666ee14d9fe2fb8cf806',
                          'a37d7aca8f84d897db94ee8e3df70de17927bd4592d6fdc0917553c6d8a80305'),
 'hgemm_kernel': ('1e6bbc90dc157c22d8210890bfd147638d16fc72e57648d33088e4c3d9f3974f',
                  '89ca4fce55817fdf5fcbaea925a1639f9b96cd809dcda8c06cd70f9e1033372e'),
 'jagged_dense_bmm_kernel': ('674878b422bb6d6a7259225898d3ed954676ff55f0a7ad05c65391f5e65aec8c',
                             '1b446ea35fee03f47ae16121fb7ba8aa933e9e99c48f2d90584c770d56065186'),
 'layernorm2d_kernel': ('844bca314b686fd2bb17b421fe25d38253f7987c9a8eb1a83eaf7434ff9e1465',
                        '435756584089bd62e3f2748d63a0eec3423740efc16d34eeb39364be59b07f90'),
 'layernorm2d_with_add_kernel': ('19fb3c3dce3e3b86abb87b79fcf00222f7ac1fc43ef382151c2b0a3578936d07',
                                 '6f29579e002744b3563e49132736f0d674bf884a35344e2d39842f21670348c7'),
 'moe_2stage_generic_kernel': ('906e05622df2f18e18cc06d35e595a11d6e0cba2ec19bb9ed72af6303088a9d3',
                               '28ffd84ad08759c7777af852d1596af211c154370f066e3c36f981ec445d9ca4'),
 'moe_biased_grouped_topk_kernel': ('f83ca84c4d16402b40cc85e5b7ea1d517ed3740b63fbe8f10af54b353f922173',
                                    'aea11fbf30e568dcd6f5e1c91ef7aaa50bfa1d209f5b24e93aa7d03f1e946f13'),
 'moe_sorting_kernel': ('9291ca50ecb38e07d95db1555ff8b4047854ed947ab0b26644a4871b88dfff71',
                        'c874911efc1c947437d5c7c62019e7c52b34458a0ec1560ecdde72aa971de271'),
 'moe_topk_sigmoid_kernel': ('15f51f440ac28443db52c54c31f43603395c95d9735bbc34a5a561ef0c9b775a',
                             '1f4d4d4e49717fc2093ba94cf5a0a87e9035f2db000125426d8226078984aa66'),
 'moe_topk_softmax_kernel': ('d5175a8ca84b620feaa25ee53d5798daf789bdacc04dd29a0e95a58225ded5af',
                             '4b44d0858365a99e907cf9ad902dce929a547a720fab3f7a7c426c3ce521f5b9'),
 'moe_topk_softplus_kernel': ('3704af9f75990d728cb64f6796a98123fc35fa676541d6a081ea3a270dbefffc',
                              'e02faeb19cdf9303c8fc7c8e61e28f2d588e713b8561a309aa2a70ea148e0fe1'),
 'per_1x128_fp8_quant_kernel': ('3bf7e1c84dc956913eb7ab7bd8eba0819a18340e3a8ae1a2a1c9555e6e2fedd9',
                                'bea46d8e57ac4d470da4a9429f07477ab443bf2dcb6b36dd359ef86a48358832'),
 'per_tensor_fp8_quant_kernel': ('7af7567c82fad169919609cdd500d2943e8028f521efd3b3368cdca011bb737a',
                                 '9e4fa8b1f616fad9c450c2b1a7bf807a70eea38ce008d82b46183b60b60a47fb'),
 'per_token_fp8_quant_kernel': ('e3f475ddaaab590f0625621d9e8cb655726c588f0212c628934b5101dda2ace9',
                                'e42ad94f8b72d6095477292ec018a45663452f5127d57ec6f855de15cf0f540d'),
 'per_token_i8_quant_kernel': ('bba6fe4d2085fc716d90a38bda8cec2c592c46c982e7caee4590f7d0c8a5de45',
                               'ed02e9cd668486366652e23865d621fb2ccb90b52e13bf37ce9fb73b591dd889'),
 'qk_norm_rope_quant_kernel': ('14156307fd446e4177cb2208a81fe7698a27750f5808ab4a36c384c8d2bdd465',
                               '152a32140302f1264c555fbd9b6f9d8362583fd08b0344290a67d6b1bb849ee2'),
 'quant_mxfp4_kernel': ('5e20bc87fde14cad8e2c5f7a2a0dc2741892f07727e5ad941c231fe203b6250d',
                        'd08a672059c4046d0f960ec5fe951ffa7fe151f27ce56e36189aeda162b293bc'),
 'rmsnorm2d_dynamicquant_kernel': ('d535eec1cf8919cb36a2e395721ce3fd60f0d06814f149040c76a7529b7f0520',
                                   '5cd3abaf088651f8f2ed9db44513bfd8c5238a45145aaf3dcd3168d63e3ff080'),
 'rmsnorm2d_kernel': ('511e9ece3ec9dab7056427e28f7b33cb46c1b1115200fb9b9f0ca1181e6d0fba',
                      '8442cb4d63444e7dfa9db1fa2d6253ceee0463b1debfd6919fc5219181de3b13'),
 'rmsnorm2d_smoothquant_kernel': ('e45fba78d1264de6a3533cbf766bd6acb4400f9747d610de79ebcf5536649149',
                                  '5496289dc3f8f72c1b9deefce1e39b7c1a00dc5726ad708abe65641fee4edf0e'),
 'rope_2d_fwd_kernel': ('c4a9e9c4223234127fdceeb2141853b0a43998373c977dd98140d6c16f02c115',
                        'dfcd006f67b8386be3a11f314ed3f64547a3b3413c60ae05d70a1a515cb55c6a'),
 'rope_fwd_kernel': ('9dc3b1def153c9fa64303310e1a2fb466a3b6297ec507e1c2c91aba1bbb99966',
                     'e129cd9500e057860e1cc9ed5a46326bd930102cdad8b854d88f7296fdd290b5'),
 'rope_thd_fwd_kernel': ('afa5038eb76ea2922b2f65284297945424de23f382f152fd1c86ac61b0d53deb',
                         '9f177fc09413c3439b57f048e83eb4b587053422fedb966042d7402d599c4bf9'),
 'silu_and_mul_kernel': ('652e2fc9469250e65070ef11d3bd5195b6b62c2b2a38f574d4aa82910172fc45',
                         'c2326fe7af8aa09d4b34eec6f31635569c510c727ed69bb6c45049604e0d7475'),
 'silu_and_mul_quant_kernel': ('95df5680d4ff20956b8ed11e75306df2ed4f046408578fa0ef48ecbdc69750d3',
                               'ad1e01617408f093753c795116745f9d9431c5eb8771fde6c4616b9ee4de764b'),
 'smoothquant_kernel': ('fee0a5e7f99fabe59c2639406c87d3d4f5cc64d8a8f84e79a458f563768716bf',
                        '989f11c99744efffc0ac8ab43cb828a7a11b23f1cce3a7958e323845be88695d'),
 'swiglu_and_mul_kernel': ('54369dafc10c09e918eda94322eb8aed5708b302f880b49d778565ede721c65e',
                           '74765a3a6e469d27926a92b0d710231f2bb7604850186fca782dc5446a8224b4')}


def test_torch_numerical_gates_cases_and_models_preserved():
    for name,(expected,model_hash) in TORCH_ORIGINAL_CORRECTNESS_AND_MODEL_SHA256.items():
        task=ROOT/"tasks/torch2flydsl"/name
        tree=ast.parse((task/"test_kernel_harness.py").read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_correctness")
        for handler in ast.walk(fn):
            if isinstance(handler,ast.ExceptHandler) and isinstance(handler.type,ast.Name) and handler.type.id=="NotImplementedError":
                assert isinstance(handler.body[0],ast.Raise)
                assert "no baseline fallback" in ast.unparse(handler.body[0])
                handler.body.pop(0)
        if name in {"silu_and_mul_kernel", "batched_gemm_bf16_kernel", "hgemm_kernel", "rmsnorm2d_kernel", "moe_topk_softmax_kernel", "moe_topk_sigmoid_kernel", "moe_topk_softplus_kernel", "moe_biased_grouped_topk_kernel"}:
            fn = _RemoveAddedReplayChecks().visit(fn)
        if name in {"silu_and_mul_quant_kernel", "smoothquant_kernel"}:
            fn = _RemoveFusedQuantChecks().visit(fn)
        if name in {"rope_fwd_kernel", "rope_thd_fwd_kernel"}:
            fn = _RemoveRopeChecks().visit(fn)
        if name in {"layernorm2d_kernel", "layernorm2d_with_add_kernel"}:
            fn = _RemoveLayernormChecks().visit(fn)
        if name in _STANDARD_QUANT_NAMES or name in {"quant_mxfp4_kernel", "rope_2d_fwd_kernel"}:
            fn = _RemoveStandardQuantChecks().visit(fn)
        if name == "rmsnorm2d_dynamicquant_kernel":
            fn = _RemoveRmsDynamicQuantChecks().visit(fn)
        if name == "moe_2stage_generic_kernel":
            fn = _RemoveGenericMoeChecks().visit(fn)
        if name in _QUANT_GEMM_CONTROL_NAMES:
            fn = _RemoveQuantGemmChecks().visit(fn)
        if name == "fused_add_rmsnorm_kernel":
            fn = _RemoveAddRmsnormChecks().visit(fn)
        if name in {"fmoe_fp8_blockscale_g1u1_kernel", "fmoe_g1u1_tkw1_kernel"}:
            fn = _RemoveFmoeTimingChecks().visit(fn)
        if name == "dynamic_mxfp8_quant_kernel":
            fn = _RemoveMxfp8Checks().visit(fn)
        if name == "batched_gemm_a8w8_kernel":
            fn = _RemoveBatchedInt8Checks().visit(fn)
        if name in {"gelu_fast_kernel", "gelu_and_mul_kernel", "gelu_tanh_and_mul_kernel", "swiglu_and_mul_kernel"}:
            fn = _RemoveActivationReplayChecks().visit(fn)
        assert hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()==expected,name
        assert hashlib.sha256((task/"model.py").read_bytes()).hexdigest()==model_hash,name
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_benchmark")
        direct=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="arena_benchmark")
        def calls(fn):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(fn) if isinstance(n,ast.Call) and getattr(n.func,"id","") in {"benchmark_cuda_graph_or_events","_mean_ms"}]
        assert calls(original)==calls(direct),name

# Base fingerprints cover original input generation, cases, tolerances, output
# checks, timing, warmups and resets. Only declaration/loader plumbing and the
# explicitly extracted reference expressions are normalized below.
class _InlineTritonReference(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            name=node.func.id
            if name=="_reference_softmax":
                return ast.parse("torch.softmax(x, axis=1)",mode="eval").body
            if name=="_reference_gemm":
                return ast.parse("F.linear(x, w, bias=None)",mode="eval").body
            if name=="_reference_layernorm":
                return ast.parse('F.layer_norm(x, (shape["N"],), weight=weight, bias=bias, eps=EPS)',mode="eval").body
        return node

    def visit_Assign(self,node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,"id",None)=="_reference_quant":
            mode=ast.literal_eval(node.value.args[2])
            source={
                "static":"ref = (x / scale).to(qdtype)",
                "dyn_tensor":"x_f32 = x.to(torch.float32)\nx_max = torch.max(torch.abs(x_f32))\nscale_ref = x_max / _dtype_max(qdtype)\nref = (x_f32 / scale_ref).to(qdtype)",
                "dyn_token":"x_max, _ = torch.max(torch.abs(x), axis=-1)\nscale_ref = x_max.to(torch.float32) / _dtype_max(qdtype)\nref = (x * (1 / scale_ref[:, None])).to(qdtype)",
            }[mode]
            return ast.parse(source).body
        return self.generic_visit(node)


def _protected_triton_fingerprint(source, *, added_replay_checks=False, sglang_elementwise=False, attention_two=False, mla_checks=False, gr_checks=False, state_checks=False, flat_checks=False, router_checks=False, combine_checks=False, gdn_checks=False, unified_checks=False, recurrent_checks=False, prepared_two=False):
    tree=ast.parse(source)
    if prepared_two:
        tree = _RemovePreparedTwoChecks().visit(tree)
    if recurrent_checks:
        tree = _RemoveRecurrentChecks().visit(tree)
    if unified_checks:
        tree = _RemoveUnifiedChecks().visit(tree)
    if gdn_checks:
        tree = _RemoveGdnChecks().visit(tree)
    if combine_checks:
        tree = _RemoveCombineChecks().visit(tree)
    if router_checks:
        tree = _RemoveRouterChecks().visit(tree)
    if flat_checks:
        tree = _RemoveFlatChecks().visit(tree)
    if state_checks:
        tree = _RemoveStateChecks().visit(tree)
    if gr_checks:
        tree = _RemoveGrChecks().visit(tree)
    if mla_checks:
        tree = _RemoveMlaChecks().visit(tree)
    if attention_two:
        tree = _RemoveAttentionChecks().visit(tree)
    if sglang_elementwise:
        tree = _RemoveSglangElementwiseChecks().visit(tree)
    if added_replay_checks:
        tree = _RemoveAddedReplayChecks().visit(tree)
        tree = _RemoveSglangReplayChecks().visit(tree)
        tree = _RemoveTritonBatchedChecks().visit(tree)
        tree = _RemoveTritonQuantChecks().visit(tree)
        tree = _RemoveMqaChecks().visit(tree)
        tree = _RemoveElementwiseChecks().visit(tree)
    excluded={"_checked_mrope_output", "_compare_mrope_output", "_reference_glm", "_check_glm_axis_map", "_mrope_replay_validator", "_checked_lightning_output", "_lightning_gate", "_compare_lightning_output", "_check_lightning_slots", "_lightning_replay_validator", "_checked_recurrent_output", "_compare_recurrent_output", "_recurrent_replay_validator", "_require_unused_state", "_checked_unified_output", "_compare_unified_output", "_unified_replay_validator", "_checked_gdn_output", "_compare_gdn_output", "_gdn_replay_validator", "_checked_combine_output", "_compare_combine_output", "_combine_replay_validator", "_check_output_buffer", "_checked_router_output", "_compare_router_output", "_router_replay_validator", "_checked_flat_output", "_compare_flat_output", "_flat_replay_validator", "_checked_state_output", "_compare_state_output", "_state_replay_validator", "_checked_gr_output", "_compare_gr_output", "_gr_replay_validator", "_checked_mla_output", "_compare_mla_output", "_mla_replay_validator", "_checked_attention_output", "_compare_attention_output", "_attention_replay_validator", "_checked_pair_output", "_compare_routing_pair", "_pair_replay_validator", "_checked_scaled_gemm_output", "_scaled_gemm_replay_validator", "_checked_sglang_output", "_compare_sglang_output", "_sglang_replay_validator", "_checked_elementwise_output","_elementwise_replay_validator","_checked_mqa_output","_compare_mqa_output","_verify_mqa_timed","_checked_mx_pair","_mx_reference","_compare_mx_pair","_verify_quant_timed","_checked_quant_output","_compare_token_outputs","_batched_replay_validator","_load_source","load_module","run_compile","_prepare_kernel","_make_prepared_fused_moe_runner",
              "_reference_softmax","_reference_gemm","_reference_layernorm","_reference_quant"}
    nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name not in excluded:
            nodes.append(_InlineTritonReference().visit(n))
        elif isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) and t.id not in {"SOURCE_FILE","DTYPE_NAME","_HERE"} for t in n.targets):
            nodes.append(n)
    return hashlib.sha256(ast.dump(ast.Module(nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def test_triton_preserves_original_harness_semantics_inputs_and_timing():
    for name,expected in TRITON_PROTECTED_SHA256.items():
        task=ROOT/"tasks/triton2flydsl"/name
        assert _protected_triton_fingerprint((task/"test_kernel_harness.py").read_text(), added_replay_checks=name in {"aiter/fused_add_rmsnorm", "aiter/moe_routing_sigmoid_top1", "aiter/gemm_a8w8", "aiter/gemm_a16w8_blockscale", "aiter/gemm_a8w8_blockscale", "aiter/gemm_afp8wfp8", "aiter/ff_a16w16", "aiter/fused_silu_mul", "aiter/fused_clamp_act_mul", "aiter/rmsnorm", "aiter/fp8_mqa_logits", "aiter/dynamic_mxfp8_quant", "aiter/dynamic_quant_fp8", "aiter/batched_gemm_a8w8", "aiter/batched_gemm_bf16", "aiter/gemm_a16w16", "aiter/softmax", "aiter/layernorm", "sglang/decode_attention", "sglang/sglang_fused_moe"}, sglang_elementwise=name in {"sglang/gdn_l2norm_fwd", "sglang/fused_norm_gate", "sglang/chunk_local_cumsum"}, attention_two=name in {"aiter/mha", "sglang/prefill_attention"}, mla_checks=name == "aiter/mla", gr_checks=name.startswith("generative_recommenders/"), state_checks=name in {"sglang/merge_state", "sglang/ssd_chunk_state", "sglang/fused_dual_residual_rmsnorm"}, flat_checks=name in {"aiter/rope_fwd", "aiter/moe_fused_gemm"}, router_checks=name in {"sglang/fused_gdn_gating", "sglang/fused_moe_router"}, combine_checks=name in {"sglang/experts_combine", "sglang/gdn_chunk_fwd_o"}, gdn_checks=name in {"sglang/chunk_scaled_dot_kkt_fwd", "sglang/wy_fast"}, unified_checks=name in {"aiter/unified_attention", "aiter/unified_attention_sparse_mla"}, recurrent_checks=name in {"sglang/gdn_chunk_fwd_h", "sglang/gdn_fused_recurrent_decode"}, prepared_two=name in {"sglang/triton_mrope_fused", "sglang/lightning_attn"})==expected,name
        cfg=yaml.safe_load((task/"config.yaml").read_text())
        assert cfg["baseline"]["kind"]=="initial_candidate"
        assert cfg["baseline"]["language"]=="triton"
        assert cfg["candidate"]["initial_language"]=="triton"
        assert cfg["candidate"]["initial_state"]=="implemented"


@pytest.mark.parametrize("task",[t for t in TASKS if "triton2flydsl" in t.parts],ids=lambda t:t.name)
def test_triton_initial_syntax_compiles_but_is_not_a_final_flydsl_candidate(task):
    result=invoke(task,"baseline","compile")
    assert result.passed,result.reason
    assert result.metadata["compile_kind"]=="python_bytecode"
    result=invoke(task,"candidate","compile")
    assert not result.passed
    assert "FlyDSL" in result.reason,result.reason


def test_triton_manifest_covers_every_original_variant():
    tasks=[t for t in TASKS if "triton2flydsl" in t.parts]
    assert len(tasks)==51
    counts=[0,0]
    for task in tasks:
        rows=json.loads((task/"cases.json").read_text())["cases"]
        actions=module(task/"scripts/task_actions.py")
        assert len(rows)==actions.CORRECTNESS_COUNT
        assert set(actions.PERFORMANCE_IDS.values())=={r["test_case_id"] for r in rows if "performance" in r["checks"]}
        counts[0]+=len(rows);counts[1]+=len(actions.PERFORMANCE_IDS)
        for row in rows:
            assert "correctness" in row["checks"]
    assert counts==[593,363]
    def variants(name,key):
        rows=json.loads((ROOT/"tasks/triton2flydsl/aiter"/name/"cases.json").read_text())["cases"]
        return {r["params"][key] for r in rows}
    assert variants("dynamic_quant_fp8","mode")=={"static","dyn_tensor","dyn_token"}
    assert variants("dynamic_quant_fp8","quant_dtype")=={"int8","fp8"}
    assert variants("batched_gemm_bf16","with_bias")=={False,True}
    assert variants("ff_a16w16","activation")=={None,"gelu_tanh","relu","silu_exp2"}
    assert variants("moe_fused_gemm","mul_routed_weight")=={False,True}
    assert variants("rope_fwd","style")=={"NEOX","GPTJ"}


def test_triton_actions_reject_partial_or_failed_correctness():
    import types
    task=ROOT/"tasks/triton2flydsl/sglang/gdn_chunk_fwd_h"
    actions=module(task/"scripts/task_actions.py")
    for result in (False,None,(True,None,[]),(False,"wrong state",[{}]*actions.CORRECTNESS_COUNT)):
        with pytest.raises(RuntimeError):actions.check(types.SimpleNamespace(run_correctness=lambda:result))
    actions.check(types.SimpleNamespace(run_correctness=lambda:(True,None,[{}]*actions.CORRECTNESS_COUNT)))


def test_real_moe_host_dtype_bridge_preserves_original_triton_tokens():
    import types
    import torch
    for rel,name in [("aiter/moe_fused_gemm/moe_fused_gemm.py","fused_moe"),("sglang/sglang_fused_moe/sglang_fused_moe.py","invoke_fused_moe_kernel")]:
        p=ROOT/"tasks/triton2flydsl"/rel
        fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        body=[n for n in fn.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        bridge=body[0]
        assert isinstance(bridge,ast.If) and "compute_type" in ast.unparse(bridge)
        tl=types.SimpleNamespace(bfloat16=object(),float16=object())
        for source,expected in [(torch.bfloat16,tl.bfloat16),(torch.float16,tl.float16),(tl.bfloat16,tl.bfloat16),(tl.float16,tl.float16)]:
            ns={"torch":torch,"tl":tl,"compute_type":source}
            exec(compile(ast.Module([bridge],type_ignores=[]),str(p),"exec"),ns)
            assert ns["compute_type"] is expected


def test_triton_known_answer_detects_a_state_update_bug(tmp_path):
    src=ROOT/"tasks/triton2flydsl/sglang/gdn_chunk_fwd_h"
    task=tmp_path/"task";shutil.copytree(src,task)
    p=task/"test_kernel_harness.py"
    source=p.read_text()
    old="state = state + bv_bf.T @ k_c"
    assert old in source
    p.write_text(source.replace(old,"state = state - bv_bf.T @ k_c"))
    result=invoke(task,"validate-task")
    assert not result.passed and "known answer" in result.reason


def test_triton_negative_control_detects_broken_task_comparator(tmp_path):
    src=ROOT/"tasks/triton2flydsl/generative_recommenders/swiglu"
    task=tmp_path/"task";shutil.copytree(src,task)
    p=task/"test_kernel_harness.py";s=p.read_text();lines=s.splitlines()
    fn=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=="_close")
    lines[fn.lineno-1:fn.end_lineno]=['def _close(ref, out):','    return True, 1., 0.']
    p.write_text('\n'.join(lines)+'\n')
    result=invoke(task,"validate-task")
    assert not result.passed and "deliberately incorrect" in result.reason


def test_benchmark_metadata_expansion_is_accepted_without_inventing_method():
    runtime=module(ROOT/"tasks/triton2flydsl/aiter/gemm_a16w16/task_runtime.py")
    cases=runtime.manifest();actions=module(ROOT/"tasks/triton2flydsl/aiter/gemm_a16w16/scripts/task_actions.py")
    cases=[c for c in cases if "performance" in c["checks"]]
    # Public helper returns (ms, metadata); legacy harness expands that mapping.
    meta={"benchmark_method":"cuda_event_fallback","fallback_reason":"fixture"}
    records=[{"test_case_id":old,"execution_time_ms":.1,**meta} for old in actions.PERFORMANCE_IDS]
    rows=runtime.require_result_rows(records,cases,actions.PERFORMANCE_IDS)
    assert len(rows)==len(cases)
    assert all(r["benchmark_method"]=="cuda_event_fallback" for r in rows)
    assert all(r["metadata"]["fallback_reason"]=="fixture" for r in rows)


TRITON_PROTECTED_SHA256 = {'aiter/batched_gemm_a8w8': 'd8740097efba2676164543599ebf3b1a6f6dff9b3ce2b47429a559f88b97226c',
 'aiter/batched_gemm_bf16': 'a0331afcc0b7e04b7df26cd339ae02bc408a4c8ad53e8dc6ca0e515faa7747fe',
 'aiter/dynamic_mxfp8_quant': '2b7fd1deb4cd4510eec598d674844d95f93242ae87dffe38d29dfeebf23d50b7',
 'aiter/dynamic_quant_fp8': '2f48c1322f84ec9b3b62feb05510f57d448d19c8c8aa6afafb623a5b5b4474ca',
 'aiter/fav3_sage': '5ede8776efee40a6ceb8f989bcec3f5d9e859e60fddcc346dc3eb19bfc94c19c',
 'aiter/fav3_sage_mxfp4': 'd238e3d4ae01cb47cd8324ed7af06442e475369d3855c7e504bf9611126b1a2a',
 'aiter/ff_a16w16': '85661a08ea4feaa4bf121ed30380abff7893f782263bf09f909a5002b83e3fba',
 'aiter/fp8_mqa_logits': 'd3b8d3059b8a76c20d37903501dcdf28dea6672edb320dae5401c92e8ecdc8a0',
 'aiter/fused_add_rmsnorm': '106759ede5640fb35d0b3ef7f4e1bbe91fa2a8659e70d391b0d9db7b2f9e9bbd',
 'aiter/fused_clamp_act_mul': '889f2343ef464df5b06a467ddba02e92b360fca1a28dd0574b85dc83f2dc6cc2',
 'aiter/fused_silu_mul': '79fdf195148b9f089b910d85f7217c1fa07481dac46e4dbb80c2b80670a981c2',
 'aiter/gemm_a16w16': '282a1199cc83a47a57f02e7b3f7ad84bc3ba804efa5b2714a3249e388b3fd3ee',
 'aiter/gemm_a16w8_blockscale': '2eaa821a6f6ad09ff2ad5cd42ff98fea704576c15d7d66d7d4bc2b51e51b6075',
 'aiter/gemm_a8w8': '0699185f68711faafff6caac01859f38d807def2d9d4b652962f95bb3fe04aa1',
 'aiter/gemm_a8w8_blockscale': '0736be31cb99af41a8d00ebccd24a11153a040bf291337f526d4b4469c174674',
 'aiter/gemm_afp8wfp8': 'e23e2d7e24bf8965fd43ab10521debeb6df4c7caee75e4b08dce812f6c4d5a2b',
 'aiter/layernorm': '6ce5fe3a4c299f3da1b738d7edd4180a6bd85cfe7fde7fb1079ca3fc229c2de9',
 'aiter/mha': 'baa74537ab3aa7abccd2e3a756f11b16d765baa9e2f0b32b93f8a3369715ac2e',
 'aiter/mla': 'dfddf7034947d7f4e4854e664487f1f9d8594d675e2d54dde55b006b03048059',
 'aiter/moe_fused_gemm': 'f0f85aa10cef13505d6b5ff9c4fa82b57916d76ae3b0b10ed6e2dd7a8527df99',
 'aiter/moe_routing_sigmoid_top1': '4c6cdef84f310dc209fe27e5962cfd32c7905def5c5cc5c9c2f178d0d1124a84',
 'aiter/rmsnorm': 'dfa1a9ad60af1cfbf595e8bb3c5095636b531bd4618ef4007e9ca16d93d1eb54',
 'aiter/rope_fwd': '7f344854df63ee74dbe8ed66a5857237d7483092da5d8569b5d75725b15a90c5',
 'aiter/softmax': '0824d5bcbf1efbcb0ef00b772506d500fbe66320bc081d4ecdd37567dc3daf0a',
 'aiter/unified_attention': 'dc52a073b202c5cec75a4d672766a7115d7461e426bc3d28527ab6bf11c4397e',
 'aiter/unified_attention_sparse_mla': '730c276f56b2313e2e08090ff597c4a8e5196c9e667fc98257596e92f5ca05ec',
 'generative_recommenders/jagged_dense_bmm_broadcast_add': 'a134469a78b7d386deaad6106d6443fa26dba1490529b32ab71beeed2fe42a9e',
 'generative_recommenders/jagged_dense_broadcast_add': '8070ed47d66de6dc07a80fb89f1379be5a51a148a633760a77eb02f11e91c7be',
 'generative_recommenders/layer_norm': '91e577e22379b564b07fc980b09b381ba12bc4c7ff2e3f537795b101e1693506',
 'generative_recommenders/swiglu': '8cff11ca2f29f2d20b70f672bd08f5c978eb84e4d0685e5987b974124d68bc8d',
 'sglang/chunk_local_cumsum': 'c19460a041ef88d6d1bda3f5dc273e06547083b66ecab5812e3fc487cc61ef6d',
 'sglang/chunk_scaled_dot_kkt_fwd': 'd5494cea25747437be1f6a7a2e498bf48a0c88617148c8c6c636254d75fa0ba9',
 'sglang/decode_attention': '841ae724d2fef05e643d830eadf8ecb87811b769af5c0e7462b79718ee940436',
 'sglang/dsv4_fp4_indexer': '2f44475812d6ff8ee771b3f7ea13eac0d2b904e6af769a2e92814322f32cb684',
 'sglang/experts_combine': 'c2b615576def29c1873a2d6e47a1e27211a54fb3d11e2536916adf40bba8563c',
 'sglang/extend_attention': 'dfe3ced0bea49210a61dc05b274397796dfd1e64f2087b98bf3007dd8e4542b4',
 'sglang/fused_dual_residual_rmsnorm': '1d8e18122149c7d81e7fa7319c2f5b6e000b43a205dd96d703ce5aafc925b727',
 'sglang/fused_gdn_gating': '647ddcb0a163d4eabf7583d81602e253b343ff2bc0d3306b9498a730e8a136c9',
 'sglang/fused_moe_router': 'a75dacd36cc4d930f0f36c8ef412953a798de35f76bffd8a64dc2657bcf607c2',
 'sglang/fused_norm_gate': 'f8e383d43355b3d261abf8ad6a1479e30fabb9fe458f4d068b993e1a3700ecb9',
 'sglang/gdn_chunk_fwd_h': 'e483d4c746630d49b5a70fcdc9c484d5a0f81493ab742fa8dd0e637080c6fffc',
 'sglang/gdn_chunk_fwd_o': '21fa391b4f23d75c555b0d08adff5c88b9f2fc07ec8f52ac8179ec9e945ea447',
 'sglang/gdn_fused_recurrent_decode': 'a2cf95190db9acb593241a181cd22df4621672a29debc6dbd03f237d57103b45',
 'sglang/gdn_l2norm_fwd': 'b9eea5fb7b300894b925f551bac954f220bad6fe6a1ecd7d803a12f268ecea96',
 'sglang/lightning_attn': '2f0f517ad6145f8129529c223624d14edad55b3060007ff7aa16ffb73345fbff',
 'sglang/merge_state': 'c2d19e307d111ddf8d6e26ad087b5f59f4cf6e67cd203bc308b10cdc8ad457c9',
 'sglang/prefill_attention': '5fb5a5b43e867327cf5a112689fbfc6103fed43c6eeabe3d7a9db0f1ad20c61d',
 'sglang/sglang_fused_moe': '005a78bdc7551901bf06507c31882ba1a9ba3479b08606526be1692dc323d758',
 'sglang/ssd_chunk_state': '88f7c12cefc1ac6ad586fd0e7b7ae109ff47242e7b4b989f63103c32a600c96e',
 'sglang/triton_mrope_fused': '408a924868dc1ec3114e8d97b48598c95dd6033f1b7296f1e9a5fe7e5c5b94c0',
 'sglang/wy_fast': 'a7bbbb6b8d61f6caa3c300bf7b299fc679b767abccde60833a72401d3517c799'}


def test_triton_initial_implementation_preserves_original_compute():
    bridges={"aiter/moe_fused_gemm/moe_fused_gemm.py":"fused_moe", "sglang/sglang_fused_moe/sglang_fused_moe.py":"invoke_fused_moe_kernel"}
    for rel,expected in TRITON_SOURCE_SHA256.items():
        tree=ast.parse((ROOT/"tasks/triton2flydsl"/rel).read_text())
        if rel in bridges:
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==bridges[rel])
            # The independently exercised neutral host dtype bridge is the sole
            # source change; original Triton kernels/launches remain bytecode-equivalent.
            bridge=fn.body[1]
            assert isinstance(bridge,ast.If) and "compute_type" in ast.unparse(bridge)
            fn.body.pop(1)
        assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==expected,rel


def test_invalid_candidate_syntax_is_reported_in_protocol(tmp_path):
    task=tmp_path/"task";shutil.copytree(ROOT/"tasks/triton2flydsl/aiter/softmax",task)
    (task/"softmax.py").write_text("def broken(:\n")
    result=invoke(task,"baseline","compile")
    assert not result.passed and "SyntaxError" in result.reason

TRITON_SOURCE_SHA256 = {'aiter/batched_gemm_a8w8/batched_gemm_a8w8.py': 'd10dcaef6641455d478c865bd1862ba8bd6d33d0b698538cd136c2a9ea3472dc',
 'aiter/batched_gemm_bf16/batched_gemm_bf16.py': '9b4cbeb1af22308f4888d93e2527e569803c9ef53fe3905545c19dc0333cd505',
 'aiter/dynamic_mxfp8_quant/dynamic_mxfp8_quant.py': 'f62a40d00fdfdce03664e2fe265881eff3596d82baf6e29ff9bb52c55809fba5',
 'aiter/dynamic_quant_fp8/dynamic_quant_fp8.py': '91eeb794415a8699273d5a9c1158debce795348eada895778deb8af324829c8f',
 'aiter/fav3_sage/fav3_sage.py': '134c3ca42fa42c77110dcaa3d0b30bdef260fa37e3316214f2776c355dad4964',
 'aiter/fav3_sage_mxfp4/fav3_sage_mxfp4.py': '69bd08c06f42d90c5881281ab752d52f86b162b7119a02945392ba6d68a3cb1b',
 'aiter/ff_a16w16/ff_a16w16.py': 'ccbb25db76cb6de4913c553856a3dbdc3218d645eb48298ade6e096152046917',
 'aiter/fp8_mqa_logits/fp8_mqa_logits.py': 'bfbaf7644d3b8e9bbde7e7cde46b6ae6dd16b97b115c8abb946b4e2ef6a30acb',
 'aiter/fused_add_rmsnorm/fused_add_rmsnorm.py': '77b6199eec5e5c891b16a48d50726a8ea42954633a9989596546e6ea8cb9c555',
 'aiter/fused_clamp_act_mul/fused_clamp_act_mul.py': 'bb0447f6ce9c58eb73cde2fd097d1fedca01db7e79bb8a9c8a162c5172af12f6',
 'aiter/fused_silu_mul/fused_silu_mul.py': '56542d48a81297de542bc6899664de6ee31f1ca0faf7758c9e6dbf3e0f4120eb',
 'aiter/gemm_a16w16/gemm_a16w16.py': 'af467b6c10f9a39982dfa4b67932ac3987df7c5e37e7bd660b5d7ac614135789',
 'aiter/gemm_a16w8_blockscale/gemm_a16w8_blockscale.py': 'e461e0f6b0c4065dadb3aa7f6b8d42090e45a2caafedaf2e632e604bcc7c3733',
 'aiter/gemm_a8w8/gemm_a8w8.py': '76a4497ad2258280ca2e7dbcf70bf2e75cdc513ac6a727e06b2c581c8e57acc1',
 'aiter/gemm_a8w8_blockscale/gemm_a8w8_blockscale.py': 'c273807cf10546c5a77f90338129dac7b6f9bfcf4dfe53298d555e17ec584850',
 'aiter/gemm_afp8wfp8/gemm_afp8wfp8.py': '63631b70d3cf97f3dc2c682f322a7f1e71272bbc78e6760693ad34465b737786',
 'aiter/layernorm/layernorm.py': '42035db0231238ef54943cc30dfb986c7994bf50cee40d2b4a7951680ec27136',
 'aiter/mha/mha.py': 'b567863379b1d0133c67035dcc238b4a837cb989c3687a110d28d3d505a26070',
 'aiter/mla/mla.py': 'e2e25d493b2fd95423a1b264e145180c81b3dc4a0990627b6661ad7c8b9f51b3',
 'aiter/moe_fused_gemm/moe_fused_gemm.py': '9762ceea15b969bcbfc482a25f581be621ca171807b479ff4eea4961c2aff379',
 'aiter/moe_routing_sigmoid_top1/moe_routing_sigmoid_top1.py': '882d4404d5e585ae98f9c0f7d2e3828eb4bf6b10d08ee280530ad95d8f5f83be',
 'aiter/rmsnorm/rmsnorm.py': '641e4b54a0be80e2d9b24c33f0006c9e16e8a4388941b80d6b17b6f9200ba725',
 'aiter/rope_fwd/rope_fwd.py': 'e98a37cb518f93e3fb40f29c63a23996403adb0f540787e9a41932c93a2457c3',
 'aiter/softmax/softmax.py': '829494ee4a57cd87892d89fabb20400d7e92467e17721861114bb2354e7426ab',
 'aiter/unified_attention/unified_attention.py': '9d889380751998cc6b518a1db5e166669ca36e125e335ef77e7bc379ac470bac',
 'aiter/unified_attention_sparse_mla/unified_attention_sparse_mla.py': 'e3a90181251c6a9d5995f63d638cf70b4badaaeda1c1696f82414276a768441e',
 'generative_recommenders/jagged_dense_bmm_broadcast_add/jagged_dense_bmm_broadcast_add.py': 'f42f3f4fa08c588907a749551a7463e4b8e77f5dfca92a9e1f1ed1958bda2e3c',
 'generative_recommenders/jagged_dense_broadcast_add/jagged_dense_broadcast_add.py': '05ac1765b2138ac27c8ca59fa1bf57fbac2965f6f6c44dc7cf26789cda653373',
 'generative_recommenders/layer_norm/layer_norm.py': '0118682f3a8700e1e8c735806aed8b20945e7592cacfa24d857ea836c344045b',
 'generative_recommenders/swiglu/swiglu.py': '96f77e03f9da841209d4603f1d3d308e05b973a12f5220761c907e92dbe6d712',
 'sglang/chunk_local_cumsum/chunk_local_cumsum.py': 'ecc8ac458425000fd12617bd84cbd393c2d89bf1d838398e1ea1cf719e1f254f',
 'sglang/chunk_scaled_dot_kkt_fwd/chunk_scaled_dot_kkt_fwd.py': 'cba5027df64812f48adfc41db29cafd1722218be22612ceb5ae3c6340464fadc',
 'sglang/decode_attention/decode_attention.py': '02421eccd7c46ed7e07fc8edbda2bed5f0f7fc67dd85288dee8d4660585a87d5',
 'sglang/dsv4_fp4_indexer/dsv4_fp4_indexer.py': 'abef459c844dc66404d09c9152d82288da1e19a3a380807f97452edddcb03ed5',
 'sglang/experts_combine/experts_combine.py': '165b80ce60456b5bc0d1125aaba9e06fd983d2a6ab77040b6b447aa4c7fdd31e',
 'sglang/extend_attention/extend_attention.py': '7cc30ba6bae3f45b1a300c7f34e37a71d8a39afb9c9eb451c607ddc27ac0b642',
 'sglang/fused_dual_residual_rmsnorm/fused_dual_residual_rmsnorm.py': 'de66df3f2b5d3984cd772c7c3e8b055944ff90e0ab1c58850724183b978417b7',
 'sglang/fused_gdn_gating/fused_gdn_gating.py': 'a7dfcef320613e938f56848f7c277647a2cc8faa2c51f7aa4d6496036269aa08',
 'sglang/fused_moe_router/fused_moe_router.py': 'a15625819a74cda7858bcbba0b821a240d852657f279fc572b2f404540f44309',
 'sglang/fused_norm_gate/fused_norm_gate.py': 'ca5b89fe8cf05428b505238049f4de4ccc635ed095a9168d7479d9ba25d4c7d0',
 'sglang/gdn_chunk_fwd_h/gdn_chunk_fwd_h.py': 'ca6f0993262fb1553d19eaffc15892956568b28ef812e5b0db4bfdf3d30ba37b',
 'sglang/gdn_chunk_fwd_o/gdn_chunk_fwd_o.py': '4291d6f8e3f5b97aad01cd6df5ff8ae5fd142ae38d3b5467a893296a231a1fe2',
 'sglang/gdn_fused_recurrent_decode/gdn_fused_recurrent_decode.py': '74257035ccbf4b4d4b3414d9fbded34e555cc967159b0e8a0acdfda70c56fd20',
 'sglang/gdn_l2norm_fwd/gdn_l2norm_fwd.py': 'f9d3a410e435bddbdefc7d9a1783e49096910d4f1d0e1a84eecdddfc99ce3690',
 'sglang/lightning_attn/lightning_attn.py': '56ab2e6b9acda62d47f31d5857eaa6717a09eb986207ab0c616c7e442f94fa47',
 'sglang/merge_state/merge_state.py': 'd465567a07f58e8fd2fcb3a01ec5cf218342cca2d1c26780efb0dfe36f7640d6',
 'sglang/prefill_attention/prefill_attention.py': '5a82ede42327451bf8860eb33364547fb9009b629fda53b2a0c352e5f0126802',
 'sglang/sglang_fused_moe/sglang_fused_moe.py': '195ac64c3f673af2c3192ec4f4461491e675a38901b9b627bb9842c990f7d194',
 'sglang/ssd_chunk_state/ssd_chunk_state.py': '1d9741ea3f23217ad6c2dced5d82928413b60501fcc200f2497d3b6b56bb6c50',
 'sglang/triton_mrope_fused/triton_mrope_fused.py': '80c9daa2ec4a3723983ada981f2363d49f8cd466b1b8226ef3b8814bf85f4a50',
 'sglang/wy_fast/wy_fast.py': '2e7c2c821cdec65faf508ffe2e573f80b76b7cbb2b06f534a059b9ba8f244da2'}


def test_declared_gpu_constraints_are_retained_without_unverified_widening():
    observed={}
    for task in TASKS:
        cfg=yaml.safe_load((task/"config.yaml").read_text())
        arch=cfg.get("platform_support",{}).get("required_arch")
        if arch:observed[str(task.relative_to(ROOT/"tasks"))]=arch
    expected = dict(ORIGINAL_REQUIRED_ARCH)
    # Explicit ports retain gfx942 and add gfx950 after real GPU actions.
    # Full task_validator qualification is tracked separately per source/runtime;
    # this CPU assertion is not GPU evidence itself.
    for name in ("rmsnorm_kernel", "softmax_kernel", "layernorm_kernel", "topk_gating_softmax_kernel",
                 "flash_attn_func_kernel", "hgemm_splitk_kernel", "fused_rope_cache_kernel", "silu_and_mul_fq_kernel",
                 "moe_sorting_kernel", "pa_decode_swa_kernel"):
        expected["flydsl2flydsl/" + name] = ["gfx942", "gfx950"]
    assert observed == expected

ORIGINAL_REQUIRED_ARCH = {'flydsl2flydsl/flash_attn_func_kernel': 'gfx942',
 'flydsl2flydsl/fp8_gemm_4wave_kernel': 'gfx950',
 'flydsl2flydsl/fp8_gemm_8wave_kernel': 'gfx950',
 'flydsl2flydsl/fused_rope_cache_kernel': 'gfx942',
 'flydsl2flydsl/hgemm_splitk_kernel': 'gfx942',
 'flydsl2flydsl/layernorm_kernel': 'gfx942',
 'flydsl2flydsl/moe_sorting_kernel': 'gfx942',
 'flydsl2flydsl/pa_decode_swa_kernel': 'gfx942',
 'flydsl2flydsl/rmsnorm_kernel': 'gfx942',
 'flydsl2flydsl/silu_and_mul_fq_kernel': 'gfx942',
 'flydsl2flydsl/softmax_kernel': 'gfx942',
 'flydsl2flydsl/topk_gating_softmax_kernel': 'gfx942',
 'torch2flydsl/gemm_a16wfp4_kernel': 'gfx950',
 'torch2flydsl/gemm_a4w4_kernel': 'gfx950',
 'torch2flydsl/gemm_a8wfp4_kernel': 'gfx950',
 'torch2flydsl/gemm_afp4wfp4_kernel': 'gfx950',
 'torch2flydsl/gemm_afp8wfp8_kernel': 'gfx950',
 'torch2flydsl/quant_mxfp4_kernel': 'gfx950',
 'triton2flydsl/aiter/fav3_sage_mxfp4': 'gfx950',
 'triton2flydsl/aiter/gemm_afp8wfp8': 'gfx950'}


class _RemoveAddedReplayChecks(ast.NodeTransformer):
    """Normalize only the added controls; retain the original numeric/timing AST.

    The controls themselves are exercised below. No original comparison,
    tolerance, seed, input generator, warmup or sample count is removed here.
    """
    def visit_Expr(self, node):
        value = node.value
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name) and value.func.id in {"require_tensor_contract", "require_unchanged", "_validate_pa_contract", "_checked_gemm_output", "_checked_rms_output", "_require_routing_contract"}:
                return None
            if (isinstance(value.func, ast.Attribute) and value.func.attr == "update"
                    and len(value.args) == 1 and isinstance(value.args[0], ast.Call)
                    and isinstance(value.args[0].func, ast.Name)
                    and value.args[0].func.id in {"verify_timed_run", "_verify_routing_timed"}):
                return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"originals", "expected", "timed"}:
                return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "_checked_silu_result":
            return self.visit(node.args[0])
        if isinstance(node.func, ast.Name) and node.func.id == "benchmark_cuda_graph_or_events":
            node.keywords = [k for k in node.keywords if k.arg != "timed_run"]
        return self.generic_visit(node)


_REPLAY_TASKS = ["triton2flydsl/aiter/gemm_a16w16",
                 "flydsl2flydsl/fp8_gemm_4wave_kernel",
                 "torch2flydsl/silu_and_mul_kernel",
                 "flydsl2flydsl/fp8_gemm_8wave_kernel",
                 "flydsl2flydsl/blockscale_preshuffle_gemm_kernel",
                 "flydsl2flydsl/preshuffle_gemm_v2_kernel",
                 "flydsl2flydsl/pa_decode_fp8_kernel",
                 "triton2flydsl/aiter/softmax", "triton2flydsl/aiter/layernorm"]


@pytest.mark.parametrize("task_name", _REPLAY_TASKS)
@pytest.mark.parametrize("behavior", ["correct", "cached", "cheap_wrong", "last_measured_wrong", "input_modified"])
def test_measured_and_replayed_outputs_use_real_numerical_controls(task_name, behavior):
    import torch
    from types import SimpleNamespace

    checks = module(ROOT / "tasks" / task_name / "scripts/replay_checks.py")
    x = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    w = torch.tensor([[3., 4.], [5., 6.]], dtype=torch.bfloat16)
    originals = (x.clone(), w.clone())
    expected = x @ w.T
    output = expected.clone()
    timed = SimpleNamespace(bound=True, outputs=output)
    if behavior == "last_measured_wrong":
        output.zero_()
    if behavior == "input_modified":
        x.add_(1)

    def replay():
        if behavior == "cached":
            output.copy_(expected)
        elif behavior == "cheap_wrong":
            output.fill_(x[0, 0].item())
        else:
            output.copy_(x @ w.T)
        return output

    timed.rerun = replay
    if "silu" in task_name:
        compare = lambda actual, ref: checks.normalized_output(actual, ref, tolerance=1e-2)
    else:
        atol, rtol = (1e-1, 1e-2) if "triton2flydsl" in task_name else (2e-2, 2e-2)
        compare = lambda actual, ref: checks.allclose_output(actual, ref, atol=atol, rtol=rtol)
    def check():
        return checks.verify_timed_run(
            timed, inputs=(x, w), originals=originals, expected=expected,
            perturb=lambda: x.neg_(), reference=lambda: x @ w.T, compare=compare,
        )
    if behavior == "correct":
        evidence = check()
        assert evidence["replay_correctness"] == "PASS"
        assert torch.equal(x, originals[0])
        assert torch.equal(w, originals[1])
    else:
        with pytest.raises(AssertionError):
            check()


@pytest.mark.parametrize("task_name", _REPLAY_TASKS)
@pytest.mark.parametrize("bad", ["broadcast_shape", "dtype", "device", "none", "nan"])
def test_replay_rejects_invalid_output_contract(task_name, bad):
    import torch
    checks = module(ROOT / "tasks" / task_name / "scripts/replay_checks.py")
    expected = torch.ones((2, 3), dtype=torch.bfloat16)
    actual = {"broadcast_shape": expected[:1], "dtype": expected.float(),
              "device": torch.empty((2, 3), dtype=torch.bfloat16, device="meta"),
              "none": None, "nan": torch.full_like(expected, float("nan"))}[bad]
    with pytest.raises(AssertionError):
        checks.allclose_output(actual, expected, atol=1e-1, rtol=1e-2)


def _harness_functions(task, names, namespace):
    tree = ast.parse((task / "test_kernel_harness.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(nodes, type_ignores=[]), "actual_harness_functions", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("wrong_shape", [False, True])
def test_triton_gemm_actual_correctness_rejects_broadcast_output(monkeypatch, wrong_shape):
    import torch
    from types import SimpleNamespace
    task = ROOT / "tasks/triton2flydsl/aiter/gemm_a16w16"
    checks = module(task / "scripts/replay_checks.py")
    x = torch.tensor([[1., 2.], [1., 2.]], dtype=torch.bfloat16)
    w = torch.tensor([[3., 4.], [5., 6.]], dtype=torch.bfloat16)
    def kernel(a, b):
        out = a @ b.T
        return out[:1] if wrong_shape else out
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    ns = _harness_functions(task, {"run_correctness", "_reference_gemm"}, {
        "TEST_SHAPES": [{"name": "controlled", "M": 2, "N": 2, "K": 2}],
        "_make_inputs": lambda *args: (x, w),
        "_load_source": lambda: SimpleNamespace(gemm_a16w16=kernel),
        "require_tensor_contract": checks.require_tensor_contract,
    })
    assert ns["run_correctness"](verbose=False) is (not wrong_shape)


def test_silu_replay_uses_original_normalized_max_gate():
    import torch
    task = ROOT / "tasks/torch2flydsl/silu_and_mul_kernel"
    checks = module(task / "scripts/replay_checks.py")
    expected = torch.tensor([[1., 100.]], dtype=torch.float32)
    # Global normalized error accepts a local 50% error at the small element.
    # Replacing this with per-element allclose would silently tighten the gate.
    actual = torch.tensor([[1.5, 100.]], dtype=torch.float32)
    checks.normalized_output(actual, expected, tolerance=1e-2)
    with pytest.raises(AssertionError):
        checks.normalized_output(torch.tensor([[1., 102.]]), expected, tolerance=1e-2)


def test_silu_provided_baseline_replay_oracle_is_independent():
    import torch
    from types import SimpleNamespace
    task = ROOT / "tasks/torch2flydsl/silu_and_mul_kernel"
    checks = module(task / "scripts/replay_checks.py")
    model = module(task / "model.py").Model()
    ns = _harness_functions(task, {"_silu_replay_validator"}, {
        "verify_timed_run": checks.verify_timed_run,
        "normalized_output": checks.normalized_output, "REL_TOL": 1e-2,
    })
    inp = torch.tensor([[0., 1., 2., 3.]], dtype=torch.bfloat16)
    # Handwritten sigmoid arithmetic is independent of Model.forward/F.silu.
    def oracle(value):
        gate, up = value.float().chunk(2, dim=-1)
        return ((gate / (1 + (-gate).exp())) * up).to(torch.bfloat16)
    validate = ns["_silu_replay_validator"](inp, oracle)
    output = model(inp)
    timed = SimpleNamespace(bound=True, outputs=output)
    def replay():
        output.copy_(model(inp))
        return output
    timed.rerun = replay
    assert validate(timed)["timed_output_correctness"] == "PASS"


@pytest.mark.parametrize("task_name", [
    "triton2flydsl/aiter/gemm_a16w16",
    "flydsl2flydsl/fp8_gemm_4wave_kernel",
    "flydsl2flydsl/fp8_gemm_8wave_kernel",
    "flydsl2flydsl/blockscale_preshuffle_gemm_kernel",
    "flydsl2flydsl/preshuffle_gemm_v2_kernel",
])
@pytest.mark.parametrize("bad_phase", [None, "measured", "replay"])
def test_actual_gemm_benchmark_binds_and_validates_timed_output(task_name, bad_phase, monkeypatch, tmp_path):
    """Execute task benchmark orchestration with CPU tensor / timing doubles.

    This checks that the real harness calls the controls on the measured output;
    it is not a CUDA graph or GPU timing test.
    """
    import math
    import types
    import torch
    task = ROOT / "tasks" / task_name
    checks = module(task / "scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    phase = {"name": "setup"}
    seen = []

    class Collector:
        bound = False
        outputs = None

    def benchmark(fn, warmup, repetition, timed_run=None):
        seen.append((warmup, repetition, timed_run is not None))
        phase["name"] = "measured"
        result = fn()
        if timed_run is not None:
            timed_run.outputs = result
            timed_run.bound = True
            def replay():
                phase["name"] = "replay"
                result.copy_(fn())
                return result
            timed_run.rerun = replay
        phase["name"] = "setup"
        return .1, {"benchmark_method": "cuda_graph"}

    a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    b = torch.tensor([[3., 4.], [5., 6.]], dtype=torch.bfloat16)
    if "blockscale_preshuffle" in task_name:
        b = b.repeat(64, 1)
    def compute():
        if bad_phase is not None and phase["name"] == bad_phase:
            return torch.full((a.shape[0], b.shape[0]), a[0, 0].item(), dtype=torch.bfloat16)
        return a @ b.T
    ns = {"TimedRun": Collector, "benchmark_cuda_graph_or_events": benchmark,
          "verify_timed_run": checks.verify_timed_run, "allclose_output": checks.allclose_output,
          "math": math, "json": json, "Path": Path}
    if "triton2flydsl" in task_name:
        ns.update({"_HERE": str(tmp_path), "WARMUP": 10, "ITERS": 100,
            "TEST_SHAPES": [{"name": "controlled", "M": 2, "N": 2, "K": 2}],
            "_make_inputs": lambda *args: (a, b),
            "_load_source": lambda: types.SimpleNamespace(gemm_a16w16=lambda *args: compute()),
        })
        _harness_functions(task, {"run_benchmark", "_reference_gemm"}, ns)
        run = lambda: ns["run_benchmark"](verbose=False)
    else:
        flydsl = types.ModuleType("flydsl")
        compiler = types.ModuleType("flydsl.compiler")
        flydsl.compiler = compiler
        monkeypatch.setitem(sys.modules, "flydsl", flydsl)
        monkeypatch.setitem(sys.modules, "flydsl.compiler", compiler)
        c = torch.zeros((a.shape[0], b.shape[0]), dtype=torch.bfloat16)
        scale = torch.ones(2)
        def compiled(*args):
            c.copy_(compute())
        ns.update({"_CANDIDATE_DIR": str(tmp_path), "HARNESS_SHAPES": [(2, 2, 2)],
            "ATOL": 2e-2, "RTOL": 2e-2,
            "_load_kernel": lambda *args: object(),
            "_make_inputs": lambda *args, **kwargs: (a, b, c, scale, scale.clone()),
            "_kernel_b": lambda mod, value: value,
            "_compile_and_run_once": lambda *args: (compiled, None),
            "_kernel_args": lambda *args: (None,),
        })
        names = {"arena_benchmark", "_torch_reference"}
        if "blockscale_preshuffle" in task_name:
            inp = {"M": 2, "N": 128, "K": 2, "scale_k": 1, "scale_n": 1,
                   "a_fp8": a, "b_fp8": b, "b_shuf": b.clone(),
                   "scale_a": torch.ones((1, 2)), "scale_b": torch.ones((1, 1)), "c": c}
            ns.update({"_KERNEL_DIR": str(tmp_path), "HARNESS_SHAPES": [(2, 128, 2)],
                       "_make_inputs": lambda *args, **kwargs: inp,
                       "_launch_args": lambda *args: (None,), "SCALE_BLOCK_K": 2, "OUT_DTYPE": "bf16", "TILE_M": 2, "TILE_N": 128, "TILE_K": 2})
            names = {"arena_benchmark", "_torch_blockscale_reference", "_time_mean_ms"}
        elif "preshuffle_gemm_v2" in task_name:
            ns["_select_config"] = lambda *args, **kwargs: (
                (2, 2, 2), compiled, None, (a, b, b.clone(), scale, scale.clone(), c))
        _harness_functions(task, names, ns)
        run = lambda: ns["arena_benchmark"](verbose=False)
    if bad_phase is None:
        records = run()
        assert records[0]["timed_output_correctness"] == "PASS"
        assert records[0]["replay_correctness"] == "PASS"
        assert seen[0] == (10 if "blockscale_preshuffle" in task_name else 0, 100, True)
    else:
        with pytest.raises(AssertionError, match="Numerical mismatch"):
            run()



def test_silu_output_contract_applies_to_public_operator_not_private_helpers():
    import torch
    from types import SimpleNamespace
    task = ROOT / "tasks/torch2flydsl/silu_and_mul_kernel"
    checks = module(task / "scripts/replay_checks.py")
    ns = _harness_functions(task, {"_require_candidate_outputs", "_checked_silu_result"}, {
        "KERNEL_ENTRY": "flydsl_silu_and_mul", "require_tensor_contract": checks.require_tensor_contract,
    })
    inp = torch.ones((2, 4), dtype=torch.bfloat16)
    # A helper is free to prepare differently shaped FP32 intermediates.
    candidate = SimpleNamespace(flydsl_prepare=lambda x: x.float().sum(0),
                                flydsl_silu_and_mul=lambda x, limit: x[:, :2])
    ns["_require_candidate_outputs"](candidate)
    assert candidate.flydsl_prepare(inp).dtype == torch.float32
    assert candidate.flydsl_silu_and_mul(inp, 0).shape == (2, 2)
    candidate.flydsl_silu_and_mul = lambda x, limit: x[:1, :2]
    ns["_require_candidate_outputs"](candidate)
    with pytest.raises(AssertionError, match="shape"):
        candidate.flydsl_silu_and_mul(inp, 0)


@pytest.mark.parametrize("function", ["arena_benchmark", "run_benchmark"])
@pytest.mark.parametrize("behavior", ["correct", "measured_wrong", "replay_wrong", "cached", "input_modified", "setup_error"])
def test_pa_benchmark_checks_actual_output_and_replay(function, behavior, monkeypatch, tmp_path):
    """CPU orchestration test of the real PA benchmark, not a GPU timing claim."""
    import math
    import types
    import torch
    task = ROOT / "tasks/flydsl2flydsl/pa_decode_fp8_kernel"
    checks = module(task / "scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    q = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
    k = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.bfloat16)
    v = torch.tensor([[3., 1.], [0., 4.]], dtype=torch.bfloat16)
    def reference():
        return (torch.softmax(q.float() @ k.float().T, -1) @ v.float()).bfloat16()
    ref = reference()
    out = torch.empty_like(ref)
    originals = tuple(x.clone() for x in (q, k, v))
    phase = {"name": "setup"}
    def launch():
        if behavior == "cached" and phase["name"] == "replay":
            return out
        if behavior == "input_modified" and phase["name"] == "measured":
            k.add_(1)
        bad = behavior == phase["name"] + "_wrong"
        out.copy_(torch.zeros_like(out) if bad else reference())
        return out
    launch.arena_inputs = (q, k, v)
    launch.arena_originals = originals
    launch.arena_perturb = lambda: q.neg_()
    launch.arena_reference = reference
    def build(*args, **kwargs):
        if behavior == "setup_error":
            raise ValueError("broken metadata")
        return launch, out, ref
    seen = []
    def benchmark(fn, warmup, repetition, timed_run=None):
        seen.append((warmup, repetition, timed_run is not None))
        phase["name"] = "measured"
        result = fn()
        if timed_run is not None:
            timed_run.bound = True
            timed_run.outputs = result
            def replay():
                phase["name"] = "replay"
                return fn()
            timed_run.rerun = replay
        phase["name"] = "setup"
        return .1, {"benchmark_method": "cuda_graph"}
    ns = {"math": math, "json": json, "Path": Path, "_KERNEL_DIR": str(tmp_path),
          "HARNESS_SHAPES": [(1, 1, (1, 1), "per_token")],
          "_load_kernel": lambda *args: object(), "_build_case": build,
          "TimedRun": types.SimpleNamespace, "benchmark_cuda_graph_or_events": benchmark,
          "verify_timed_run": checks.verify_timed_run, "require_tensor_contract": checks.require_tensor_contract}
    _harness_functions(task, {function, "_validate_pa_timing", "_compare_pa_output"}, ns)
    if behavior == "correct":
        result = ns[function](verbose=False)
        if function == "arena_benchmark":
            assert result[0]["timed_output_correctness"] == "PASS"
            assert result[0]["replay_correctness"] == "PASS"
        assert seen == [(0, 100, True), (0, 20, False)]
        checks.require_unchanged((q, k, v), originals)
    elif behavior == "setup_error":
        with pytest.raises(RuntimeError, match="PA setup failed"):
            ns[function](verbose=False)
        assert not seen
    else:
        with pytest.raises(AssertionError):
            ns[function](verbose=False)


def test_pa_replay_keeps_original_absolute_error_gate():
    import torch
    task = ROOT / "tasks/flydsl2flydsl/pa_decode_fp8_kernel"
    checks = module(task / "scripts/replay_checks.py")
    compare = _harness_functions(task, {"_compare_pa_output"}, {
        "require_tensor_contract": checks.require_tensor_contract})["_compare_pa_output"]
    # Absolute tolerance stays constant even when the reference magnitude changes.
    ref = torch.tensor([0., 1000.], dtype=torch.bfloat16)
    compare(ref + torch.tensor([.004, 0.], dtype=torch.bfloat16), ref)
    for delta in ([.006, 0.], [0., 4.]):
        with pytest.raises(AssertionError, match="Numerical mismatch"):
            compare(ref + torch.tensor(delta, dtype=torch.bfloat16), ref)
    with pytest.raises(AssertionError, match="dtype"):
        compare(ref.float(), ref)


@pytest.mark.parametrize("mutate", [False, True])
def test_pa_build_case_records_inputs_before_metadata_and_excludes_writable_scratch(monkeypatch, mutate):
    import types
    import torch
    task = ROOT / "tasks/flydsl2flydsl/pa_decode_fp8_kernel"
    checks = module(task / "scripts/replay_checks.py")
    cpu = torch.device("cpu")
    monkeypatch.setattr(torch, "device", lambda *args: cpu)
    monkeypatch.setattr(torch, "set_default_device", lambda *args: None)
    monkeypatch.setitem(sys.modules, "aiter", types.SimpleNamespace(dtypes=types.SimpleNamespace(fp8=torch.float32)))
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(cdiv=lambda a,b: (a+b-1)//b))
    cache = torch.ones((2, 1, 2, 2))
    scale = torch.ones((2, 1, 2))
    scratch = torch.empty(2)
    def metadata(query, *args):
        if mutate:
            query.add_(1)
        # A valid implementation may use a different private metadata format.
        return types.SimpleNamespace(private_scratch=scratch)
    mod = types.SimpleNamespace(get_pa_metadata=metadata, get_sw_ps_max_context_partition_num=lambda *args: 1,
                                pa_decode_ps_launch=lambda *args, **kwargs: None)
    ns = {"HEAD_SIZE": 2, "BLOCK_SIZE": 2, "CONTEXT_LENGTH": 2, "CONTEXT_PARTITION_SIZE": 2,
          "SLIDING_WINDOW": 0, "TRANS_V": True, "UNIFORM_RANGE": (-1, 1),
          "setup_seed": lambda *args: None, "random": __import__("random"),
          "create_kv_cache": lambda *args: ([cache.clone()], [cache.clone()]),
          "quantize_kv_cache_symmetric": lambda k,v,**kwargs: (k, scale, v, scale, scale, scale),
          "torch_mha_extend": lambda q,*args,**kwargs: q + 1,
          "shuffle_value_cache_layout": lambda x: x.clone(),
          "build_ps_page_data": lambda *args: (torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)),
          "require_tensor_contract": checks.require_tensor_contract, "require_unchanged": checks.require_unchanged}
    _harness_functions(task, {"_build_case", "_validate_pa_contract"}, ns)
    launch, out, ref = ns["_build_case"](mod, (1, 1), 1, 1, "per_token")
    assert all(t is not scratch for t in launch.arena_inputs)
    if mutate:
        with pytest.raises(AssertionError, match="read-only"):
            ns["_validate_pa_contract"](launch, out, ref)
    else:
        ns["_validate_pa_contract"](launch, out, ref)
        before = launch.arena_reference().clone()
        launch.arena_perturb()
        assert not torch.equal(before, launch.arena_reference())
        assert launch() is out


@pytest.mark.parametrize("name", ["softmax", "layernorm"])
@pytest.mark.parametrize("bad_phase", [None, "measured", "replay", "input_modified"])
def test_normalization_benchmark_validates_measured_and_replayed_values(name, bad_phase, monkeypatch, tmp_path):
    import math
    import types
    import torch
    import torch.nn.functional as F
    task = ROOT / "tasks/triton2flydsl/aiter" / name
    checks = module(task / "scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    x = torch.tensor([[1., 2., 4.], [-2., 3., 1.]], dtype=torch.bfloat16)
    weight, bias = torch.ones(3, dtype=x.dtype), torch.ones(3, dtype=x.dtype)
    originals = tuple(t.clone() for t in (x, weight, bias))
    def reference():
        return torch.softmax(x, 1) if name == "softmax" else F.layer_norm(x, (3,), weight, bias, 1e-5)
    phase = {"name": "setup"}
    def compute(*args):
        if bad_phase == "input_modified" and phase["name"] == "measured":
            x.add_(1)
        return torch.zeros_like(x) if phase["name"] == bad_phase else reference()
    seen = []
    def benchmark(fn, warmup, repetition, timed_run):
        seen.append((warmup, repetition))
        phase["name"] = "measured"
        timed_run.outputs = fn()
        timed_run.bound = True
        def replay():
            phase["name"] = "replay"
            timed_run.outputs.copy_(fn())
            return timed_run.outputs
        timed_run.rerun = replay
        return .1, {"benchmark_method": "cuda_graph"}
    ns = {"math": math, "json": json, "Path": Path, "_HERE": str(tmp_path),
          "EPS": 1e-5, "WARMUP": 10, "ITERS": 100,
          "TEST_SHAPES": [{"name": "controlled", "M": 2, "N": 3}],
          "_torch_dtype": lambda dt: torch.bfloat16,
          "_make_inputs": lambda *args: x if name == "softmax" else (x, weight, bias),
          "_load_source": lambda: types.SimpleNamespace(softmax=compute, layer_norm=compute),
          "TimedRun": types.SimpleNamespace, "benchmark_cuda_graph_or_events": benchmark,
          "verify_timed_run": checks.verify_timed_run, "allclose_output": checks.allclose_output}
    _harness_functions(task, {"run_benchmark", "_reference_softmax", "_reference_layernorm"}, ns)
    if bad_phase is None:
        records = ns["run_benchmark"](verbose=False)
        assert records[0]["timed_output_correctness"] == "PASS"
        assert records[0]["replay_correctness"] == "PASS"
        assert seen == [(0, 100)]
        checks.require_unchanged((x, weight, bias), originals)
    else:
        with pytest.raises(AssertionError):
            ns["run_benchmark"](verbose=False)


@pytest.mark.parametrize("name", ["softmax", "layernorm"])
@pytest.mark.parametrize("behavior", ["correct", "input_modified", "wrong_shape", "wrong_dtype"])
def test_normalization_correctness_enforces_input_and_output_contract(name, behavior, monkeypatch):
    import types
    import torch
    import torch.nn.functional as F
    task = ROOT / "tasks/triton2flydsl/aiter" / name
    checks = module(task / "scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    x = torch.tensor([[1., 2., 4.], [-2., 3., 1.]], dtype=torch.bfloat16)
    weight, bias = torch.ones(3, dtype=x.dtype), torch.ones(3, dtype=x.dtype)
    def compute(*args):
        if behavior == "input_modified":
            x.add_(1)
        result = torch.softmax(x, 1) if name == "softmax" else F.layer_norm(x, (3,), weight, bias, 1e-5)
        if behavior == "wrong_shape":
            return result.unsqueeze(0)
        return result.float() if behavior == "wrong_dtype" else result
    ns = {"EPS": 1e-5, "DTYPES": ["bf16"],
          "TEST_SHAPES": [{"name": "controlled", "M": 2, "N": 3}],
          "_torch_dtype": lambda dt: torch.bfloat16,
          "_make_inputs": lambda *args: x if name == "softmax" else (x, weight, bias),
          "_load_source": lambda: types.SimpleNamespace(softmax=compute, layer_norm=compute),
          "require_tensor_contract": checks.require_tensor_contract, "require_unchanged": checks.require_unchanged}
    _harness_functions(task, {"run_correctness", "_reference_softmax", "_reference_layernorm"}, ns)
    assert ns["run_correctness"](verbose=False) is (behavior == "correct")


@pytest.mark.parametrize("name,provided", [("batched_gemm_bf16_kernel", False), ("batched_gemm_bf16_kernel", True), ("hgemm_kernel", False)])
@pytest.mark.parametrize("function", ["run_benchmark", "arena_benchmark"])
@pytest.mark.parametrize("behavior", ["correct", "measured_wrong", "replay_wrong", "cached", "input_modified", "shape", "dtype", "nonfinite"])
def test_torch_gemm_measured_output_and_eager_reinvocation(name, provided, function, behavior, monkeypatch, tmp_path):
    import math
    import types
    import torch
    task = ROOT / "tasks/torch2flydsl" / name
    checks = module(task / "scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setitem(sys.modules, "aiter", types.SimpleNamespace())
    a = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    b = torch.tensor([[2., 4.], [6., 8.]], dtype=torch.bfloat16)
    batched = name.startswith("batched")
    if batched:a, b = a.unsqueeze(0), b.unsqueeze(0)
    original = (a.clone(), b.clone())
    phase = {"value": "setup"}
    cached = (a.float() @ b.float().transpose(-1, -2)).to(a.dtype)
    def compute(*args, **kwargs):
        if behavior == "input_modified" and phase["value"] == "replay":a.add_(1)
        out = (a.float() @ b.float().transpose(-1, -2)).to(a.dtype)
        if behavior == phase["value"] + "_wrong":out.fill_(a.flatten()[0].item())
        if behavior == "cached" and phase["value"] == "replay":out = cached.clone()
        if phase["value"] == "measured":
            if behavior == "shape":out = out[..., :1]
            if behavior == "dtype":out = out.float()
            if behavior == "nonfinite":out.flatten()[0] = float("nan")
        return out
    calls = []
    class Collector:
        bound = False
        outputs = None
    def benchmark(fn, *, warmup, repetition, use_cuda_graph, fallback_reason, timed_run=None):
        calls.append((warmup, repetition, use_cuda_graph, timed_run is not None))
        phase["value"] = "measured"
        out = fn()
        if timed_run is not None:
            timed_run.outputs = out
            timed_run.bound = True
            def rerun():
                phase["value"] = "replay"
                result = fn()  # eager Event path returns a fresh allocation
                assert result is not out
                return result
            timed_run.rerun = rerun
        phase["value"] = "setup"
        return .1, {"benchmark_method": "cuda_graph" if use_cuda_graph else "cuda_event_fallback"}
    kmod = types.SimpleNamespace(flydsl_batched_gemm_bf16=compute, flydsl_hgemm=compute)
    monkeypatch.setitem(sys.modules, "aiter", types.SimpleNamespace(batched_gemm_bf16_CK=compute))
    ns = {"TimedRun": Collector, "benchmark_cuda_graph_or_events": benchmark,
          "verify_timed_run": checks.verify_timed_run, "require_tensor_contract": checks.require_tensor_contract,
          "math": math, "json": json, "Path": Path, "_KERNEL_DIR": str(tmp_path),
          "KERNEL_FILE": "kernel.py", "MODEL_FILE": "model.py", "KERNEL_ENTRY": "flydsl_batched_gemm_bf16",
          "SHAPES": [{"name": "controlled", "b": 1, "m": 2, "n": 2, "k": 2}],
          "TOL": .01, "ATOL": .01, "RTOL": .01, "PASS_PCT": 99.9, "TILING_KEYS": (),
          "_make_inputs": lambda *args: (a,b), "_load_module": lambda directory,filename,alias: None if provided and filename == "kernel.py" else kmod,
          "_retry": lambda fn, **kwargs: fn()}
    _harness_functions(task, {function, "_norm_worst", "_checked_gemm_output", "_gemm_reference", "_compare_gemm_output"}, ns)
    if behavior == "correct":
        result = ns[function](verbose=False)
        if function == "run_benchmark":result = json.loads((tmp_path/"build/performance_report.json").read_text())
        assert result[0]["timed_output_correctness"] == result[0]["replay_correctness"] == "PASS"
        assert calls == [(0,100,False,True),(0,100,False,False)]
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(a, original[0]) and torch.equal(b, original[1])


@pytest.mark.parametrize("name", ["batched_gemm_bf16_kernel", "hgemm_kernel"])
@pytest.mark.parametrize("behavior", ["correct", "shape", "dtype", "device", "nonfinite", "input_modified"])
def test_torch_gemm_correctness_rejects_invalid_contracts(name, behavior, monkeypatch):
    import types
    import torch
    task = ROOT / "tasks/torch2flydsl" / name
    checks = module(task / "scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    a = torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)
    b = a.clone()
    if name.startswith("batched"):a,b=a.unsqueeze(0),b.unsqueeze(0)
    reference = lambda *args: (a.float() @ b.float().transpose(-1,-2)).to(a.dtype)
    class Model:
        def to(self,*args):return self
        def eval(self):return self
        def __call__(self,*args):return reference()
    def compute(*args,**kwargs):
        out = reference()
        if behavior == "shape":out=out[..., :1]
        if behavior == "dtype":out=out.float()
        if behavior == "device":out=out.to("meta")
        if behavior == "nonfinite":out.flatten()[0]=float("nan")
        if behavior == "input_modified":a.add_(1)
        return out
    monkeypatch.setitem(sys.modules,"aiter",types.SimpleNamespace(batched_gemm_bf16_CK=reference))
    mmod=types.SimpleNamespace(Model=Model,get_init_inputs=lambda:[])
    kmod=types.SimpleNamespace(flydsl_batched_gemm_bf16=compute,flydsl_hgemm=compute)
    ns={"require_tensor_contract":checks.require_tensor_contract,"require_unchanged":checks.require_unchanged,
        "_KERNEL_DIR":".","KERNEL_FILE":"kernel.py","MODEL_FILE":"model.py","KERNEL_ENTRY":"flydsl_batched_gemm_bf16",
        "SHAPES":[{"name":"controlled","b":1,"m":2,"n":2,"k":2}],
        "TOL":.01,"ATOL":.01,"RTOL":.01,"PASS_PCT":99.9,"TILING_KEYS":(),
        "_make_inputs":lambda *args:(a,b),"_retry":lambda fn,**kwargs:fn(),
        "_load_module":lambda directory,filename,alias:mmod if filename=="model.py" else kmod}
    _harness_functions(task,{"run_correctness","_norm_worst","_checked_gemm_output"},ns)
    if behavior=="correct":assert ns["run_correctness"](verbose=False) is True
    else:
        with pytest.raises(AssertionError,match="correctness FAILED"):ns["run_correctness"](verbose=False)


def test_torch_gemm_replay_numerical_rules_keep_zero_scale_and_percent_boundaries():
    import torch
    for name in ("batched_gemm_bf16_kernel","hgemm_kernel"):
        task=ROOT/"tasks/torch2flydsl"/name
        checks=module(task/"scripts/replay_checks.py")
        ns={"require_tensor_contract":checks.require_tensor_contract,"TOL":.01,"ATOL":.01,"RTOL":.01,"PASS_PCT":99.9}
        _harness_functions(task,{"_norm_worst","_checked_gemm_output","_compare_gemm_output"},ns)
        compare=ns["_compare_gemm_output"]
        if name.startswith("batched"):
            expected=torch.zeros((2,2),dtype=torch.bfloat16)
            compare(torch.full_like(expected,.009),expected)
            with pytest.raises(AssertionError,match="Numerical mismatch"):compare(torch.full_like(expected,.011),expected)
        else:
            expected=torch.ones(10000,dtype=torch.bfloat16)
            actual=expected.clone();actual[:9]=100
            compare(actual,expected)
            actual[:11]=100
            with pytest.raises(AssertionError,match="Numerical mismatch"):compare(actual,expected)
            actual=expected.clone();actual[0]=float("nan")
            with pytest.raises(AssertionError,match="Non-finite"):compare(actual,expected)


_TORCH_GEMM_ORIGINAL_BENCHMARKS = {('batched_gemm_bf16_kernel', 'run_benchmark'): 'c90d460059918323dbece546e3f557219b2e9384260a6661383d6aaec3b26b82', ('batched_gemm_bf16_kernel', 'arena_benchmark'): '06e9e400f31817921ef212bd186f685bbf94420a7e0c2f946af5b30f2307f254', ('hgemm_kernel', 'run_benchmark'): '472f2bde0f1462d720bf12509600c1b4fc10fae4580c5c5828695ca4a8afbf4a', ('hgemm_kernel', 'arena_benchmark'): '9094d767825dd1f9d77aa7a9d54cb4175626165b185a666fc7a49fb297afaa4a'}


def test_torch_gemm_original_benchmark_work_and_sampling_preserved():
    for (name, function), expected_hash in _TORCH_GEMM_ORIGINAL_BENCHMARKS.items():
        tree = ast.parse((ROOT / "tasks/torch2flydsl" / name / "test_kernel_harness.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        fn = _RemoveAddedReplayChecks().visit(fn)
        if name == "batched_gemm_bf16_kernel":
            from test_gemm_paired_timing import normalize_former_role_policy
            fn = normalize_former_role_policy(fn)
        assert hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() == expected_hash


_ACTIVATION_REPLAY_TASKS = ["gelu_fast", "gelu_and_mul", "gelu_tanh_and_mul", "swiglu_and_mul"]


@pytest.mark.parametrize("name", _ACTIVATION_REPLAY_TASKS)
@pytest.mark.parametrize("function", ["run_benchmark", "arena_benchmark"])
@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("behavior", ["correct", "measured_wrong", "replay_wrong", "cached", "input_modified", "shape", "dtype", "nonfinite"])
def test_activation_actual_measured_and_replayed_invocations(name, function, provided, behavior, monkeypatch, tmp_path):
    """Run real harness orchestration with CPU outputs; this is not GPU timing."""
    import math
    import types
    import torch
    task = ROOT / "tasks/torch2flydsl" / (name + "_kernel")
    checks = module(task / "scripts/replay_checks.py")
    actual_model = module(task / "model.py").Model()
    inp = torch.tensor([[-2., 10., -9., 4.], [1., 2., 3., 4.]], dtype=torch.bfloat16)
    original = inp.clone()
    phase = {"name": "setup"}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def oracle(value):
        if name == "swiglu_and_mul":
            gate, linear = value.float().chunk(2, -1)
            gate = torch.minimum(gate, torch.tensor(7.))
            linear = torch.maximum(torch.minimum(linear, torch.tensor(7.)), torch.tensor(-7.))
            return (gate / (1 + torch.exp(-1.702 * gate)) * (linear + 1)).to(value.dtype)
        x = value.float() if name == "gelu_fast" else value.float().chunk(2, -1)[0]
        if name == "gelu_and_mul":
            out = .5 * x * (1 + torch.erf(x / math.sqrt(2)))
        else:
            out = .5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + .044715 * x ** 3)))
        if name != "gelu_fast": out *= value.float().chunk(2, -1)[1]
        return out.to(value.dtype)

    cached = actual_model(inp)
    def compute(value, is_model):
        out = actual_model(value)
        if is_model == provided:
            if behavior == phase["name"] + "_wrong": out.fill_(20.)
            if phase["name"] == "replay":
                if behavior == "cached": out = cached.clone()
                if behavior == "input_modified": value.add_(1)
            if phase["name"] == "measured":
                if behavior == "shape": out = out[:1]
                if behavior == "dtype": out = out.float()
                if behavior == "nonfinite": out[0, 0] = float("nan")
        return out
    class Model:
        def __call__(self, value): return compute(value, True)
    mmod = types.SimpleNamespace(Model=Model, get_init_inputs=lambda: [])
    kmod = types.SimpleNamespace(**{"flydsl_" + name: lambda value: compute(value, False)})
    class Collector:
        bound = False
        outputs = None
    calls = []
    def benchmark(fn, *, warmup, repetition, timed_run):
        calls.append((warmup, repetition, timed_run is not None))
        phase["name"] = "measured"
        out = fn()
        if timed_run is not None:
            timed_run.outputs = out
            timed_run.bound = True
            def replay():
                phase["name"] = "replay"
                result = fn()
                phase["name"] = "setup"
                return result
            timed_run.rerun = replay
        phase["name"] = "setup"
        return .1, {"benchmark_method": "cuda_graph", "benchmark_timed_run_kind": "captured_graph"}
    ns = {"TimedRun": Collector, "benchmark_cuda_graph_or_events": benchmark,
          "normalized_output": checks.normalized_output, "require_tensor_contract": checks.require_tensor_contract,
          "require_unchanged": checks.require_unchanged, "verify_timed_run": checks.verify_timed_run,
          "_KERNEL_DIR": str(tmp_path), "MODEL_FILE": "model.py", "KERNEL_FILE": "kernel.py",
          "KERNEL_ENTRY": "flydsl_" + name, "REL_TOL": .01, "_aiter_op": oracle,
          "_load_module": lambda directory, filename, alias: mmod if filename == "model.py" else (None if provided else kmod),
          "SHAPES": [{"name": "controlled", "m": 2, "n": 4}], "_make_inputs": lambda *args: inp,
          "math": math, "json": json, "Path": Path}
    if name == "swiglu_and_mul":
        ns["saturation_input_"] = module(task / "scripts/swiglu_controls.py").saturation_input_
    _harness_functions(task, {function, "_mean_ms", "_activation_replay_validator", "_checked_activation_output"}, ns)
    if behavior == "correct":
        report = ns[function](verbose=False)
        if function == "run_benchmark": report = json.loads((tmp_path / "build/performance_report.json").read_text())
        assert report[0]["timed_output_correctness"] == report[0]["replay_correctness"] == "PASS"
        assert calls == [(10, 100, False), (10, 100, provided)] + ([] if provided else [(10, 100, True)])
    else:
        with pytest.raises(AssertionError): ns[function](verbose=False)
    assert torch.equal(inp, original)


@pytest.mark.parametrize("name", _ACTIVATION_REPLAY_TASKS)
@pytest.mark.parametrize("behavior", ["correct", "shape", "dtype", "device", "nonfinite", "input_modified"])
def test_activation_correctness_retains_output_and_input_contract(name, behavior, monkeypatch):
    import types
    import torch
    task = ROOT / "tasks/torch2flydsl" / (name + "_kernel")
    model = module(task / "model.py").Model()
    checks = module(task / "scripts/replay_checks.py")
    inp = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.]], dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    def compute(value):
        out = model(value)
        if behavior == "shape": out = out[:1]
        if behavior == "dtype": out = out.float()
        if behavior == "device": out = out.to("meta")
        if behavior == "nonfinite": out[0, 0] = float("nan")
        if behavior == "input_modified": value.add_(1)
        return out
    mmod = types.SimpleNamespace(Model=lambda: model, get_init_inputs=lambda: [])
    kmod = types.SimpleNamespace(**{"flydsl_" + name: compute})
    ns = {"require_tensor_contract": checks.require_tensor_contract, "require_unchanged": checks.require_unchanged,
          "_KERNEL_DIR": ".", "MODEL_FILE": "model.py", "KERNEL_FILE": "kernel.py", "KERNEL_ENTRY": "flydsl_" + name,
          "_load_module": lambda directory, filename, alias: mmod if filename == "model.py" else kmod,
          "SHAPES": [{"name": "controlled", "m": 2, "n": 4}], "_make_inputs": lambda *args: inp,
          "_aiter_op": model, "_retry": lambda fn, **kwargs: fn(), "REL_TOL": .01}
    _harness_functions(task, {"run_correctness", "_checked_activation_output"}, ns)
    if behavior == "correct": assert ns["run_correctness"](verbose=False)
    else:
        with pytest.raises(AssertionError): ns["run_correctness"](verbose=False)


class _RemoveActivationReplayChecks(_RemoveAddedReplayChecks):
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"original", "ref_validate", "ker_validate"}:
                return None
        return super().visit_Assign(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "_checked_activation_output":
            return self.visit(node.args[0])
        if isinstance(node.func, ast.Name) and node.func.id == "_mean_ms":
            node.keywords = [k for k in node.keywords if k.arg != "validate"]
        return super().visit_Call(node)


_ACTIVATION_ORIGINAL_BENCHMARKS = {('gelu_fast', 'run_benchmark'): '9c380844988351c657b3be9ac29948bbd4825f8f8146dd118b99279522a29f6e', ('gelu_fast', 'arena_benchmark'): '8af86be049fad9ef00f0a01e2e5635a5589f413e75a34cb243e38b70a63df54a', ('gelu_and_mul', 'run_benchmark'): 'cfffddd747d47293fc3de6f7e16f3b9575613cba5132650e8e072228fc92b28f', ('gelu_and_mul', 'arena_benchmark'): 'a90c92fa4f4d7adfb6e55be9c60e0d4adabaf1e2f9e4b3617315e33b00495675', ('gelu_tanh_and_mul', 'run_benchmark'): '26d26e61238601cc73c39909701917635a7f897e87cf4aacd65f175a60526b73', ('gelu_tanh_and_mul', 'arena_benchmark'): 'b9264e16963ec7cb727939225b47f18c36d219bc0f3d296a933fb9c866fe8959', ('swiglu_and_mul', 'run_benchmark'): '74280c1ee802db9d83bf2cc16df46965207e00abc77a59039b3da58f699a8d43', ('swiglu_and_mul', 'arena_benchmark'): 'e9e502e4752c6f7cc261231c5f63065e435f3d2a7555921f7bc9454de6357434'}


def test_activation_original_timed_work_sampling_and_reference_boundaries_preserved():
    for (name, function), expected in _ACTIVATION_ORIGINAL_BENCHMARKS.items():
        tree = ast.parse((ROOT / "tasks/torch2flydsl" / (name + "_kernel") / "test_kernel_harness.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        fn = _RemoveActivationReplayChecks().visit(fn)
        assert hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() == expected


def test_preshuffle_preload_policy_extraction_preserves_every_entry_and_default():
    task = ROOT / "tasks/flydsl2flydsl/preshuffle_gemm_v2_kernel"
    source = (task / "kernels/preshuffle_gemm.py").read_text()
    wanted = {"_TILE_PRELOAD_TABLE", "_TILE_PRELOAD_DEFAULT", "_get_preload"}
    nodes = [n for n in ast.parse(source).body if
             isinstance(n, ast.FunctionDef) and n.name in wanted or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)]
    original = {}
    exec(compile(ast.Module(nodes, type_ignores=[]), "original_preload_policy", "exec"), original)
    extracted = module(task / "kernels/preload.py")
    for key, value in original["_TILE_PRELOAD_TABLE"].items():
        assert extracted._get_preload(*key) == value
        assert extracted._get_preload(*(str(x) for x in key)) == value
    for key in [(1, 2, 3), (999, 1, 64), (-1, 0, 0)]:
        assert extracted._get_preload(*key) == original["_get_preload"](*key)


def test_preshuffle_vector_api_port_preserves_compilation_algorithm():
    task = ROOT / "tasks/flydsl2flydsl/preshuffle_gemm_v2_kernel"
    tree = ast.parse((task / "kernel.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compile_preshuffle_gemm_v2")
    # The typed vector assembly is the only executable change. Keep all layout,
    # scaling order, MFMA choice, tile selection and launch geometry unchanged.
    converted = []
    class OriginalAssembly(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "Vec" and node.func.attr == "from_elements":
                converted.append(node)
                return ast.parse("vector.from_elements(T.vec(acc_size, out_elem_cls.ir_type), scaled_elems)", mode="eval").body
            return self.generic_visit(node)
    fn = OriginalAssembly().visit(fn)
    assert len(converted) == 1
    assert hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() == "4ffbe6d2fb5813bf4c4e6c663556e752f1ddb9319d6179e16657b8f95791092f"


@pytest.mark.parametrize("function", ["run_benchmark", "arena_benchmark"])
@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("behavior", ["correct", "measured_wrong", "replay_wrong", "cached", "input_modified", "dtype", "shape", "nonfinite"])
def test_rmsnorm_primary_baseline_and_candidate_timed_controls(function, provided, behavior, monkeypatch, tmp_path):
    import math
    import types
    import torch
    task=ROOT/"tasks/torch2flydsl/rmsnorm2d_kernel"
    checks=module(task/"scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    inp=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)
    weight=torch.tensor([2.,3.],dtype=torch.bfloat16)
    originals=(inp.clone(),weight.clone())
    phase={"value":"setup"}
    def oracle():
        x=inp.float();return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*weight.float()).to(inp.dtype)
    cached=oracle()
    def compute(is_model):
        target_under_test = is_model == provided
        out=oracle()
        if target_under_test:
            if behavior==phase["value"]+"_wrong":out.fill_(2.)
            if phase["value"]=="replay":
                if behavior=="cached":out=cached.clone()
                if behavior=="input_modified":inp.add_(1)
            if phase["value"]=="measured":
                if behavior=="dtype":out=out.float()
                if behavior=="shape":out=out[:1]
                if behavior=="nonfinite":out.flatten()[0]=float("inf")
        return out
    class Model:
        def __init__(self,*args):pass
        def to(self,*args):return self
        def eval(self):return self
        def __call__(self,*args):return compute(True)
    target=lambda *args:compute(False)
    monkeypatch.setitem(sys.modules,"aiter",types.SimpleNamespace(rms_norm=lambda *args:oracle()))
    class Collector:
        bound=False
        outputs=None
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((fn.__name__,warmup,repetition))
        phase["value"]="measured";out=fn();timed_run.outputs=out;timed_run.bound=True
        def replay():
            phase["value"]="replay";result=fn();phase["value"]="setup";return result
        timed_run.rerun=replay;phase["value"]="setup"
        return .1,{"benchmark_method":"cuda_graph","benchmark_timed_run_kind":"captured_graph"}
    ns={"TimedRun":Collector,"benchmark_cuda_graph_or_events":benchmark,
        "verify_timed_run":checks.verify_timed_run,"require_tensor_contract":checks.require_tensor_contract,
        "math":math,"json":json,"Path":Path,"_KERNEL_DIR":str(tmp_path),"MODEL_FILE":"model.py",
        "KERNEL_ENTRY":"flydsl_rmsnorm2d","REL_TOL":.01,"EPS":1e-5,
        "SHAPES":[{"name":"controlled","m":2,"n":2}],"_make_inputs":lambda *args:(inp,weight),
        "_load_module":lambda *args:types.SimpleNamespace(Model=Model),"_load_target":lambda:target,
        "_is_pure_starter":lambda:provided,"_probe_target":lambda *args:(False,None) if provided else (True,target()),
        "_retry":lambda fn,**kwargs:fn()}
    _harness_functions(task,{function,"_norm_max_err","_checked_rms_output","_compare_rms_output"},ns)
    if behavior=="correct":
        result=ns[function](verbose=False)
        if function=="run_benchmark":result=json.loads((tmp_path/"build/performance_report.json").read_text())
        assert result[0]["timed_output_correctness"]==result[0]["replay_correctness"]=="PASS"
        assert calls==[("run_ref",10,100),("run_truth",10,100)]+([] if provided else [("run_target",10,100)])
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(inp,originals[0]) and torch.equal(weight,originals[1])


@pytest.mark.parametrize("bad", ["shape", "dtype", "device", "nonfinite"])
def test_rmsnorm_output_contract_and_original_normalized_gate(bad):
    import torch
    task=ROOT/"tasks/torch2flydsl/rmsnorm2d_kernel"
    checks=module(task/"scripts/replay_checks.py")
    ns={"require_tensor_contract":checks.require_tensor_contract,"REL_TOL":.01}
    _harness_functions(task,{"_norm_max_err","_checked_rms_output","_compare_rms_output"},ns)
    expected=torch.tensor([[100.,1.],[2.,3.]],dtype=torch.bfloat16)
    # Original global max normalization allows .5 absolute error near a small element.
    actual=expected.clone();actual[0,1]+=.5
    ns["_compare_rms_output"](actual,expected)
    actual[0,1]+=2.
    with pytest.raises(AssertionError,match="Numerical mismatch"):ns["_compare_rms_output"](actual,expected)
    actual=expected.clone()
    if bad=="shape":actual=actual[:1]
    if bad=="dtype":actual=actual.float()
    if bad=="device":actual=actual.to("meta")
    if bad=="nonfinite":actual[0,0]=float("nan")
    with pytest.raises(AssertionError):ns["_compare_rms_output"](actual,expected)


@pytest.mark.parametrize("function", ["run_benchmark", "arena_benchmark"])
@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("behavior", ["correct", "measured_wrong", "replay_wrong", "cached", "input_modified", "bad_ids", "nonfinite"])
@pytest.mark.parametrize("routing", ["softmax", "sigmoid", "softplus"])
def test_moe_routing_benchmark_checks_both_real_timed_outputs(routing, function, provided, behavior, monkeypatch, tmp_path):
    import math
    import types
    import torch
    task=ROOT/"tasks/torch2flydsl"/("moe_topk_"+routing+"_kernel")
    checks=module(task/"scripts/replay_checks.py")
    mmod=module(task/"model.py")
    model = {"softmax": lambda: mmod.Model(6,2,1.,False),
             "sigmoid": lambda: mmod.Model(6,2),
             "softplus": lambda: mmod.Model(6,2,True,2.5)}[routing]()
    if routing == "softplus":
        with torch.no_grad(): model.correction_bias.copy_(torch.tensor([.05,.1,0.,.15,.2,.25]))
    original_bias = model.correction_bias.detach().clone() if routing == "softplus" else None
    gating=torch.tensor([[1.,3.,2.,5.,6.,4.],[6.,2.,4.,3.,1.,5.]],dtype=torch.bfloat16)
    original=gating.clone();cached=model(gating)
    phase={"value":"setup"}
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    def compute():
        w,ids=model(gating)
        if behavior==phase["value"]+"_wrong":w=w+1.
        if phase["value"]=="replay":
            if behavior=="cached":w,ids=(x.clone() for x in cached)
            if behavior=="input_modified":gating.add_(1)
        if phase["value"]=="measured":
            if behavior=="bad_ids":ids[:,0]=-1
            if behavior=="nonfinite":w[:,0]=float("nan")
        return w,ids
    def aiter_op(w,ids,*args,**kwargs):
        actual_w,actual_ids=compute();w.copy_(actual_w);ids.copy_(actual_ids)
    monkeypatch.setitem(sys.modules,"aiter",types.SimpleNamespace(topk_gating=aiter_op,topk_softplus=aiter_op))
    kmod=types.SimpleNamespace(**{"flydsl_topk_"+routing:lambda *args:compute()})
    class Collector:
        bound=False
        outputs=None
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run=None):
        calls.append((warmup,repetition,timed_run is not None))
        phase["value"]="measured";out=fn()
        if timed_run is not None:
            timed_run.outputs=out;timed_run.bound=True
            def replay():
                phase["value"]="replay";result=fn();phase["value"]="setup";return result
            timed_run.rerun=replay
        phase["value"]="setup"
        return .1,{"benchmark_method":"cuda_graph","benchmark_timed_run_kind":"captured_graph"}
    ns={"TimedRun":Collector,"benchmark_cuda_graph_or_events":benchmark,"require_unchanged":checks.require_unchanged,
        "math":math,"json":json,"Path":Path,"_KERNEL_DIR":str(tmp_path),"MODEL_FILE":"model.py","KERNEL_FILE":"kernel.py",
        "_TIE_TOL":1e-4,"_WEIGHT_ATOL":1e-2,"_BIAS_ID_ERR_TOL":.05,"REL_TOL":1e-2,
        "SHAPES":[{"name":"controlled","tokens":2,"experts":6,"topk":2,"route_scale":2.5 if routing=="softplus" else 1.,"use_bias":False,"renormalize":True}],
        "_load_module":lambda directory,filename,alias:mmod if filename=="model.py" else (None if provided else kmod),
        "_build_model":lambda *args:(model,gating),"_retry":lambda fn,**kwargs:fn()}
    _harness_functions(task,{function,"_require_routing_contract","_routing_reference","_verify_routing_timed","_compare_routing"},ns)
    if behavior=="correct":
        result=ns[function](verbose=False)
        if function=="run_benchmark":result=json.loads((tmp_path/"build/performance_report.json").read_text())
        assert result[0]["timed_output_correctness"]==result[0]["replay_correctness"]=="PASS"
        assert result[0]["replay_checked_outputs"]==["weights","expert_ids"]
        assert calls==[(0,100,True),(0,100,False)]
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(gating,original)

    if original_bias is not None: assert torch.equal(model.correction_bias, original_bias)


@pytest.mark.parametrize("bad", ["shape", "weight_dtype", "id_dtype", "device", "out_of_range", "duplicate", "nonfinite"])
@pytest.mark.parametrize("routing", ["softmax", "sigmoid", "softplus"])
def test_moe_routing_contract_rejects_invalid_outputs_even_with_bias_allowance(routing, bad):
    import torch
    task=ROOT/"tasks/torch2flydsl"/("moe_topk_"+routing+"_kernel")
    ns={}
    _harness_functions(task,{"_require_routing_contract"},ns)
    gating=torch.zeros(2,4,dtype=torch.bfloat16)
    w=torch.ones(2,2,dtype=torch.float32);ids=torch.tensor([[1,2],[2,3]],dtype=torch.int32)
    ns["_require_routing_contract"](w,ids,gating,2)
    if bad=="shape":w=w[:1]
    if bad=="weight_dtype":w=w.bfloat16()
    if bad=="id_dtype":ids=ids.long()
    if bad=="device":w=w.to("meta")
    if bad=="out_of_range":ids[0,0]=4
    if bad=="duplicate":ids[0,0]=ids[0,1]
    if bad=="nonfinite":w[0,0]=float("nan")
    with pytest.raises(AssertionError):ns["_require_routing_contract"](w,ids,gating,2)


_MOE_ROUTING_ORIGINAL_PROTECTED_FUNCTIONS = {'_compare_routing': 'ae7b784aee35b1eed3868157123f5b9dd5be69c15eb3b489f7d6996d9b8edd14', 'run_benchmark': '833ba05fe35dc7b25512a7fbbf74d34955c40a63a63d385dffab17ec158dad06', 'arena_benchmark': 'a096d9bad6085aaeee32e01f2b49bebdcec7cd7f8d7c3fe36bcf317b0949d858'}


def test_moe_routing_original_tie_weight_policy_and_benchmark_work_preserved():
    tree = ast.parse((ROOT / "tasks/torch2flydsl/moe_topk_softmax_kernel/test_kernel_harness.py").read_text())
    for function, expected_hash in _MOE_ROUTING_ORIGINAL_PROTECTED_FUNCTIONS.items():
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        fn = _RemoveAddedReplayChecks().visit(fn)
        assert hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() == expected_hash


_OTHER_MOE_ROUTING_ORIGINAL_FUNCTIONS = {('sigmoid', '_compare_routing'): '633545e95be340bac4c9750e9a2be314b79d118723b5e233f46e1f821331a5af', ('sigmoid', 'run_benchmark'): 'c72855278a7ddc458fe3cd89290d2236efae0db51812659e2724e80b3720a5ab', ('sigmoid', 'arena_benchmark'): 'ac343e5c79731541b946538b2de1e74ada082f41952400f7bcc992b374d3d2eb', ('softplus', '_compare_routing'): '633545e95be340bac4c9750e9a2be314b79d118723b5e233f46e1f821331a5af', ('softplus', 'run_benchmark'): 'c73426d4d99892d1e90f6f88f9098166342365b5966aca1b424974001e865ff5', ('softplus', 'arena_benchmark'): '260c5a769350229dca4f07831c8cb71a6c5b34957c649e699f738c7aa04891a4'}


def test_other_moe_original_tie_weight_policy_and_benchmark_work_preserved():
    for (routing, function), expected in _OTHER_MOE_ROUTING_ORIGINAL_FUNCTIONS.items():
        task = ROOT / "tasks/torch2flydsl" / ("moe_topk_" + routing + "_kernel")
        fn = next(n for n in ast.parse((task / "test_kernel_harness.py").read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == function)
        fn = _RemoveAddedReplayChecks().visit(fn)
        assert hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() == expected


@pytest.mark.parametrize("name", ["decode_attention", "sglang_fused_moe"])
@pytest.mark.parametrize("behavior", ["correct", "measured_wrong", "replay_wrong", "cached", "input_modified", "nonfinite"])
def test_sglang_five_each_actual_performance_observes_output(name, behavior, monkeypatch):
    import types
    import torch
    task = ROOT / "tasks/triton2flydsl/sglang" / name
    checks = module(task / "scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    phase = {"name": "setup"}
    ns = {"load_module": lambda: object(), "TimedRun": types.SimpleNamespace,
          "require_tensor_contract": checks.require_tensor_contract,
          "require_unchanged": checks.require_unchanged, "verify_timed_run": checks.verify_timed_run,
          "_retry_oom": lambda fn: fn(), "WARMUP_ITERATIONS": 10, "BENCHMARK_ITERATIONS": 100,
          "DTYPE_NAME": "bfloat16", "MAX_KV_SPLITS": 1}
    _harness_functions(task, {"run_performance", "reference", "reference_moe", "_shape_of",
                              "_compare_decode_output", "_compare_prepared_moe_output"}, ns)
    if name == "decode_attention":
        cfg = {"seqs": [2, 1], "head": 1, "kv_head": 1, "Lk": 2, "Lv": 2}
        q = torch.tensor([[[1., 2.]], [[3., 4.]]], dtype=torch.bfloat16)
        k = torch.tensor([[[1., 0.]], [[0., 1.]], [[1., 1.]]], dtype=torch.bfloat16)
        v = k + 1
        o = torch.empty_like(q)
        kvp = torch.tensor([0, 2, 3], dtype=torch.int32)
        kvi = torch.arange(3, dtype=torch.int64)
        nks = torch.ones(2, dtype=torch.int32)
        al = torch.empty(2, 1, 1, 2)
        alse = torch.empty(2, 1, 1)
        inputs = (q, k, v, kvp, kvi, nks)
        ns.update(TEST_SHAPES=[cfg], make_inputs=lambda *args: (q, k, v, o, kvp, kvi, al, alse, nks))
        oracle = lambda: ns["reference"](q, k, v, kvi, cfg)
        def launch(*args): o.copy_(compute())
        ns["load_module"] = lambda: types.SimpleNamespace(decode_attention_fwd=launch)
    else:
        inp = {"M": 2, "K": 2, "I": 2, "E": 2, "topk": 1,
               "hidden": torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16),
               "w1": torch.tensor([[[1., 0.], [0., 1.], [1., 1.], [1., 2.]],
                                   [[1., 1.], [2., 0.], [1., 0.], [0., 1.]]], dtype=torch.bfloat16),
               "w2": torch.eye(2, dtype=torch.bfloat16).repeat(2, 1, 1),
               "topk_weights": torch.ones(2, 1), "topk_ids": torch.tensor([[0], [1]], dtype=torch.int32)}
        inputs = tuple(inp[n] for n in ("hidden", "w1", "w2", "topk_weights", "topk_ids"))
        ns.update(TEST_SHAPES=[(2, 2, 2, 2, 1)], make_test_data=lambda *args: inp,
                  _make_prepared_fused_moe_runner=lambda *args: (compute, None))
        oracle = lambda: ns["reference_moe"](inp)
    originals = tuple(value.clone() for value in inputs)
    cached = oracle().to(torch.bfloat16)
    def compute():
        result = oracle().to(torch.bfloat16)
        if behavior == phase["name"] + "_wrong": result.add_(30.)
        if phase["name"] == "replay":
            if behavior == "cached": result = cached.clone()
            if behavior == "input_modified": inputs[0].add_(1)
        if phase["name"] == "measured" and behavior == "nonfinite": result.flatten()[0] = float("nan")
        return result
    calls = []
    def benchmark(fn, *, warmup, repetition, timed_run):
        calls.append((warmup, repetition))
        phase["name"] = "measured"
        timed_run.outputs = fn()
        timed_run.bound = True
        def replay():
            phase["name"] = "replay"
            out = fn()
            phase["name"] = "setup"
            return out
        timed_run.rerun = replay
        phase["name"] = "setup"
        return .1, {"benchmark_method": "cuda_graph", "benchmark_timed_run_kind": "captured_graph"}
    ns["benchmark_cuda_graph_or_events"] = benchmark
    rows = ns["run_performance"]()
    assert len(rows) == 1 and calls == [(0, 100)]
    if behavior == "correct":
        assert rows[0]["timed_output_correctness"] == rows[0]["replay_correctness"] == "PASS"
    else:
        assert rows[0]["execution_time_ms"] < 0
        assert rows[0]["benchmark_method"] == "benchmark_failed"
    checks.require_unchanged(inputs, originals)


@pytest.mark.parametrize("name", ["decode_attention", "sglang_fused_moe"])
def test_sglang_output_contracts_keep_original_fraction_and_zero_reference_rules(name):
    import torch
    task = ROOT / "tasks/triton2flydsl/sglang" / name
    checks = module(task / "scripts/replay_checks.py")
    fn = "_compare_decode_output" if name == "decode_attention" else "_compare_prepared_moe_output"
    ns = _harness_functions(task, {fn}, {"require_tensor_contract": checks.require_tensor_contract})
    compare = ns[fn]
    expected = torch.ones(10000, dtype=torch.bfloat16)
    if name == "decode_attention":
        actual = expected.clone(); actual[:9] = 50
        compare(actual, expected)  # Original >=99.9% disjunct.
        actual[:11] = 50
        with pytest.raises(AssertionError, match="Numerical mismatch"): compare(actual, expected)
        # Original global normalization disjunct and zero-reference fallback.
        expected = torch.ones(10000); expected[0] = 100
        actual = expected.bfloat16(); actual[1:] += .5
        compare(actual, expected)
        compare(torch.full((2, 2), .009, dtype=torch.bfloat16), torch.zeros(2, 2))
        with pytest.raises(AssertionError): compare(torch.full((2, 2), .011, dtype=torch.bfloat16), torch.zeros(2, 2))
    else:
        actual = expected.clone(); actual[:199] = 50
        compare(actual, expected)  # Original <=2% mismatch fraction.
        actual[:201] = 50
        with pytest.raises(AssertionError, match="Numerical mismatch"): compare(actual, expected)
        expected = torch.ones(10000, dtype=torch.float32)
        actual = expected.clone(); actual[0] += .01
        with pytest.raises(AssertionError): compare(actual, expected)  # FP32 permits no outliers.
    expected = torch.ones((2, 2), dtype=torch.bfloat16)
    for bad in (expected[:1], expected.float(), expected.to("meta"), torch.full_like(expected, float("nan"))):
        with pytest.raises(AssertionError): compare(bad, expected)


class _RemoveSglangReplayChecks(ast.NodeTransformer):
    """Remove only new checks and the host-only output return for AST comparison."""
    def visit_FunctionDef(self, node):
        if node.name in {"_compare_decode_output", "_compare_prepared_moe_output"}:
            return None
        self.generic_visit(node)
        if (node.name == "fn" and isinstance(node.body[-1], ast.Return)
                and isinstance(node.body[-1].value, ast.Name) and node.body[-1].value.id == "o"):
            node.body.pop()
        return node

    def visit_If(self, node):
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Raise)
                and isinstance(node.body[0].exc, ast.Call)
                and len(node.body[0].exc.args) == 1
                and isinstance(node.body[0].exc.args[0], ast.Constant)
                and node.body[0].exc.args[0].value == "Non-finite fused MoE output/reference"):
            return None
        return self.generic_visit(node)


@pytest.mark.parametrize("name", ["decode_attention", "sglang_fused_moe"])
@pytest.mark.parametrize("mutate_input", [False, True])
def test_sglang_correctness_uses_pristine_inputs_before_computing_reference(name, mutate_input, monkeypatch):
    import types
    import torch
    task = ROOT / "tasks/triton2flydsl/sglang" / name
    checks = module(task / "scripts/replay_checks.py")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    ns = {"require_unchanged": checks.require_unchanged, "require_tensor_contract": checks.require_tensor_contract,
          "_retry_oom": lambda fn: fn(), "DTYPE_NAME": "bfloat16", "MAX_KV_SPLITS": 1}
    _harness_functions(task, {"run_correctness", "reference", "reference_moe", "_shape_of"}, ns)
    if name == "decode_attention":
        cfg = {"seqs": [1], "head": 1, "kv_head": 1, "Lk": 2, "Lv": 2}
        q = torch.ones(1, 1, 2, dtype=torch.bfloat16)
        k, v, out = q.clone(), q.clone(), torch.empty_like(q)
        kvp, kvi, nks = torch.tensor([0, 1], dtype=torch.int32), torch.tensor([0]), torch.ones(1, dtype=torch.int32)
        al, alse = torch.empty(1, 1, 1, 2), torch.empty(1, 1, 1)
        def compute(*args):
            if mutate_input: q.add_(1)
            out.copy_(ns["reference"](q, k, v, kvi, cfg))
        ns.update(TEST_SHAPES=[cfg], make_inputs=lambda *args: (q, k, v, out, kvp, kvi, al, alse, nks),
                  load_module=lambda: types.SimpleNamespace(decode_attention_fwd=compute))
    else:
        inp = {"M": 1, "K": 2, "I": 2, "E": 1, "topk": 1,
               "hidden": torch.ones(1, 2, dtype=torch.bfloat16),
               "w1": torch.ones(1, 4, 2, dtype=torch.bfloat16), "w2": torch.ones(1, 2, 2, dtype=torch.bfloat16),
               "topk_weights": torch.ones(1, 1), "topk_ids": torch.zeros(1, 1, dtype=torch.int32)}
        def compute(*args):
            if mutate_input: inp["hidden"].add_(1)
            return ns["reference_moe"](inp)
        ns.update(TEST_SHAPES=[(1, 2, 2, 1, 1)], make_test_data=lambda *args: inp,
                  load_module=lambda: types.SimpleNamespace(fused_moe=compute))
    ok, error, details = ns["run_correctness"]()
    assert ok is (not mutate_input)
    if mutate_input: assert "read-only input" in error
    assert len(details) == 1


@pytest.mark.parametrize("omit", [None, "gate", "linear"])
def test_swiglu_saturation_controls_reject_missing_clamps(omit):
    import math
    import types
    import torch
    task = ROOT / "tasks/torch2flydsl/swiglu_and_mul_kernel"
    controls = module(task / "scripts/swiglu_controls.py")
    checks = module(task / "scripts/replay_checks.py")
    model = module(task / "model.py").Model()
    inputs = torch.zeros((2, 20), dtype=torch.bfloat16)
    def oracle(inp):
        values = inp.tolist()
        return torch.tensor([[
            min(g, 7.) / (1 + math.exp(-1.702 * min(g, 7.))) * (max(-7., min(y, 7.)) + 1.)
            for g, y in zip(row[:10], row[10:])
        ] for row in values], dtype=inp.dtype)
    def candidate(inp):
        gate, linear = inp.float().chunk(2, -1)
        if omit != "gate": gate = gate.clamp(max=7.)
        if omit != "linear": linear = linear.clamp(-7., 7.)
        return (gate * torch.sigmoid(1.702 * gate) * (linear + 1)).to(inp.dtype)
    mmod = types.SimpleNamespace(Model=lambda: model, get_init_inputs=lambda: [])
    kmod = types.SimpleNamespace(flydsl_swiglu_and_mul=candidate)
    h = types.SimpleNamespace(
        ARENA_PROVIDED_BASELINE=False, _KERNEL_DIR=".", MODEL_FILE="model.py", KERNEL_FILE="kernel.py",
        KERNEL_ENTRY="flydsl_swiglu_and_mul", REL_TOL=.01,
        SHAPES=[{"name":"boundary", "m":2,"n":20}], _make_inputs=lambda shape: inputs.clone(),
        _load_module=lambda directory, filename, alias: mmod if filename=="model.py" else kmod,
        _aiter_op=oracle, _checked_activation_output=lambda result, inp: result,
        require_unchanged=checks.require_unchanged, normalized_output=checks.normalized_output,
    )
    if omit is None:
        controls.check_saturation(h)
    else:
        with pytest.raises(AssertionError, match="Numerical mismatch"):
            controls.check_saturation(h)


@pytest.mark.parametrize("omit", [None, "gate", "linear"])
def test_swiglu_measured_replay_detects_missing_saturation(omit):
    import types
    import torch
    task = ROOT / "tasks/torch2flydsl/swiglu_and_mul_kernel"
    controls = module(task / "scripts/swiglu_controls.py")
    checks = module(task / "scripts/replay_checks.py")
    model = module(task / "model.py").Model()
    inp = torch.full((2, 20), .5, dtype=torch.bfloat16)
    original = inp.clone()
    def run():
        gate, linear = inp.float().chunk(2, -1)
        if omit != "gate": gate = gate.clamp(max=7.)
        if omit != "linear": linear = linear.clamp(-7., 7.)
        return (gate * torch.sigmoid(1.702 * gate) * (linear + 1)).to(inp.dtype)
    ns = {"normalized_output":checks.normalized_output, "require_tensor_contract":checks.require_tensor_contract,
          "require_unchanged":checks.require_unchanged, "verify_timed_run":checks.verify_timed_run,
          "saturation_input_":controls.saturation_input_, "REL_TOL":.01}
    _harness_functions(task, {"_activation_replay_validator", "_checked_activation_output"}, ns)
    timed = types.SimpleNamespace(bound=True, outputs=run(), rerun=run)
    # Original low-valued measured input cannot distinguish a missing clamp.
    torch.testing.assert_close(timed.outputs, model(inp))
    validate = ns["_activation_replay_validator"](inp, model)
    if omit is None:
        assert validate(timed)["replay_correctness"] == "PASS"
    else:
        with pytest.raises(AssertionError, match="Numerical mismatch"):
            validate(timed)
    assert torch.equal(inp, original)



class _RemoveBatchedInt8Checks(_RemoveAddedReplayChecks):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) == '_checked_batched_output':
            return None
        return super().visit_Expr(node)

    def visit_Call(self, node):
        if getattr(node.func, 'id', None) == '_checked_batched_output':
            return self.visit(node.args[0])
        return super().visit_Call(node)


@pytest.mark.parametrize('function,provided', [('run_benchmark',False),('arena_benchmark',False),('run_benchmark',True),('arena_benchmark',True)])
@pytest.mark.parametrize('behavior', ['correct','measured_wrong','replay_wrong','cached','input_modified','shape','dtype','nonfinite'])
def test_batched_int8_measured_outputs_and_replay(function, provided, behavior, monkeypatch, tmp_path):
    import math
    import types
    import torch
    task = ROOT/'tasks/torch2flydsl/batched_gemm_a8w8_kernel'
    checks = module(task/'scripts/replay_checks.py')
    mmod = module(task/'model.py')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    x = torch.tensor([[[1.,2.],[3.,4.]]],dtype=torch.bfloat16)
    w = torch.tensor([[[2.,3.],[4.,6.]]],dtype=torch.bfloat16)
    original=(x.clone(),w.clone());cached=mmod.Model()(x,w)
    phase={'value':'setup'}
    def compute(*args):
        result=mmod.Model()(x,w)
        if behavior==phase['value']+'_wrong':result.fill_(25)
        if phase['value']=='measured':
            if behavior=='shape':result=result[...,:1]
            if behavior=='dtype':result=result.float()
            if behavior=='nonfinite':result.flatten()[0]=float('nan')
        if phase['value']=='replay':
            if behavior=='cached':result=cached.clone()
            if behavior=='input_modified':x.add_(1)
        return result
    monkeypatch.setitem(sys.modules,'aiter',types.SimpleNamespace(batched_gemm_a8w8_CK=compute))
    kmod=types.SimpleNamespace(flydsl_batched_gemm_a8w8=compute)
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,use_cuda_graph,fallback_reason,timed_run=None):
        calls.append((warmup,repetition,use_cuda_graph,timed_run is not None))
        phase['value']='measured';out=fn();phase['value']='setup'
        if timed_run is not None:
            timed_run.bound=True;timed_run.outputs=out
            def replay():
                phase['value']='replay'
                try:return fn()
                finally:phase['value']='setup'
            timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph' if use_cuda_graph else 'cuda_event_fallback'}
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,
        'require_tensor_contract':checks.require_tensor_contract,'require_unchanged':checks.require_unchanged,'verify_timed_run':checks.verify_timed_run,
        '_KERNEL_DIR':str(tmp_path),'KERNEL_FILE':'kernel.py','MODEL_FILE':'model.py','KERNEL_ENTRY':'flydsl_batched_gemm_a8w8',
        '_load_module':lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod,
        '_make_inputs':lambda *a:(x,w),'_retry':lambda fn,**kwargs:fn(),
        'SHAPES':[{'name':'controlled','b':1,'m':2,'n':2,'k':2}],'TOL':.01,'math':math,'json':json,'Path':Path}
    _harness_functions(task,{function,'_norm_worst','_checked_batched_output','_compare_batched_output'},ns)
    if behavior=='correct':
        result=ns[function](verbose=False)
        if function=='run_benchmark':result=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        assert calls==[(0,100,False,True),(0,100,False,False)]
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(x,original[0]) and torch.equal(w,original[1])


@pytest.mark.parametrize('behavior',['correct','shape','dtype','device','nonfinite','input_modified'])
def test_batched_int8_correctness_output_and_input_contracts(behavior,monkeypatch):
    import types
    import torch
    task=ROOT/'tasks/torch2flydsl/batched_gemm_a8w8_kernel'
    checks=module(task/'scripts/replay_checks.py');mmod=module(task/'model.py')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    x=torch.tensor([[[1.,2.],[3.,4.]]],dtype=torch.bfloat16);w=x.clone()
    # Exercise the protected CPU quantizer/model without a fabricated CUDA device.
    model=mmod.Model();model.to=lambda *args:model
    original_model=mmod.Model;mmod.Model=lambda:model
    def compute(*args):
        result=model(x,w)
        if behavior=='shape':result=result[...,:1]
        if behavior=='dtype':result=result.float()
        if behavior=='device':result=result.to('meta')
        if behavior=='nonfinite':result.flatten()[0]=float('nan')
        if behavior=='input_modified':x.add_(1)
        return result
    def independent_int8_product(xi,wi,xs,ws,output):
        return (torch.bmm(xi.float(),wi.float().transpose(-1,-2))*xs*ws).to(torch.bfloat16)
    monkeypatch.setitem(sys.modules,'aiter',types.SimpleNamespace(batched_gemm_a8w8_CK=independent_int8_product))
    kmod=types.SimpleNamespace(flydsl_batched_gemm_a8w8=compute)
    ns={'require_tensor_contract':checks.require_tensor_contract,'require_unchanged':checks.require_unchanged,
        '_KERNEL_DIR':'.','KERNEL_FILE':'kernel.py','MODEL_FILE':'model.py','KERNEL_ENTRY':'flydsl_batched_gemm_a8w8',
        '_load_module':lambda directory,filename,alias:mmod if filename=='model.py' else kmod,
        '_make_inputs':lambda *a:(x,w),'_retry':lambda fn,**kwargs:fn(),
        'SHAPES':[{'name':'controlled','b':1,'m':2,'n':2,'k':2}],'TOL':.01}
    _harness_functions(task,{'run_correctness','_norm_worst','_checked_batched_output'},ns)
    if behavior=='correct':assert ns['run_correctness'](verbose=False)
    else:
        with pytest.raises(AssertionError,match='correctness FAILED'):ns['run_correctness'](verbose=False)


def test_batched_int8_zero_reference_gate_and_original_work_unchanged():
    import torch
    task=ROOT/'tasks/torch2flydsl/batched_gemm_a8w8_kernel'
    checks=module(task/'scripts/replay_checks.py');ns={'require_tensor_contract':checks.require_tensor_contract,'TOL':.01}
    _harness_functions(task,{'_norm_worst','_compare_batched_output'},ns)
    expected=torch.zeros((2,2),dtype=torch.bfloat16)
    ns['_compare_batched_output'](torch.full_like(expected,.009),expected)
    with pytest.raises(AssertionError,match='Numerical mismatch'):
        ns['_compare_batched_output'](torch.full_like(expected,.011),expected)
    hashes={'_make_inputs':'233b2a557543202859e73ecd66bcc69b905a88908b0cc31c0e8c1330a3823924',
            '_norm_worst':'30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014',
            'run_correctness':'edb1cbd2ed256c7cb1d858c04969a9c9b91a26f8473b0b2ae2232709f9dc3011',
            'run_benchmark':'cde32db449b949672b4d02cba0f5c3858e3553a3e9f53515034cc7dca9435674',
            'arena_benchmark':'794b1125a25e7e3a0b4d255fa19e263e086b6c0d22e7c16a7d414f131b54687b'}
    tree=ast.parse((task/'test_kernel_harness.py').read_text())
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name in hashes:
            normalized=_RemoveBatchedInt8Checks().visit(fn)
            if fn.name in {"run_benchmark", "arena_benchmark"}:
                from test_gemm_paired_timing import normalize_former_role_policy
                normalized = normalize_former_role_policy(normalized)
            assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==hashes[fn.name],fn.name


class _RemoveMxfp8Checks(_RemoveActivationReplayChecks):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='_checked_mxfp8_output':return None
        return super().visit_Expr(node)


def _mxfp8_check_namespace():
    task=ROOT/'tasks/torch2flydsl/dynamic_mxfp8_quant_kernel'
    checks=module(task/'scripts/replay_checks.py')
    ns={'CODE_TOL':1,'require_unchanged':checks.require_unchanged}
    _harness_functions(task,{'_compare','_checked_mxfp8_output','_compare_mxfp8_output','_quant_replay_validator'},ns)
    return ns


@pytest.mark.parametrize('bad',['shape','code_dtype','scale_dtype','device','nonfinite','scale','code','count'])
def test_mxfp8_output_contract_and_original_byte_gate(bad):
    import torch
    ns=_mxfp8_check_namespace();inp=(torch.ones((1,64),dtype=torch.bfloat16),)
    y=torch.full((1,64),2.,dtype=torch.float8_e4m3fn);s=torch.full((1,2),127,dtype=torch.uint8)
    expected=(y.clone(),s.clone());actual=(y,s)
    ns['_compare_mxfp8_output'](actual,expected,inp)
    if bad=='shape':actual=(y.reshape(-1),s)
    if bad=='code_dtype':actual=(y.view(torch.uint8),s)
    if bad=='scale_dtype':actual=(y,s.view(torch.int8))
    if bad=='device':actual=(y.to('meta'),s)
    if bad=='nonfinite':y.view(torch.uint8)[0,0]=127
    if bad=='scale':s[0,0]+=1
    if bad=='code':y.view(torch.uint8)[0,0]+=2
    if bad=='count':actual=(y,)
    with pytest.raises(AssertionError):ns['_compare_mxfp8_output'](actual,expected,inp)
    near=expected[0].clone();near.view(torch.uint8)[0,0]+=1
    ns['_compare_mxfp8_output']((near,expected[1]),expected,inp)


@pytest.mark.parametrize('function,provided',[('run_benchmark',False),('arena_benchmark',False),('run_benchmark',True),('arena_benchmark',True)])
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached','stale_scale','input_modified','shape','dtype','nonfinite'])
def test_mxfp8_actual_measured_and_tuple_replayed_outputs(function,provided,behavior,monkeypatch,tmp_path):
    import math
    import types
    import torch
    task=ROOT/'tasks/torch2flydsl/dynamic_mxfp8_quant_kernel';ns=_mxfp8_check_namespace()
    actual_model=module(task/'model.py').Model()
    x=torch.arange(1,129,dtype=torch.float32).reshape(2,64).to(torch.bfloat16)/16
    original=x.clone();phase={'name':'setup'};cached=actual_model(x)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    def compute(value,is_model):
        y,s=actual_model(value)
        if is_model==provided:
            if behavior==phase['name']+'_wrong':y.view(torch.uint8).fill_(0)
            if phase['name']=='replay':
                if behavior=='cached':y,s=(v.clone() for v in cached)
                if behavior=='stale_scale':s=cached[1].clone()
                if behavior=='input_modified':value.add_(1)
            if phase['name']=='measured':
                if behavior=='shape':y=y.reshape(-1)
                if behavior=='dtype':y=y.view(torch.uint8)
                if behavior=='nonfinite':y.view(torch.uint8).fill_(127)
        return y,s
    class Model:
        def to(self,*a):return self
        def __call__(self,value):return compute(value,True)
    mmod=types.SimpleNamespace(Model=Model,get_init_inputs=lambda:[])
    kmod=types.SimpleNamespace(flydsl_dynamic_mxfp8_quant=lambda value:compute(value,False))
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run=None):
        calls.append((warmup,repetition,timed_run is not None))
        phase['name']='measured';out=fn();phase['name']='setup'
        if timed_run is not None:
            timed_run.bound=True;timed_run.outputs=out
            def replay():
                phase['name']='replay'
                try:return fn()
                finally:phase['name']='setup'
            timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    ns.update(TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,_aiter_op=actual_model,
        _KERNEL_DIR=str(tmp_path),KERNEL_FILE='kernel.py',MODEL_FILE='model.py',KERNEL_ENTRY='flydsl_dynamic_mxfp8_quant',
        _load_module=lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod,
        _make_inputs=lambda shape:(x,),_retry=lambda fn,**kwargs:fn(),
        SHAPES=[{'name':'controlled','m':2,'n':64}],math=math,json=json,Path=Path)
    _harness_functions(task,{function,'_mean_ms'},ns)
    if behavior=='correct':
        report=ns[function](verbose=False)
        if function=='run_benchmark':report=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert report[0]['timed_output_correctness']==report[0]['replay_correctness']=='PASS'
        assert calls==[(10,100,False),(10,100,provided)]+([] if provided else [(10,100,True)])
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(x,original)


def test_mxfp8_original_compute_cases_gates_and_timed_work_unchanged():
    task=ROOT/'tasks/torch2flydsl/dynamic_mxfp8_quant_kernel'
    hashes={'_make_inputs': 'bd724715d203212c52011f3384ae5455007eee6d195a5f3a620e19c9643226b3', '_compare': 'c4ba881e0c4b71948d9b5fcbe2ad7743cd2207ad8def5e61afd8c313f5d58a3d', 'run_correctness': '163a59da1d6e0e44c216f0ef15040750dde1c9b84a24607ec03c352814430f88', 'run_benchmark': 'd05f4e55d364d37b42480de755ff16708a425be476cebda880f252b85aa5f366', 'arena_benchmark': '1cd996e9365de12e133caf89003413a12fad7c9f6226dce5952ae2c8ea69da3e'}
    tree=ast.parse((task/'test_kernel_harness.py').read_text())
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name in hashes:
            normalized=_RemoveMxfp8Checks().visit(fn)
            assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==hashes[fn.name],fn.name


@pytest.mark.parametrize('behavior',['correct','ignore_limit','wrong_up_clamp','input_modified'])
def test_silu_positive_limit_controls_and_restore(behavior):
    import math
    import types
    import torch
    task=ROOT/'tasks/torch2flydsl/silu_and_mul_kernel'
    control=module(task/'scripts/limit_controls.py');checks=module(task/'scripts/replay_checks.py');mmod=module(task/'model.py')
    original=torch.zeros((2,16),dtype=torch.bfloat16)
    h=types.SimpleNamespace(ARENA_PROVIDED_BASELINE=False,LIMIT=0.,REL_TOL=.01,
        _KERNEL_DIR='.',MODEL_FILE='model.py',KERNEL_FILE='kernel.py',KERNEL_ENTRY='flydsl_silu_and_mul',
        SHAPES=[{'name':'controlled','m':2,'n':16}],_make_inputs=lambda shape:original.clone(),
        require_unchanged=checks.require_unchanged,normalized_output=checks.normalized_output)
    def scalar_reference(inp):
        values=[]
        for row in inp.tolist():
            values.append([])
            for gate,up in zip(row[:8],row[8:]):
                g=float(torch.tensor(min(gate,h.LIMIT),dtype=torch.bfloat16))
                u=max(-h.LIMIT,min(up,h.LIMIT))
                values[-1].append(g/(1+math.exp(-g))*u)
        return torch.tensor(values,dtype=inp.dtype)
    def candidate(inp,limit):
        out=mmod.Model(0 if behavior=='ignore_limit' else limit)(inp)
        if behavior=='wrong_up_clamp':
            gate,up=inp.float().chunk(2,-1);gate=gate.clamp(max=limit).to(inp.dtype).float()
            out=(gate.sigmoid()*gate*up).to(inp.dtype)
        if behavior=='input_modified':inp.add_(1)
        return out
    h._aiter_op=scalar_reference
    h._load_module=lambda directory,filename,alias:mmod if filename=='model.py' else types.SimpleNamespace(flydsl_silu_and_mul=candidate)
    def checked(result,inp):
        checks.require_tensor_contract(result,inp[:,:inp.shape[1]//2]);return result
    h._checked_silu_result=checked
    if behavior=='correct':control.check_positive_limits(h)
    else:
        with pytest.raises(AssertionError):control.check_positive_limits(h)
    assert h.LIMIT==0.


def test_silu_only_requires_the_executed_public_operator(tmp_path):
    task=tmp_path/'silu';shutil.copytree(ROOT/'tasks/torch2flydsl/silu_and_mul_kernel',task)
    runtime=module(task/'task_runtime.py');cfg=runtime.config()
    (task/'kernel.py').write_text('def flydsl_silu_and_mul(inp, limit):\n return inp\ndef build_silu_and_mul_module(*args):\n raise NotImplementedError\n')
    # State/contract evidence only; identity output is not GPU correctness.
    assert runtime.source_state(cfg)==('implemented',[True])
    spec=load_task_spec(task/'config.yaml', task_id='torch2flydsl/silu_and_mul_kernel')
    assert [e['symbol'] for e in spec.to_mapping()['candidate']['entrypoints']]==['flydsl_silu_and_mul']


class _RemoveFmoeTimingChecks(_RemoveAddedReplayChecks):
    def visit_Global(self,node):
        if node.names==["ARENA_CORRECTNESS_RESULTS"]:return None
        return node
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call):
            if getattr(call.func,'id',None) in {'_checked_fmoe_output','_record_fmoe_case'}:return None
            if isinstance(call.func,ast.Attribute) and call.func.attr=='update' and call.args and isinstance(call.args[0],ast.Call) and getattr(call.args[0].func,'id',None)=='replay_validate':return None
        return super().visit_Expr(node)
    def visit_Assign(self,node):
        if len(node.targets)==1:
            target=node.targets[0]
            if isinstance(target,ast.Name) and target.id in {'routing_originals','replay_validate','ARENA_CORRECTNESS_RESULTS'}:return None
            if isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant) and target.slice.value=='operator_timing_inputs':return None
        return super().visit_Assign(node)
    def visit_Call(self,node):
        if getattr(node.func,'id',None)=='_checked_fmoe_output':return self.visit(node.args[0])
        return super().visit_Call(node)
    def visit_If(self,node):
        node=self.generic_visit(node)
        if len(node.orelse)==1 and isinstance(node.orelse[0],ast.FunctionDef):
            fn=node.orelse[0]
            if fn.name=='device_op' and len(fn.body)==1 and isinstance(fn.body[0],ast.Return):
                call=fn.body[0].value
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Call) and getattr(call.func.func,'id',None)=='_make_prepared_aiter_op':
                    # This one intentional boundary change is independently
                    # exercised by the preparation-in-measured-call test below.
                    node.orelse=[ast.Assign([ast.Name('device_op',ast.Store())],call.func)]
        return node


_FMOE_REPLAY_NAMES=['fmoe_fp8_blockscale_g1u1_kernel','fmoe_g1u1_tkw1_kernel']


@pytest.mark.parametrize('name',_FMOE_REPLAY_NAMES)
@pytest.mark.parametrize('function,provided',[('run_benchmark',False),('arena_benchmark',False),('run_benchmark',True),('arena_benchmark',True)])
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached','weight_mutation','shape','dtype','nonfinite'])
def test_fmoe_raw_input_timing_boundary_and_replay(name,function,provided,behavior,monkeypatch,tmp_path):
    import math
    import types
    import torch
    task=ROOT/'tasks/torch2flydsl'/name;checks=module(task/'scripts/replay_checks.py')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None);monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    hidden=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)
    phase={'value':'setup'};models=[];preparations=[]
    def reference(h,w1,w2):return ((h.float() @ w1[0].float().T) @ w2[0].float().T).to(h.dtype)
    class Model:
        experts=1
        def __init__(self):
            self.w1=torch.tensor([[[1.,2.],[3.,4.]]],dtype=torch.bfloat16)
            self.w2=torch.tensor([[[2.,1.],[1.,3.]]],dtype=torch.bfloat16)
            self.originals=(self.w1.clone(),self.w2.clone());models.append(self)
        def gate(self,h):return torch.zeros((h.shape[0],1))
        def forward_with_routing(self,h,weights,ids,plan):return reference(h,self.w1,self.w2)
    cached={}
    def compute(h,w1,w2,weights,ids):
        out=reference(h,w1,w2)
        cached.setdefault('output',out.clone())
        if behavior==phase['value']+'_wrong':out.fill_(1)
        if phase['value']=='replay':
            if behavior=='cached':out=cached['output'].clone()
            if behavior=='weight_mutation':w2.add_(1)
        if phase['value']=='measured':
            if behavior=='shape':out=out[:1]
            if behavior=='dtype':out=out.float()
            if behavior=='nonfinite':out[0,0]=float('nan')
        return out
    mmod=types.SimpleNamespace(route_topk=lambda logits,k:(torch.ones((2,1)),torch.zeros((2,1),dtype=torch.int32)),prepare_expert_plan=lambda *args:None)
    entry='flydsl_'+name.removesuffix('_kernel');kmod=types.SimpleNamespace(**{entry:compute})
    def prepared(*args):
        preparations.append(phase['value'])
        model,h,weights,ids=args[1:5] if 'blockscale' in name else args
        return lambda:compute(h,model.w1,model.w2,weights,ids)
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run=None):
        calls.append((warmup,repetition,timed_run is not None));phase['value']='measured';out=fn();phase['value']='setup'
        if timed_run is not None:
            timed_run.outputs=out;timed_run.bound=True
            def replay():
                phase['value']='replay'
                try:return fn()
                finally:phase['value']='setup'
            timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph'}
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_tensor_contract':checks.require_tensor_contract,
        'require_unchanged':checks.require_unchanged,'verify_timed_run':checks.verify_timed_run,
        '_KERNEL_DIR':str(tmp_path),'KERNEL_FILE':'kernel.py','MODEL_FILE':'model.py','KERNEL_ENTRY':entry,
        '_load_module':lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod,
        '_build_model':lambda *a:(Model(),hidden.clone()),'_make_prepared_aiter_op':prepared,'_retry':lambda fn,**kwargs:fn(),
        'SHAPES':[{'name':'controlled','tokens':2,'model_dim':2,'inter_dim':2,'experts':1,'topk':1}],
        'TOL':.035 if 'tkw1' in name else .01,'math':math,'json':json,'Path':Path}
    _harness_functions(task,{function,'_fmoe_inputs','_fmoe_replay_validator','_checked_fmoe_output','_compare_fmoe_output','_norm_worst'},ns)
    if behavior=='correct':
        result=ns[function](verbose=False)
        if function=='run_benchmark':result=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        assert result[0]['operator_timing_inputs']=='raw_hidden_raw_weights_selected_routing'
        assert calls==[(0,100,True),(0,100,False)]
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    if provided:
        assert 'measured' in preparations  # baseline preprocessing is timed
        if behavior not in ('measured_wrong','shape','dtype','nonfinite'):assert 'replay' in preparations
    else:assert not preparations
    for model in models:
        assert torch.equal(model.w1,model.originals[0]) and torch.equal(model.w2,model.originals[1])


@pytest.mark.parametrize('name',_FMOE_REPLAY_NAMES)
def test_fmoe_original_normalized_error_floor_kept(name):
    import torch
    task=ROOT/'tasks/torch2flydsl'/name;checks=module(task/'scripts/replay_checks.py')
    tol=.035 if 'tkw1' in name else .01
    ns={'TOL':tol,'require_tensor_contract':checks.require_tensor_contract}
    _harness_functions(task,{'_norm_worst','_compare_fmoe_output'},ns)
    expected=torch.zeros((2,2),dtype=torch.bfloat16)
    ns['_compare_fmoe_output'](torch.full_like(expected,tol*.9),expected)
    with pytest.raises(AssertionError,match='Numerical mismatch'):
        ns['_compare_fmoe_output'](torch.full_like(expected,tol*1.1),expected)


def test_fmoe_preserves_original_references_shapes_and_sampling_except_fair_input_boundary():
    hashes={'fmoe_fp8_blockscale_g1u1_kernel': {'_build_model': '9281e5efbec053c54f509ce05e3e5c92d09a444559d9eecaf9039b907275a9cb', '_make_prepared_aiter_op': '272d71b7299739875b02d5562c34f12ba884bbac5753988afda631c39270a813', '_aiter_op': '2fd80bb48d290b2330945cf09b82192748521355705941841e60fe0e17b9fa2e', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': '42adf960ecf89b23e26be1622dda86724a84c450467769726c33dccfe1c4a4b7', 'run_benchmark': '5279b64452e54271b408a75fa18a82a495ecacb6386669297871d0a948ded4a6', 'arena_benchmark': 'f98f295a876311be1650c7a3a32b9d25268e7c9cfffd6ce552be84a060ee813c'}, 'fmoe_g1u1_tkw1_kernel': {'_build_model': '9281e5efbec053c54f509ce05e3e5c92d09a444559d9eecaf9039b907275a9cb', '_make_prepared_aiter_op': 'a1ebb29f1b2ef25b96caa8e8d3807182f0ed8a8294e7385c41940156255e4867', '_aiter_op': '13e15634ca8882794989f39477de8c57653c2ea6c99933a3df89491163bfaf6a', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', '_cos_diff': 'a85a0c38f32e78c891cbde03d0fccc8d2b9d36d31de559a16b569d6b5db2595f', 'run_correctness': 'a1bc2faba011bef6f24e8fdc243f95924211e56d72a253973dab706397af3685', 'run_benchmark': 'f881b723e9dbc1197afdce84d53c3f644d50b739bb6046ef318179f28a228bc7', 'arena_benchmark': 'cd5793dd29587975382f8e91328dc0a7eec34974a1221b9cc28883561636be59'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/torch2flydsl'/name
        tree=ast.parse((task/'test_kernel_harness.py').read_text())
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                restored=_RemoveFmoeTimingChecks().visit(fn)
                assert hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


class _RemoveAddRmsnormChecks(_RemoveAddedReplayChecks):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='_checked_add_rmsnorm_pair':
            return None
        return super().visit_Expr(node)
    def visit_Call(self,node):
        if getattr(node.func,'id',None)=='_checked_add_rmsnorm_pair':return self.visit(node.args[0])
        return super().visit_Call(node)
    def visit_FunctionDef(self,node):
        if node.name=='_mean':
            node.body=ast.parse('return benchmark_cuda_graph_or_events(fn, warmup=warmup, repetition=iters)').body
            return node
        return self.generic_visit(node)


@pytest.mark.parametrize('function',['run_benchmark','arena_benchmark'])
@pytest.mark.parametrize('provided',[False,True])
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached_residual','cached_output','input_mutated','weight_mutated','residual_mutated','shape','dtype','nonfinite','missing'])
def test_fused_add_rmsnorm_measured_pair_and_replay(function,provided,behavior,monkeypatch,tmp_path):
    import torch,types,math
    t=ROOT/'tasks/torch2flydsl/fused_add_rmsnorm_kernel';checks=module(t/'scripts/replay_checks.py')
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    inp=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16);weight=torch.tensor([2.,3.],dtype=torch.bfloat16);residual=torch.tensor([[2.,-1.],[.5,2.]],dtype=torch.bfloat16);inputs=(inp,weight,residual);originals=tuple(x.clone() for x in inputs)
    phase={'value':'setup'}
    def oracle(*args):
        x,w,res=args or inputs;ro=x+res;rf=ro.float();return (rf*torch.rsqrt(rf.square().mean(-1,keepdim=True)+1e-5)*w.float()).to(x.dtype),ro
    cached=oracle()
    def compute(is_model):
        out=list(oracle())
        if is_model==provided:
            if behavior==phase['value']+'_wrong':out[0].fill_(5)
            if phase['value']=='replay':
                if behavior=='cached_output':out[0]=cached[0].clone()
                if behavior=='cached_residual':out[1]=cached[1].clone()
                for name,value in zip(('input','weight','residual'),inputs):
                    if behavior==name+'_mutated':value.add_(1)
            if phase['value']=='measured':
                if behavior=='shape':out[1]=out[1][:1]
                if behavior=='dtype':out[0]=out[0].float()
                if behavior=='nonfinite':out[1][0,0]=float('nan')
                if behavior=='missing':out=out[:1]
        return tuple(out)
    class Model:
        def __init__(self,*a):pass
        def to(self,*a):return self
        def eval(self):return self
        def __call__(self,*a):return compute(True)
    target=lambda *a:compute(False)
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((fn.__name__,warmup,repetition));phase['value']='measured';timed_run.outputs=fn();timed_run.bound=True;phase['value']='setup'
        def replay():
            phase['value']='replay'
            try:return fn()
            finally:phase['value']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_tensor_contract':checks.require_tensor_contract,'require_unchanged':checks.require_unchanged,
        'math':math,'json':json,'Path':Path,'_KERNEL_DIR':str(tmp_path),'MODEL_FILE':'model.py','KERNEL_ENTRY':'flydsl_fused_add_rmsnorm',
        'REL_TOL':.01,'EPS':1e-5,'SHAPES':[{'name':'controlled','m':2,'n':2}], '_make_inputs':lambda *a:inputs,
        '_load_module':lambda *a:types.SimpleNamespace(Model=Model),'_load_target':lambda:target,'_is_pure_starter':lambda:provided,
        '_probe_target':lambda *a:(False,None) if provided else (True,target()),'_retry':lambda fn,**kw:fn(),'_aiter_add_rmsnorm':oracle}
    _harness_functions(t,{function,'_norm_max_err','_checked_add_rmsnorm_pair','_compare_add_rmsnorm_pair','_verify_add_rmsnorm_timed'},ns)
    if behavior=='correct':
        result=ns[function](verbose=False)
        if function=='run_benchmark':result=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        assert calls==[('run_ref',10,100),('run_truth',10,100)]+([] if provided else [('run_target',10,100)])
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    for value,original in zip(inputs,originals):assert torch.equal(value,original)


def test_fused_add_rmsnorm_pair_original_gate_and_interface():
    import torch
    t=ROOT/'tasks/torch2flydsl/fused_add_rmsnorm_kernel';checks=module(t/'scripts/replay_checks.py');ns={'require_tensor_contract':checks.require_tensor_contract,'REL_TOL':.01}
    _harness_functions(t,{'_norm_max_err','_checked_add_rmsnorm_pair','_compare_add_rmsnorm_pair'},ns)
    expected=(torch.tensor([[100.,1.]],dtype=torch.bfloat16),torch.tensor([[40.,2.]],dtype=torch.bfloat16))
    actual=tuple(x.clone() for x in expected);actual[0][0,1]+=.5;actual[1][0,1]+=.25
    ns['_compare_add_rmsnorm_pair'](actual,expected)
    actual[1][0,1]+=1
    with pytest.raises(AssertionError,match='Numerical mismatch'):ns['_compare_add_rmsnorm_pair'](actual,expected)
    cfg=yaml.safe_load((t/'config.yaml').read_text());assert [e['symbol'] for e in cfg['candidate']['entrypoints']]==['flydsl_fused_add_rmsnorm']


def test_fused_add_rmsnorm_original_numeric_and_timing_work_preserved():
    hashes={'_make_inputs': 'c3edf63cd1900b43c1cb36892301a24a9e091233ad4417048b4efed591ff0b76', '_aiter_add_rmsnorm': '8a493e418f637db8c313094e185560ec0c021db90c2e2d2122bd3219a89c0a8b', '_norm_max_err': '750a488ebe381cf762ba31a41890c0fe3dd87c2dc703955ad218dd85a2d9862e', 'run_correctness': '70bf0b8f5c08b726579825cc5ed3f859c62e75a05a268f8b32b49362dfc34c84', 'run_benchmark': '46a7dd12bae5d7acf6e00a97ddb5d2ff170b0cd69c65b73cde3dc3fbe2a01099', 'arena_benchmark': '5695d3712ae1a91fa293b9cd61ea29bb50c52d43e8f193e8b67191e846b0d738'}
    t=ROOT/'tasks/torch2flydsl/fused_add_rmsnorm_kernel';tree=ast.parse((t/'test_kernel_harness.py').read_text())
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name in hashes:
            restored=_RemoveAddRmsnormChecks().visit(fn)
            assert hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()==hashes[fn.name],fn.name


@pytest.mark.parametrize('name',_FMOE_REPLAY_NAMES)
@pytest.mark.parametrize('fault',['numerical','dtype','launch'])
def test_fmoe_case_evidence_preserves_real_passes_and_failure_kind(name,fault,monkeypatch,capsys):
    import torch,types
    t=ROOT/'tasks/torch2flydsl'/name;runtime=module(t/'task_runtime.py');actions=module(t/'scripts/task_actions.py');checks=module(t/'scripts/replay_checks.py')
    candidate_checks=module(t/'scripts/candidate_checks.py');monkeypatch.setitem(sys.modules,'scripts.candidate_checks',candidate_checks)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    h=types.ModuleType('controlled_fmoe_harness');h.__dict__.update(ARENA_PROVIDED_BASELINE=True,
        _KERNEL_DIR='.',MODEL_FILE='model.py',KERNEL_FILE='kernel.py',KERNEL_ENTRY='flydsl_'+name.removesuffix('_kernel'),
        SHAPES=actions.EXPECTED_CASES,TOL=.035 if 'tkw1' in name else .01,
        require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged)
    class Model:
        def __init__(self,i):self.i=i;self.w1=torch.ones((1,2,2),dtype=torch.bfloat16);self.w2=self.w1.clone()
        def __call__(self,hidden):return hidden.clone()
    def build(m,shape):return Model(h.SHAPES.index(shape)),torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)
    def baseline(m,model,hidden,k):
        if model.i==0:
            if fault=='launch':raise RuntimeError('controlled GPU launch failure')
            if fault=='dtype':return hidden.float()
            return hidden+1
        return hidden.clone()
    h._build_model=build;h._aiter_op=baseline;h._retry=lambda fn,**kw:fn();h._load_module=lambda directory,filename,alias:None if filename=='kernel.py' else types.SimpleNamespace()
    _harness_functions(t,{'run_correctness','_fmoe_inputs','_checked_fmoe_output','_norm_worst','_cos_diff','_record_fmoe_case'},h.__dict__)
    real_loader=runtime.load_module
    monkeypatch.setattr(runtime,'load_module',lambda key,path:h if key=='arena_harness' else actions if key=='arena_task_actions' else real_loader(key,path))
    assert runtime.run(['baseline','correctness'])==1
    report=json.loads(next(line.split('=',1)[1] for line in capsys.readouterr().out.splitlines() if line.startswith('ARENA_EVAL_RESULT=')))
    assert report['status']=='FAIL'
    assert [row['status'] for row in report['cases']]==['FAIL','PASS','PASS']
    first=report['cases'][0]
    if fault=='numerical':
        assert report['failure_kind']==first['failure_kind']=='numerical_mismatch'
        assert first['metadata']['baseline_max_abs_error']==1.
        assert first['metadata']['baseline_normalized_max_error']==.25
        assert first['metadata']['tolerance']==h.TOL
    else:
        assert report.get('failure_kind')!='numerical_mismatch'
        assert first['failure_kind']=='execution_or_contract_error'
    for row in report['cases'][1:]:
        assert 'failure_kind' not in row and 'reason' not in row
        assert row['metadata']['baseline_normalized_max_error']==0.
    manifest=runtime.manifest()
    assert [{k:row[k] for k in ('test_case_id','params')} for row in report['cases']]==[{k:row[k] for k in ('test_case_id','params')} for row in manifest]


_QUANT_GEMM_CONTROL_NAMES=['gemm_a16w8_blockscale_kernel', 'gemm_a16wfp4_kernel', 'gemm_a4w4_kernel', 'gemm_a8w8_blockscale_kernel', 'gemm_a8w8_kernel', 'gemm_a8w8_per_token_scale_kernel', 'gemm_a8wfp4_kernel', 'gemm_afp4wfp4_kernel', 'gemm_afp8wfp8_kernel']


class _RemoveQuantGemmChecks(_RemoveAddedReplayChecks):
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call):
            if getattr(call.func,'id',None) in {'_checked_quant_gemm_output','_record_gemm_case'}:return None
            if isinstance(call.func,ast.Attribute) and call.func.attr=='update' and call.args and isinstance(call.args[0],ast.Call) and getattr(call.args[0].func,'id',None)=='replay_validate':return None
        return super().visit_Expr(node)
    def visit_Global(self,node):
        if node.names==['ARENA_CORRECTNESS_RESULTS']:return None
        return node
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None)=='ARENA_CORRECTNESS_RESULTS':return None
        if len(node.targets)==1 and getattr(node.targets[0],'id',None)=='knorm' and isinstance(node.value,ast.Constant) and node.value.value is None:return None
        if len(node.targets)==1 and getattr(node.targets[0],'id',None)=='replay_validate':return None
        return super().visit_Assign(node)
    def visit_Call(self,node):
        if getattr(node.func,'id',None)=='_checked_quant_gemm_output':return self.visit(node.args[0])
        if getattr(node.func,'id',None)=='gemm_afp8wfp8':
            for keyword in node.keywords:
                if keyword.arg=='x_scale_group_size':
                    assert isinstance(keyword.value,ast.Constant) and keyword.value.value==32
            node.keywords=[kw for kw in node.keywords if kw.arg!='x_scale_group_size']
        return super().visit_Call(node)


@pytest.mark.parametrize('name',_QUANT_GEMM_CONTROL_NAMES)
@pytest.mark.parametrize('function,provided',[('run_benchmark',False),('arena_benchmark',False),('run_benchmark',True),('arena_benchmark',True)])
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached','input_mutation','weight_mutation','shape','dtype','nonfinite'])
def test_quant_gemm_actual_timed_output_and_original_quantized_oracle(name,function,provided,behavior,monkeypatch,tmp_path):
    import torch,types,math
    t=ROOT/'tasks/torch2flydsl'/name;checks=module(t/'scripts/replay_checks.py');phase={'value':'setup'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    a=torch.tensor([[.4,1.7],[2.3,3.8]],dtype=torch.bfloat16);w=torch.tensor([[2.2,.7],[1.3,2.8]],dtype=torch.bfloat16);originals=(a.clone(),w.clone())
    # Controlled CPU quantizer differs from raw matrix multiplication. This tests
    # oracle plumbing, not GPU quantizer accuracy; original models are fingerprinted.
    def quant(x):return (x.float()*2).round()/2
    def oracle(x,y):return (quant(x) @ quant(y).T).to(torch.bfloat16)
    cached=oracle(a,w)
    assert not torch.equal(cached,a@w.T)
    def compute(x,y):
        out=oracle(x,y)
        if behavior==phase['value']+'_wrong':out.fill_(1)
        if phase['value']=='replay':
            if behavior=='cached':out=cached.clone()
            if behavior=='input_mutation':x.add_(1)
            if behavior=='weight_mutation':y.add_(1)
        if phase['value']=='measured':
            if behavior=='shape':out=out[:1]
            if behavior=='dtype':out=out.float()
            if behavior=='nonfinite':out[0,0]=float('nan')
        return out
    class Model:
        def to(self,*a):return self
        def eval(self):return self
        def __call__(self,x,y):return oracle(x,y)
    mmod=types.SimpleNamespace(Model=Model,prepare_mxfp4_values=lambda *a:None,quantize_a8w8_blockscale=lambda x,y:(x,None,y,None),quantize_a8w8=lambda x,y:(x,None,y,None))
    kmod=types.SimpleNamespace(**{'flydsl_'+name.removesuffix('_kernel'):compute})
    monkeypatch.setitem(sys.modules,'aiter',types.SimpleNamespace(gemm_a8w8_blockscale=lambda x,y,*a:compute(x,y),gemm_a8w8=lambda x,y,*a:compute(x,y)))
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run=None,**kwargs):
        calls.append((warmup,repetition,timed_run is not None,kwargs));phase['value']='measured';out=fn();phase['value']='setup'
        if timed_run is not None:
            timed_run.outputs=out;timed_run.bound=True
            def replay():
                phase['value']='replay'
                try:return fn()
                finally:phase['value']='setup'
            timed_run.rerun=replay
        graph=kwargs.get('use_cuda_graph',True)
        return .1,{'benchmark_method':'cuda_graph' if graph else 'cuda_event_fallback','benchmark_timed_run_kind':'captured_graph' if graph else 'eager_callable'}
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_tensor_contract':checks.require_tensor_contract,'require_unchanged':checks.require_unchanged,'verify_timed_run':checks.verify_timed_run,
        '_load_module':lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod,
        '_make_inputs':lambda *a:(originals[0].clone(),originals[1].clone()),'_retry':lambda fn,**kw:fn(),
        '_aiter_ground_truth':lambda *args:compute(*args[-2:]),'_KERNEL_DIR':str(tmp_path),'MODEL_FILE':'model.py','KERNEL_FILE':'kernel.py',
        'KERNEL_ENTRY':'flydsl_'+name.removesuffix('_kernel'),'SHAPES':[{'name':'controlled','m':2,'n':2,'k':2}],'TOL':.01,'math':math,'json':json,'Path':Path}
    _harness_functions(t,{function,'_checked_quant_gemm_output','_compare_quant_gemm_output','_gemm_replay_validator','_norm_worst'},ns)
    if behavior=='correct':
        result=ns[function](verbose=False)
        if function=='run_benchmark':result=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        assert [(a,b,c) for a,b,c,_ in calls]==[(0,100,True),(0,100,False)]
        if name!='gemm_a8w8_blockscale_kernel':
            assert calls[0][3]['use_cuda_graph'] is False
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)


@pytest.mark.parametrize('name',_QUANT_GEMM_CONTROL_NAMES)
def test_quant_gemm_preserves_zero_reference_denominator_and_normalized_gate(name):
    import torch
    t=ROOT/'tasks/torch2flydsl'/name;checks=module(t/'scripts/replay_checks.py');ns={'require_tensor_contract':checks.require_tensor_contract,'TOL':.01}
    _harness_functions(t,{'_norm_worst','_compare_quant_gemm_output'},ns)
    expected=torch.zeros((2,2),dtype=torch.bfloat16)
    ns['_compare_quant_gemm_output'](torch.full_like(expected,.009),expected)
    with pytest.raises(AssertionError,match='Numerical mismatch'):ns['_compare_quant_gemm_output'](torch.full_like(expected,.02),expected)


def test_quant_gemm_original_quantization_inputs_numeric_and_timing_functions_preserved():
    hashes={'gemm_a16w8_blockscale_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_aiter_ground_truth': '93bff37a12d09580cc31aa6ffa370e5a2078a48b39f6b45c1660cdb85ceed836', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': 'ce7fc05173a63f6dd8a3c8ed28cf80406fa0c219cc622b76d50250e844d22053', 'run_benchmark': '9a48b014da77dbd608d90d612a6388e18f95fd8f33e482171a4700e5686f725d', 'arena_benchmark': 'defc1227e2c5eb6db2c8bb530c0d3d20493271b8a1f3b504c24154573cba059e'}, 'gemm_a16wfp4_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_aiter_ground_truth': '94c9fb8daa6c36fddc997ac668a6cb45a36d3ccd3f3b3e7d2b61f2eaa261debd', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': 'efa4bd4a9ad4138146bf604144086cb01d969ea5eb213385f83996ff690dd00f', 'run_benchmark': '4a94553fec39201aa1aa248165c57f38edc28334c765cca94b19a1fbe589ca33', 'arena_benchmark': '3818b99f737ad1ec43e6d4c575da11642801346bb036cfe9ecfa143aedff8060'}, 'gemm_a4w4_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_aiter_ground_truth': '5865e0e578d5b704f9b8e14647b8012315927edda4d707a434273d7f56929ac1', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': '5fde44afc9fbb877a379ff91caee3f2a33e890b70689428476e93c1b957db2b9', 'run_benchmark': '4ca5e661c5d35618e6dc4591984dc4a4369ffe793043fdea4dac133186f43f8c', 'arena_benchmark': 'c8853015a53f2f888f974a53e4a02f7b22c1aeceeb0d2c6aa2957d4d3cca1ad4'}, 'gemm_a8w8_blockscale_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_norm_worst': 'd7003c9ba7525de49603baf59e23bdb58e69a36796ae75fa7cad95fb292025a6', 'run_correctness': '6af71206b39f7aba58d882103352b64382317d1529a9fba71e8265fec33b7f0c', 'run_benchmark': 'b172161ae8a4c4cba052938c63aa5c3b4c6f674bc66f4c0f20551c3b7614f556', 'arena_benchmark': '7a62ff1081019435b838188ad687efdcd029b0a9c87720302f162ed78643dd47'}}
    for name,functions in hashes.items():
        tree=ast.parse((ROOT/'tasks/torch2flydsl'/name/'test_kernel_harness.py').read_text())
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                restored=_RemoveQuantGemmChecks().visit(fn)
                if fn.name in {"run_benchmark", "arena_benchmark"}:
                    from test_gemm_paired_timing import normalize_former_role_policy
                    restored = normalize_former_role_policy(restored)
                assert hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


@pytest.mark.parametrize('behavior',['correct','cached','scale_mutated','reference_error'])
@pytest.mark.parametrize('expanded_dims',[1,2])
def test_pa_replay_restores_expanded_scale_storage_without_changing_layout(behavior,expanded_dims):
    import torch,types
    checks=module(ROOT/'tasks/flydsl2flydsl/pa_decode_fp8_kernel/scripts/replay_checks.py')
    query=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)
    storage=torch.tensor([2.]) if expanded_dims==2 else torch.tensor([2.,3.])
    scale=storage.expand(2,2);strides=scale.stride();pointer=scale.data_ptr()
    originals=(query.clone(),scale.clone());expected=(query*scale).to(query.dtype);output=expected.clone()
    def replay():
        if behavior=='cached':output.copy_(expected)
        else:output.copy_((query*scale).to(query.dtype))
        if behavior=='scale_mutated':storage.add_(1)
        return output
    def reference():
        if behavior=='reference_error':raise RuntimeError('controlled reference failure')
        return (query*scale).to(query.dtype)
    timed=types.SimpleNamespace(bound=True,outputs=output,rerun=replay)
    def compare(actual,ref):checks.allclose_output(actual,ref,atol=.005,rtol=0)
    kwargs=dict(inputs=(query,scale),originals=originals,expected=expected,perturb=lambda:query.neg_(),reference=reference,compare=compare)
    if behavior=='correct':assert checks.verify_timed_run(timed,**kwargs)['replay_correctness']=='PASS'
    else:
        with pytest.raises((AssertionError,RuntimeError)):checks.verify_timed_run(timed,**kwargs)
    checks.require_unchanged((query,scale),originals)
    assert scale.stride()==strides and scale.data_ptr()==pointer


def test_remaining_quant_gemm_original_inputs_numerics_and_timing_preserved():
    hashes={'gemm_a8w8_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': '49c41d1fba71eb54e664e556e6081ceb5938aa08214341987d89e65b26cbe740', 'run_benchmark': 'e9cbc03267bff29c1aaf61240ab3363b389563fc7f3dea389ceb7f96f3dd84dd', 'arena_benchmark': '1b94d6280a483fb04e1c8aaebc9848c2ebea4202635ffb20f0b525bf9311e61c'}, 'gemm_a8w8_per_token_scale_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_aiter_ground_truth': 'b7f7bb34e20c863b280d9183cf5fe2e53be4ca6e1d77a2b9ba6628b3eaf6a2fd', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': '6126445a23348efcbf4ce6ecb3f7257b13022201408973b491a532002ef510a8', 'run_benchmark': 'dca6bb6cf3914593cc7ae6e3e23b008a3266b9d18cd78dbcb39a5349858b104e', 'arena_benchmark': '363964d370db9089b89b36950d53c1768c03b9eb9b256e4cd691ebb014f3e55d'}, 'gemm_a8wfp4_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_aiter_ground_truth': '1121be4746b34f62737523a4c3f59090dbf9d1424e471afcf28a02f6345cbdeb', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': 'f3943c326b31dd2b5a5af19af08207cd9ca4adf0e3fb3c8f6bf54bd87a3c1daa', 'run_benchmark': 'ba914faf92c506e87a2884cadf492d0cf88ccebd796267fee9ccdcb2e25e7b6a', 'arena_benchmark': '3f765494cebcd6e7e272c46c087c5deef2da8930eb863fcf3218af19326b1467'}, 'gemm_afp4wfp4_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_aiter_ground_truth': '58db98b7753e81a8641b5b32bf411d5edef8ad09aafa6f49d8a5a7eee1d7f7f7', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': '91ade3a412df56e545036bf500ce1f9a44153414d0ef684e971ac9381a45aa3b', 'run_benchmark': '28293637bec4ef2e181d77971151d2f0ff1a428b107b8542bb5ec8db9360cac2', 'arena_benchmark': '5aab1be9ee10f65ac072ce7b0565edcf54faef708d1acc4225df52abb9dcaf40'}, 'gemm_afp8wfp8_kernel': {'_make_inputs': '1ad09792077e031b36ddc25108380668ae78fb442487fca064c831df7d79a0ec', '_aiter_ground_truth': 'c25e2507a4a265cc74e92c592123cc75772bed23a6fb5af6de9a604261a783f0', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': 'e073bf26df3162f6752189650fb9854efb27d39f954b39ea97efde3eb8cdce66', 'run_benchmark': 'd9172d3f2b94921e9574d5eabe8d2c99b521eb12f0d6b4ef8a206bb97e10ad7b', 'arena_benchmark': 'e87e8dbc1508828b5f42cfaa941575f2327ac030a04e1eadd315ee815e707450'}}
    for name,functions in hashes.items():
        tree=ast.parse((ROOT/'tasks/torch2flydsl'/name/'test_kernel_harness.py').read_text())
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                restored=_RemoveQuantGemmChecks().visit(fn)
                if fn.name in {"run_benchmark", "arena_benchmark"}:
                    from test_gemm_paired_timing import normalize_former_role_policy
                    restored = normalize_former_role_policy(restored)
                assert hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


@pytest.mark.parametrize('k',[128,256])
def test_afp8wfp8_production_binding_preserves_actual_mx32_scales(k,monkeypatch):
    import torch,types
    task=ROOT/'tasks/torch2flydsl/gemm_afp8wfp8_kernel';model=module(task/'model.py')
    # Each group has a different power-of-two range, so treating four MX32
    # groups as one 128-wide scale cannot accidentally produce the same data.
    a=torch.arange(1,2*k+1,dtype=torch.float32).reshape(2,k)
    a=(a*torch.exp2(torch.arange(k,dtype=torch.float32)//32-3)).to(torch.bfloat16)
    w=(torch.arange(128*k,dtype=torch.float32).reshape(128,k)%11-5).to(torch.bfloat16)
    actual_inputs=model.quantize_afp8wfp8(a,w);calls=[]
    def production(x,weight,x_scale,w_scale,*,dtype=torch.bfloat16,x_scale_group_size=128):
        assert x_scale.shape==(x.shape[0],x.shape[1]//x_scale_group_size)
        calls.append(x_scale_group_size)
        # Independently decode the supplied bytes and perform a CPU/FP64 dot.
        dx=x.double()*torch.exp2(x_scale.double()-127).repeat_interleave(32,1)
        dw=weight.double()*torch.exp2(w_scale.double()-127).repeat_interleave(128,0).repeat_interleave(128,1)
        return (dx@dw.T).to(dtype)
    with pytest.raises(AssertionError):production(actual_inputs[0],actual_inputs[2],actual_inputs[1],actual_inputs[3])
    monkeypatch.setitem(sys.modules,'aiter.ops.triton.gemm.basic.gemm_afp8wfp8',types.SimpleNamespace(gemm_afp8wfp8=production))
    ns={};_harness_functions(task,{'_aiter_ground_truth'},ns)
    actual=ns['_aiter_ground_truth'](model,a,w);expected=model.Model()(a,w)
    assert calls==[32]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


def test_afp8wfp8_production_binding_propagates_runtime_error(monkeypatch):
    import types
    def production(*args,**kwargs):
        assert kwargs['x_scale_group_size']==32
        raise RuntimeError('device launch failed')
    monkeypatch.setitem(sys.modules,'aiter.ops.triton.gemm.basic.gemm_afp8wfp8',types.SimpleNamespace(gemm_afp8wfp8=production))
    model=types.SimpleNamespace(quantize_afp8wfp8=lambda a,w:(1,2,3,4));ns={}
    _harness_functions(ROOT/'tasks/torch2flydsl/gemm_afp8wfp8_kernel',{'_aiter_ground_truth'},ns)
    with pytest.raises(RuntimeError,match='device launch failed'):ns['_aiter_ground_truth'](model,None,None)


class _RemoveLayernormChecks(_RemoveAddedReplayChecks):
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call):
            if getattr(call.func,'id',None) in {'_checked_layernorm_output','_checked_layernorm_pair'}:return None
            if isinstance(call.func,ast.Attribute) and call.func.attr=='update' and call.args and isinstance(call.args[0],ast.Call) and getattr(call.args[0].func,'id',None) in {'_verify_layernorm_timed','replay_validate'}:return None
        return super().visit_Expr(node)
    def visit_Call(self,node):
        if getattr(node.func,'id',None) in {'_checked_layernorm_output','_checked_layernorm_pair'}:return self.visit(node.args[0])
        if getattr(node.func,'id',None)=='_mean_ms':node.keywords=[x for x in node.keywords if x.arg!='replay_validate']
        return super().visit_Call(node)
    def visit_With(self,node):
        node=self.generic_visit(node)
        return node if node.body else None
    def visit_FunctionDef(self,node):
        if node.name=='replay_validate':return None
        if node.name=='_mean':
            node.body=ast.parse('return benchmark_cuda_graph_or_events(fn,warmup=warmup,repetition=iters)').body
            return node
        if node.name=='_mean_ms':
            node.args.kwonlyargs=[];node.args.kw_defaults=[]
        return self.generic_visit(node)


@pytest.mark.parametrize('name',['layernorm2d_kernel','layernorm2d_with_add_kernel'])
@pytest.mark.parametrize('function',['run_benchmark','arena_benchmark'])
@pytest.mark.parametrize('provided',[True,False])
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached','input_mutated','weight_mutated','bias_mutated','shape','dtype','nonfinite'])
def test_layernorm_actual_measured_output_and_replay(name,function,provided,behavior,monkeypatch,tmp_path):
    import torch,types,math
    t=ROOT/'tasks/torch2flydsl'/name;checks=module(t/'scripts/replay_checks.py');added='with_add' in name
    x=torch.tensor([[1.,2.,4.,-1.],[3.,-4.,2.,1.]],dtype=torch.bfloat16)
    residual=torch.tensor([[.5,-1.,2.,3.],[2.,1.,-.5,2.]],dtype=torch.bfloat16)
    weight=torch.tensor([2.,-1.,.5,3.],dtype=torch.bfloat16);bias=torch.tensor([.5,1.,-2.,.25],dtype=torch.bfloat16)
    inputs=(x,residual,weight,bias) if added else (x,weight,bias);originals=tuple(v.clone() for v in inputs)
    actual_model=module(t/'model.py').Model();phase={'value':'setup'}
    def oracle(*args):
        inp,res,w,b=args if added else (args[0],None,args[1],args[2])
        summed=inp+res if added else inp
        out=torch.nn.functional.layer_norm(summed.float(),[4],w.float(),b.float(),1e-5).to(inp.dtype)
        return (out,summed) if added else out
    cache=oracle(*inputs);cached=tuple(v.clone() for v in cache) if added else (cache.clone(),)
    def compute(is_model):
        raw=actual_model(*inputs);values=list(raw) if added else [raw]
        if is_model==provided:
            if behavior==phase['value']+'_wrong':values[0].fill_(5)
            if phase['value']=='replay':
                if behavior=='cached':values=[v.clone() for v in cached]
                for label,value in [('input',x),('weight',weight),('bias',bias)]:
                    if behavior==label+'_mutated':value.add_(1)
            if phase['value']=='measured':
                if behavior=='shape':values[0]=values[0][:1]
                if behavior=='dtype':values[0]=values[0].float()
                if behavior=='nonfinite':values[0][0,0]=float('nan')
        return tuple(values) if added else values[0]
    class Model:
        def __init__(self,*a):pass
        def to(self,*a):return self
        def eval(self):return self
        def __call__(self,*a):return compute(True)
    target=lambda *a:compute(False);mmod=types.SimpleNamespace(Model=Model,get_init_inputs=lambda:[1e-5]);kmod=types.SimpleNamespace(**{'flydsl_'+name.removesuffix('_kernel'):target})
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setitem(sys.modules,'aiter',types.SimpleNamespace(layer_norm=lambda *a:oracle(*a[:-1])))
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));phase['value']='measured';timed_run.outputs=fn();timed_run.bound=True;phase['value']='setup'
        def replay():
            phase['value']='replay'
            try:return fn()
            finally:phase['value']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_tensor_contract':checks.require_tensor_contract,'require_unchanged':checks.require_unchanged,
        'math':math,'json':json,'Path':Path,'_KERNEL_DIR':str(tmp_path),'MODEL_FILE':'model.py','KERNEL_FILE':'kernel.py','KERNEL_ENTRY':'flydsl_'+name.removesuffix('_kernel'),
        'REL_TOL':.01,'PASS_PCT':99.9,'EPS':1e-5,'SHAPES':[{'name':'controlled','m':2,'n':4}], '_make_inputs':lambda *a:inputs,
        '_load_module':lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod,
        '_load_target':lambda:target,'_is_pure_starter':lambda:provided,'_probe_target':lambda *a:(False,None) if provided else (True,target()),'_retry':lambda fn,**kw:fn(),'_aiter_op':oracle}
    _harness_functions(t,{function,'_mean_ms','_norm_max_err','_tensor_ok','_compare','_checked_layernorm_output','_checked_layernorm_pair','_compare_layernorm_output','_verify_layernorm_timed'},ns)
    if behavior=='correct':
        report=ns[function](verbose=False)
        if function=='run_benchmark':report=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert report[0]['timed_output_correctness']==report[0]['replay_correctness']=='PASS'
        assert calls==[(10,100)]*(2 if provided else 3)
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    for value,original in zip(inputs,originals):assert torch.equal(value,original)


def test_layernorm_add_original_or_gate_and_residual_contract():
    import torch
    t=ROOT/'tasks/torch2flydsl/layernorm2d_with_add_kernel';checks=module(t/'scripts/replay_checks.py')
    ns={'require_tensor_contract':checks.require_tensor_contract,'REL_TOL':.01,'PASS_PCT':99.9};_harness_functions(t,{'_tensor_ok','_compare','_checked_layernorm_pair','_compare_layernorm_output'},ns)
    # 1/2000 deliberately large errors fail the normalized bound but satisfy
    # the ORIGINAL 99.9% gate. Requiring both conditions would change the task.
    actual=(torch.ones((1,2000),dtype=torch.bfloat16),torch.ones((1,2000),dtype=torch.bfloat16));truth=tuple(v.clone() for v in actual);truth[0][0,0]=10
    ok,error,pct=ns['_tensor_ok'](actual[0],truth[0]);assert ok and error>.01 and pct>=99.9
    ns['_compare_layernorm_output'](actual,truth)
    bad=(actual[0],torch.zeros_like(actual[1]))
    with pytest.raises(AssertionError,match='Numerical mismatch'):ns['_compare_layernorm_output'](bad,truth)
    for bad in [(actual[0],), (actual[0],actual[1][:,:1]),(actual[0],actual[1].float()),(actual[0],torch.full_like(actual[1],float('nan')))]:
        with pytest.raises(AssertionError):ns['_compare_layernorm_output'](bad,actual)


def test_layernorm_original_numeric_inputs_and_timing_preserved():
    hashes={'layernorm2d_kernel': {'_make_inputs': '96047c11ef94bb5e3124f1cd5de22b1f6959ac99d9336927143baca4baaa21c3', '_norm_max_err': '750a488ebe381cf762ba31a41890c0fe3dd87c2dc703955ad218dd85a2d9862e', 'run_correctness': '844bca314b686fd2bb17b421fe25d38253f7987c9a8eb1a83eaf7434ff9e1465', 'run_benchmark': '34ec3583d9970fae7690374cb18b28f3087da202dfd8fff8fe5fd5320f975b27', 'arena_benchmark': '7db8114697de0bf8efb6774c9bae5be11c662b0d760095ff9051c5890d8a945b'}, 'layernorm2d_with_add_kernel': {'_make_inputs': 'a04871a09de4827358e0bc8db07faff135fb9e0d1a855fe01c5d81635f7df318', '_aiter_op': '82f1bc3a8fcdb3588d6ed5f2fa17f3dfb8e152846a57b1097394c366f6bb7f68', '_tensor_ok': 'c00fb86d889b0e7ce78e148f7a61bc1ce6fb16b2817a572f1229f87aa577942f', '_compare': 'e0bc09cd3e2d00997e194b1366075a6908596af9ee8629946f1a0491178d8f0f', 'run_correctness': 'f706a5821b6653d4629be4af9244c4d1da9df5fdf581a44ac93bc28c2b431c46', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': 'e2dc0275cc3ae5807f6053bbf4379776d51e6cea87c3ec9f5ac9e97cdcc9c584', 'arena_benchmark': '72452bbe2421a0564c40d125346bb07b7cc69ec3b564ef3705f160c6155d71b1'}}
    for name,functions in hashes.items():
        tree=ast.parse((ROOT/'tasks/torch2flydsl'/name/'test_kernel_harness.py').read_text())
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                restored=_RemoveLayernormChecks().visit(fn)
                assert hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_STANDARD_QUANT_NAMES=['per_tensor_fp8_quant_kernel', 'per_token_fp8_quant_kernel', 'per_1x128_fp8_quant_kernel', 'per_token_i8_quant_kernel']


class _RemoveStandardQuantChecks(_RemoveAddedReplayChecks):
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call):
            if getattr(call.func,'id',None)=='_checked_quant_pair':return None
            if isinstance(call.func,ast.Attribute) and call.func.attr=='update' and call.args and isinstance(call.args[0],ast.Call) and getattr(call.args[0].func,'id',None)=='validate':return None
        return super().visit_Expr(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None)=='validate':return None
        return super().visit_Assign(node)
    def visit_Call(self,node):
        if getattr(node.func,'id',None)=='_checked_quant_pair':return self.visit(node.args[0])
        if getattr(node.func,'id',None)=='_mean_ms':node.keywords=[x for x in node.keywords if x.arg!='validate']
        return super().visit_Call(node)
    def visit_FunctionDef(self,node):
        if node.name=='_mean_ms':node.args.kwonlyargs=[];node.args.kw_defaults=[]
        return self.generic_visit(node)


@pytest.mark.parametrize('name',_STANDARD_QUANT_NAMES)
@pytest.mark.parametrize('function',['run_benchmark','arena_benchmark'])
@pytest.mark.parametrize('provided',[True,False])
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached_codes','cached_scale','input_modified','shape','code_dtype','scale_dtype','nonfinite'])
def test_standard_quantizer_actual_measured_pair_and_replay(name,function,provided,behavior,monkeypatch,tmp_path):
    import math,types,torch
    t=ROOT/'tasks/torch2flydsl'/name;checks=module(t/'scripts/replay_checks.py');real_model=module(t/'model.py');oracle=real_model.Model()
    # CPU plumbing test: real original quantization produces the expected
    # codes/scales for each changed input. Deliberately wrong paths must fail;
    # full GPU task checks still compare independently against AITER.
    inp=torch.linspace(-8,7,512,dtype=torch.float32).reshape(2,256).to(torch.bfloat16);original=inp.clone();cached=oracle(inp);phase={'name':'setup'}
    def compute(is_model):
        y,scale=oracle(inp)
        if is_model==provided:
            if behavior==phase['name']+'_wrong':y.view(torch.uint8).zero_()
            if phase['name']=='replay':
                if behavior=='cached_codes':y=cached[0].clone()
                if behavior=='cached_scale':scale=cached[1].clone()
                if behavior=='input_modified':inp.add_(1)
            if phase['name']=='measured':
                if behavior=='shape':scale=scale.reshape(-1) if scale.ndim==2 else scale.reshape(1,1)
                if behavior=='code_dtype':y=y.view(torch.uint8)
                if behavior=='scale_dtype':scale=scale.to(torch.bfloat16)
                if behavior=='nonfinite':scale.fill_(float('nan'))
        return y,scale
    class Model:
        def to(self,*a):return self
        def __call__(self,*a):return compute(True)
    mmod=types.SimpleNamespace(Model=Model,get_init_inputs=lambda:[],_FP8_DTYPE=getattr(real_model,'_FP8_DTYPE',None));kmod=types.SimpleNamespace(**{'flydsl_'+name.removesuffix('_kernel'):lambda *a:compute(False)})
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));phase['name']='measured';timed_run.outputs=fn();timed_run.bound=True;phase['name']='setup'
        def replay():
            phase['name']='replay'
            try:return fn()
            finally:phase['name']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_unchanged':checks.require_unchanged,
        'CODE_TOL':1,'SCALE_RTOL':.001,'_aiter_op':oracle,'_make_inputs':lambda shape:inp if name=='per_token_i8_quant_kernel' else (inp,),
        '_load_module':lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod,
        '_KERNEL_DIR':str(tmp_path),'MODEL_FILE':'model.py','KERNEL_FILE':'kernel.py','KERNEL_ENTRY':'flydsl_'+name.removesuffix('_kernel'),
        'SHAPES':[{'name':'controlled','m':2,'n':256}], 'math':math,'json':json,'Path':Path}
    _harness_functions(t,{function,'_mean_ms','_compare','_checked_quant_pair','_compare_quant_outputs','_quant_replay_validator'},ns)
    if behavior=='correct':
        report=ns[function](verbose=False)
        if function=='run_benchmark':report=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert report[0]['timed_output_correctness']==report[0]['replay_correctness']=='PASS'
        assert calls==[(10,100)]*(2 if provided else 3)
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(inp,original)


@pytest.mark.parametrize('name',_STANDARD_QUANT_NAMES)
def test_standard_quantizer_keeps_one_code_step_and_original_scale_gate(name):
    import torch
    t=ROOT/'tasks/torch2flydsl'/name;mmod=module(t/'model.py');x=torch.full((2,256),2.,dtype=torch.bfloat16)
    expected=list(mmod.Model()(x));ns={'CODE_TOL':1,'SCALE_RTOL':.001};_harness_functions(t,{'_compare','_checked_quant_pair','_compare_quant_outputs'},ns)
    actual=[v.clone() for v in expected]
    # Avoid endpoint wrapping: compare two adjacent finite code values.
    if name=='per_token_i8_quant_kernel':expected[0].fill_(2);actual[0].fill_(3)
    else:expected[0].view(torch.uint8).fill_(56);actual[0].view(torch.uint8).fill_(57)
    ns['_compare_quant_outputs'](tuple(actual),tuple(expected),(x,),mmod)
    actual[1].mul_(1.01)
    with pytest.raises(AssertionError,match='Numerical mismatch'):ns['_compare_quant_outputs'](tuple(actual),tuple(expected),(x,),mmod)
    actual[1].copy_(expected[1]);actual[0].view(torch.uint8).add_(1)
    with pytest.raises(AssertionError,match='Numerical mismatch'):ns['_compare_quant_outputs'](tuple(actual),tuple(expected),(x,),mmod)


def test_standard_quantizers_preserve_original_numerics_inputs_and_timing():
    hashes={'per_tensor_fp8_quant_kernel': {'_make_inputs': '17800253aebd6dc6f79a2f102d9127e1bf787de9708ee39e29e408d421377b3d', '_aiter_op': '337fc0e0288af0d8c6e08c202c9815167a08a54a62d99721132814b3d6f040c7', '_compare': '7370859da62e853ba8a197c5ba6e4f07f7f41f1f815a55c24cb4c54ed73d3390', 'run_correctness': 'c7991fb82df9aee3ff159b03aef748e6d72ee670fbd2e7eaa46e9c04c6d53d01', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': '90bfab17dbc2c9bf422a8deca038a9ba100c6791af0ab642cb95c09c66a4b538', 'arena_benchmark': 'ca5e3fcd67f5d57c95ee648c27b44432f7d32cb354dbc8c1eaccc897d81f4df4'}, 'per_token_fp8_quant_kernel': {'_make_inputs': '17800253aebd6dc6f79a2f102d9127e1bf787de9708ee39e29e408d421377b3d', '_aiter_op': 'b7962f1966999d19c5161d0d50e996868b44268d6e556b8fabc8eb6a4a4f0139', '_compare': '7370859da62e853ba8a197c5ba6e4f07f7f41f1f815a55c24cb4c54ed73d3390', 'run_correctness': '09320c212eec2c7c965d155721e4c0037ff143ed3b470ef83fcdd3b745b45d05', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': '71bba4b5222624e432ddbec8bd71b1df401e07b7785c58d2fffde407847a181c', 'arena_benchmark': '4b10fcd809f486c9cd23059979c72f62c292fe01f7a8d34296126ac5119ed493'}, 'per_1x128_fp8_quant_kernel': {'_make_inputs': '17800253aebd6dc6f79a2f102d9127e1bf787de9708ee39e29e408d421377b3d', '_aiter_op': '3ebcae67fe0961a1b266815022db4de2222e31a9ebce95b4b052ead1fc38286c', '_compare': '7370859da62e853ba8a197c5ba6e4f07f7f41f1f815a55c24cb4c54ed73d3390', 'run_correctness': '9f141b10b3eae32f373b1605c510dda09ca44e70432155362cfb102d3716bbc9', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': '2ac80fb49c0a10f04026bf05926c1ecce39e9bb173251c5c20cb7355bd7e2ca8', 'arena_benchmark': 'fe43be52eeae8e690f57386df4ba9a0a5aed3a8c3956c33c008d937898ea7bd4'}, 'per_token_i8_quant_kernel': {'_make_inputs': 'b189ce05262cdd417908cc9746babb21a576bc68927754e8beb7e1e36c9e2038', '_aiter_op': 'e1367df215a4f2c7d0ffb3b9b2a52f9b13e6e7baafd7babad17d814b35a765da', '_compare': '26b0aed52720cd3ac91782410828b02e1947aed8ed614f5b2ab148c5dabb5be2', 'run_correctness': 'ee742cf41d63d6b6b1fe879acdcd45b08907e722b13c982c434b75c56d533aee', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': '49e51bb6201effc168702a851ca698e99d3c3ae2daf77eb9cb8a3ee549519b9d', 'arena_benchmark': 'a9f07227c9f5d46eb6cd3398115fa4e0b0bbe622815cd0deb44eba127940fb91'}}
    for name,functions in hashes.items():
        tree=ast.parse((ROOT/'tasks/torch2flydsl'/name/'test_kernel_harness.py').read_text())
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                restored=_RemoveStandardQuantChecks().visit(fn)
                assert hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


@pytest.mark.parametrize("function", ["run_benchmark", "arena_benchmark"])
@pytest.mark.parametrize("provided", [False, True])
@pytest.mark.parametrize("behavior", ["correct", "measured_wrong", "replay_wrong", "cached", "input_modified", "bad_ids", "nonfinite"])
def test_biased_grouped_routing_actual_measured_and_replay( function, provided, behavior, monkeypatch, tmp_path):
    import math
    import types
    import torch
    task=ROOT/"tasks/torch2flydsl"/"moe_biased_grouped_topk_kernel"
    checks=module(task/"scripts/replay_checks.py")
    mmod=module(task/"model.py")
    model = mmod.Model(8,2,4,2,True,2.5)
    with torch.no_grad():model.correction_bias.copy_(torch.tensor([-.1,.2,.05,.1,.15,-.2,.05,.25]))
    original_bias = model.correction_bias.detach().clone()
    gating=torch.tensor([[-2.,1.,-1.,2.,0.,3.,-3.,4.],[4.,-1.,3.,-2.,2.,-3.,1.,0.]],dtype=torch.float32)
    original=gating.clone();cached=model(gating)
    phase={"value":"setup"}
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    def compute():
        w,ids=model(gating)
        if behavior==phase["value"]+"_wrong":w=w+1.
        if phase["value"]=="replay":
            if behavior=="cached":w,ids=(x.clone() for x in cached)
            if behavior=="input_modified":gating.add_(1)
        if phase["value"]=="measured":
            if behavior=="bad_ids":ids[:,0]=-1
            if behavior=="nonfinite":w[:,0]=float("nan")
        return w,ids
    def aiter_op(gating,bias,w,ids,*args,**kwargs):
        actual_w,actual_ids=compute();w.copy_(actual_w);ids.copy_(actual_ids)
    monkeypatch.setitem(sys.modules,"aiter",types.SimpleNamespace(biased_grouped_topk_hip=aiter_op))
    kmod=types.SimpleNamespace(flydsl_biased_grouped_topk=lambda *args:compute())
    class Collector:
        bound=False
        outputs=None
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run=None):
        calls.append((warmup,repetition,timed_run is not None))
        phase["value"]="measured";out=fn()
        if timed_run is not None:
            timed_run.outputs=out;timed_run.bound=True
            def replay():
                phase["value"]="replay";result=fn();phase["value"]="setup";return result
            timed_run.rerun=replay
        phase["value"]="setup"
        return .1,{"benchmark_method":"cuda_graph","benchmark_timed_run_kind":"captured_graph"}
    ns={"TimedRun":Collector,"benchmark_cuda_graph_or_events":benchmark,"require_unchanged":checks.require_unchanged,
        "math":math,"json":json,"Path":Path,"_KERNEL_DIR":str(tmp_path),"MODEL_FILE":"model.py","KERNEL_FILE":"kernel.py",
        "_TIE_TOL":1e-4,"_WEIGHT_ATOL":1e-2,"_BIAS_ID_ERR_TOL":.05,"REL_TOL":1e-2,
        "SHAPES":[{"name":"controlled","tokens":2,"experts":8,"topk":2,"num_expert_group":4,"topk_group":2,"route_scale":2.5,"renormalize":True}],
        "_load_module":lambda directory,filename,alias:mmod if filename=="model.py" else (None if provided else kmod),
        "_build_model":lambda *args:(model,gating),"_retry":lambda fn,**kwargs:fn()}
    _harness_functions(task,{function,"_require_routing_contract","_routing_reference","_verify_routing_timed","_compare_routing"},ns)
    if behavior=="correct":
        result=ns[function](verbose=False)
        if function=="run_benchmark":result=json.loads((tmp_path/"build/performance_report.json").read_text())
        assert result[0]["timed_output_correctness"]==result[0]["replay_correctness"]=="PASS"
        assert result[0]["replay_checked_outputs"]==["weights","expert_ids"]
        assert calls==[(0,100,True),(0,100,False)]
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(gating,original)

    if original_bias is not None: assert torch.equal(model.correction_bias, original_bias)


@pytest.mark.parametrize("bad", ["shape", "weight_dtype", "id_dtype", "device", "out_of_range", "duplicate", "nonfinite"])
def test_biased_grouped_routing_contract_rejects_invalid_outputs( bad):
    import torch
    task=ROOT/"tasks/torch2flydsl"/"moe_biased_grouped_topk_kernel"
    ns={}
    _harness_functions(task,{"_require_routing_contract"},ns)
    gating=torch.zeros(2,4,dtype=torch.bfloat16)
    w=torch.ones(2,2,dtype=torch.float32);ids=torch.tensor([[1,2],[2,3]],dtype=torch.int32)
    ns["_require_routing_contract"](w,ids,gating,2)
    if bad=="shape":w=w[:1]
    if bad=="weight_dtype":w=w.bfloat16()
    if bad=="id_dtype":ids=ids.long()
    if bad=="device":w=w.to("meta")
    if bad=="out_of_range":ids[0,0]=4
    if bad=="duplicate":ids[0,0]=ids[0,1]
    if bad=="nonfinite":w[0,0]=float("nan")
    with pytest.raises(AssertionError):ns["_require_routing_contract"](w,ids,gating,2)


_BIASED_GROUPED_ORIGINAL_FUNCTIONS = {'_make_gating': '1ccf0c7f77ad760039746631cb1c44c10cab0fadec179ee31014f43d06bf55da', '_build_model': '9a013352090c4df22a6384224b63461e67ae4a87ce781fa66f8b8edb41ed7afc', '_aiter_grouped': 'd7c1f3ea52ef575594530ff2537e1552b735e98f094d88adce8e2a9aa37ebb46', '_compare_routing': '85889d660dbf6741317e8f6ce46fc02a7693e1197d96b25a5e0dcf0dd9e5ed30', 'run_correctness': '1a6dc48db9799bbde6206651c1600a14410bced99b0fb07b260650d16ccf7897', 'run_benchmark': '2ec3211da94b77e9e5eb1f90a9add21390e3a103618a618b24a4916c923407e2', 'arena_benchmark': '75cef4eddb3c911259a6835be33147226b8144360103e50f1cffbef0f9435432'}


def test_biased_grouped_preserves_original_tie_policy_inputs_and_timing():
    task=ROOT/'tasks/torch2flydsl/moe_biased_grouped_topk_kernel'
    tree=ast.parse((task/'test_kernel_harness.py').read_text())
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name in _BIASED_GROUPED_ORIGINAL_FUNCTIONS:
            restored=_RemoveAddedReplayChecks().visit(fn)
            assert hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()==_BIASED_GROUPED_ORIGINAL_FUNCTIONS[fn.name],fn.name


def test_biased_grouped_scalar_known_answer_and_comparator_negative_control():
    import math,torch
    task=ROOT/'tasks/torch2flydsl/moe_biased_grouped_topk_kernel';mmod=module(task/'model.py')
    x=torch.tensor([[-2.,1.,-1.,2.,0.,3.,-3.,4.],[4.,-1.,3.,-2.,2.,-3.,1.,0.]])
    bias=torch.tensor([-.1,.2,.05,.1,.15,-.2,.05,.25])
    expected_w=[];expected_id=[]
    for row in x.tolist():
        scores=[1/(1+math.exp(-v)) for v in row];choice=[v+b for v,b in zip(scores,bias.tolist())]
        groups=sorted(range(4),key=lambda g:sum(choice[2*g:2*g+2]),reverse=True)[:2]
        ids=sorted([i for g in groups for i in (2*g,2*g+1)],key=choice.__getitem__,reverse=True)[:2]
        denom=sum(scores[i] for i in ids);expected_id.append(ids);expected_w.append([2.5*scores[i]/denom for i in ids])
    expected=(torch.tensor(expected_w),torch.tensor(expected_id,dtype=torch.int32))
    actual_w,actual_ids,selection=mmod.grouped_route(x,bias,2,True,4,2,2.5)
    ns={'_TIE_TOL':1e-4};_harness_functions(task,{'_compare_routing'},ns);compare=ns['_compare_routing']
    mismatch,error=compare(*expected,actual_w,actual_ids,selection,2)
    assert mismatch==0 and error<1e-6
    wrong_ids=actual_ids.clone();wrong_ids[:,0]=0
    assert compare(*expected,actual_w,wrong_ids,selection,2)[0]>0
    assert compare(*expected,actual_w+1,actual_ids,selection,2)[1]>.01
    # Keep genuine boundary-tie acceptance and reject an expert away from it.
    weights=torch.tensor([[1.25,1.25]]);ids=torch.tensor([[0,1]],dtype=torch.int32);sel=torch.tensor([[.8,.5,.5,.1]])
    assert compare(weights,ids,weights,torch.tensor([[0,2]],dtype=torch.int32),sel,2)==(0,0.)
    assert compare(weights,ids,weights,torch.tensor([[0,3]],dtype=torch.int32),sel,2)[0]==1


@pytest.mark.parametrize('fault',['numerical','dtype','launch','nonfinite'])
@pytest.mark.parametrize('action',['correctness','performance'])
def test_a8w8_gemm_real_case_evidence_and_failed_timing_precheck(fault,action,monkeypatch,capsys):
    import torch,types
    from src.task_protocol import parse_command_result
    t=ROOT/'tasks/torch2flydsl/gemm_a8w8_kernel';runtime=module(t/'task_runtime.py');actions=module(t/'scripts/task_actions.py');checks=module(t/'scripts/replay_checks.py')
    candidate_checks=module(t/'scripts/candidate_checks.py');monkeypatch.setitem(sys.modules,'scripts.candidate_checks',candidate_checks)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    h=types.ModuleType('controlled_gemm_harness');h.__dict__.update(ARENA_PROVIDED_BASELINE=True,
        _KERNEL_DIR='.',MODEL_FILE='model.py',KERNEL_FILE='kernel.py',KERNEL_ENTRY='flydsl_gemm_a8w8',
        SHAPES=actions.EXPECTED_CASES,TOL=.01,require_unchanged=checks.require_unchanged)
    # Small CPU tensors exercise the original complete five-case orchestration;
    # numeric GPU checks and full shapes remain in the actual frozen campaign.
    class Model:
        def to(self,*a):return self
        def eval(self):return self
        def __call__(self,a,w):return a.clone()
    mmod=types.SimpleNamespace(Model=Model,quantize_a8w8=lambda a,w:(a,None,w,None));state={'case':-1}
    def inputs(*args):
        state['case']+=1
        return torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16),torch.ones((2,2),dtype=torch.bfloat16)
    def baseline(a,w,*args):
        if state['case']==0:
            if fault=='launch':raise RuntimeError('controlled GPU launch failure')
            if fault=='dtype':return a.float()
            if fault=='nonfinite':return torch.full_like(a,float('nan'))
            return a+1
        return a.clone()
    monkeypatch.setitem(sys.modules,'aiter',types.SimpleNamespace(gemm_a8w8=baseline))
    h._make_inputs=inputs;h._retry=lambda fn,**kw:fn();h._load_module=lambda directory,filename,alias:mmod if filename=='model.py' else None
    _harness_functions(t,{'run_correctness','_checked_quant_gemm_output','_norm_worst','_record_gemm_case'},h.__dict__)
    real_loader=runtime.load_module
    monkeypatch.setattr(runtime,'load_module',lambda key,path:h if key=='arena_harness' else actions if key=='arena_task_actions' else real_loader(key,path))
    assert runtime.run(['baseline',action])==1
    output=capsys.readouterr().out
    result=parse_command_result(output,role='baseline',action=action,returncode=1)
    report=result.to_mapping();assert report['status']=='FAIL'
    if action=='correctness':
        rows=report['cases'];assert [row['status'] for row in rows]==['FAIL','PASS','PASS','PASS','PASS']
    else:
        assert all(row['status']=='FAIL' and 'execution_time_ms' not in row for row in report['cases'])
        assert report['metadata']['failed_stage']=='correctness_precheck'
        rows=list(report['metadata']['correctness_precheck_cases'].values())
    first=rows[0]
    if fault=='numerical':
        assert first['failure_kind']=='numerical_mismatch'
        assert first['metadata']['baseline_max_abs_error']==1.
        assert first['metadata']['baseline_normalized_max_error']==.25
        assert first['metadata']['tolerance']==.01
    else:assert first['failure_kind']=='execution_or_contract_error'
    assert all(row['status']=='PASS' and row['metadata']['baseline_normalized_max_error']==0 for row in rows[1:])


@pytest.mark.parametrize('name',_FMOE_REPLAY_NAMES)
def test_fmoe_failed_correctness_precheck_cannot_emit_passing_performance_rows(name,monkeypatch,capsys):
    import types
    from src.task_protocol import parse_command_result
    t=ROOT/'tasks/torch2flydsl'/name;runtime=module(t/'task_runtime.py')
    evidence={'case_0000':{'status':'FAIL','failure_kind':'numerical_mismatch','reason':'controlled numerical failure'},
              'case_0001':{'status':'PASS'},'case_0002':{'status':'PASS'}}
    def fail(h):
        exc=AssertionError('correctness precheck failed');exc.arena_case_results=evidence;raise exc
    actions=types.SimpleNamespace(check=fail,select_role=lambda *a:None)
    monkeypatch.setattr(runtime,'load_module',lambda key,path:actions if key=='arena_task_actions' else types.SimpleNamespace())
    assert runtime.run(['baseline','performance'])==1
    result=parse_command_result(capsys.readouterr().out,role='baseline',action='performance',returncode=1)
    assert not result.passed and all(row['status']=='FAIL' and 'execution_time_ms' not in row for row in result.cases)
    assert result.metadata['correctness_precheck_cases']==evidence


@pytest.mark.parametrize('function',['run_benchmark','arena_benchmark'])
@pytest.mark.parametrize('provided',[True,False])
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached_codes','cached_scale','input_modified','shape','code_dtype','scale_dtype','nonfinite'])
def test_mxfp4_actual_measured_pair_and_exact_replay(function,provided,behavior,monkeypatch,tmp_path):
    import math,types,torch
    name='quant_mxfp4_kernel'
    t=ROOT/'tasks/torch2flydsl'/name;checks=module(t/'scripts/replay_checks.py');real_model=module(t/'model.py');oracle=real_model.Model()
    # CPU plumbing test: real original quantization produces the expected
    # codes/scales for each changed input. Deliberately wrong paths must fail;
    # full GPU task checks still compare independently against AITER.
    inp=torch.linspace(-8,7,512,dtype=torch.float32).reshape(2,256).to(torch.bfloat16);original=inp.clone();cached=oracle(inp);phase={'name':'setup'}
    def compute(is_model):
        y,scale=oracle(inp)
        if is_model==provided:
            if behavior==phase['name']+'_wrong':y.view(torch.uint8).zero_()
            if phase['name']=='replay':
                if behavior=='cached_codes':y=cached[0].clone()
                if behavior=='cached_scale':scale=cached[1].clone()
                if behavior=='input_modified':inp.add_(1)
            if phase['name']=='measured':
                if behavior=='shape':scale=scale.reshape(-1) if scale.ndim==2 else scale.reshape(1,1)
                if behavior=='code_dtype':y=y.view(torch.int8)
                if behavior=='scale_dtype':scale=scale.view(torch.int8)
                if behavior=='nonfinite':scale.view(torch.uint8).fill_(255)
        return y,scale
    class Model:
        def to(self,*a):return self
        def __call__(self,*a):return compute(True)
    mmod=types.SimpleNamespace(Model=Model,get_init_inputs=lambda:[],_FP4X2=real_model._FP4X2,_FP8_E8M0=real_model._FP8_E8M0);kmod=types.SimpleNamespace(**{'flydsl_'+name.removesuffix('_kernel'):lambda *a,**kw:compute(False)})
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));phase['name']='measured';timed_run.outputs=fn();timed_run.bound=True;phase['name']='setup'
        def replay():
            phase['name']='replay'
            try:return fn()
            finally:phase['name']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_unchanged':checks.require_unchanged,
        'GROUP_SIZE':32,'_aiter_op':oracle,'_make_inputs':lambda shape:inp if name=='per_token_i8_quant_kernel' else (inp,),
        '_load_module':lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod,
        '_KERNEL_DIR':str(tmp_path),'MODEL_FILE':'model.py','KERNEL_FILE':'kernel.py','KERNEL_ENTRY':'flydsl_'+name.removesuffix('_kernel'),
        'SHAPES':[{'name':'controlled','m':2,'n':256}], 'math':math,'json':json,'Path':Path}
    _harness_functions(t,{function,'_mean_ms','_compare','_checked_quant_pair','_compare_quant_outputs','_quant_replay_validator'},ns)
    if behavior=='correct':
        report=ns[function](verbose=False)
        if function=='run_benchmark':report=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert report[0]['timed_output_correctness']==report[0]['replay_correctness']=='PASS'
        assert calls==[(10,100)]*(2 if provided else 3)
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(inp,original)


@pytest.mark.parametrize('wrong',['one_code_bit','one_scale_bit','scale_nan','shape','device','code_dtype'])
def test_mxfp4_original_byte_exact_gate_and_output_contract(wrong):
    import torch
    t=ROOT/'tasks/torch2flydsl/quant_mxfp4_kernel';mmod=module(t/'model.py');x=torch.linspace(-4,6,128).reshape(2,64).to(torch.bfloat16)
    expected=mmod.Model()(x);actual=[v.clone() for v in expected];ns={'GROUP_SIZE':32}
    _harness_functions(t,{'_compare','_checked_quant_pair','_compare_quant_outputs'},ns)
    ns['_compare_quant_outputs'](actual,expected,(x,),mmod)
    if wrong=='one_code_bit':actual[0].view(torch.uint8)[0,0]^=1
    if wrong=='one_scale_bit':actual[1].view(torch.uint8)[0,0]^=1
    if wrong=='scale_nan':actual[1].view(torch.uint8)[0,0]=255
    if wrong=='shape':actual[0]=actual[0].reshape(-1)
    if wrong=='device':actual[0]=actual[0].to('meta')
    if wrong=='code_dtype':actual[0]=actual[0].view(torch.int8)
    with pytest.raises(AssertionError):ns['_compare_quant_outputs'](actual,expected,(x,),mmod)


def test_mxfp4_original_math_inputs_and_timing_preserved():
    hashes={'_make_inputs': '17800253aebd6dc6f79a2f102d9127e1bf787de9708ee39e29e408d421377b3d', '_aiter_op': 'b0cf4572a62ec1dd55f0d724b867f34f659fee1168defdee1e5ec18ff5a91fd1', '_compare': '7c1418c398e1df2b3e5ec14b1ab311b37694b2fc766b52dfebb452a1033d6acb', 'run_correctness': '204f18a5d399f889da36f10ae1056a4ba9d80854e1b7d05d3fb89b5e7b424fc9', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': '99ad3efe2c60267a7aebbc308d6d4885b8986ec1d1a84f480c136c0338478889', 'arena_benchmark': '3c4c85b24ef9f8f6aef8f8ae22207c0882054c964c8a3bf4bd23c4ad6befce0d'}
    task=ROOT/'tasks/torch2flydsl/quant_mxfp4_kernel';tree=ast.parse((task/'test_kernel_harness.py').read_text())
    for fn in tree.body:
        if isinstance(fn,ast.FunctionDef) and fn.name in hashes:
            normalized=_RemoveStandardQuantChecks().visit(fn)
            assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==hashes[fn.name],fn.name
    cfg=yaml.safe_load((task/'config.yaml').read_text())
    assert [e['symbol'] for e in cfg['candidate']['entrypoints']]==['flydsl_quant_mxfp4']
    assert cfg['platform_support']['required_arch']=='gfx950'


_TRITON_BATCHED_NAMES=['batched_gemm_a8w8','batched_gemm_bf16']


class _RemoveTritonBatchedChecks(_RemoveAddedReplayChecks):
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None) in {'protected_inputs','replay_validate'}:return None
        return super().visit_Assign(node)
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call, ast.Call) and getattr(call.func, 'id', None) in {'_checked_scaled_gemm_output', '_checked_pair_output'}:return None
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='update' and call.args and isinstance(call.args[0],ast.Call) and getattr(call.args[0].func,'id',None)=='replay_validate':return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('name,phase,behavior', [
    (name,phase,behavior)
    for name in _TRITON_BATCHED_NAMES for phase in ['correctness','benchmark']
    for behavior in ['correct','shape','dtype','device','nan','input_modified','weight_modified','scale_modified','bias_modified','measured_wrong','replay_wrong','cached']
    if not (name.endswith('bf16') and behavior=='scale_modified')
    and not (phase=='benchmark' and behavior=='bias_modified')
    and not (phase=='correctness' and behavior in {'measured_wrong','replay_wrong','cached'})
])
def test_triton_batched_original_numeric_gate_and_measured_replay(name,phase,behavior,monkeypatch,tmp_path):
    import torch,types,math
    t=ROOT/'tasks/triton2flydsl/aiter'/name;checks=module(t/'scripts/replay_checks.py');quant=name.endswith('a8w8')
    x=torch.tensor([[[1.,2.],[1.,2.]],[[3.,4.],[3.,4.]]],dtype=torch.int8 if quant else torch.bfloat16)
    w=torch.tensor([[[3.,4.],[5.,6.]],[[4.,2.],[7.,1.]]],dtype=x.dtype)
    xs=torch.tensor([[[.5],[.25]],[[.75],[.625]]]);ws=torch.tensor([[[.125,.25]],[[.5,.75]]]);bias=torch.ones((2,1,2),dtype=torch.bfloat16)
    originals=tuple(v.clone() for v in (x,w,xs,ws,bias));state={'value':'setup'}
    ns={};_harness_functions(t,{'_torch_ref'},ns);oracle=ns['_torch_ref']
    def args(with_bias=False):return (x,w,xs,ws,bias if with_bias else None,torch.bfloat16) if quant else (x,w,bias if with_bias else None,torch.bfloat16)
    cached=oracle(*args())
    def compute(*values):
        out=oracle(*values);active=phase=='correctness' or state['value']=='measured'
        if active:
            if behavior=='shape':out=out[:,:1]
            if behavior=='dtype':out=out.float()
            if behavior=='device':out=out.to('meta')
            if behavior=='nan':out.fill_(float('nan'))
            if behavior=='input_modified':x.add_(1)
            if behavior=='weight_modified':w.add_(1)
            if behavior=='scale_modified':xs.mul_(.5)
            if behavior=='bias_modified' and values[-2] is not None:values[-2].add_(1)
        if behavior==state['value']+'_wrong':out.add_(10)
        if state['value']=='replay' and behavior=='cached':out=cached.clone()
        return out
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['value']='measured';timed_run.outputs=fn();timed_run.bound=True;state['value']='setup'
        def replay():
            state['value']='replay'
            try:return fn()
            finally:state['value']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,require_tensor_contract=checks.require_tensor_contract,
        require_unchanged=checks.require_unchanged,verify_timed_run=checks.verify_timed_run,allclose_output=checks.allclose_output,
        _HERE=str(tmp_path),Path=Path,json=json,math=math,WARMUP=10,ITERS=100,
        TEST_SHAPES=[{'name':'controlled','B':2,'M':2,'N':2,'K':2}],_load_source=lambda:types.SimpleNamespace(**{name:compute}),
        _make_inputs=lambda *params:args(params[-1])[:-1])
    _harness_functions(t,{'run_correctness','run_benchmark','_batched_replay_validator'},ns)
    if phase=='correctness':assert ns['run_correctness'](verbose=False) is (behavior=='correct')
    elif behavior=='correct':
        report=ns['run_benchmark'](verbose=False)
        assert report[0]['timed_output_correctness']==report[0]['replay_correctness']=='PASS'
        assert calls==[(0,100)]
    else:
        with pytest.raises(AssertionError):ns['run_benchmark'](verbose=False)
    if phase=='benchmark' and behavior not in {'input_modified','weight_modified','scale_modified'}:
        for a,b in zip((x,w,xs,ws,bias),originals):assert torch.equal(a,b)


@pytest.mark.parametrize('name',_TRITON_BATCHED_NAMES)
def test_triton_batched_final_candidate_audit_does_not_instrument_initial_triton(name,monkeypatch):
    import torch,types
    t=ROOT/'tasks/triton2flydsl/aiter'/name;audit=module(t/'scripts/candidate_checks.py');actions=module(t/'scripts/task_actions.py')
    x=torch.tensor([1.,2.]);source=lambda:types.SimpleNamespace(**{name:lambda value:value.square()})
    h=types.SimpleNamespace(ENTRY=name,_load_source=source,TEST_SHAPES=actions.EXPECTED_SHAPES)
    actions.select_role(h,'baseline',False)
    with audit.audit_candidate_calls(h):torch.testing.assert_close(getattr(h._load_source(),name)(x),x.square())
    monkeypatch.setenv('ARENA_EVAL_PHASE','final_candidate');actions.select_role(h,'candidate',False)
    with pytest.raises(RuntimeError,match='non-preparation PyTorch operation'):
        with audit.audit_candidate_calls(h):getattr(h._load_source(),name)(x)
    assert h._load_source is source
    # Restored ordinary timing loader receives no dispatch profiler.
    torch.testing.assert_close(getattr(h._load_source(),name)(x),x.square())


def test_triton_batched_all_original_arithmetic_cases_and_timing_retained():
    hashes={'batched_gemm_a8w8': {'_make_inputs': 'aeb4337e39fb87caa6ba38a24255d94d3bee67811bb6fdc4e12ff6018d647f6f', '_torch_ref': 'd63c9e3286b09330e4390662ca6e958491a1305717360e9223896bb18e00e582', 'run_correctness': 'ba9465885ea684b71bb6fd3660ec11f309fbf2825ce5770dc50febb13a1105bd', 'run_benchmark': 'cb1b2218d06139101d04225a7ad6efcceabd74cbdefb68ba9b5f20173edbece4'}, 'batched_gemm_bf16': {'_make_inputs': '042d7e1122c3d72947342b4faa216cbaddaa17348ca65e7c9e74e3b4aa26a5f9', '_torch_ref': '699640c43ffad755291064fe51c1c5d5da5645164be088ae25fe2c23d082e9a8', 'run_correctness': '4822a2e8973b8f169c8d42a91efcd136db602c323f27ef38ea531ab0ad039050', 'run_benchmark': 'f2f8b72cf696760ac642475c7818e6f047d2dd65c9f0345e148cf6751c337000'}}
    for name,functions in hashes.items():
        tree=ast.parse((ROOT/'tasks/triton2flydsl/aiter'/name/'test_kernel_harness.py').read_text())
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveTritonBatchedChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_TRITON_QUANT_NAMES=['dynamic_mxfp8_quant','dynamic_quant_fp8']


class _RemoveTritonQuantChecks(_RemoveAddedReplayChecks):
    def visit_FunctionDef(self, node):
        if node.name == '_reference_quant':
            # Only undo the explicitly documented FP8 per-token oracle repair
            # for historical arithmetic fingerprints, never update old hashes.
            node.body = [statement for statement in node.body if not (
                isinstance(statement, ast.If)
                and isinstance(statement.test, ast.Compare)
                and getattr(statement.test.left, 'id', None) == 'qdtype'
                and len(statement.test.ops) == 1
                and isinstance(statement.test.ops[0], ast.NotEq)
                and ast.unparse(statement.test.comparators[0]) == 'torch.int8')]
            if ast.get_docstring(node) == 'Quantization oracle; FP64 resolves FP8 per-token rounding boundaries.':
                node.body[0] = ast.Expr(ast.Constant('Original reference expressions, including token-path dtype rounding.'))
        return self.generic_visit(node)

    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call):
            if getattr(call.func,'id',None) in {'_checked_mx_pair','_checked_quant_output'}:return None
            if isinstance(call.func,ast.Attribute) and call.func.attr=='update' and call.args and isinstance(call.args[0],ast.Call) and getattr(call.args[0].func,'id',None)=='_verify_quant_timed':return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('name',_TRITON_QUANT_NAMES)
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached','stale_scale','input_modified','shape','dtype','scale_dtype','nonfinite'])
def test_triton_quant_actual_measured_pair_and_original_replay_gate(name,behavior,monkeypatch,tmp_path):
    import torch,types,math
    task=ROOT/'tasks/triton2flydsl/aiter'/name;checks=module(task/'scripts/replay_checks.py');mx=name=='dynamic_mxfp8_quant'
    ns={'QUANT_BLOCK_SIZE':32,'_E8M0_MASK_INT32':-8388608}
    _harness_functions(task,{'_torch_mxfp8_quant_from_fp32','_mx_reference','_reference_quant','_dtype_max'},ns)
    original=(torch.linspace(-2048,1536,128).reshape(2,64)).to(torch.bfloat16);state={'phase':'setup'}
    def oracle(x):return ns['_mx_reference'](x) if mx else ns['_reference_quant'](x,torch.float8_e4m3fn,'dyn_token')
    cached=oracle(original);inputs=[]
    def randn(*args,**kwargs):
        x=original.clone();inputs.append(x);return x
    original_zeros=torch.zeros
    monkeypatch.setattr(torch,'randn',randn)
    monkeypatch.setattr(torch,'zeros',lambda *args,**kwargs:original_zeros(*args,**{**kwargs,'device':'cpu'}))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    def compute(*args):
        x=args[0] if mx else args[1];y,scale=oracle(x)
        if behavior==state['phase']+'_wrong':y.view(torch.uint8).zero_()
        if state['phase']=='replay':
            if behavior=='cached':y,scale=(v.clone() for v in cached)
            if behavior=='stale_scale':scale=cached[1].clone()
            if behavior=='input_modified':x.add_(1)
        if not mx:args[0].copy_(y);args[2].copy_(scale);y,scale=args[0],args[2]
        if state['phase']=='measured':
            if behavior=='shape':y=y.reshape(-1)
            if behavior=='dtype':y=y.view(torch.uint8)
            if behavior=='scale_dtype':scale=scale.to(torch.bfloat16)
            if behavior=='nonfinite':y.view(torch.uint8).fill_(127)
        return y,scale
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    ns.update(TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,require_unchanged=checks.require_unchanged,
        _HERE=str(tmp_path),Path=Path,json=json,math=math,WARMUP=10,ITERS=100,SEED=20,
        TEST_SHAPES=[{'name':'controlled','shape':(2,64),'M':2,'N':64}],_fp8_e4m3_dtype=lambda:torch.float8_e4m3fn,
        _load_source=lambda:types.SimpleNamespace(dynamic_mxfp8_quant=compute,dynamic_per_token_quant_fp8_i8=compute))
    _harness_functions(task,{'run_benchmark','_checked_mx_pair','_mx_reference','_compare_mx_pair','_checked_quant_output','_compare_token_outputs','_verify_quant_timed'},ns)
    if behavior=='correct':
        result=ns['run_benchmark'](verbose=False);assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        assert calls==[(0,100)]
    else:
        with pytest.raises(AssertionError):ns['run_benchmark'](verbose=False)
    assert len(inputs)==1 and torch.equal(inputs[0],original)


def test_triton_mxfp8_preserves_three_dimensional_output_and_exact_scale_rule():
    import torch
    task=ROOT/'tasks/triton2flydsl/aiter/dynamic_mxfp8_quant';ns={'QUANT_BLOCK_SIZE':32,'_E8M0_MASK_INT32':-8388608}
    _harness_functions(task,{'_torch_mxfp8_quant_from_fp32','_checked_mx_pair','_mx_reference','_compare_mx_pair'},ns)
    x=torch.full((4,8,128),2.,dtype=torch.bfloat16);expected=ns['_mx_reference'](x)
    assert expected[0].shape==(4,8,128) and expected[1].shape==(4,8,4)
    actual=[v.clone() for v in expected];actual[0].view(torch.uint8)[0,0,0]+=1
    ns['_compare_mx_pair'](actual,expected,x)
    actual[0].view(torch.uint8)[0,0,0]+=1
    with pytest.raises(AssertionError):ns['_compare_mx_pair'](actual,expected,x)
    actual=[v.clone() for v in expected];actual[1][0,0,0]+=1
    with pytest.raises(AssertionError):ns['_compare_mx_pair'](actual,expected,x)
    with pytest.raises(AssertionError):ns['_compare_mx_pair']((expected[0].reshape(-1,128),expected[1]),expected,x)


@pytest.mark.parametrize('mode',['static','dyn_tensor','dyn_token'])
@pytest.mark.parametrize('fault',['correct','shape','dtype','unwritten_output','unwritten_scale','input_modified'])
def test_triton_quant_original_check_enforces_caller_buffers_and_input_contract(mode,fault,monkeypatch):
    import torch,types
    task=ROOT/'tasks/triton2flydsl/aiter/dynamic_quant_fp8';checks=module(task/'scripts/replay_checks.py')
    ns={'SEED':20,'require_unchanged':checks.require_unchanged}
    _harness_functions(task,{'_check','_reference_quant','_dtype_max','_checked_quant_output'},ns)
    # Real CPU original arithmetic; only GPU tensor placement is redirected.
    for op in ['randn','rand','zeros','tensor']:
        original=getattr(torch,op)
        monkeypatch.setattr(torch,op,lambda *args,__op=original,**kw:__op(*args,**({**kw,'device':'cpu'} if 'device' in kw else kw)))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    def call(qx,x,scale):
        y,s=ns['_reference_quant'](x,torch.int8,mode,scale if mode=='static' else None)
        if mode!='static':scale.copy_(s)
        qx.copy_(y);out=qx
        if fault=='shape':out=out[:1]
        if fault=='dtype':out=out.float()
        if fault=='unwritten_output':out=out.clone();qx.zero_()
        if fault=='unwritten_scale' and mode!='static':s=scale.clone();scale.fill_(1000)
        else:s=scale
        if fault=='input_modified':x.add_(1)
        return out if mode=='static' else (out,s)
    mod=types.SimpleNamespace(static_per_tensor_quant_fp8_i8=call,dynamic_per_tensor_quant_fp8_i8=call,dynamic_per_token_quant_fp8_i8=call)
    if fault=='correct' or (fault=='unwritten_scale' and mode=='static'):
        assert ns['_check'](mode,mod,2,64,torch.int8,False)[0]
    else:
        with pytest.raises(AssertionError):ns['_check'](mode,mod,2,64,torch.int8,False)


def test_triton_quant_multi_entry_candidate_audit_keeps_nested_valid_launch_scope():
    import torch,types
    task=ROOT/'tasks/triton2flydsl/aiter/dynamic_quant_fp8';audit=module(task/'scripts/candidate_checks.py')
    scope={'__name__':'flydsl.cpu_profile_fixture'}
    exec('class CompiledKernel:\n def __call__(self,x):return x\n',scope)
    launch=scope['CompiledKernel']();original=types.SimpleNamespace(inner=lambda x:launch(x))
    original.outer=lambda x:original.inner(x)
    h=types.SimpleNamespace(ARENA_FINAL_CANDIDATE=True,ENTRIES=('inner','outer'),_load_source=lambda:original)
    with audit.audit_candidate_calls(h) as observed:
        proxy=h._load_source();x=torch.tensor([1.]);assert proxy.outer(x) is x
        assert original.inner is not proxy.inner
    assert observed=={'flydsl.cpu_profile_fixture.CompiledKernel'}
    original.outer=lambda x: x.square()
    with pytest.raises(RuntimeError,match='non-preparation PyTorch operation'):
        with audit.audit_candidate_calls(h):h._load_source().outer(x)


def test_triton_quant_original_inputs_arithmetic_tolerances_and_timing_preserved():
    hashes={'dynamic_mxfp8_quant': {'_torch_mxfp8_quant_from_fp32': '5fd469d7046f8984c262098d559492bcf0ac0bc126efb9b5a1b1b2f8efd4a26d', 'run_correctness': 'c63fc2198d788e2d285a88d5c7040448d89c1e7270b9a1262dc067db18200f51', 'run_benchmark': '3060e07030b082af72987af732af1b6657933c25a27cbdddbd77afc9a2235445'}, 'dynamic_quant_fp8': {'_reference_quant': 'ad89f40181f40e0ac1ab7e54952a3b896117c6681c2e69b5e968994fcb87fffd', '_check': '5931205268555e732dc4cc23a254efb9a30e42535d983488385ae9feee7653b0', 'run_correctness': '3f0ff796227f37c3053e57da55ad8d709b5f7b2d96a7cc7df8132f5cabeae895', 'run_benchmark': 'b01afa95ca0efbc085dc327633f9167fb952fb90d9a2cb81800646efda97cb5c'}}
    for name,functions in hashes.items():
        tree=ast.parse((ROOT/'tasks/triton2flydsl/aiter'/name/'test_kernel_harness.py').read_text())
        for fn in tree.body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveTritonQuantChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


class _RemoveMqaChecks(_RemoveAddedReplayChecks):
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None)=='protected_inputs':return None
        return super().visit_Assign(node)
    def visit_Expr(self,node):
        call=node.value
        if isinstance(call,ast.Call):
            if getattr(call.func,'id',None)=='_checked_mqa_output':return None
            if isinstance(call.func,ast.Attribute) and call.func.attr=='update' and call.args and isinstance(call.args[0],ast.Call) and getattr(call.args[0].func,'id',None)=='_verify_mqa_timed':return None
        return super().visit_Expr(node)
    def visit_Return(self,node):
        if isinstance(node.value,ast.Dict):
            pairs=[(k,v) for k,v in zip(node.value.keys,node.value.values) if not isinstance(k,ast.Constant) or k.value!='cases']
            node.value.keys=[k for k,v in pairs];node.value.values=[v for k,v in pairs]
        return self.generic_visit(node)


@pytest.mark.parametrize('window',['full','causal','band'])
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','benchmark'] for behavior in ['correct','shape','dtype','device','nan','mask','input_modified','measured_wrong','replay_wrong','cached'] if phase=='benchmark' or behavior not in {'measured_wrong','replay_wrong','cached'}])
def test_mqa_actual_harness_contract_and_measured_replay(window,phase,behavior,monkeypatch,tmp_path):
    import torch,math,types
    t=ROOT/'tasks/triton2flydsl/aiter/fp8_mqa_logits';checks=module(t/'scripts/replay_checks.py');ns={}
    _harness_functions(t,{'_window_mask','_build_windows','_ref_fp8_mqa_logits','_calc_diff','_checked_mqa_output','_compare_mqa_output','_verify_mqa_timed'},ns)
    q=torch.tensor([[[1.,0.],[0.,1.]],[[1.,2.],[3.,4.]]]).to(torch.float8_e4m3fn)
    kv=torch.tensor([[2.,4.],[-1.,3.],[2.,1.]]).to(torch.float8_e4m3fn)
    inputs=(q,kv,torch.tensor([2.,3.,.5]),torch.tensor([[.5,2.],[1.,.5]]),*ns['_build_windows'](2,3,window,'cpu'))
    originals=tuple(x.clone() for x in inputs);cached=ns['_ref_fp8_mqa_logits'](*inputs);state={'value':'setup'}
    def compute(*args,**kwargs):
        out=ns['_ref_fp8_mqa_logits'](*args)
        active=phase=='correctness' or state['value']=='measured'
        if active:
            if behavior=='shape':out=out[:1]
            if behavior=='dtype':out=out.double()
            if behavior=='device':out=out.to('meta')
            if behavior=='nan':out[0,0]=float('nan')
            if behavior=='mask':out[0,0]=float('-inf') if window=='full' else 0.
            if behavior=='input_modified':args[3].mul_(.5)
        if behavior==state['value']+'_wrong':out.zero_()
        if state['value']=='replay' and behavior=='cached':out=cached.clone()
        return out
    class Collector:bound=False
    samples=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        samples.append((warmup,repetition));state['value']='measured';timed_run.outputs=fn();timed_run.bound=True;state['value']='setup'
        def replay():
            state['value']='replay'
            try:return fn()
            finally:state['value']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    ns.update(TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,require_unchanged=checks.require_unchanged,
        TEST_SHAPES=[(2,3,2,2,window)],_TASK_DIR=str(tmp_path),SOURCE_FILE='candidate.py',Path=Path,json=json,math=math,
        WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100,load_module=lambda:types.SimpleNamespace(fp8_mqa_logits=compute),make_inputs=lambda *a:inputs)
    _harness_functions(t,{'run_correctness','run_benchmark'},ns)
    if phase=='correctness':
        result=ns['run_correctness'](verbose=False)
        assert result['correct'] is (behavior=='correct')
    else:
        result=ns['run_benchmark'](verbose=False);rows=result['cases']
        assert len(rows)==1 and rows[0]['test_case_id']=='perf1'
        if behavior=='correct':
            assert rows[0]['execution_time_ms']==.1
            assert rows[0]['timed_output_correctness']==rows[0]['replay_correctness']=='PASS'
            assert samples==[(0,100)]
        else:assert rows[0]['execution_time_ms']<0 and rows[0]['benchmark_method']=='benchmark_failed'
        if behavior!='input_modified':checks.require_unchanged(inputs,originals)


@pytest.mark.parametrize('fault',['none','partial','duplicate','wrong_shape','failed','error','aggregate_only'])
def test_mqa_adapter_accepts_real_dictionary_and_rejects_incomplete_evidence(fault,monkeypatch):
    import types
    t=ROOT/'tasks/triton2flydsl/aiter/fp8_mqa_logits';a=module(t/'scripts/task_actions.py');checks=module(t/'scripts/candidate_checks.py')
    monkeypatch.setitem(sys.modules,'scripts.candidate_checks',checks)
    rows=[{'shape_id':i+1,'shape':list(sh),'passed':True} for i,sh in enumerate(a.EXPECTED_SHAPES)]
    if fault=='partial':rows.pop()
    if fault=='duplicate':rows[-1]=rows[0]
    if fault=='wrong_shape':rows[0]['shape']=[1,2]
    if fault=='failed':rows[0]['passed']=False
    if fault=='error':rows[0]['error']='launch failed'
    h=types.SimpleNamespace(run_correctness=lambda:{'correct':True,'details':rows})
    if fault in {'none','aggregate_only'}:assert a.check(h)==[]
    else:
        with pytest.raises(RuntimeError):a.check(h)
    perf=[{'test_case_id':'perf'+str(i+1),'execution_time_ms':.1,'benchmark_method':'cuda_graph'} for i in range(5)]
    h.run_benchmark=lambda:{'geomean_latency_ms':.1,**({} if fault=='aggregate_only' else {'cases':perf})}
    if fault=='aggregate_only':
        with pytest.raises(RuntimeError,match='per-case'):a.performance(h)
    else:assert a.performance(h)==perf
    runtime=module(t/'task_runtime.py');manifest=json.loads((t/'cases.json').read_text())['cases']
    assert len(runtime.require_result_rows(perf,manifest,a.PERFORMANCE_IDS))==5
    with pytest.raises(ValueError,match='Incomplete'):runtime.require_result_rows(perf[:-1],manifest,a.PERFORMANCE_IDS)


def test_mqa_original_gate_allows_diagnostic_allclose_failure_and_exact_mask():
    import torch
    t=ROOT/'tasks/triton2flydsl/aiter/fp8_mqa_logits';ns={}
    _harness_functions(t,{'_window_mask','_calc_diff','_checked_mqa_output','_compare_mqa_output'},ns)
    inputs=(torch.ones(1,1,1),torch.ones(3,1),None,None,torch.tensor([0]),torch.tensor([2]))
    expected=torch.tensor([[100.,0.,float('-inf')]]);actual=torch.tensor([[100.,1.,float('-inf')]])
    assert not torch.allclose(actual[:,:2],expected[:,:2],atol=.05,rtol=.05)
    ns['_compare_mqa_output'](actual,expected,inputs)
    for wrong in [torch.tensor([[100.,10.,float('-inf')]]),torch.tensor([[100.,0.,0.]]),torch.tensor([[100.,float('nan'),float('-inf')]])]:
        with pytest.raises(AssertionError):ns['_compare_mqa_output'](wrong,expected,inputs)


def test_mqa_original_reference_inputs_numerics_and_timing_unchanged():
    task=ROOT/'tasks/triton2flydsl/aiter/fp8_mqa_logits'
    hashes={'_resolve_dir': 'f6c6d7a4dab024924282dc98fa368522cdf52317955ae50ab0f16a15c976d3a5', '_build_windows': 'bb31644715d1679422d40812492532011b72de480c68c2667f2053c5b0350fdc', 'make_inputs': '9dfd71d62b949164de323b2356fe778ab9706aef83231a0fdcb728217df3ae2c', '_window_mask': 'c6335be54b85da98e91cbf731c9bac2f744a4bc849a1f6365d96c241b58d4a5f', '_ref_fp8_mqa_logits': '937f4764d129d02cf4276b40a9c6ab8e578ed2b4ec3fa8bb9e750d55b889918c', '_calc_diff': '470182b3a4f81f2f3598f3ae11e9c51293b59cfa9dbdd826acff3ad7e82233ac', 'run_correctness': 'f4e131b9ce78144724bccd3a4273a1ef43f29ae10dc1ecc520f87cccce10e6c5', 'run_benchmark': 'a9ab39fa473a0997b7e62b3d07279856df37a35c0cee0a0b21d19c764df65055'}
    for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
        if isinstance(fn,ast.FunctionDef) and fn.name in hashes:
            normalized=_RemoveMqaChecks().visit(fn)
            assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==hashes[fn.name],fn.name


@pytest.mark.parametrize('fault',['none','partial_correctness','partial_performance','invalid_latency','unknown_method'])
def test_mqa_real_runner_emits_complete_or_failed_protocol(fault,monkeypatch,capsys):
    import types
    t=ROOT/'tasks/triton2flydsl/aiter/fp8_mqa_logits';a=module(t/'scripts/task_actions.py');runtime=module(t/'task_runtime.py');checks=module(t/'scripts/candidate_checks.py')
    monkeypatch.setitem(sys.modules,'scripts.candidate_checks',checks)
    details=[{'shape_id':i+1,'shape':list(sh),'passed':True} for i,sh in enumerate(a.EXPECTED_SHAPES)]
    perf=[{'test_case_id':'perf'+str(i+1),'execution_time_ms':.1,'benchmark_method':'cuda_graph'} for i in range(5)]
    if fault=='partial_correctness':details.pop()
    if fault=='partial_performance':perf.pop()
    if fault=='invalid_latency':perf[0]['execution_time_ms']=-1.
    if fault=='unknown_method':perf[0]['benchmark_method']='wall_clock'
    h=types.SimpleNamespace(TEST_SHAPES=a.EXPECTED_SHAPES,run_correctness=lambda:{'correct':True,'details':details},run_benchmark=lambda:{'geomean_latency_ms':.1,'cases':perf})
    monkeypatch.setattr(runtime,'load_module',lambda name,path:a if name=='arena_task_actions' else h)
    rc=runtime.run(['baseline','performance'])
    report=json.loads(capsys.readouterr().out.split('ARENA_EVAL_RESULT=')[-1])
    assert report['protocol']=='arena-eval-v1' and report['action']=='performance' and report['role']=='baseline'
    assert rc==(0 if fault=='none' else 1)
    assert report['status']==('PASS' if fault=='none' else 'FAIL')
    assert len(report['cases'])==5
    assert all(row['status']==report['status'] for row in report['cases'])
    if fault=='none':
        manifest=json.loads((t/'cases.json').read_text())['cases']
        assert [row['params'] for row in report['cases']]==[row['params'] for row in manifest]
        assert all(row['execution_time_ms']==.1 and row['benchmark_method']=='cuda_graph' for row in report['cases'])
    else:assert all('execution_time_ms' not in row for row in report['cases'])


_TRITON_ELEMENTWISE_NAMES=['ff_a16w16','fused_silu_mul','fused_clamp_act_mul','rmsnorm']


class _RemoveElementwiseChecks(_RemoveTritonBatchedChecks):
    def visit_FunctionDef(self, node):
        if node.name == '_torch_rmsnorm':
            # Explicit reference repair: preserve the historical fingerprint
            # except for the missing stabilizer, tested independently below.
            for statement in node.body:
                if (isinstance(statement, ast.Assign)
                        and len(statement.targets) == 1
                        and getattr(statement.targets[0], 'id', None) == 'rms'):
                    argument = statement.value.args[0]
                    if (isinstance(argument, ast.BinOp)
                            and isinstance(argument.op, ast.Add)
                            and getattr(argument.right, 'id', None) == 'EPS'):
                        statement.value.args[0] = argument.left
        return self.generic_visit(node)

    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='_checked_elementwise_output':return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('name',_TRITON_ELEMENTWISE_NAMES)
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','benchmark'] for behavior in ['correct','shape','dtype','device','nan','input_modified','weight_modified','measured_wrong','replay_wrong','cached'] if phase=='benchmark' or behavior not in {'measured_wrong','replay_wrong','cached'}])
def test_triton_elementwise_original_numeric_gate_actual_measured_replay(name,phase,behavior,monkeypatch,tmp_path):
    import torch,types,math
    t=ROOT/'tasks/triton2flydsl/aiter'/name;checks=module(t/'scripts/replay_checks.py');ns={'LOG2_E':1.44269504089}
    _harness_functions(t,{'_torch_ref','_torch_silu_mul','_torch_reference','_torch_rmsnorm','_torch_dtype','_cases'},ns)
    current={};state={'value':'setup'}
    def make_inputs(*args):
        dtype=args[-1] if isinstance(args[-1],torch.dtype) else torch.bfloat16
        if name=='ff_a16w16':result=(torch.tensor([[1.,-1.],[.5,2.]],dtype=dtype),torch.eye(2,dtype=dtype),torch.tensor([[2.,1.],[-1.,3.]],dtype=dtype))
        elif name=='rmsnorm':result=(torch.tensor([[1.,2.,3.,4.],[-2.,1.,4.,1.]],dtype=dtype),torch.tensor([.5,1.,-.5,2.],dtype=dtype))
        elif name=='fused_clamp_act_mul':
            wm=args[-1];result=(torch.tensor([[20.,-1.,-12.,4.],[.5,1.,-1.,2.]],dtype=dtype),None if wm=='none' else torch.tensor([[.5],[1.25]],dtype=torch.float32) if wm=='broadcast' else torch.tensor([[.5,1.25],[.25,.75]],dtype=torch.float32))
        else:result=(torch.tensor([[2.,-1.,3.,4.],[.5,1.,-1.,2.]],dtype=dtype),)
        current['values']=tuple(v for v in result if v is not None);current['originals']=tuple(v.clone() for v in current['values']);current.pop('cached',None)
        return result[0] if name=='fused_silu_mul' else result
    def compute(*args,**kwargs):
        if name=='ff_a16w16':out=ns['_torch_ref'](*args[:3],kwargs['activation'])
        elif name=='fused_silu_mul':out=ns['_torch_silu_mul'](args[0])
        elif name=='fused_clamp_act_mul':out=ns['_torch_reference'](args[0],kwargs['swiglu_limit'],kwargs['weights'])
        else:out=ns['_torch_rmsnorm'](*args[:2],args[0].dtype)
        if 'cached' not in current:current['cached']=out.clone()
        if phase=='correctness' or state['value']=='measured':
            if behavior=='shape':out=out[:1]
            if behavior=='dtype':out=out.float()
            if behavior=='device':out=out.to('meta')
            if behavior=='nan':out.fill_(float('nan'))
            if behavior=='input_modified':args[0].add_(1)
            if behavior=='weight_modified':current['values'][-1].add_(1)
        if behavior==state['value']+'_wrong':out.add_(100)
        if state['value']=='replay' and behavior=='cached':out=current['cached'].clone()
        return out
    class Collector:bound=False
    samples=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        samples.append((warmup,repetition));state['value']='measured';timed_run.outputs=fn();timed_run.bound=True;state['value']='setup'
        def replay():
            state['value']='replay'
            try:return fn()
            finally:state['value']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    entries={'ff_a16w16':'ff_a16w16_nogate','fused_silu_mul':'fused_silu_mul','fused_clamp_act_mul':'fused_clamp_act_mul','rmsnorm':'rms_norm'}
    ns.update(TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,require_unchanged=checks.require_unchanged,
        verify_timed_run=checks.verify_timed_run,allclose_output=checks.allclose_output,_HERE=str(tmp_path),Path=Path,json=json,math=math,WARMUP=10,ITERS=100,
        TEST_SHAPES=[{'name':'controlled','batch':2,'hidden':2,'intermediate':2,'rows':2,'last':4,'M':2,'N':4}],
        ACTIVATIONS=['gelu_tanh','silu_exp2','relu',None],DTYPES=['bf16','fp16'],EPS=1e-5,
        MS=[2],DS=[4],LIMITS=[0.,7.],WEIGHT_MODES=['none','broadcast','elementwise'],
        _load_source=lambda:types.SimpleNamespace(**{entries[name]:compute}),_make_inputs=make_inputs)
    _harness_functions(t,{'_checked_elementwise_output','_elementwise_replay_validator','run_correctness','run_benchmark'},ns)
    if phase=='correctness':assert ns['run_correctness'](verbose=False) is (behavior=='correct')
    elif behavior=='correct':
        rows=ns['run_benchmark'](verbose=False)
        assert all(row['timed_output_correctness']==row['replay_correctness']=='PASS' for row in rows)
        assert samples==[(0,100)]*(2 if name=='fused_clamp_act_mul' else 1)
        checks.require_unchanged(current['values'],current['originals'])
    else:
        with pytest.raises(AssertionError):ns['run_benchmark'](verbose=False)


def test_triton_elementwise_original_math_cases_and_timing_preserved():
    hashes={'ff_a16w16': {'_make_inputs': '1d20457066b97fb6d915307a899281b0ca4158fbcda9a95c1a44b5f8248fc0f7', '_torch_ref': 'e9997ea7b309abd3b0d775ea166838be4742859f0fd36d2a20dbfb3dd02f2f06', 'run_correctness': 'c3664fc353a1d6ae2f7237f21b56e931e3d58d2b6c14734e041c2b3554746cb0', 'run_benchmark': '4d17a4fde8eb9009731cce6f94d7edd74bd5c681036c5ba35007a995ee70756a'}, 'fused_silu_mul': {'_torch_dtype': '664e9bfa3e0f1b752c5ff297485baf851e06ceaa2bc5873012159e8ccff77add', '_make_inputs': 'cb3d2a74241c7e5e43322f9ad9012d3d15f78b982ff9b30252db15088031f7fe', '_torch_silu_mul': '5101cf3a84972ba87ee20666543fe377bbbc2cec45d007d2e57d8a539ac8abd1', 'run_correctness': 'ae04a207e6407d253191b44a55e157dc014c1a36e7c4e80dfc059d1610f9e59c', 'run_benchmark': '699d90764342286d45dbbc6c167ad605612ee39253449946a818c5641c4941ec'}, 'fused_clamp_act_mul': {'_make_inputs': '4141c4cd053618d4c28302405ea076c203dd61d3c869619acb541f54f5e16927', '_torch_reference': '21a38febcca29b3929afce48f5665aab1d11224bcc5b7307b57310d703fac298', '_cases': '7e046fa5de5141a3f79224ee5b1609fba75f5361ef221b91dfd64942abc641ab', 'run_correctness': 'e8c9cbaff5f22a5084f0c42d34adfd50badea4721a1401d23e94aad017026ef7', 'run_benchmark': '2d2790a34d14b3f9d8a0f6a97bba0665fd3a3de446329947951d997ad191ca9b'}, 'rmsnorm': {'_torch_dtype': '664e9bfa3e0f1b752c5ff297485baf851e06ceaa2bc5873012159e8ccff77add', '_make_inputs': 'a580372ddba80a3ab421ca36d1bc22ada43dd0de2cb2ddeded1660176980e219', '_torch_rmsnorm': '3cefb6753eee682122bf267e3d587f5d1d7765545ecc36122d8fe82befbac3a4', 'run_correctness': 'cec48cbe43d807df0a3ac556b3bfb55354a9b80063796da549585ca535fe851f', 'run_benchmark': 'd76c41e25a86d0f9391405eed6fb88399e2789dcbb244f2a5ef4b7c57371a30f'}}
    for name,functions in hashes.items():
        t=ROOT/'tasks/triton2flydsl/aiter'/name
        for fn in ast.parse((t/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveElementwiseChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_ROPE_TWO_NAMES=['rope_fwd_kernel','rope_thd_fwd_kernel']


class _RemoveRopeChecks(_RemoveAddedReplayChecks):
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None) in {'protected_inputs','replay_validate'}:return None
        return super().visit_Assign(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='_checked_rope_output':return None
        return super().visit_Expr(node)
    def visit_FunctionDef(self,node):
        if node.name=='_mean':
            call=next(n for n in ast.walk(node) if isinstance(n,ast.Call) and getattr(n.func,'id',None)=='benchmark_cuda_graph_or_events')
            call.keywords=[k for k in call.keywords if k.arg!='timed_run']
            node.body=[ast.Return(value=call)]
            return node
        return self.generic_visit(node)


@pytest.mark.parametrize('name',_ROPE_TWO_NAMES)
@pytest.mark.parametrize('provided',[True,False])
@pytest.mark.parametrize('function,behavior',[(function,behavior) for function in ['run_correctness','run_benchmark','arena_benchmark'] for behavior in ['correct','shape','dtype','device','nan','input_modified','position_modified','measured_wrong','replay_wrong','cached'] if function!='run_correctness' or behavior not in {'measured_wrong','replay_wrong','cached'}])
def test_rope_two_actual_benchmark_preserves_original_truth_and_timed_replay(name,function,provided,behavior,monkeypatch,tmp_path):
    import torch,types,math
    t=ROOT/'tasks/torch2flydsl'/name;model=module(t/'model.py');checks=module(t/'scripts/replay_checks.py');packed=name=='rope_thd_fwd_kernel'
    shape={'name':'controlled','s':2,'b':1,'h':1,'d':4,'rotary_pct':1.,'rotate_style':0,'reuse':True,'nope_first':False,'cu':[0,1,2]}
    x=torch.tensor([[[[1.,2.,3.,4.]]],[[[5.,6.,7.,8.]]]],dtype=torch.bfloat16)
    if packed:inputs=(x.squeeze(1),torch.tensor([0,1,2],dtype=torch.int32),torch.ones((2,1,1,2),dtype=torch.bfloat16)*.5)
    else:inputs=(x,torch.ones((2,1,1,2),dtype=torch.bfloat16)*.75,torch.ones((2,1,1,2),dtype=torch.bfloat16)*.5)
    originals=tuple(v.clone() for v in inputs);state={'phase':'setup','role':'setup'}
    class Model(model.Model):
        def to(self,*a,**k):return self
        def forward(self,*args):return apply_fault(super().forward(*args),'baseline')
    def oracle(*args):return model.Model(0,True,False)(*args[:3])
    cached=oracle(*inputs)
    def apply_fault(out,role):
        if role!=('baseline' if provided else 'candidate'):return out
        if function!='run_correctness' and state['role'] not in ('run_ref','run_kernel','run_target'):return out
        if function=='run_correctness' or state['phase']=='measured':
            if behavior=='shape':out=out[:1]
            if behavior=='dtype':out=out.float()
            if behavior=='device':out=out.to('meta')
            if behavior=='nan':out.fill_(float('nan'))
            if behavior=='input_modified':inputs[0].add_(1)
            if behavior=='position_modified':inputs[-1].add_(1)
        if behavior==state['phase']+'_wrong':out.add_(100)
        if state['phase']=='replay' and behavior=='cached':out=cached.clone()
        return out
    def candidate(*args):return apply_fault(oracle(*args),'candidate')
    entry='flydsl_rope_thd_fwd' if packed else 'flydsl_rope_cached_fwd'
    kmod=types.SimpleNamespace(**{entry:candidate});mmod=types.SimpleNamespace(Model=Model,get_init_inputs=lambda:[0,True,False],get_inputs=lambda:[v.clone() for v in inputs])
    def load(directory,filename,alias):return mmod if filename=='model.py' else None if provided else kmod
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((fn.__name__,warmup,repetition));state.update(phase='measured',role=fn.__name__);timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    original_to=torch.Tensor.to
    def cpu_to(tensor,*args,**kwargs):
        if args and args[0]=='cuda':args=('cpu',)+args[1:]
        return original_to(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    monkeypatch.setitem(sys.modules,'aiter',types.SimpleNamespace(rope_cached_fwd=oracle))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_unchanged':checks.require_unchanged,'verify_timed_run':checks.verify_timed_run,
        '_load_module':load,'_load_target':lambda:None if provided else candidate,'_is_pure_starter':lambda:provided,'_probe_target':lambda target,pure,*a:(False,None) if provided else (True,target(*a)),
        '_KERNEL_DIR':str(tmp_path),'MODEL_FILE':'model.py','KERNEL_FILE':'kernel.py','KERNEL_ENTRY':entry,'SHAPES':[shape],
        '_make_inputs':lambda *a:inputs,'_retry':lambda fn,**kw:fn(),'_aiter_op':oracle,'REL_TOL':.01,'ROTATE_STYLE':0,'REUSE_FREQS_FRONT_PART':True,'NOPE_FIRST':False,'Path':Path,'json':json,'math':math}
    # Keep the exact task Model per-sequence arithmetic in this CPU simulation.
    def prepared(model_instance,input,freqs,cu):
        def run_ref():return model_instance(input,inputs[1],freqs)
        return run_ref
    ns['_make_reference_runner']=prepared
    _harness_functions(t,{'_norm_max_err','_checked_rope_output','_rope_replay_validator',function},ns)
    if behavior=='correct' and function=='run_correctness':
        assert ns[function](verbose=False) is True
    elif behavior=='correct':
        rows=ns[function](verbose=False)
        if function=='run_benchmark':rows=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert rows[0]['timed_output_correctness']==rows[0]['replay_correctness']=='PASS'
        assert [(a,b) for _,a,b in calls]==[(10,100)]*(2 if provided else 3)
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    if behavior not in {'input_modified','position_modified'}:checks.require_unchanged(inputs,originals)


@pytest.mark.parametrize('name',_ROPE_TWO_NAMES)
def test_rope_two_original_numerical_gate_not_elementwise_allclose(name):
    import torch
    t=ROOT/'tasks/torch2flydsl'/name;ns={'REL_TOL':.01};checks=module(t/'scripts/replay_checks.py')
    _harness_functions(t,{'_norm_max_err','_checked_rope_output','_rope_replay_validator'},ns)
    x=torch.tensor([[100.,0.]],dtype=torch.bfloat16);positions=torch.ones(1);expected=x.clone();originals=(x.clone(),positions.clone())
    ns.update(require_unchanged=checks.require_unchanged,verify_timed_run=checks.verify_timed_run)
    def run_ref():return x.clone()
    validate=ns['_rope_replay_validator']((x,positions),run_ref)
    class Timed:
        bound=True
        outputs=torch.tensor([[100.,.5]],dtype=torch.bfloat16)
        def rerun(self):return x.clone()
    assert not torch.allclose(Timed.outputs,expected,atol=.01,rtol=.01)
    validate(Timed())
    checks.require_unchanged((x,positions),originals)
    bad=Timed();bad.outputs=torch.tensor([[100.,2.]],dtype=torch.bfloat16)
    with pytest.raises(AssertionError):validate(bad)


def test_rope_two_original_inputs_models_cases_comparisons_timing_unchanged():
    hashes={'rope_fwd_kernel': {'_resolve_kernel_dir': 'ebfabcdd05c1a3f254b035e42b1783890c241f92613bda21ccb77fa064039f1a', '_load_module': 'caae8222257891caee15a695bfc7cff029ec363f792c4331f20ff92c86efb520', '_load_target': '664064a72dbdde3b5c2394280c9a9d80391753ca5fc208d893241c5f2bc10a0f', '_is_pure_starter_source': '5b07c5fdd33728bba2c41c84e320d6651222029356660ea8530ff85879ed5540', '_is_pure_starter': '2c5ca5b331dcdc34e4a5d17071b172f672ba9e26746d0bcb992e92c93db9377f', '_probe_target': 'cc64692e7e510e4df88628a2b7c738aa9b9835556ecb72542f0c09c9a2f1d047', '_retry': 'fcef3f3d7f904af49c79f1a81e6eff2f8cfbbebc651c459cbbc27878c84c996c', '_make_inputs': '4ce69054a8087de0d9fff0a3b9b4841d032a2ae26dab89be5b3f79ca37f89602', '_norm_max_err': '750a488ebe381cf762ba31a41890c0fe3dd87c2dc703955ad218dd85a2d9862e', 'run_correctness': '9dc3b1def153c9fa64303310e1a2fb466a3b6297ec507e1c2c91aba1bbb99966', 'run_benchmark': '7191c9821325124b0d0ef5abe8aa32bcb1bbf67cada1258dd79e57a4f6341b5f', 'run_compile': '8c93e69f8c091b1f53595790626186ba5aaeade2edd3fd4f4998f9276849565e', '_require_candidate_outputs': 'e5f69466d2ec4497fe192bf53ce8d7b034bb14cd59272fc7b1dea160a574e09e', 'arena_benchmark': 'b8da654bd8eee3a75aee19564f022b964cea9c6c69499783d53a4437938580b0'}, 'rope_thd_fwd_kernel': {'_resolve_kernel_dir': 'ebfabcdd05c1a3f254b035e42b1783890c241f92613bda21ccb77fa064039f1a', '_load_module': 'caae8222257891caee15a695bfc7cff029ec363f792c4331f20ff92c86efb520', '_retry': 'd0533f7305c7aa3bf15eab61ce33ab1ae23ec313c13f2699ba25d3f82d6d1882', '_make_inputs': 'f96e2f9263118813081056ec5f070955dea608032a8ad939cd106c18f57a3cc9', '_aiter_op': '3db252eb117ec48fa10ce2d58eb7e98f3ec35fc7dbcb9cb9febc1a8454ca88ad', '_make_reference_runner': '7a1b6f4ad67f8710ffedbff4cdb4ee984e753f4f3ccd90f020d20d8460a3d6ba', '_norm_max_err': '750a488ebe381cf762ba31a41890c0fe3dd87c2dc703955ad218dd85a2d9862e', 'run_correctness': '4c07b38fb34f84197e3b1d91b1db6bfe3fc4be05a8d99d47b042c1948ee45f5a', 'run_benchmark': '9690fc88a66e140c86254bb89f133886f1835d3a8b176005090597fcdb4eb0a5', 'run_compile': 'edf94a982dec224427540a11121a0a7b73de4f369de59ec0b0ddf7c657c0ff36', '_require_candidate_outputs': 'e5f69466d2ec4497fe192bf53ce8d7b034bb14cd59272fc7b1dea160a574e09e', 'arena_benchmark': '78d16a4f559bd35849a0e9eaae69efa9af4139f490a2e9c05eba813a7775a139'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/torch2flydsl'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveRopeChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_FUSED_QUANT_TWO_NAMES=['silu_and_mul_quant_kernel','smoothquant_kernel']


class _RemoveFusedQuantChecks(_RemoveStandardQuantChecks):
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None)=='protected_inputs':return None
        return super().visit_Assign(node)


@pytest.mark.parametrize('name',_FUSED_QUANT_TWO_NAMES)
@pytest.mark.parametrize('provided',[True,False])
@pytest.mark.parametrize('function,behavior',[(function,behavior) for function in ['run_correctness','run_benchmark','arena_benchmark'] for behavior in ['correct','shape','code_dtype','scale_dtype','scale_shape','nan','input_modified','scale_input_modified','measured_wrong','replay_wrong','cached_codes','cached_scale'] if function!='run_correctness' or behavior not in {'measured_wrong','replay_wrong','cached_codes','cached_scale'}])
def test_fused_quant_real_original_gate_and_actual_measured_pair(name,provided,function,behavior,monkeypatch,tmp_path):
    import torch,math,types
    task=ROOT/'tasks/torch2flydsl'/name;mmod=module(task/'model.py');checks=module(task/'scripts/replay_checks.py');silu=name.startswith('silu');oracle=mmod.Model(*mmod.get_init_inputs())
    x=torch.linspace(-8,7,512).reshape(2,256).to(torch.bfloat16);channel_scale=torch.linspace(.5,2.,256)
    inp=(x,) if silu else (x,channel_scale);originals=tuple(v.clone() for v in inp);cached=oracle(*inp);phase={'value':'setup'}
    def compute(is_model):
        y,scale=oracle(*inp)
        if is_model==provided:
            active=function=='run_correctness' or phase['value']=='measured'
            if active:
                if behavior=='shape':y=y[:1]
                if behavior=='code_dtype':y=y.float()
                if behavior=='scale_dtype':scale=scale.double()
                if behavior=='scale_shape':scale=scale.reshape(-1)
                if behavior=='nan':scale.fill_(float('nan'))
                if behavior=='input_modified':x.add_(1)
                if behavior=='scale_input_modified':inp[-1].mul_(.5)
            if behavior==phase['value']+'_wrong':y.view(torch.uint8).zero_()
            if phase['value']=='replay':
                if behavior=='cached_codes':y=cached[0].clone()
                if behavior=='cached_scale':scale=cached[1].clone()
        return y,scale
    class Model:
        def __init__(self,*a):pass
        def to(self,*a,**k):return self
        def __call__(self,*a):return compute(True)
    model_module=types.SimpleNamespace(Model=Model,get_init_inputs=mmod.get_init_inputs,_FP8_DTYPE=getattr(mmod,'_FP8_DTYPE',None))
    entry='flydsl_'+name.removesuffix('_kernel');candidate=types.SimpleNamespace(**{entry:lambda *a:compute(False)})
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));phase['value']='measured';timed_run.outputs=fn();timed_run.bound=True;phase['value']='setup'
        def replay():
            phase['value']='replay'
            try:return fn()
            finally:phase['value']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_unchanged':checks.require_unchanged,
        '_make_inputs':lambda *a:x if silu else inp,'_aiter_op':lambda *a:oracle(*inp),'_retry':lambda fn,**kw:fn(),
        '_load_module':lambda directory,filename,alias:model_module if filename=='model.py' else None if provided else candidate,
        '_KERNEL_DIR':str(tmp_path),'MODEL_FILE':'model.py','KERNEL_FILE':'kernel.py','KERNEL_ENTRY':entry,'SHAPES':[{'name':'controlled','m':2,'n':256}],
        'GROUP_SIZE':128,'LIMIT':0.,'CODE_TOL':1,'SCALE_RTOL':1e-3,'Path':Path,'json':json,'math':math}
    _harness_functions(task,{'_compare','_checked_quant_pair','_compare_quant_outputs','_quant_replay_validator','_mean_ms',function},ns)
    if behavior=='correct':
        result=ns[function](verbose=False)
        if function=='run_correctness':assert result is True
        else:
            if function=='run_benchmark':result=json.loads((tmp_path/'build/performance_report.json').read_text())
            assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
            assert calls==[(10,100)]*(2 if provided else 3)
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    if behavior not in {'input_modified','scale_input_modified'}:checks.require_unchanged(inp,originals)


def test_fused_quant_two_original_inputs_models_comparators_and_timing_retained():
    hashes={'silu_and_mul_quant_kernel': {'_resolve_kernel_dir': 'ebfabcdd05c1a3f254b035e42b1783890c241f92613bda21ccb77fa064039f1a', '_load_module': 'caae8222257891caee15a695bfc7cff029ec363f792c4331f20ff92c86efb520', '_make_inputs': 'b189ce05262cdd417908cc9746babb21a576bc68927754e8beb7e1e36c9e2038', '_aiter_op': '0082a34698dd9082c8d48a1decf3046633ce29bb72d7e316562c37a2fb6b96df', '_compare': '7370859da62e853ba8a197c5ba6e4f07f7f41f1f815a55c24cb4c54ed73d3390', '_retry': '1ac6a6d4264ec7293e454d1721c31e136efdb38788ec8e16081461ece902fd2a', 'run_compile': 'dafa99d58c67b18fcdcd9fae2c81b87505f7e0487b351837744a828e0a147328', 'run_correctness': '56d4b423f3fd74d8c98c8a38042c8d5e89f942ee9d6fdc3d4c70a8e146ec1260', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': '0f743381713215eb7b13801d171208ff277c7ec3018962efef6a6b667e11a3f9', '_require_candidate_outputs': 'e5f69466d2ec4497fe192bf53ce8d7b034bb14cd59272fc7b1dea160a574e09e', 'arena_benchmark': 'b397c6b04fcc8c4a7e61af22a14bb4a112c188dde2f2c75f1f085e27d9f7268b'}, 'smoothquant_kernel': {'_resolve_kernel_dir': 'ebfabcdd05c1a3f254b035e42b1783890c241f92613bda21ccb77fa064039f1a', '_load_module': 'caae8222257891caee15a695bfc7cff029ec363f792c4331f20ff92c86efb520', '_make_inputs': '8503f4be576083cc00f8040f453ec57700f13c5ea3efd59e917aaa878f517bf2', '_aiter_op': '1b8156d73d386fff80bb5ec7f30145d953bcda12cc353b70151b012cd5dee5ea', '_compare': '26b0aed52720cd3ac91782410828b02e1947aed8ed614f5b2ab148c5dabb5be2', '_retry': '1ac6a6d4264ec7293e454d1721c31e136efdb38788ec8e16081461ece902fd2a', 'run_compile': 'dafa99d58c67b18fcdcd9fae2c81b87505f7e0487b351837744a828e0a147328', 'run_correctness': '1d3750228c51807a019c5ee85beddfbd10ba710455ba12c69768401cff0faf7f', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': 'cd88a8ef49765a1af92d2479df336632e06aa028f859ec698767dd016d0415bc', '_require_candidate_outputs': 'e5f69466d2ec4497fe192bf53ce8d7b034bb14cd59272fc7b1dea160a574e09e', 'arena_benchmark': 'c7a397c370af7236f8aee2551f052dcdc1ae2a1d0495b8133925ec4d8d394db0'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/torch2flydsl'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveFusedQuantChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)
        cfg=yaml.safe_load((task/'config.yaml').read_text())
        assert [e['symbol'] for e in cfg['candidate']['entrypoints']]==['flydsl_'+name.removesuffix('_kernel')]


@pytest.mark.parametrize("name", _ROPE_TWO_NAMES)
@pytest.mark.parametrize("correct", [True, False])
def test_rope_action_imports_are_task_local_and_propagate_correctness(name, correct):
    # Execute the actual action and audit imports from the isolated task cwd.
    # No fake scripts package can conceal a missing copied dependency here.
    task = ROOT / "tasks/torch2flydsl" / name
    program = """
from types import SimpleNamespace
from scripts.task_actions import check
calls = []
def run_correctness(verbose):
    calls.append(verbose)
    return EXPECTED
h = SimpleNamespace(ARENA_PROVIDED_BASELINE=True, run_correctness=run_correctness)
try:
    observed = check(h)
except RuntimeError as error:
    assert not EXPECTED and 'Correctness/output-contract' in str(error)
else:
    assert EXPECTED and observed == []
assert calls == [True]
""".replace("EXPECTED", repr(correct))
    result = subprocess.run([sys.executable, "-c", program], cwd=task, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('fault', ['none', 'missing_eps', 'squared_eps'])
def test_rmsnorm_reference_controls_reject_missing_or_misplaced_epsilon(fault, tmp_path):
    task = tmp_path / 'rmsnorm'
    shutil.copytree(ROOT / 'tasks/triton2flydsl/aiter/rmsnorm', task)
    if fault != 'none':
        path = task / 'test_kernel_harness.py'
        source = path.read_text()
        assert source.count(' * (1.0 / N) + EPS)') == 1
        replacement = ' * (1.0 / N) + ' + ('0.0)' if fault == 'missing_eps' else 'EPS * EPS)')
        path.write_text(source.replace(' * (1.0 / N) + EPS)', replacement))
    result = invoke(task, 'validate-task')
    assert result.passed == (fault == 'none'), result.reason
    if fault == 'none':
        assert len(result.cases) == 20
        controls = result.metadata['reference_controls']
        assert len(controls) == 3
        assert all(row['negative_output'] == 'rejected' for row in controls)
    else:
        assert 'reference output' in result.reason or 'known answer' in result.reason


@pytest.mark.parametrize('dtype_name', ['float32', 'float16', 'bfloat16'])
def test_rmsnorm_reference_epsilon_scale_matches_independent_fp64_math(dtype_name):
    import math
    import torch
    task = ROOT / 'tasks/triton2flydsl/aiter/rmsnorm'
    tree = ast.parse((task / 'test_kernel_harness.py').read_text())
    eps = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'EPS' for t in n.targets))
    ns = {'EPS': eps}
    _harness_functions(task, {'_torch_rmsnorm'}, ns)
    dtype = getattr(torch, dtype_name)
    v = math.sqrt(ns['EPS'])
    x = torch.tensor([[0., 0.], [v, -v], [1., 3.]], dtype=dtype)
    weight = torch.tensor([2., 4.], dtype=dtype)
    expected = []
    for row in x.tolist():
        denominator = math.sqrt(sum(value * value for value in row) / 2 + ns['EPS'])
        expected.append([value * w / denominator for value, w in zip(row, weight.tolist())])
    expected = torch.tensor(expected, dtype=dtype)
    actual = ns['_torch_rmsnorm'](x, weight, dtype)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


_SGLANG_ELEMENTWISE_NAMES = ['gdn_l2norm_fwd', 'fused_norm_gate', 'chunk_local_cumsum']


class _RemoveSglangElementwiseChecks(_RemoveAddedReplayChecks):
    def visit_Assign(self, node):
        if len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in {'ENTRY', 'protected_inputs', 'replay_validate'}:
            return None
        return super().visit_Assign(node)

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call):
            if getattr(call.func, 'id', None) == '_checked_sglang_output':
                return None
            if (isinstance(call.func, ast.Attribute) and call.func.attr == 'update'
                    and call.args and isinstance(call.args[0], ast.Call)
                    and getattr(call.args[0].func, 'id', None) == 'replay_validate'):
                return None
        return super().visit_Expr(node)

    def visit_FunctionDef(self, node):
        if node.name == 'fn':
            node.body = [ast.Expr(statement.value) if isinstance(statement, ast.Return) else statement for statement in node.body]
        return self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name == 'error' and any(isinstance(n, ast.Constant) and n.value == 'performance case failed: ' for n in ast.walk(node)):
            node.name = None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        for i, key in enumerate(node.keys):
            if isinstance(key, ast.Constant) and key.value == 'benchmark_fallback_reason':
                node.values[i] = ast.Constant('performance case failed before timing completed')
        return self.generic_visit(node)


@pytest.mark.parametrize('name', _SGLANG_ELEMENTWISE_NAMES)
@pytest.mark.parametrize('variant', [0, 1, 2])
@pytest.mark.parametrize('phase,behavior', [(phase, behavior) for phase in ['correctness', 'performance'] for behavior in ['correct', 'shape', 'dtype', 'device', 'nan', 'input_modified', 'measured_wrong', 'replay_wrong', 'cached'] if phase == 'performance' or behavior not in {'measured_wrong', 'replay_wrong', 'cached'}])
def test_sglang_elementwise_real_original_gates_and_measured_replay(name, variant, phase, behavior, monkeypatch):
    import torch
    import types
    task = ROOT / 'tasks/triton2flydsl/sglang' / name
    checks = module(task / 'scripts/replay_checks.py')
    ns = {'EPS': 1e-6 if name == 'gdn_l2norm_fwd' else 1e-5, 'CHUNK_SIZE': 2,
          'DTYPE_NAME': 'bfloat16', 'WARMUP_ITERATIONS': 10, 'BENCHMARK_ITERATIONS': 100,
          'require_tensor_contract': checks.require_tensor_contract,
          'require_unchanged': checks.require_unchanged, 'verify_timed_run': checks.verify_timed_run}
    _harness_functions(task, {'reference_l2norm', 'reference_norm_gate', 'reference_cumsum', '_shape_of', '_checked_sglang_output', '_compare_sglang_output', '_sglang_replay_validator'}, ns)
    if name == 'gdn_l2norm_fwd':
        x = torch.tensor([[3., 4., 1., -2.], [-1., 2., 3., 4.]], dtype=torch.bfloat16)
        inputs = (x,)
        shape = (2, 4)
        entry = 'l2norm_fwd'
        oracle = lambda: ns['reference_l2norm'](x, ns['EPS'])
        ns['make_x'] = lambda *args: x
    elif name == 'fused_norm_gate':
        x = torch.tensor([[1., 3., -2., 4.], [-1., 2., 3., 4.]], dtype=torch.bfloat16)
        inp = dict(x=x, g=x.clone().mul_(.3), weight=torch.tensor([2., 1., -.5, 3.], dtype=x.dtype), bias=torch.ones(4, dtype=x.dtype) if variant == 2 else None)
        inputs = tuple(v for v in inp.values() if v is not None)
        shape = (2, 4, variant != 2, 'sigmoid' if variant == 1 else 'swish', variant == 2)
        entry = 'layer_norm_gated_fwd'
        oracle = lambda: ns['reference_norm_gate'](inp, shape[2], shape[3])
        ns['make_test_data'] = lambda *args: inp
    else:
        shape = dict(ndim=4 if variant == 2 else 3, B=1, T=5, H=1, S=2, reverse=variant == 1, scale=.5 if variant == 2 else None)
        x = -torch.arange(1, 11 if variant == 2 else 6, dtype=torch.float32).reshape(ns['_shape_of'](shape))
        inputs = (x,)
        entry = 'chunk_local_cumsum'
        oracle = lambda: ns['reference_cumsum'](x, shape)
        ns['make_g'] = lambda *args: x
    originals = tuple(v.clone() for v in inputs)
    cached = oracle().clone()
    state = {'phase': 'setup'}
    def compute(*args, **kwargs):
        out = oracle()
        active = phase == 'correctness' or state['phase'] == 'measured'
        if active:
            if behavior == 'shape': out = out.reshape(-1)
            if behavior == 'dtype': out = out.double()
            if behavior == 'device': out = out.to('meta')
            if behavior == 'nan': out.fill_(float('nan'))
            if behavior == 'input_modified': x.mul_(.5)
        if behavior == state['phase'] + '_wrong': out.zero_()
        if state['phase'] == 'replay' and behavior == 'cached': out = cached.clone()
        return (out, None, None, x) if name == 'fused_norm_gate' else out
    class Collector:
        bound = False
    calls = []
    def benchmark(fn, *, warmup, repetition, timed_run):
        calls.append((warmup, repetition))
        state['phase'] = 'measured'
        timed_run.outputs = fn()
        timed_run.bound = True
        state['phase'] = 'setup'
        def replay():
            state['phase'] = 'replay'
            try:
                return fn()
            finally:
                state['phase'] = 'setup'
        timed_run.rerun = replay
        return .1, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    ns.update(TEST_SHAPES=[shape], load_module=lambda: types.SimpleNamespace(**{entry: compute}), _retry_oom=lambda fn: fn(), TimedRun=Collector, benchmark_cuda_graph_or_events=benchmark)
    _harness_functions(task, {'run_correctness', 'run_performance'}, ns)
    result = ns['run_' + phase]()
    if phase == 'correctness':
        assert result[0] == (behavior == 'correct'), result
        assert len(result[2]) == 1
    else:
        assert len(result) == 1
        assert calls == [(0, 100)]
        if behavior == 'correct':
            assert result[0]['timed_output_correctness'] == result[0]['replay_correctness'] == 'PASS'
            assert result[0]['benchmark_timed_run_kind'] == 'captured_graph'
        else:
            assert result[0]['execution_time_ms'] == -1
            assert 'PASS' not in result[0].values()
    if behavior != 'input_modified':
        checks.require_unchanged(inputs, originals)


def test_sglang_norm_gate_preserves_original_error_fraction_policy():
    import torch
    task = ROOT / 'tasks/triton2flydsl/sglang/fused_norm_gate'
    checks = module(task / 'scripts/replay_checks.py')
    ns = {'require_tensor_contract': checks.require_tensor_contract}
    _harness_functions(task, {'_checked_sglang_output', '_compare_sglang_output'}, ns)
    expected = torch.ones((1, 200), dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, :3] = 2
    assert not torch.allclose(actual.float(), expected.float(), atol=.02, rtol=.01)
    ns['_compare_sglang_output'](actual, expected)
    actual[0, :5] = 2
    with pytest.raises(AssertionError, match='error fraction'):
        ns['_compare_sglang_output'](actual, expected)
    actual = expected.clone()
    actual[0, 0] = float('nan')
    with pytest.raises(AssertionError, match='Non-finite'):
        ns['_compare_sglang_output'](actual, expected)


@pytest.mark.parametrize('name', _SGLANG_ELEMENTWISE_NAMES)
def test_sglang_elementwise_copied_audit_and_action_propagation(name, monkeypatch):
    import torch
    import types
    task = ROOT / 'tasks/triton2flydsl/sglang' / name
    audit = module(task / 'scripts/candidate_checks.py')
    actions = module(task / 'scripts/task_actions.py')
    monkeypatch.setitem(sys.modules, 'scripts.candidate_checks', audit)
    details = [{'shape_id': i + 1, 'passed': True} for i in range(actions.CORRECTNESS_COUNT)]
    h = types.SimpleNamespace(ARENA_FINAL_CANDIDATE=False, run_correctness=lambda: (True, None, details))
    assert actions.check(h) == []
    details[-1]['passed'] = False
    with pytest.raises(RuntimeError, match='Invalid or failed'):
        actions.check(h)
    with pytest.raises(RuntimeError, match='non-preparation PyTorch operation'):
        with audit.candidate_preparation_only():
            torch.ones(2).square()
    assert (task / 'scripts/candidate_checks.py').read_bytes() == (ROOT / 'tasks/triton2flydsl/aiter/fp8_mqa_logits/scripts/candidate_checks.py').read_bytes()
    assert (task / 'scripts/replay_checks.py').read_bytes() == (ROOT / 'tasks/triton2flydsl/aiter/fp8_mqa_logits/scripts/replay_checks.py').read_bytes()


def test_sglang_elementwise_original_inputs_seeds_references_gates_and_timing():
    hashes = {'gdn_l2norm_fwd': {'load_module': 'c36c53e82ca1a8838919c427b1a370d2ef0438789a451e3a3df7c2e0f5c6df0a', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_x': '43a8c24b405e3c222c5d716f8a4c7d8fd8a736f98935914f8b66c2f497be37c8', 'reference_l2norm': 'e84d9abe9ec04b64a76502104c61db81b66aa1a2c972fe14dd759b28ba0dbb01', 'run_compile': '0b48af5b7f311fb4f4219c09fee8175b1becb7715f1ba47e0ada836dfc6969d0', 'run_correctness': '67db83bf8dc83a9cff2faad8dba9d710e575dc6bd973e7e558bcd6ab0093ba97', 'run_performance': 'b2f5193d24a348ef8517fdbaaf6f8d1c6a4ed689daec1f508b6972e77b2fa886', 'main': 'f71a0eca447c73e79099d35ccf944d04173587ec5c41a316b6245efb0b6fca4f'}, 'fused_norm_gate': {'load_module': '6c0eaa337759666cb08631be7de786cad7efd062fa04f0e064d89fdc985fc30a', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_test_data': '69948fb1e87722247812dc5068f0cbbf694ed31ac703e8a4c831e637b7b97d9b', 'reference_norm_gate': '41640fb658ce94671a055326711c46a6f642f62bf1b638f8d0a4ccbad06cb53c', 'run_compile': '1db90d55271c579ee45aec9eda512006a49b31b88e77f044af0994324903ca5c', 'run_correctness': '6b7d95841f019489e89caa957fe7ae21746a228c1889da759763f529cc955769', 'run_performance': '6de4d4747db4e2da5d1a2364ec88fe3f9825e50763dfb18075654abaa9c8033f', 'main': '4cd9b931b1bb38cc5868ed2f101b083f973cd9663e4429f8870faead285981f6'}, 'chunk_local_cumsum': {'load_module': '4ca849e18c8b68f3a230283dd813b8085f55b7863e18cc025b00370a714397e7', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_g': 'f60c65683f65c46600f3cc871ec82c15a616d3f58aae311158b32fbbdcc5608c', 'reference_cumsum': 'ef53138ec4c7abfc028b8096edad2e4d3fcf0edb706a16890860c5fbf9d85e67', '_shape_of': 'c111669b72d1d44501e511a2873860b9be4827e8b6de9e20bad6eca53c4548d8', 'run_compile': 'f567feb9581378fd43f738b902c3dc860aae0f55197336d9bbe3622e4919d4cc', 'run_correctness': 'ef65af08b9b5aff5ab8a58e4842d3d1cde2e13c851b248f13904740a43f9a2a6', 'run_performance': 'e0f378044e5ece6aeac90876a689387d636e5a9ea7dfec307282b3ef69199ae7', 'main': '619e83159bd006e04c6f20f5a206688e1f05896109808711979b45aa6427e8e3'}}
    for name, functions in hashes.items():
        task = ROOT / 'tasks/triton2flydsl/sglang' / name
        for fn in ast.parse((task / 'test_kernel_harness.py').read_text()).body:
            if isinstance(fn, ast.FunctionDef) and fn.name in functions:
                normalized = _RemoveSglangElementwiseChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest() == functions[fn.name], (name, fn.name)


@pytest.mark.parametrize('factor', [0.5, 1., 2.])
def test_fp8_token_reference_exact_rational_ties_and_neighbors(factor):
    from fractions import Fraction
    import torch
    task = ROOT / 'tasks/triton2flydsl/aiter/dynamic_quant_fp8'
    ns = {}
    _harness_functions(task, {'_reference_quant', '_dtype_max'}, ns)
    values = [[.4453125, .443359375, .447265625, 2.625],
              [-.4453125, -.443359375, -.447265625, -2.625]]
    # Independent rational arithmetic determines each side of the 76 midpoint.
    scale = Fraction(21, 8) * Fraction(factor) / 448
    assert Fraction(values[0][0]) * Fraction(factor) / scale == 76
    assert Fraction(values[0][1]) * Fraction(factor) / scale < 76
    assert Fraction(values[0][2]) * Fraction(factor) / scale > 76
    x = (torch.tensor(values) * factor).to(torch.bfloat16)
    q, observed_scale = ns['_reference_quant'](x, torch.float8_e4m3fn, 'dyn_token')
    expected = torch.tensor([[80., 72., 80., 448.], [-80., -72., -80., -448.]])
    assert torch.equal(q.float(), expected)
    assert observed_scale.dtype == torch.float32
    assert torch.equal(observed_scale, torch.full((2,), float(scale)))
    assert torch.equal(x, (torch.tensor(values) * factor).to(torch.bfloat16))


@pytest.mark.parametrize('bad_rounding', [False, True])
def test_fp8_token_runner_known_answer_rejects_wrong_tie_rounding(bad_rounding, tmp_path):
    task = tmp_path / 'quantizer'
    shutil.copytree(ROOT / 'tasks/triton2flydsl/aiter/dynamic_quant_fp8', task)
    if bad_rounding:
        p = task / 'test_kernel_harness.py'
        source = p.read_text()
        old = 'return torch.div(x_f64, scale_f64[:, None]).to(qdtype), scale_f64.float()'
        assert old in source
        # Preserve unit-scale extrema while corrupting exact76 ties only.
        new = 'normalized = torch.div(x_f64, scale_f64[:, None])\n        normalized = torch.where(normalized.abs() == 76., normalized * .99, normalized)\n        return normalized.to(qdtype), scale_f64.float()'
        p.write_text(source.replace(old, new))
    result = invoke(task, 'validate-task')
    assert result.passed == (not bad_rounding), result.reason
    if bad_rounding:
        assert 'known answer' in result.reason and 'half-way' in result.reason
    else:
        assert len(result.cases) == 48
        assert len(result.metadata['reference_controls']) == 14
        assert all(row['negative_output'] == 'rejected' for row in result.metadata['reference_controls'])


_TRITON_SCALED_GEMM_NAMES = ['gemm_a8w8', 'gemm_a16w8_blockscale', 'gemm_a8w8_blockscale', 'gemm_afp8wfp8']


@pytest.mark.parametrize('name,phase,behavior', [(name, phase, behavior) for name in _TRITON_SCALED_GEMM_NAMES for phase in ['correctness', 'benchmark'] for behavior in ['correct', 'shape', 'dtype', 'device', 'nan', 'input_modified', 'weight_modified', 'scale_modified', 'bias_modified', 'measured_wrong', 'replay_wrong', 'cached'] if (behavior != 'bias_modified' or name == 'gemm_a8w8' and phase == 'correctness') and (phase != 'correctness' or behavior not in {'measured_wrong', 'replay_wrong', 'cached'})])
def test_triton_scaled_gemm_real_original_rules_and_measured_replay(name, phase, behavior, monkeypatch, tmp_path):
    import torch
    import math
    import types
    task = ROOT / 'tasks/triton2flydsl/aiter' / name
    checks = module(task / 'scripts/replay_checks.py')
    ns = {}
    for n in ast.parse((task / 'test_kernel_harness.py').read_text()).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try: ns[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError): pass
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple):
            try: values = ast.literal_eval(n.value)
            except (ValueError, TypeError): continue
            ns.update({target.id: value for target, value in zip(n.targets[0].elts, values) if isinstance(target, ast.Name)})
    _harness_functions(task, {'_torch_ref', '_e8m0_to_f32'}, ns)
    k = 128
    n = 128 if name == 'gemm_afp8wfp8' else 2
    x = torch.arange(1, 2*k+1).reshape(2, k).remainder(7).add(1).to(torch.bfloat16 if name == 'gemm_a16w8_blockscale' else torch.float8_e4m3fn)
    w = torch.arange(1, n*k+1).reshape(n, k).remainder(5).add(1).to(torch.float8_e4m3fn)
    bias = torch.ones((1, n), dtype=torch.float32)
    if name == 'gemm_afp8wfp8':
        scales = (torch.full((2, k//32), 127, dtype=torch.uint8), torch.full((n//128, k//128), 128, dtype=torch.uint8))
    elif name == 'gemm_a16w8_blockscale': scales = (torch.tensor([[.5]]),)
    elif name == 'gemm_a8w8_blockscale': scales = (torch.tensor([[.5], [.25]]), torch.tensor([[.75]]))
    else: scales = (torch.tensor([[.5], [.25]]), torch.tensor([[.125, .25]]))
    inputs = (x, w, *scales, bias)
    originals = tuple(v.clone() for v in inputs)
    def args(with_bias=False):
        return (x, w, *scales, bias if with_bias else None, torch.bfloat16) if name == 'gemm_a8w8' else (x, w, *scales, torch.bfloat16)
    cached = ns['_torch_ref'](*args())
    state = {'phase': 'setup'}
    def compute(*values, **kw):
        if 'dtype' in kw: values = (*values, kw['dtype'])
        out = ns['_torch_ref'](*values)
        active = phase == 'correctness' or state['phase'] == 'measured'
        if active:
            if behavior == 'shape': out = out[:1]
            if behavior == 'dtype': out = out.float()
            if behavior == 'device': out = out.to('meta')
            if behavior == 'nan': out.fill_(float('nan'))
            if behavior == 'input_modified': x.view(torch.uint8).bitwise_xor_(1)
            if behavior == 'weight_modified': w.view(torch.uint8).bitwise_xor_(1)
            if behavior == 'scale_modified': scales[0].add_(1)
            if behavior == 'bias_modified' and values[-2] is not None: bias.add_(1)
        if behavior == state['phase'] + '_wrong': out.zero_()
        if state['phase'] == 'replay' and behavior == 'cached': out = cached.clone()
        return out
    class Collector:
        bound = False
    calls = []
    def benchmark(fn, *, warmup, repetition, timed_run):
        calls.append((warmup, repetition))
        state['phase'] = 'measured'; timed_run.outputs = fn(); timed_run.bound = True; state['phase'] = 'setup'
        def replay():
            state['phase'] = 'replay'
            try: return fn()
            finally: state['phase'] = 'setup'
        timed_run.rerun = replay
        return .1, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    ns.update(TimedRun=Collector, benchmark_cuda_graph_or_events=benchmark, require_unchanged=checks.require_unchanged, verify_timed_run=checks.verify_timed_run, allclose_output=checks.allclose_output,
              _load_source=lambda: types.SimpleNamespace(**{name: compute}), _fp8_e4m3_dtype=lambda: torch.float8_e4m3fn,
              _make_inputs=lambda *a: args(a[-1] if name == 'gemm_a8w8' else False)[:-1],
              TEST_SHAPES=[dict(name='controlled', M=2, N=n, K=k)], WARMUP=10, ITERS=100, _HERE=str(tmp_path), Path=Path, json=json, math=math)
    _harness_functions(task, {'_checked_scaled_gemm_output', '_scaled_gemm_replay_validator', 'run_correctness', 'run_benchmark'}, ns)
    if phase == 'correctness':
        assert ns['run_correctness'](verbose=False) == (behavior == 'correct')
    elif behavior == 'correct':
        rows = ns['run_benchmark'](verbose=False)
        assert rows[0]['timed_output_correctness'] == rows[0]['replay_correctness'] == 'PASS'
        assert rows[0]['benchmark_timed_run_kind'] == 'captured_graph'
    else:
        with pytest.raises(AssertionError): ns['run_benchmark'](verbose=False)
    if phase == 'benchmark': assert calls == [(0, 100)]
    if not behavior.endswith('_modified'): checks.require_unchanged(inputs, originals)


@pytest.mark.parametrize('name', _TRITON_SCALED_GEMM_NAMES)
def test_triton_scaled_gemm_audit_helper_and_static_dependencies(name, tmp_path):
    task = ROOT / 'tasks/triton2flydsl/aiter' / name
    runtime = module(task / 'task_runtime.py')
    candidate = tmp_path / 'candidate.py'
    candidate.write_text('import flydsl\nfrom aiter import gemm_a8w8\n')
    with pytest.raises(ValueError, match='Final operator must execute FlyDSL'):
        runtime.check_dependencies([candidate], True)
    for helper in ['candidate_checks.py', 'replay_checks.py']:
        assert (task / 'scripts' / helper).read_bytes() == (ROOT / 'tasks/triton2flydsl/aiter/rmsnorm/scripts' / helper).read_bytes()


def test_triton_scaled_gemm_original_refs_inputs_gates_bias_and_timing():
    hashes = {'gemm_a8w8': {'_fp8_e4m3_dtype': 'dce8a3e9b226a3bc53cc87697bd3973be028c96ccf5c71e61ec7667cfba2292a', '_load_source': '0109b3040e9f8cd2dd24ee21b710aa28eafca60be1a275b3481c54bead0535f4', '_make_inputs': '267dfecc9adb861fa25922aa1a683fc927c6732f4c9abf30d9896ab9767e81f7', '_torch_ref': '825e9cfcea4d5ab06de921ccfc00567cb9f010f5312bbe6a3f41e30e93e486b6', 'run_compile': '8f829e9dafe3fa510b965656f9805ccbd0727c1cb47081513760f5bd2d68c083', 'run_correctness': '36fd9f15a29a2efd01f60d65502f57af0978322148f19d3513112ab67d97e846', 'run_benchmark': '42711707b4bbda7e3b855c1ffa95bec3cabd6c56336a660859269bf8df3b7afd'}, 'gemm_a16w8_blockscale': {'_fp8_e4m3_dtype': 'dce8a3e9b226a3bc53cc87697bd3973be028c96ccf5c71e61ec7667cfba2292a', '_load_source': 'b4c9fd411a1dc0a557d5d7e34ebeb217dd8764c81c29b98dd38adb834c132ec0', '_make_inputs': '94ce346bf7f3e6f7694e32c17cfc137fa1e6158ff1ba820c6b922df63dab1e6c', '_torch_ref': 'cf0a0ef309f0a880ffb08ae655c4b080318d571162bb29cf41a4368e10d379e8', 'run_compile': '8f829e9dafe3fa510b965656f9805ccbd0727c1cb47081513760f5bd2d68c083', 'run_correctness': '04248b0bb97a7d4453f15b327d971d883fd0469f478eb79690d56ff9c69aa059', 'run_benchmark': 'a89f9c6b1a5f8dd4afafb2864bb674f9e6e5917b8b7811336417ea9cccd90a4f'}, 'gemm_a8w8_blockscale': {'_fp8_e4m3_dtype': 'dce8a3e9b226a3bc53cc87697bd3973be028c96ccf5c71e61ec7667cfba2292a', '_load_source': '41ba03fd742602a208651c5ecd7eabb4be2c2c52e8bc60cffb745b01e4d1a52d', '_make_inputs': '5a5bd4923a985c42de6f7a34471eb8322cb7ea58b6ae50bb3e1509b8a351569d', '_torch_ref': '9d24f0a69145ed3a490a564f817be94e7b21368a011304f41e289132e2055ff0', 'run_compile': '8f829e9dafe3fa510b965656f9805ccbd0727c1cb47081513760f5bd2d68c083', 'run_correctness': '5d9de3cafd955c9ee89b80ca5ed13a16abff4c4b60c8ba9863f31d6ed4edb605', 'run_benchmark': '72e2c9cc1bb490b7e273959fd0fe3c5fcd67c9ce1bc04ecd44db45d2674f478e'}, 'gemm_afp8wfp8': {'_load_source': '3d369c1f5bbcee527cb2dc79d301e71afaef8119ab0bc49ea64769a39a53a956', '_e8m0_to_f32': 'ffc2caaa5468c8c2a7b505ef046dfde68643edeb9921a53c1e63de6fa44a9f8e', '_make_inputs': 'bceee075948c1fae518a33c6b930db1172a09a9318f09b6bf6f3ac1f58b9eb99', '_torch_ref': 'f19986d79754472dd0cf416134b5671e187bae77583cfdf5398970ef847c6e12', 'run_compile': '8f829e9dafe3fa510b965656f9805ccbd0727c1cb47081513760f5bd2d68c083', 'run_correctness': '7849c97309a0b5b051c4e307c05bb6904b80a6917477eb56308d5839e1f44b56', 'run_benchmark': 'ffcc455e8d48031987a57ac9edae19f37e907706c6653db6f3a490a6e3c34120'}}
    for name, functions in hashes.items():
        task = ROOT / 'tasks/triton2flydsl/aiter' / name
        for fn in ast.parse((task / 'test_kernel_harness.py').read_text()).body:
            if isinstance(fn, ast.FunctionDef) and fn.name in functions:
                normalized = _RemoveTritonBatchedChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest() == functions[fn.name], (name, fn.name)


_TRITON_PAIR_NAMES = ['fused_add_rmsnorm', 'moe_routing_sigmoid_top1']


@pytest.mark.parametrize('name,shared', [('fused_add_rmsnorm', False), ('moe_routing_sigmoid_top1', False), ('moe_routing_sigmoid_top1', True)])
@pytest.mark.parametrize('phase,behavior', [(phase, behavior) for phase in ['correctness', 'benchmark'] for behavior in ['correct', 'wrong_first', 'wrong_second', 'dtype', 'shape', 'nan', 'input_modified', 'weight_modified', 'measured_wrong', 'replay_wrong', 'cached_first', 'cached_second'] if phase != 'correctness' or behavior not in {'measured_wrong', 'replay_wrong', 'cached_first', 'cached_second'}])
def test_triton_pair_actual_both_outputs_original_rules_and_replay(name, shared, phase, behavior, monkeypatch, tmp_path):
    import torch
    import types
    import math
    task = ROOT / 'tasks/triton2flydsl/aiter' / name
    checks = module(task / 'scripts/replay_checks.py')
    add = name == 'fused_add_rmsnorm'
    ns = {'EPS': 1e-5, 'require_tensor_contract': checks.require_tensor_contract, 'require_unchanged': checks.require_unchanged,
          'verify_timed_pair': checks.verify_timed_pair, 'allclose_output': checks.allclose_output}
    _harness_functions(task, {'_torch_rmsnorm', '_torch_dtype', '_torch_routing_ref', '_checked_pair_output', '_compare_routing_pair', '_pair_replay_validator'}, ns)
    dtype = torch.bfloat16 if add else torch.float16
    x = torch.tensor([[1., 2.], [-1., 3.]], dtype=dtype)
    residual = torch.tensor([[3., -1.], [1., 2.]], dtype=dtype)
    weight = torch.tensor([2., 4.], dtype=dtype) if add else torch.tensor([[.5, -.5], [0., .25]], dtype=dtype)
    inputs = (x, residual, weight) if add else (x, weight)
    originals = tuple(v.clone() for v in inputs)
    def reference():
        if add:
            summed = x + residual
            return ns['_torch_rmsnorm'](summed, weight, x.dtype), summed
        return ns['_torch_routing_ref'](x, weight, 2, shared)[:2]
    cached = tuple(v.clone() for v in reference())
    state = {'phase': 'setup'}
    def outputs():
        first, second = reference()
        active = phase == 'correctness' or state['phase'] == 'measured'
        if active:
            if behavior == 'wrong_first': first.fill_(-100)
            if behavior == 'wrong_second': second.zero_()
            if behavior == 'dtype': first = first.float() if add else first.long()
            if behavior == 'shape': first = first[:1]
            if behavior == 'nan': second = second.float(); second.fill_(float('nan'))
            if behavior == 'input_modified': x.mul_(.5)
            if behavior == 'weight_modified': weight.mul_(.5)
        if behavior == state['phase'] + '_wrong': first.fill_(-100)
        if state['phase'] == 'replay':
            if behavior == 'cached_first': first = cached[0].clone()
            if behavior == 'cached_second': second = cached[1].clone()
        return first, second
    if add:
        def kernel(out, _x, _residual, residual_out, _weight, eps):
            a, b = outputs()
            # The real _run_kernel provides the two destination buffers.
            # Altering their Tensor metadata represents a malformed submission.
            out.data = a
            residual_out.data = b
        mod = types.SimpleNamespace(rmsnorm2d_fwd_with_add=kernel)
    else:
        mod = types.SimpleNamespace(routing_sigmoid_top1=lambda *a, **kw: outputs())
    class Collector:
        bound = False
    calls = []
    def benchmark(fn, *, warmup, repetition, timed_run):
        calls.append((warmup, repetition)); state['phase'] = 'measured'; timed_run.outputs = fn(); timed_run.bound = True; state['phase'] = 'setup'
        def replay():
            state['phase'] = 'replay'
            try: return fn()
            finally: state['phase'] = 'setup'
        timed_run.rerun = replay
        return .1, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    ns.update(TimedRun=Collector, benchmark_cuda_graph_or_events=benchmark, _load_source=lambda: mod, _make_inputs=lambda *a: inputs,
              TEST_SHAPES=[dict(name='controlled', M=2, N=2, K=2, shared=shared)], DTYPES=['bf16'], WARMUP=10, ITERS=100,
              _HERE=str(tmp_path), Path=Path, json=json, math=math)
    _harness_functions(task, {'_run_kernel', 'run_correctness', 'run_benchmark'}, ns)
    if phase == 'correctness':
        assert ns['run_correctness'](verbose=False) == (behavior == 'correct')
    elif behavior == 'correct':
        rows = ns['run_benchmark'](verbose=False)
        assert rows[0]['timed_output_correctness'] == rows[0]['replay_correctness'] == 'PASS'
    else:
        with pytest.raises(AssertionError): ns['run_benchmark'](verbose=False)
    if phase == 'benchmark': assert calls == [(0, 100)]
    if not behavior.endswith('_modified'): checks.require_unchanged(inputs, originals)


def test_triton_router_preserves_near_tie_policy_but_rejects_invalid_ids_and_shared_column():
    import torch
    task = ROOT / 'tasks/triton2flydsl/aiter/moe_routing_sigmoid_top1'
    ns = {}
    _harness_functions(task, {'_checked_pair_output', '_compare_routing_pair'}, ns)
    x = torch.zeros((2, 2), dtype=torch.float16)
    scores = torch.tensor([[.8, .795], [.5, .499]])
    ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.int32)
    weights = torch.tensor([[.8, 1.], [.5, 1.]])
    expected = (torch.tensor([[0, 2], [0, 2]], dtype=torch.int32), weights.clone(), scores)
    ns['_compare_routing_pair']((ids, weights), expected, x, 2, True)
    bad = ids.clone(); bad[0, 0] = 3
    with pytest.raises(AssertionError, match='out-of-range'):
        ns['_compare_routing_pair']((bad, weights), expected, x, 2, True)
    bad = ids.clone(); bad[0, 1] = 0
    with pytest.raises(AssertionError, match='shared-expert'):
        ns['_compare_routing_pair']((bad, weights), expected, x, 2, True)
    wrong = weights.clone(); wrong[0, 1] = .995
    # General0.01 tolerance accepts this; stricter shared0.001 still rejects it.
    with pytest.raises(AssertionError, match='shared-expert'):
        ns['_compare_routing_pair']((ids, wrong), expected, x, 2, True)


def test_triton_pair_original_refs_inputs_cases_gates_and_timing():
    hashes = {'fused_add_rmsnorm': {'_torch_dtype': '664e9bfa3e0f1b752c5ff297485baf851e06ceaa2bc5873012159e8ccff77add', '_load_source': '272886dc4f96f1966857ceb0edebc52609ecd2345cb569ea981ca9fedd83a70b', '_make_inputs': '75bff69f5f24c8488f117d47cfc47b9d46e774519cc432ddb7761b552e2709b5', '_torch_rmsnorm': 'd88d784ecb418ebe8356af643af74e0c66fc9e712a25e516463ffbb64cead7b2', '_run_kernel': '8f1dbde7a551c9423a793f77fb117939ed149eb5003404d72c66d2026e70a474', 'run_compile': '8f829e9dafe3fa510b965656f9805ccbd0727c1cb47081513760f5bd2d68c083', 'run_correctness': '7446b51344422331bfdef144a566e8146ff37463d3231537c1ca00d1e89b85f3', 'run_benchmark': 'facd33a51e5d921806af64cee7fe82b6d1aad8d42ea4e47620ae35fbe3a390be'}, 'moe_routing_sigmoid_top1': {'_load_source': 'a48ac9a4135e768cb20bcc94aa00bd263bc400eeb763316f286479a63796d3d3', '_make_inputs': '754cfc98f97cc42452f7abb867e482dd1f7cabfa034b59c07c8defeca5a8f5db', '_torch_routing_ref': '791cc240d003114cc7188f9274fbbb927b3d90554f8e65bfe3f4d90f73cb5b16', 'run_compile': '8f829e9dafe3fa510b965656f9805ccbd0727c1cb47081513760f5bd2d68c083', 'run_correctness': '3fb21949bfa8b1f3f7fe86fcccc5ebd69ad71793a374960b40db02b981b3f60e', 'run_benchmark': 'b8e88f1e82a7c40372ab49703c0b858298dbb5a4483dd3395d8b177e0d3e56fa'}}
    for name, functions in hashes.items():
        task = ROOT / 'tasks/triton2flydsl/aiter' / name
        for fn in ast.parse((task / 'test_kernel_harness.py').read_text()).body:
            if isinstance(fn, ast.FunctionDef) and fn.name in functions:
                normalized = _RemoveTritonBatchedChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest() == functions[fn.name], (name, fn.name)
    a = ROOT / 'tasks/triton2flydsl/aiter/fused_add_rmsnorm/scripts/replay_checks.py'
    b = ROOT / 'tasks/triton2flydsl/aiter/moe_routing_sigmoid_top1/scripts/replay_checks.py'
    assert a.read_bytes() == b.read_bytes()


_ATTENTION_TWO = ['aiter/mha', 'sglang/prefill_attention']


class _RemoveAttentionChecks(_RemoveSglangElementwiseChecks):
    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and getattr(call.func, 'id', None) == '_checked_attention_output':
            return None
        if ast.unparse(node) == "o.fill_(float('nan'))":
            return None
        return super().visit_Expr(node)

    def visit_FunctionDef(self, node):
        if node.name == 'fn':
            node.body = [s for s in node.body if not (isinstance(s, ast.Return) and getattr(s.value, 'id', None) == 'o')]
        return self.generic_visit(node)


@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('name,phase,behavior', [(name, phase, behavior) for name in _ATTENTION_TWO for phase in ['correctness', 'performance'] for behavior in ['correct', 'wrong', 'shape', 'dtype', 'device', 'nan', 'input_modified', 'kv_modified', 'metadata_modified', 'measured_wrong', 'replay_wrong', 'cached', 'unwritten'] if (phase == 'performance' or behavior not in {'measured_wrong', 'replay_wrong', 'cached'}) and not (name.endswith('/mha') and behavior in {'metadata_modified', 'unwritten'})])
def test_attention_two_real_contracts_original_gates_and_measured_replay(name, causal, phase, behavior, monkeypatch):
    import torch
    import types
    prefill = name.endswith('prefill_attention')
    task = ROOT / 'tasks/triton2flydsl' / name
    checks = module(task / 'scripts/replay_checks.py')
    ns = dict(NORM_ERR_TOL=.01, ALLCLOSE_TOL=.01, WARMUP_ITERATIONS=10, BENCHMARK_ITERATIONS=100,
              require_tensor_contract=checks.require_tensor_contract, require_unchanged=checks.require_unchanged,
              verify_timed_run=checks.verify_timed_run)
    _harness_functions(task, {'torch_mha_ref', '_compare', 'reference', '_shape_of', '_checked_attention_output', '_compare_attention_output', '_attention_replay_validator', '_call_kernel'}, ns)
    dtype = torch.bfloat16 if prefill else torch.float16
    q = torch.zeros((3, 2, 2) if prefill else (1, 3, 2, 2), dtype=dtype)
    k = torch.zeros((3, 1, 2) if prefill else (1, 3, 1, 2), dtype=dtype)
    v = torch.tensor([2., 4., 6., 8., 10., 12.], dtype=dtype).reshape(k.shape)
    o = torch.empty_like(q)
    bsl = torch.tensor([0, 2], dtype=torch.int32); bseq = torch.tensor([2, 1], dtype=torch.int32)
    cfg = dict(seqs=[2, 1], head=2, kv_head=1, d=2, causal=causal)
    shape = cfg if prefill else (1, 3, 2, 1, 2, causal)
    inputs = (q, k, v, bsl, bseq) if prefill else (q, k, v)
    originals = tuple(x.clone() for x in inputs)
    def oracle():
        return ns['reference'](q, k, v, cfg).to(dtype) if prefill else ns['torch_mha_ref'](q, k, v, .5, causal)
    cached = oracle().clone(); state = {'phase': 'setup'}
    def compute(*args, **kwargs):
        out = oracle()
        if phase == 'correctness' or state['phase'] == 'measured':
            if behavior == 'wrong': out.zero_()
            if behavior == 'shape': out = out.reshape(-1)
            if behavior == 'dtype': out = out.double()
            if behavior == 'device': out = out.to('meta')
            if behavior == 'nan': out.fill_(float('nan'))
            if behavior == 'input_modified': q.add_(1)
            if behavior == 'kv_modified': v.mul_(.5)
            if behavior == 'metadata_modified': bseq.add_(1)
        if behavior == state['phase'] + '_wrong': out.zero_()
        if state['phase'] == 'replay' and behavior == 'cached': out = cached.clone()
        if not prefill:
            return out
        if behavior == 'unwritten':
            return out  # Returning a tensor must not substitute for writing o.
        if behavior == 'device' and (phase == 'correctness' or state['phase'] == 'measured'):
            # Tensor.data cannot move to meta: a malformed output buffer still fails.
            raise AssertionError('wrong output device')
        o.data = out
    class Collector:
        bound = False
    calls = []
    def benchmark(fn, *, warmup, repetition, timed_run):
        calls.append((warmup, repetition)); state['phase'] = 'measured'
        timed_run.outputs = fn(); timed_run.bound = True; state['phase'] = 'setup'
        def replay():
            state['phase'] = 'replay'
            try: return fn()
            finally: state['phase'] = 'setup'
        timed_run.rerun = replay
        return .1, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    ns.update(TEST_SHAPES=[shape], TimedRun=Collector, benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda: types.SimpleNamespace(**{'context_attention_fwd' if prefill else 'flash_attn_func': compute}),
              _with_oom_retry=lambda fn: fn(), _retry_oom=lambda fn: fn(),
              make_test_data=lambda *a: (q, k, v, .5), make_inputs=lambda *a: (q, k, v, o, bsl, bseq, 2))
    _harness_functions(task, {'run_correctness', 'run_performance'}, ns)
    result = ns['run_' + phase]()
    if phase == 'correctness':
        assert result[0] == (behavior == 'correct'), result
        assert len(result[2]) == 1
    else:
        assert len(result) == 1 and calls == [(0, 100)]
        if behavior == 'correct':
            assert result[0]['timed_output_correctness'] == result[0]['replay_correctness'] == 'PASS'
            assert result[0]['benchmark_timed_run_kind'] == 'captured_graph'
        else:
            assert result[0]['execution_time_ms'] == -1
            assert 'PASS' not in result[0].values()
    if not behavior.endswith('_modified'):
        checks.require_unchanged(inputs, originals)


@pytest.mark.parametrize('name', _ATTENTION_TWO)
def test_attention_two_original_comparator_policies_and_controls(name):
    import torch
    task = ROOT / 'tasks/triton2flydsl' / name
    checks = module(task / 'scripts/replay_checks.py')
    ns = dict(require_tensor_contract=checks.require_tensor_contract, NORM_ERR_TOL=.01, ALLCLOSE_TOL=.01)
    _harness_functions(task, {'_compare', '_compare_attention_output'}, ns)
    dtype = torch.float16 if name.endswith('/mha') else torch.bfloat16
    expected = torch.ones((1, 2000), dtype=dtype)
    actual = expected.clone(); actual[0, 0] = 10
    if name.endswith('/mha'):
        with pytest.raises(AssertionError): ns['_compare_attention_output'](actual, expected)
        expected[0, 0] = 100; actual = expected.clone(); actual[0, 1] = 1.5
        assert not torch.allclose(actual, expected, atol=.01, rtol=.01)
        ns['_compare_attention_output'](actual, expected)  # allclose is diagnostic only.
    else:
        ns['_compare_attention_output'](actual, expected)  # Original 99.9% OR branch.
        actual[0, :10] = 10
        with pytest.raises(AssertionError): ns['_compare_attention_output'](actual, expected)
    with pytest.raises(AssertionError): ns['_compare_attention_output'](torch.full_like(expected, float('nan')), expected)
    with pytest.raises(AssertionError): ns['_compare_attention_output'](torch.ones_like(expected), torch.zeros_like(expected))


@pytest.mark.parametrize('name', _ATTENTION_TWO)
def test_attention_two_candidate_audit_and_complete_cpu_manifest(name, monkeypatch, tmp_path):
    task = ROOT / 'tasks/triton2flydsl' / name
    runtime = module(task / 'task_runtime.py')
    for helper in ['candidate_checks.py', 'replay_checks.py']:
        assert (task / 'scripts' / helper).read_bytes() == (ROOT / 'tasks/triton2flydsl/aiter/fp8_mqa_logits/scripts' / helper).read_bytes()
    bad = tmp_path / 'candidate.py'; bad.write_text('import flydsl\nfrom aiter import flash_attn_func\n')
    with pytest.raises(ValueError): runtime.check_dependencies([bad], True)
    result = invoke(task, 'validate-task')
    assert result.passed, result.reason
    assert len(result.cases) == (6 if name.endswith('/mha') else 7)
    assert result.metadata['candidate_state'] == 'implemented'
    assert all(c['known_answer'] == 'PASS' and c['negative_output'] == 'rejected' for c in result.metadata['reference_controls'])


def test_attention_two_original_inputs_refs_gates_sampling_and_seed_fingerprints():
    hashes = {'mha': {'load_module': '793850d95e2403d96a538054b9fb5d063ec1bb31edb056127faf56f160d65bca', 'make_test_data': 'fcdf094e192ab01a6072185a3f42a8d7cc5adde17fe2f97193465ebfa14c6513', '_call_kernel': '084b377ffacd2a2f3f87a86c659495a8db4ac05a3e2aad2ee2599a492233bc28', 'torch_mha_ref': '3fa2a991fc2c7ffc2d622589e0f34149201f28b4abca7dd967b3adfd860cb6c7', '_compare': 'ff61bc86d21eebb74ffa14e426d5c4907aa35276e150280692314f0e27f5d034', '_with_oom_retry': 'ba105db4902c9317e66f98405a423ac5fcca640ab7215affa8ef3a18b7a997ed', 'run_compile': '26edf1e9455cb93792f87e8ca4cb6e3559f4c04756b6a0ba8ca54c81d7892005', 'run_correctness': '7a15514bf4dc8fd42e1488d769277ffca92857b242ab219670c4e62bbc917670', 'run_performance': 'c3aff5f4991904ef81e2314761ab36c932b22bba690bd8e47d39bf04b185607e', 'main': '100ea598422d5e73bb3fd01546adb785d8757a1503fe677c1401dcf2ab8d6eb5'}, 'prefill_attention': {'load_module': '5d4fa6189cd4262db877805f6302332777b7cd0d831c31d19910ed6c7406ba0d', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_inputs': '6d1e18356f3bc48d646cbacdf1334b88b165cce7f6734b9fedb50f6aa9a5bf57', 'reference': '0804d331f89afea0f94fa648ec2f7804d470f09d883ff28f09966ac8b0c9bf62', '_shape_of': 'feed359089564f0ce804ffe17af6ee2d94695946fc8f62229480c567cd8fb01f', 'run_compile': '46b663ec253bc02d8ac7a4f4d8daa50cb8bfc98defabb6d1da2899e9f1883d97', 'run_correctness': '19a550a8a4723382d807d6765615bc006c9630a3b9611afe19d3a4fb4b50f618', 'run_performance': '30f59390278fb3f019f2bd1db3852838333e5ef9dcd9ddc864ee3ae6d3380603', 'main': 'b8fc9aa7f4f55d0112581240ecc2b58ac31e49aba320feedd0067792a46fca38'}}
    for task_name in _ATTENTION_TWO:
        task = ROOT / 'tasks/triton2flydsl' / task_name
        functions = hashes[task.name]
        for fn in ast.parse((task / 'test_kernel_harness.py').read_text()).body:
            if isinstance(fn, ast.FunctionDef) and fn.name in functions:
                normalized = _RemoveAttentionChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest() == functions[fn.name], (task_name, fn.name)


class _RemoveMlaChecks(_RemoveAttentionChecks):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) == '_checked_mla_output':
            return None
        if ast.unparse(node) == "out.fill_(float('nan'))":
            return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('queries', [1, 2])
@pytest.mark.parametrize('phase,behavior', [(phase, behavior) for phase in ['correctness', 'performance'] for behavior in ['correct', 'wrong', 'shape', 'dtype', 'nan', 'input_modified', 'kv_modified', 'metadata_modified', 'measured_wrong', 'replay_wrong', 'cached', 'unwritten', 'separate_return'] if phase == 'performance' or behavior not in {'measured_wrong', 'replay_wrong', 'cached'}])
def test_mla_actual_paged_buffer_contract_original_gate_and_measured_replay(queries, phase, behavior, monkeypatch):
    import torch
    import types
    task = ROOT / 'tasks/triton2flydsl/aiter/mla'
    checks = module(task / 'scripts/replay_checks.py')
    ns = dict(NORM_ERR_TOL=.01, ALLCLOSE_TOL=.01, WARMUP_ITERATIONS=10, BENCHMARK_ITERATIONS=100,
              require_tensor_contract=checks.require_tensor_contract, require_unchanged=checks.require_unchanged,
              verify_timed_run=checks.verify_timed_run)
    _harness_functions(task, {'_ref_masked_attention', 'torch_mla_extend', '_compare', '_call_kernel', '_checked_mla_output', '_compare_mla_output', '_mla_replay_validator'}, ns)
    q = torch.zeros((queries, 2, 3), dtype=torch.bfloat16)
    kv = torch.tensor([[[[2.,4.,0.]], [[6.,8.,0.]]], [[[10.,12.,0.]], [[14.,16.,0.]]]], dtype=q.dtype)
    out = torch.empty((queries, 2, 2), dtype=q.dtype)
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    cu = torch.tensor([0, queries], dtype=torch.int32); used = torch.tensor([3], dtype=torch.int32)
    inputs = (q, kv, table, cu, used); originals = tuple(x.clone() for x in inputs)
    def oracle(): return ns['torch_mla_extend'](q, kv, cu, used, table, 2, 1., q.dtype)
    cached = oracle().clone(); state = {'phase': 'setup'}
    def compute(*args):
        result = oracle()
        if phase == 'correctness' or state['phase'] == 'measured':
            if behavior == 'wrong': result.zero_()
            if behavior == 'shape': result = result.reshape(-1)
            if behavior == 'dtype': result = result.float()
            if behavior == 'nan': result.fill_(float('nan'))
            if behavior == 'input_modified': q.add_(1)
            if behavior == 'kv_modified': kv.mul_(.5)
            if behavior == 'metadata_modified': table.zero_()
        if behavior == state['phase'] + '_wrong': result.zero_()
        if state['phase'] == 'replay' and behavior == 'cached': result = cached.clone()
        if behavior == 'unwritten': return out
        out.data = result
        return result.clone() if behavior == 'separate_return' else out
    class Collector:
        bound = False
    calls = []
    def benchmark(fn, *, warmup, repetition, timed_run):
        calls.append((warmup, repetition)); state['phase'] = 'measured'
        timed_run.outputs = fn(); timed_run.bound = True; state['phase'] = 'setup'
        def replay():
            state['phase'] = 'replay'
            try: return fn()
            finally: state['phase'] = 'setup'
        timed_run.rerun = replay
        return .1, {'benchmark_method': 'cuda_graph', 'benchmark_timed_run_kind': 'captured_graph'}
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    ns.update(TEST_SHAPES=[(1, queries, 2, 1, 2, 1, 2, 3)], TimedRun=Collector,
              benchmark_cuda_graph_or_events=benchmark, load_module=lambda: types.SimpleNamespace(mla_decode_fwd=compute),
              _retry_gpu=lambda fn: fn(), make_test_data=lambda *a: (q, kv, out, table, cu, used, 1.))
    _harness_functions(task, {'run_correctness', 'run_performance'}, ns)
    result = ns['run_' + phase]()
    if phase == 'correctness':
        assert result[0] == (behavior == 'correct'), result
        assert len(result[2]) == 1
    else:
        assert len(result) == 1 and calls == [(0, 100)]
        if behavior == 'correct':
            assert result[0]['timed_output_correctness'] == result[0]['replay_correctness'] == 'PASS'
        else: assert result[0]['execution_time_ms'] == -1
    if not behavior.endswith('_modified'): checks.require_unchanged(inputs, originals)


@pytest.mark.parametrize('fault', [None, 'page_gather', 'compare'])
def test_mla_real_cpu_paged_known_answer_rejects_wrong_gather_and_comparator(fault, tmp_path):
    task = tmp_path / 'mla'
    shutil.copytree(ROOT / 'tasks/triton2flydsl/aiter/mla', task)
    p = task / 'test_kernel_harness.py'; s = p.read_text()
    if fault == 'page_gather':
        assert 'block_indices = block_tables[i, :num_kv_blocks]' in s
        s = s.replace('block_indices = block_tables[i, :num_kv_blocks]', 'block_indices = torch.arange(num_kv_blocks)')
    if fault == 'compare':
        s = s.replace('if norm_err > NORM_ERR_TOL:', 'if False:')
    p.write_text(s)
    result = invoke(task, 'validate-task')
    assert result.passed == (fault is None), result.reason
    if fault is None: assert len(result.cases) == 6


def test_mla_original_reference_inputs_gate_source_and_sample_fingerprints():
    task = ROOT / 'tasks/triton2flydsl/aiter/mla'
    assert hashlib.sha256((task / 'mla.py').read_bytes()).hexdigest() == '22000b5f657b3d28afe38c94117f74e9343f10fb28f07b095ba3fa238e8d6baa'
    hashes = {'load_module': 'f86cec48c13505a8571991aad880c0a25a0e06119b68730f57edd92e2757653f', '_is_transient_gpu_error': '5afa68828c130e63eef28cb95296055fe9a8ce93f9e0ff74d37824d5c545adb2', '_retry_gpu': '011da24c8f60af67764f9f5800d76e003e13ed1ace6a6eb6a579f9ca0bf8d2bf', 'make_test_data': '07680d0288f0496b9542c72764fe420592ebb582be2e57d08da7eef3adbeaab2', '_call_kernel': '21f919b5d8087d332e32c9177c52891349ce9fbb569021100e57b56ab4da23eb', '_ref_masked_attention': '1e6967be2b944fda854738ab6f84b3b041d78b75ab6f4e06f53bbbdd18e0bb5b', 'torch_mla_extend': '067ea108b35498b9b6703338c112c1db69335bf979cb76c1f712e66047433a45', '_compare': 'ff61bc86d21eebb74ffa14e426d5c4907aa35276e150280692314f0e27f5d034', 'run_compile': '555938e9c5b41aa7562d4b37d0ba6d90c8244137ef345acf02c2ed25e5d148f4', 'run_correctness': '9f482f5cd5213e6b12939ece6099288ccfcc52ab24adfb97d9c81ac5106abf19', 'run_performance': 'ab467bc1c4ffa77e7a6e83403d6c8527e8e08faf9de8561c3f8a08ba879b7038', 'main': '1a078382d740bf6949578767b767d05cc47c72a5ec777174ebadc2ffb2cbb0c0'}
    for fn in ast.parse((task / 'test_kernel_harness.py').read_text()).body:
        if isinstance(fn, ast.FunctionDef) and fn.name in hashes:
            normalized = _RemoveMlaChecks().visit(fn)
            assert hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest() == hashes[fn.name], fn.name
    for helper in ['candidate_checks.py', 'replay_checks.py']:
        assert (task / 'scripts' / helper).read_bytes() == (ROOT / 'tasks/triton2flydsl/aiter/mha/scripts' / helper).read_bytes()


_GR_NAMES = ['layer_norm', 'swiglu', 'jagged_dense_broadcast_add', 'jagged_dense_bmm_broadcast_add']


class _RemoveGrChecks(_RemoveAttentionChecks):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) == '_checked_gr_output':
            return None
        return super().visit_Expr(node)

    def visit_IfExp(self, node):
        # Only the explicit zero-reference bug fix is normalized, never the
        # nonzero branch, OR connective, fraction or absolute/relative tolerances.
        if ast.unparse(node.test) == 'denom > 0' and ast.unparse(node.orelse) in {
            '(out - ref).abs().max().item()', '(rf - of).abs().max().item()'}:
            node.orelse = ast.Constant(0.0)
        return self.generic_visit(node)


@pytest.mark.parametrize('name', _GR_NAMES)
@pytest.mark.parametrize('variant', [False, True])
@pytest.mark.parametrize('phase,behavior', [(phase, behavior) for phase in ['correctness', 'performance'] for behavior in ['correct', 'wrong', 'zero_reference_wrong', 'shape', 'dtype', 'device', 'nan', 'input_modified', 'weight_modified', 'measured_wrong', 'replay_wrong', 'cached'] if phase == 'performance' or behavior not in {'measured_wrong', 'replay_wrong', 'cached'}])
def test_gr_four_real_outputs_zero_reference_fix_and_actual_measured_replay(name, variant, phase, behavior, monkeypatch):
    import torch
    import types
    task = ROOT / 'tasks/triton2flydsl/generative_recommenders' / name
    checks = module(task / 'scripts/replay_checks.py')
    ns = dict(EPS=1e-6, ATOL=.1 if name == 'swiglu' else .01, RTOL=.01, PASS_FRACTION=.999,
              WARMUP_ITERATIONS=10, BENCHMARK_ITERATIONS=100, require_tensor_contract=checks.require_tensor_contract,
              require_unchanged=checks.require_unchanged, verify_timed_run=checks.verify_timed_run)
    _harness_functions(task, {'_torch_ref', '_close', '_call_kernel', '_checked_gr_output', '_compare_gr_output', '_gr_replay_validator'}, ns)
    x = torch.tensor([[1.,3.], [-2.,6.], [2.,4.]], dtype=torch.bfloat16)
    offsets = torch.tensor([0,2,3], dtype=torch.int64)
    if name == 'layer_norm':
        weight = torch.tensor([2.,3.], dtype=x.dtype) if variant else None
        bias = torch.tensor([.5,-.5], dtype=x.dtype) if variant else None
        data = (x, weight, bias); args = data; refargs = (*data, ns['EPS']); shape = (3,2,variant); entry='triton_layer_norm'
    elif name == 'swiglu':
        weight = torch.tensor([[1.,0.],[0.,-1.]], dtype=x.dtype)
        up = torch.tensor([[1.,2.],[-1.,1.]], dtype=x.dtype)
        if variant: x.neg_()
        data = (x,weight,up); args = data; refargs=data; shape=(3,2,2); entry='triton_swiglu_fwd'
    elif name == 'jagged_dense_broadcast_add':
        weight = torch.tensor([[10.,20.],[30.,40.]],dtype=x.dtype)
        data=(2,offsets,x,weight); args=data; refargs=data[1:];shape=(2,2,2);entry='triton_jagged_dense_broadcast_add'
    else:
        weight = torch.tensor([[[1.,2.],[3.,4.]],[[2.,1.],[1.,3.]]],dtype=x.dtype)
        bias = torch.ones((3 if variant else 2,2),dtype=x.dtype)
        data=(2,offsets,x,weight,bias);args=(*data,variant);refargs=(*data[1:],variant);shape=(2,2,2,2,variant);entry='triton_jagged_dense_bmm_add'
    inputs=tuple(v for v in data if isinstance(v,torch.Tensor))
    if behavior == 'zero_reference_wrong':
        for v in inputs:
            if v.is_floating_point(): v.zero_()
    originals=tuple(v.clone() for v in inputs)
    def oracle(): return ns['_torch_ref'](*refargs)
    cached=oracle().clone();state={'phase':'setup'}
    def compute(*a,**kw):
        out=oracle()
        if phase == 'correctness' or state['phase'] == 'measured':
            if behavior in {'wrong','zero_reference_wrong'}:out.fill_(100)
            if behavior=='shape':out=out.reshape(-1)
            if behavior=='dtype':out=out.float()
            if behavior=='device':out=out.to('meta')
            if behavior=='nan':out.fill_(float('nan'))
            if behavior=='input_modified':x.add_(1)
            if behavior=='weight_modified':
                (weight if weight is not None else x).mul_(.5)
        if behavior==state['phase']+'_wrong':out.fill_(100)
        if state['phase']=='replay' and behavior=='cached':out=cached.clone()
        return out
    class Collector:
        bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[shape],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda:types.SimpleNamespace(**{entry:compute}),_retry_oom=lambda fn:fn(),make_test_data=lambda *a:data)
    _harness_functions(task,{'run_correctness','run_performance'},ns)
    result=ns['run_'+phase]()
    if phase=='correctness':
        assert result[0]==(behavior=='correct'),result
        assert len(result[2])==1
    else:
        assert len(result)==1 and calls==[(0,100)]
        if behavior=='correct':assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        else:assert result[0]['execution_time_ms']==-1
    if not behavior.endswith('_modified'):checks.require_unchanged(inputs,originals)


@pytest.mark.parametrize('name',_GR_NAMES)
def test_gr_four_preserves_nonzero_or_rule_and_rejects_wrong_zero_reference(name):
    import torch
    task=ROOT/'tasks/triton2flydsl/generative_recommenders'/name
    checks=module(task/'scripts/replay_checks.py')
    ns=dict(require_tensor_contract=checks.require_tensor_contract,ATOL=.1 if name=='swiglu' else .01,RTOL=.01,PASS_FRACTION=.999)
    _harness_functions(task,{'_close','_compare_gr_output'},ns)
    expected=torch.ones((1,2000),dtype=torch.bfloat16);actual=expected.clone();actual[0,0]=10
    if name=='jagged_dense_bmm_broadcast_add':
        with pytest.raises(AssertionError):ns['_compare_gr_output'](actual,expected)
    else:ns['_compare_gr_output'](actual,expected)
    actual[0,:10]=10
    with pytest.raises(AssertionError):ns['_compare_gr_output'](actual,expected)
    zero=torch.zeros_like(expected)
    ns['_compare_gr_output'](zero,zero)
    with pytest.raises(AssertionError):ns['_compare_gr_output'](expected,zero)
    if name!='jagged_dense_bmm_broadcast_add':assert not ns['_close'](zero,expected)[0]


@pytest.mark.parametrize('name',_GR_NAMES)
def test_gr_four_real_cpu_known_answers_and_copied_audit(name):
    task=ROOT/'tasks/triton2flydsl/generative_recommenders'/name
    result=invoke(task,'validate-task')
    assert result.passed,result.reason
    assert len(result.cases)==6
    assert result.metadata['candidate_state']=='implemented'
    assert all(c['known_answer']=='PASS' and c['negative_output']=='rejected' for c in result.metadata['reference_controls'])
    for helper in ['candidate_checks.py','replay_checks.py']:
        assert (task/'scripts'/helper).read_bytes()==(ROOT/'tasks/triton2flydsl/aiter/mha/scripts'/helper).read_bytes()


def test_gr_four_original_math_cases_seeds_and_timing_except_explicit_zero_branch_fix():
    hashes={'layer_norm': {'load_module': 'c58dc7788f06826b21777ee67278ed69d6b858941fd67020f4a2868f05b0da9f', '_is_oom': '95fe19fe4319ad374b28eaa07fe0a37e2196c5f5ca315b7149913b85f38ecea9', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_test_data': '088c7c4d1e1d64f55952e36488556f33f38d94ede6f862f409461dcb4a289e80', '_torch_ref': '09b3366f2a3a55bf32b6ec854366f0aa2bd0cb08d20a523e6db0ee68afbe8095', '_close': 'b8aa2beace023149beeceeea088b1f69caf487bad4caf2970180bcce4779af63', '_call_kernel': 'b4e7c8a5870961bcbf79aab623ca2cd1e06e81afb116e56563aec4aafa849571', 'run_compile': 'a0db3281e6d2e28a66ba9f0ea56c90fd33a3660457c6d87e8c21d67e2a184c65', 'run_correctness': 'e7f7d77f6869803dd375944b3af1f11ea2e135c077e10a5d15318d3a4e66e7f6', 'run_performance': '0f7e3371702e3ed522e9b00170c505f0eb34ac3fe3bd9032e465599f6a399611', 'main': 'e80dfd607b82df6ed8ab3f782f30c6b811c5c1397bde24c6360d1cdbad848776'}, 'swiglu': {'load_module': '1ddaca99dfee91d0120a7df23bb99ed20ed437ac22c44ab62a84d8f4578f2d34', '_is_oom': '95fe19fe4319ad374b28eaa07fe0a37e2196c5f5ca315b7149913b85f38ecea9', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_test_data': '6e2fe79872c16280457dfc1c6435ef87b8470c7025896456a505fbc82422321f', '_torch_ref': '2c624c3c5db703047a803994648ac4317999b250aefd133db051c1c6932f76d4', '_close': 'b8aa2beace023149beeceeea088b1f69caf487bad4caf2970180bcce4779af63', '_call_kernel': '24b0d424edda099c6f5cdc6ef2e2b3f2cb48d79cc8b657eec693e45d343df32e', 'run_compile': '93998b215d22169c14dbea4f8f49521b781a7b4b5512d694adb9ea425a9bc01a', 'run_correctness': '53e20d20379f2ed004f1f4eaee812541ad659306b4947fbda6ce64aaf0a552f0', 'run_performance': 'c7e2dbee6a308d179724c8e779e23ec69d87c2c1153bef847e13f89c78d725a4', 'main': 'e80dfd607b82df6ed8ab3f782f30c6b811c5c1397bde24c6360d1cdbad848776'}, 'jagged_dense_broadcast_add': {'load_module': '794cd9d96378bd275472bd825e4ad55832ae66b94c1e8b86d149f22ab6319456', '_is_oom': '95fe19fe4319ad374b28eaa07fe0a37e2196c5f5ca315b7149913b85f38ecea9', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_test_data': '94ae6f59861acd022da65787893fb58754b421060de5c9dddbbcdc00b4b93ed0', '_torch_ref': '8059f772e0b41601f32dc8c4ede58c9f5b843639e26af01186c466a117eb42a2', '_close': 'b8aa2beace023149beeceeea088b1f69caf487bad4caf2970180bcce4779af63', '_call_kernel': 'f8473afe2824b540ae242913bedb6b67e67f5bd0e016d1d69101f83fda1acb26', 'run_compile': 'f4a4aabb3ef251c135e391740c50f2ed87fa0beaa3bbc89cc6cd16ab3cd7c3cb', 'run_correctness': '759c838dde7a411a8b0fe7b783ba5c02661d954626bb8896b52b6cda03751420', 'run_performance': '9109d85dc5c136126f3e27908fa2bf37a9665589259fda4eb74ab74467f0d5cd', 'main': 'e80dfd607b82df6ed8ab3f782f30c6b811c5c1397bde24c6360d1cdbad848776'}, 'jagged_dense_bmm_broadcast_add': {'load_module': 'f499ce51895c96bcbe19d4abbd62eb1a9227da90175bf29c27c00988d72d5899', '_is_oom': '95fe19fe4319ad374b28eaa07fe0a37e2196c5f5ca315b7149913b85f38ecea9', '_retry_oom': '2270b49f7be6ae5bcc6b61e6bde2cf3a120423644d2586ac800b3aa2c0b7512b', 'make_test_data': '106ea0671490433ff62789d24b724e0c989edc0d5b2b1cae72f00c4fe1e31e13', '_call_kernel': '0d9c9498e544f4c1b04c81a7a04a0ffb2f7b4b0e705a72edb4d82dee814ea6cb', '_torch_ref': 'b814076198b485ce4882b7c3489bc4c00248d2770ae7bcf9e1928d384bd7ffb3', 'run_compile': '0d8dbd924dc17eac2b48a1968b05498df3b0c1bc1a4d1e64dac6d4a20bdca221', 'run_correctness': '8931363a0aa3223dbacb06c23fc8037db13182bad59206e8426f098a428b199f', 'run_performance': 'ee417ffa6046cf0fff59ef76028ce9c88b6eae24ec1f78c1dca7bb6650fa1b7e', 'main': 'f2076381a0045c8efa3bd6075c1255cb6e7ea55a4d256f7dfc30d7f19745fa2a'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/generative_recommenders'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveGrChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_STATE_TWO = ['merge_state', 'ssd_chunk_state']


class _RemoveStateChecks(_RemoveSglangElementwiseChecks):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) == '_checked_state_output':
            return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('dtype_name', ['bf16', 'fp16', 'fp32'])
@pytest.mark.parametrize('name,phase,behavior', [(name,phase,behavior) for name in _STATE_TWO for phase in ['correctness','performance'] for behavior in ['correct','wrong_first','wrong_second','shape','dtype','device','nan','input_modified','metadata_modified','measured_wrong','replay_wrong','cached_first','cached_second'] if (phase=='performance' or behavior not in {'measured_wrong','replay_wrong','cached_first','cached_second'}) and (name=='merge_state' or behavior not in {'wrong_second','cached_second'})])
def test_sglang_state_two_real_output_dtypes_original_gates_and_timed_replay(name,dtype_name,phase,behavior,monkeypatch):
    import torch
    import types
    merge=name=='merge_state';task=ROOT/'tasks/triton2flydsl/sglang'/name
    checks=module(task/'scripts/replay_checks.py')
    ns=dict(require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged,
            WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100)
    ns['verify_timed_pair' if merge else 'verify_timed_run']=getattr(checks,'verify_timed_pair' if merge else 'verify_timed_run')
    _harness_functions(task,{'reference_merge','reference','_shape_of','_checked_state_output','_compare_state_output','_state_replay_validator'},ns)
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[dtype_name]
    if merge:
        x=torch.tensor([[[2.,4.]],[[3.,7.]]],dtype=dtype);other=x.clone().mul_(2)
        a=torch.tensor([[.5],[1.]]);b=torch.tensor([[1.],[.25]])
        data=(x,a,other,b);cfg=dict(N=2,H=1,D=2,dtype=dtype_name);entry='merge_state_triton'
        def oracle():return ns['reference_merge'](*data)
    else:
        x=torch.tensor([2.,3.,1.,4.],dtype=dtype).reshape(1,2,1,2)
        other=torch.tensor([3.,4.,1.,2.],dtype=dtype).reshape(1,2,1,2)
        a=torch.tensor([.1,.2]).reshape(1,1,1,2);b=torch.tensor([-.2,-.6]).reshape(1,1,1,2)
        data=(x,other,a,b);cfg=dict(b=1,H=1,P=2,G=1,N=2,cs=2,C=1,dtype=dtype_name);entry='_chunk_state_fwd'
        def oracle():return ns['reference'](*data,cfg)
    originals=tuple(v.clone() for v in data)
    cached=tuple(v.clone() for v in oracle()) if merge else (oracle().clone(),)
    state={'phase':'setup'}
    def compute(*args):
        outputs=list(oracle()) if merge else [oracle()]
        if phase=='correctness' or state['phase']=='measured':
            if behavior=='wrong_first':outputs[0].fill_(100)
            if behavior=='wrong_second':outputs[1].fill_(100)
            if behavior=='shape':outputs[0]=outputs[0].reshape(-1)
            if behavior=='dtype':outputs[0]=outputs[0].double() if merge else outputs[0].bfloat16()
            if behavior=='device':outputs[0]=outputs[0].to('meta')
            if behavior=='nan':outputs[-1].fill_(float('nan'))
            if behavior=='input_modified':x.mul_(.5)
            if behavior=='metadata_modified':a.mul_(.5)
        if behavior==state['phase']+'_wrong':outputs[0].fill_(100)
        if state['phase']=='replay':
            if behavior=='cached_first':outputs[0]=cached[0].clone()
            if behavior=='cached_second':outputs[1]=cached[1].clone()
        return tuple(outputs) if merge else outputs[0]
    class Collector:
        bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[cfg],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda:types.SimpleNamespace(**{entry:compute}),_retry_oom=lambda fn:fn(),make_inputs=lambda *args:data)
    _harness_functions(task,{'run_correctness','run_performance'},ns)
    result=ns['run_'+phase]()
    if phase=='correctness':
        assert result[0]==(behavior=='correct'),result
        assert len(result[2])==1
    else:
        assert len(result)==1 and calls==[(0,100)]
        if behavior=='correct':assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        else:assert result[0]['execution_time_ms']==-1
    if not behavior.endswith('_modified'):checks.require_unchanged(data,originals)


@pytest.mark.parametrize('name',_STATE_TWO)
def test_sglang_state_two_original_fraction_or_value_lse_tolerances(name):
    import torch
    task=ROOT/'tasks/triton2flydsl/sglang'/name
    checks=module(task/'scripts/replay_checks.py');ns=dict(require_tensor_contract=checks.require_tensor_contract)
    _harness_functions(task,{'_checked_state_output','_compare_state_output'},ns)
    if name=='merge_state':
        expected=(torch.ones(1,1,2),torch.ones(1,1))
        actual=(expected[0].clone().add_(.005),expected[1].clone())
        with pytest.raises(AssertionError):ns['_compare_state_output'](actual,expected,{'dtype':'fp32'})
        for dtype in [torch.float16,torch.bfloat16]:
            e=(expected[0].to(dtype),expected[1]);a=(actual[0].to(dtype),actual[1])
            ns['_compare_state_output'](a,e,{'dtype':'bf16'})
            bad_lse=expected[1]+.01
            with pytest.raises(AssertionError):ns['_compare_state_output']((a[0],bad_lse),e,{'dtype':'bf16'})
    else:
        expected=torch.ones(1,2000);actual=expected.clone();actual[0,0]=10
        ns['_compare_state_output'](actual,expected,{})
        actual[0,:10]=10
        with pytest.raises(AssertionError):ns['_compare_state_output'](actual,expected,{})


@pytest.mark.parametrize('name',_STATE_TWO)
def test_sglang_state_two_real_cpu_controls_and_unchanged_helpers(name):
    task=ROOT/'tasks/triton2flydsl/sglang'/name;result=invoke(task,'validate-task')
    assert result.passed,result.reason
    assert len(result.cases)==(7 if name=='merge_state' else 6)
    assert all(c['known_answer']=='PASS' and c['negative_output']=='rejected' for c in result.metadata['reference_controls'])
    assert (task/'scripts/candidate_checks.py').read_bytes()==(ROOT/'tasks/triton2flydsl/aiter/mha/scripts/candidate_checks.py').read_bytes()
    source='fused_add_rmsnorm' if name=='merge_state' else 'mha'
    assert (task/'scripts/replay_checks.py').read_bytes()==(ROOT/'tasks/triton2flydsl/aiter'/source/'scripts/replay_checks.py').read_bytes()


def test_sglang_state_two_original_input_reference_dtype_gates_and_sample_fingerprints():
    hashes={'merge_state': {'load_module': 'b2073f65823e826672888d411c64138935686a2329e1a0f30a51155876cc29b0', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_inputs': 'de4b3ea00bc1895093f60a09f0f199c95d77aef0df96dbc581e3a4e0dce89cf0', 'reference_merge': 'c73d7bd53784af5b1dbc193b3216bb13bbcac6f78ee051be06ab0cc45f6fcec9', '_shape_of': '0df3f55adfeee5710acd4c4837f3202c9865cc1667dea43165482da5cf02bad6', 'run_compile': 'd27bd318034a1ceb5ca14fbfbdeba891e409e9ab3798ac57562ce9a83959b789', 'run_correctness': '1ae17daff0a24311ea46d2e78da223e4f04b71bdc131e63e5def14686c43b9e5', 'run_performance': '85ecf7c40d726c8483f8c29ae08a9137741878f51c6754efccc66a18e6d396a6', 'main': '1c5ab37d045dbe55f9c726226059e47730bb052a91021da69a0e0993d9dbb74a'}, 'ssd_chunk_state': {'load_module': '56be25f07e79240daf96e45a7b39ed133d015471318de14b9378394a712e669b', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_inputs': '686f1015d24a8b683b91881320b9db71aa5dafdf8396c269cc638d9f8c7ca60b', 'reference': 'a07e8cba5cc25b690109b3161b618dbb588269b1c4cdff9cc8b45ff09cd6cc62', '_shape_of': '9cb979edfc3fb0fa13f59c868fc7faf889972e472b7388dfbc12b0954d266a05', 'run_compile': '6a294eab63c1d4f3633cc519a2ef35a4d629de1b148b12b830100a8ca2902cdc', 'run_correctness': '368ae55914e0de5ae24b86e4e8dfd53a7748c02f9043f522fba475d6eeb0fa63', 'run_performance': '7b070a1ba496cc951c26e2678ee8957354bdeb987b657db036d2b13f885574ff', 'main': 'abc68a7888954fd841797c99341d12a64de52a2bbe8b1c78ac31cb6403f706a3'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/sglang'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveStateChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


@pytest.mark.parametrize('dtype_name',['bf16','fp16','fp32'])
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','performance'] for behavior in ['correct','wrong_first','wrong_second','shape','dtype','nan','input_modified','weight_modified','measured_wrong','replay_wrong','cached_first','cached_second'] if phase=='performance' or behavior not in {'measured_wrong','replay_wrong','cached_first','cached_second'}])
def test_dual_rms_real_intermediate_rounding_both_outputs_and_measured_replay(dtype_name,phase,behavior,monkeypatch):
    import torch
    import types
    task=ROOT/'tasks/triton2flydsl/sglang/fused_dual_residual_rmsnorm';checks=module(task/'scripts/replay_checks.py')
    ns=dict(EPS=1e-6,WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100,require_tensor_contract=checks.require_tensor_contract,
            require_unchanged=checks.require_unchanged,verify_timed_pair=checks.verify_timed_pair)
    _harness_functions(task,{'reference','_shape_of','_checked_state_output','_compare_state_output','_state_replay_validator'},ns)
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[dtype_name]
    x=torch.tensor([[1.,3.],[-2.,6.]],dtype=dtype);residual=torch.tensor([[.5,2.],[-1.,2.]],dtype=dtype)
    w1=torch.tensor([2.,3.],dtype=dtype);w2=torch.tensor([.5,1.5],dtype=dtype)
    data=(x,residual,w1,w2);originals=tuple(v.clone() for v in data);cached=tuple(v.clone() for v in ns['reference'](*data));state={'phase':'setup'}
    def compute(*args):
        outputs=list(ns['reference'](*data))
        if phase=='correctness' or state['phase']=='measured':
            if behavior=='wrong_first':outputs[0].zero_()
            if behavior=='wrong_second':outputs[1].zero_()
            if behavior=='shape':outputs[1]=outputs[1].reshape(-1)
            if behavior=='dtype':outputs[1]=outputs[1].double()
            if behavior=='nan':outputs[1].fill_(float('nan'))
            if behavior=='input_modified':x.mul_(.5)
            if behavior=='weight_modified':w1.mul_(.5)
        if behavior==state['phase']+'_wrong':outputs[0].zero_()
        if state['phase']=='replay':
            if behavior=='cached_first':outputs[0]=cached[0].clone()
            if behavior=='cached_second':outputs[1]=cached[1].clone()
        return tuple(outputs)
    class Collector:
        bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[dict(bs=2,hidden=2,dtype=dtype_name)],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda:types.SimpleNamespace(fused_dual_residual_rmsnorm=compute),_retry_oom=lambda fn:fn(),make_inputs=lambda *a:data)
    _harness_functions(task,{'run_correctness','run_performance'},ns)
    result=ns['run_'+phase]()
    if phase=='correctness':assert result[0]==(behavior=='correct'),result
    else:
        assert len(result)==1 and calls==[(0,100)]
        if behavior=='correct':assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        else:assert result[0]['execution_time_ms']==-1
    if not behavior.endswith('_modified'):checks.require_unchanged(data,originals)


@pytest.mark.parametrize('fault',[None,'first_eps','second_eps','compare'])
def test_dual_rms_real_cpu_controls_reject_each_missing_epsilon_and_wrong_comparator(fault,tmp_path):
    task=tmp_path/'dual';shutil.copytree(ROOT/'tasks/triton2flydsl/sglang/fused_dual_residual_rmsnorm',task)
    p=task/'test_kernel_harness.py';s=p.read_text()
    if fault=='first_eps':s=s.replace('(a * a).mean(dim=-1, keepdim=True) + eps','(a * a).mean(dim=-1, keepdim=True)')
    if fault=='second_eps':s=s.replace('(a2 * a2).mean(dim=-1, keepdim=True) + eps','(a2 * a2).mean(dim=-1, keepdim=True)')
    if fault=='compare':s=s.replace('if not torch.allclose(out.float(), ref.float(), atol=tolerance, rtol=tolerance):','if False:')
    p.write_text(s);result=invoke(task,'validate-task')
    assert result.passed==(fault is None),result.reason
    if fault is None:assert len(result.cases)==8


def test_dual_rms_original_references_rounding_inputs_gates_and_timing():
    task=ROOT/'tasks/triton2flydsl/sglang/fused_dual_residual_rmsnorm'
    hashes={'load_module': '6750a187a2cf3e2f809d7e49cc73013f5e4c0c2e004194d79327de14ba3f4fb6', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_inputs': '35846be59bffd0ffb4a903bc807e96ca8e10843ca857bb60b791ae41fd01df92', 'reference': '29bc8033a542e581fe6bbc430e77a8df53ee507416aaad707323aa66fd8b83d1', '_shape_of': '05532e8adeff843a63f917ec904d725165a63fa1435106976ae0eb6441e38f0b', 'run_compile': 'a266eb55aa2023d80a9212c445fc0399fdfec64ef9543605db9aa98bd4ac83b8', 'run_correctness': '0e1091829e50708fd9613f99cb424028d95f20072a3aec95bf727063827e522a', 'run_performance': '758f72d312074bc73e94a75e942686d6f0428130f770b7878011a04a5978d460', 'main': 'e0a005223e7387da498863af9b0a830c8a0a3f386098c7e9a6923a44697df4d8'}
    for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
        if isinstance(fn,ast.FunctionDef) and fn.name in hashes:
            normalized=_RemoveStateChecks().visit(fn)
            assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==hashes[fn.name],fn.name
    for helper in ['candidate_checks.py','replay_checks.py']:
        assert (task/'scripts'/helper).read_bytes()==(ROOT/'tasks/triton2flydsl/sglang/merge_state/scripts'/helper).read_bytes()


class _RemoveFlatChecks(_RemoveTritonBatchedChecks):
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None)=='ENTRIES':return None
        return super().visit_Assign(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='_checked_flat_output':return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('style,reuse,nope,nope_first',[(style,reuse,nope,first) for style in [0,1] for reuse in [False,True] for nope,first in [(False,False),(True,False),(True,True)]])
@pytest.mark.parametrize('behavior',['correct','wrong','dtype','shape','input_modified','freq_modified'])
def test_flat_rope_all_original_style_reuse_and_nope_contracts(style,reuse,nope,nope_first,behavior,monkeypatch):
    import torch,types
    task=ROOT/'tasks/triton2flydsl/aiter/rope_fwd';checks=module(task/'scripts/replay_checks.py');ns=dict(require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged)
    _harness_functions(task,{'_rotate_half_neox','_rotate_half_gptj','_ref_rope_sbhd_fwd','_checked_flat_output'},ns)
    x=torch.arange(1,9,dtype=torch.bfloat16).reshape(1,1,1,8);freqs=torch.full((1,1,1,8//(2 if nope else 1)//(2 if reuse else 1)),.5,dtype=x.dtype)
    def compute(*args,**kw):
        out=ns['_ref_rope_sbhd_fwd'](x,freqs,style,reuse,nope_first,0)
        if behavior=='wrong':out.zero_()
        if behavior=='dtype':out=out.float()
        if behavior=='shape':out=out.reshape(-1)
        if behavior=='input_modified':x.mul_(.5)
        if behavior=='freq_modified':freqs.mul_(.5)
        return out
    ns.update(_load_source=lambda:types.SimpleNamespace(RotateStyle=types.SimpleNamespace(NEOX=0,GPTJ=1),rope_fwd=compute),
              _make_inputs=lambda *args:(x,freqs),_cases=lambda:[dict(name='control',B=1,S=1,style='NEOX' if style==0 else 'GPTJ',nope=nope,nope_first=nope_first,reuse=reuse)])
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);_harness_functions(task,{'run_correctness'},ns)
    assert ns['run_correctness'](verbose=False)==(behavior=='correct')


@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached','input_modified','freq_modified'])
def test_flat_rope_actual_four_benchmark_cases_measured_and_replayed(behavior,monkeypatch,tmp_path):
    import torch,types,math
    task=ROOT/'tasks/triton2flydsl/aiter/rope_fwd';checks=module(task/'scripts/replay_checks.py');ns=dict(require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged,verify_timed_run=checks.verify_timed_run)
    _harness_functions(task,{'_rotate_half_neox','_rotate_half_gptj','_ref_rope_sbhd_fwd','_checked_flat_output','_compare_flat_output','_flat_replay_validator'},ns)
    state={'phase':'setup'};inputs=[];cached=[]
    def make_inputs(*args):
        x=torch.arange(1,9,dtype=torch.bfloat16).reshape(1,1,1,8);freqs=torch.full((1,1,1,4),.5,dtype=x.dtype)
        inputs[:]=[x,freqs];cached[:]=[ns['_ref_rope_sbhd_fwd'](x,freqs,0,True,False,0)]
        return x,freqs
    def compute(*args,**kw):
        out=ns['_ref_rope_sbhd_fwd'](*inputs,0,True,False,0)
        if behavior==state['phase']+'_wrong':out.zero_()
        if state['phase']=='replay' and behavior=='cached':out=cached[0].clone()
        if state['phase']=='measured' and behavior.endswith('_modified'):inputs[behavior=='freq_modified'].mul_(.5)
        return out
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    ns.update(_load_source=lambda:types.SimpleNamespace(RotateStyle=types.SimpleNamespace(NEOX=0,GPTJ=1),rope_fwd=compute),_make_inputs=make_inputs,
              TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,WARMUP=10,ITERS=100,H=1,D=8,_HERE=str(tmp_path),Path=Path,json=json,math=math)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);_harness_functions(task,{'run_benchmark'},ns)
    if behavior=='correct':
        rows=ns['run_benchmark'](verbose=False)
        assert len(rows)==4 and calls==[(0,100)]*4
        assert all(row['timed_output_correctness']==row['replay_correctness']=='PASS' for row in rows)
    else:
        with pytest.raises(AssertionError):ns['run_benchmark'](verbose=False)


@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','benchmark'] for behavior in ['correct','wrong','dtype','shape','nan','input_modified','ids_modified','weight_modified','measured_wrong','replay_wrong','cached'] if phase=='benchmark' or behavior not in {'measured_wrong','replay_wrong','cached'}])
def test_flat_moe_real_preparation_both_route_modes_and_timed_replay(phase,behavior,monkeypatch,tmp_path):
    import torch,types,math
    task=ROOT/'tasks/triton2flydsl/aiter/moe_fused_gemm';checks=module(task/'scripts/replay_checks.py')
    ns=dict(require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged,verify_timed_run=checks.verify_timed_run,BLOCK_SIZE_M=64)
    _harness_functions(task,{'_ref_moe_gemm','_prepare_kernel','_run_kernel','_checked_flat_output','_compare_flat_output','_flat_replay_validator'},ns)
    A=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16);B=torch.tensor([[[3.,4.]],[[5.,6.]]],dtype=A.dtype)
    ids=torch.tensor([[1,0],[0,1]],dtype=torch.int32);weights=torch.tensor([[.25,.75],[.75,.25]])
    data=(A,B,ids,weights);state={'phase':'setup'};prepared=[]
    cached=ns['_ref_moe_gemm'](*data,2,True).to(A.dtype)
    def sort(*args):
        prepared.append(state['phase']);return ids.clone().reshape(-1),torch.tensor([0,1]),torch.tensor([4])
    def compute(*args,**kw):
        C=args[2];mul=args[10];top_k=args[11];out=ns['_ref_moe_gemm'](*data,top_k,mul).to(A.dtype)
        if phase=='correctness' or state['phase']=='measured':
            if behavior=='wrong':out.zero_()
            if behavior=='dtype':out=out.float()
            if behavior=='shape':out=out.reshape(-1)
            if behavior=='nan':out.fill_(float('nan'))
            if behavior=='input_modified':A.mul_(.5)
            if behavior=='ids_modified':ids.zero_()
            if behavior=='weight_modified':weights.mul_(.5)
        if behavior==state['phase']+'_wrong':out.zero_()
        if state['phase']=='replay' and behavior=='cached':out=cached.clone()
        C.data=out
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,prepare_fn,timed_run):
        calls.append((warmup,repetition));prepare_fn();state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            prepare_fn();state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    ns.update(_load_source=lambda:types.SimpleNamespace(moe_align_block_size=sort,fused_moe=compute),_make_inputs=lambda *args:data,
              TEST_SHAPES=[dict(name='control',M=2,K=2,N=1,E=2,top_k=2)],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              WARMUP=10,ITERS=100,_HERE=str(tmp_path),Path=Path,json=json,math=math)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);_harness_functions(task,{'run_correctness','run_benchmark'},ns)
    if phase=='correctness':assert ns['run_correctness'](verbose=False)==(behavior=='correct')
    elif behavior=='correct':
        rows=ns['run_benchmark'](verbose=False);assert len(rows)==1 and calls==[(0,100)]
        assert rows[0]['timed_output_correctness']==rows[0]['replay_correctness']=='PASS'
        assert prepared==['setup']  # sorting stays outside measured work and replay.
    else:
        with pytest.raises(AssertionError):ns['run_benchmark'](verbose=False)


@pytest.mark.parametrize('name', ['rope_fwd','moe_fused_gemm'])
def test_flat_rope_moe_real_cpu_full_manifest_and_copied_candidate_checks(name):
    task=ROOT/'tasks/triton2flydsl/aiter'/name;result=invoke(task,'validate-task')
    assert result.passed,result.reason
    assert len(result.cases)==(48 if name=='rope_fwd' else 10)
    for helper in ['candidate_checks.py','replay_checks.py']:
        assert (task/'scripts'/helper).read_bytes()==(ROOT/'tasks/triton2flydsl/aiter/rmsnorm/scripts'/helper).read_bytes()


def test_flat_rope_moe_original_input_reference_rotation_and_prepare_timing_fingerprints():
    hashes={'rope_fwd': {'_load_source': '017ca74e8b8637ad01ca6762de871208612ea351fafe400285bf447f7126e1e9', '_make_inputs': '23a4c5182690aa833c119a7d0ad65828da778d22ce91b8dfa12b400b5939e2fe', '_rotate_half_neox': '8d7d330c98e9e99e6ec1387ad8386bf055d05eb58f4ed2fdf17d48c80bf39e37', '_rotate_half_gptj': '434b5d4257c29863c21c7cba46a0cba71abb02a842e384d1e622f50ffb992b9f', '_ref_rope_sbhd_fwd': '3f450126474ef56a8365d1d4c525d6725c733eb5bc0ff0f5fbcf027125ec98d6', '_cases': '4957ee8a62fd55e1069503ad693b03ec3e9a67bb9c9731acf9e18f325de00c02', 'run_compile': '8f829e9dafe3fa510b965656f9805ccbd0727c1cb47081513760f5bd2d68c083', 'run_correctness': 'cae0149f4c1b3e66b7c371411fb0c5241610cca98effd45818d4038ca900c5b5', 'run_benchmark': 'd5fe98996e6cd12947ef6efe87c61ea096c734b9ed75b0265e6cce36f4f231b1'}, 'moe_fused_gemm': {'_load_source': 'acdced07419dae8b1803a737870c38687adf7a2bf35b3f0e183c8bb4be502574', '_make_inputs': 'b9efac3a95b3d642146377e84daf3e95da946ce0f127b00e17a5fce39dd81ede', '_prepare_kernel': '8f118965ce7ad15869b75eaf60f308ea15be3887b1cbeaae879c64079d450cf6', '_run_kernel': '97e4e85950df57de831c64d15685a877f772c9280ec5a46ce0b655ae098843f3', '_ref_moe_gemm': 'bbaf201bcabbbdb6d897c8c29b493b722fd7760a7715afd3c4b868001410b873', 'run_compile': '4dde78d7144a193df6c53c38c92b102947afbf8ae15656268d197f0ea4097d56', 'run_correctness': '5105c851019f177cb65300e38f96ca7c8f1a400ca3dddd704e2bc398c6481656', 'run_benchmark': 'f1109cf5a43830a0f14d456fcf39b9caacf3acfd134ab1bc95720cf26b24ddb5'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/aiter'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveFlatChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_SGLANG_ROUTER_TWO = ['fused_gdn_gating', 'fused_moe_router']


class _RemoveRouterChecks(_RemoveSglangElementwiseChecks):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) == '_checked_router_output':
            return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('name',_SGLANG_ROUTER_TWO)
@pytest.mark.parametrize('variant',[0,1,2])
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','performance'] for behavior in ['correct','wrong_first','wrong_second','shape','dtype_first','dtype_second','device','nan','input_modified','parameter_modified','measured_wrong','replay_wrong','cached_first','cached_second'] if phase=='performance' or behavior not in {'measured_wrong','replay_wrong','cached_first','cached_second'}])
def test_sglang_router_two_real_output_contract_original_gates_measured_pair(name,variant,phase,behavior,monkeypatch):
    import torch
    import types
    task=ROOT/'tasks/triton2flydsl/sglang'/name;gating=name=='fused_gdn_gating'
    checks=module(task/'scripts/replay_checks.py');ns=dict(BETA=1.,THRESHOLD=20.,DTYPE_NAME='bfloat16',
        require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged,
        verify_timed_pair=checks.verify_timed_pair,WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100)
    _harness_functions(task,{'reference_gating','reference','_shape_of','_checked_router_output','_compare_router_output','_router_replay_validator'},ns)
    x=torch.tensor([[1.,-2.],[3.,-1.]],dtype=torch.bfloat16)
    if gating:
        inp=dict(a=x,b=x.clone().mul_(.5),A_log=torch.tensor([0.,.3]),dt_bias=torch.tensor([.5,1.]))
        if variant==1:inp['a'].add_(30.)
        if variant==2:inp['a'].mul_(.1)
        data=tuple(inp.values());shape=(2,2);entry='fused_gdn_gating'
        def oracle():return ns['reference_gating'](inp)
        ns['make_test_data']=lambda *args:inp
    else:
        w=torch.tensor([[2.,1.],[-1.,2.],[3.,-1.]],dtype=x.dtype)
        bias=torch.tensor([.5,-.25,1.]) if variant==2 else None
        shape=dict(bs=2,E=3,hidden=2,topk=1 if variant==0 else 2,cap=0 if variant==0 else 3.,bias=variant==2)
        data=tuple(v for v in (x,w,bias) if v is not None);entry='fused_moe_router_shim'
        def oracle():return ns['reference'](x,w,shape,bias)
        ns['make_inputs']=lambda *args:(x,w,bias)
    originals=tuple(v.clone() for v in data);cached=tuple(v.clone() for v in oracle());state={'phase':'setup'}
    def compute(*args,**kwargs):
        outputs=list(oracle())
        if phase=='correctness' or state['phase']=='measured':
            if behavior=='wrong_first':outputs[0].fill_(100)
            if behavior=='wrong_second':outputs[1].fill_(100)
            if behavior=='shape':outputs[0]=outputs[0].reshape(-1)
            if behavior=='dtype_first':outputs[0]=outputs[0].bfloat16()
            if behavior=='dtype_second':outputs[1]=outputs[1].bfloat16() if gating else outputs[1].long()
            if behavior=='device':outputs[1]=outputs[1].to('meta')
            if behavior=='nan':outputs[0].fill_(float('nan'))
            if behavior=='input_modified':x.mul_(.5)
            if behavior=='parameter_modified':data[-1].mul_(.5)
        if behavior==state['phase']+'_wrong':outputs[0].fill_(100)
        if state['phase']=='replay':
            if behavior=='cached_first':outputs[0]=cached[0].clone()
            if behavior=='cached_second':outputs[1]=cached[1].clone()
        return tuple(outputs)
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[shape],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda:types.SimpleNamespace(**{entry:compute}),_retry_oom=lambda fn:fn())
    _harness_functions(task,{'run_correctness','run_performance'},ns)
    result=ns['run_'+phase]()
    if phase=='correctness':
        assert result[0]==(behavior=='correct'),result
        assert len(result[2])==1
    else:
        assert len(result)==1 and calls==[(0,100)]
        if behavior=='correct':assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        else:assert result[0]['execution_time_ms']==-1,result
    if not behavior.endswith('_modified'):checks.require_unchanged(data,originals)


@pytest.mark.parametrize('name',_SGLANG_ROUTER_TWO)
@pytest.mark.parametrize('fault',['none','reference_wrong','comparator_accepts_all'])
def test_sglang_router_two_independent_controls_reject_reference_and_comparator_mutations(name,fault,tmp_path):
    task=tmp_path/name;shutil.copytree(ROOT/'tasks/triton2flydsl/sglang'/name,task)
    if fault!='none':
        p=task/'test_kernel_harness.py';s=p.read_text()
        if fault=='reference_wrong':
            old='g = -torch.exp(A_log.float())[None, :] * softplus_x' if name=='fused_gdn_gating' else 'weights = torch.gather(probs, 1, ids)'
            assert old in s
            s=s.replace(old,old+' + 10')
        else:
            assert 'def _compare_router_output(' in s
            lines=s.splitlines(True);tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_compare_router_output')
            lines[fn.lineno:fn.lineno]=['    return\n'];s=''.join(lines)
        p.write_text(s)
    result=invoke(task,'validate-task')
    assert result.passed==(fault=='none'),result.reason
    if fault=='none':
        assert len(result.cases)==(9 if name=='fused_gdn_gating' else 8)
        assert all(c['known_answer']=='PASS' and c['negative_output']=='rejected' for c in result.metadata['reference_controls'])
    else:assert 'known answer' in result.reason or 'incorrect output' in result.reason


@pytest.mark.parametrize('name',_SGLANG_ROUTER_TWO)
def test_sglang_router_two_helpers_identical_to_reviewed_measured_pair_and_audit(name):
    task=ROOT/'tasks/triton2flydsl/sglang'/name;base=ROOT/'tasks/triton2flydsl/sglang/merge_state'
    for rel in ['scripts/candidate_checks.py','scripts/replay_checks.py','task_runtime.py']:
        assert (task/rel).read_bytes()==(base/rel).read_bytes()


def test_sglang_router_two_preserves_all_original_math_inputs_shapes_gates_sampling():
    hashes={'fused_gdn_gating': {'load_module': '037642864242b26b505dfbd39aa5fa58023551b14b55d5e4f893fc3a000a80c6', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_test_data': '9665205015bcccff2084920733ec802621249d3fcb5ab04b735d3c970894618d', 'reference_gating': '2d0f6ef3d9636af2a4198438fdf564454ec972df605803c5dbaeeab84b3f5472', 'run_compile': '6f3d546d2a7a3d382b1440d69c614860580b849c1dc4d848b3fda236b791ca92', 'run_correctness': 'f3905f19c863be7dae8654763111994df870c8a3ee36f4f34bb53c5fe17c6e7f', 'run_performance': 'ff4a33ad71cfe56f1ac04ab438920cc74b1b3c054e28458e28cd1bcf636ac16f', 'main': '69f2c9d7455496f97c31a580e679c4361271c4db6f05482dbfe424df9641c15a'}, 'fused_moe_router': {'load_module': '953696d189c1fbd01e5400c59617345300e9d9c4605526653fc33b38a5b1b407', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_inputs': '5ff114739e6c974d57f187d49b768ad06a45307575b4f507207c03cc643e8ae4', 'reference': 'c6e999edcab9cd51b4b00ab58aaa847a31913a9abe008e128218ddd7548e1da4', '_shape_of': 'b96d3ad069d4f767c1afa4a64fe11645cafc2fac1a66bc72ea43ccceaded8180', 'run_compile': 'f3df7f8ad05b681d25a2f64479e7dfde1e75395283b6504922f171d10fc8c6a9', 'run_correctness': '9d319633583e26faf8f206584d0b0079ae6b4d09d884b0e22d0ceeac010d7e20', 'run_performance': '3e518aff630943bc2e91859443472f8a8fb7ef8bee16bde56a5e411574e16258', 'main': '6f1fc7fcef166b402f53f12edef0c19e168f192c2d7bbc04cd405e9bd51e7cc5'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/sglang'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveRouterChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_SGLANG_COMBINE_TWO = ['experts_combine', 'gdn_chunk_fwd_o']


class _RemoveCombineChecks(_RemoveSglangElementwiseChecks):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) in {'_checked_combine_output','_check_output_buffer'}:
            return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('name',_SGLANG_COMBINE_TWO)
@pytest.mark.parametrize('variant',[0,1,2])
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','performance'] for behavior in ['correct','wrong','shape','dtype','device','nan','input_modified','parameter_modified','measured_wrong','replay_wrong','cached'] if phase=='performance' or behavior not in {'measured_wrong','replay_wrong','cached'}])
def test_sglang_combine_two_real_original_gates_outputs_and_actual_replay(name,variant,phase,behavior,monkeypatch):
    import torch
    import types
    task=ROOT/'tasks/triton2flydsl/sglang'/name;combine=name=='experts_combine'
    checks=module(task/'scripts/replay_checks.py');ns=dict(BT=2,SQRT2=2**.5,DTYPE_NAME='bfloat16',
        require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged,
        verify_timed_run=checks.verify_timed_run,WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100)
    _harness_functions(task,{'reference_o','reference','_shape_of','_checked_combine_output','_compare_combine_output','_combine_replay_validator','_check_output_buffer'},ns)
    dtype=[torch.bfloat16,torch.float16,torch.float32][variant] if combine else torch.bfloat16
    x=torch.tensor([[1.,-2.],[3.,-1.]],dtype=dtype)
    if combine:
        mlp=x.clone().mul_(.5);moe=x if variant==0 else torch.stack([x,2*x],dim=1)
        data=(moe,mlp);shape=dict(tokens=2,combine_k=1 if variant==0 else 2,hidden=2,dtype=['bf16','fp16','fp32'][variant]);entry='experts_combine_triton'
        def oracle():return ns['reference'](moe,mlp)
        ns['make_inputs']=lambda *args:data
    else:
        # Two chunks, optionally grouped heads and V != K. Small CPU values
        # exercise the original body; no GPU case or sampling count changes.
        B,T,Hg,H,K,V=1,4,1,2 if variant==1 else 1,1,2 if variant==2 else 1
        shape=(B,T,Hg,H,K,V)
        inp=dict(B=B,T=T,Hg=Hg,H=H,K=K,V=V,NT=2,scale=.5,
                 q=torch.arange(1,5,dtype=dtype).reshape(1,4,1,1),k=torch.full((1,4,1,1),2.,dtype=dtype),
                 v=torch.arange(1,1+T*H*V,dtype=dtype).reshape(1,T,H,V),h=torch.full((1,2,H,V,1),.5,dtype=dtype),
                 g=torch.tensor([-.1,-.2,-.2,-.3]).reshape(1,4,1).expand(1,4,H).clone())
        x=inp['q'];data=tuple(inp[key] for key in ('q','k','v','h','g'));entry='chunk_fwd_o'
        def oracle():return ns['reference_o'](inp)
        ns['make_test_data']=lambda *args:inp
    originals=tuple(v.clone() for v in data);cached=oracle().clone();state={'phase':'setup'}
    def compute(*args,**kwargs):
        output=oracle()
        if phase=='correctness' or state['phase']=='measured':
            if behavior=='wrong':output.fill_(100)
            if behavior=='shape':output=output.reshape(-1)
            if behavior=='dtype':output=output.double()
            if behavior=='device':output=output.to('meta')
            if behavior=='nan':output.fill_(float('nan'))
            if behavior=='input_modified':data[0].mul_(.5)
            if behavior=='parameter_modified':data[-1].mul_(.5)
        if behavior==state['phase']+'_wrong':output.fill_(100)
        if state['phase']=='replay' and behavior=='cached':output=cached.clone()
        if kwargs.get('output_buffer') is not None:
            buffer=kwargs['output_buffer'];nbytes=mlp.numel()*mlp.element_size()
            view=buffer[:nbytes].view(mlp.dtype).reshape_as(mlp);view.copy_(output);return view
        return output
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[shape],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda:types.SimpleNamespace(**{entry:compute}),_retry_oom=lambda fn:fn())
    _harness_functions(task,{'run_correctness','run_performance'},ns)
    result=ns['run_'+phase]()
    if phase=='correctness':assert result[0]==(behavior=='correct'),result
    else:
        assert len(result)==1 and calls==[(0,100)]
        if behavior=='correct':assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        else:assert result[0]['execution_time_ms']==-1,result
    if not behavior.endswith('_modified'):checks.require_unchanged(data,originals)


@pytest.mark.parametrize('dtype_name',['bfloat16','float16','float32'])
@pytest.mark.parametrize('fault',['none','ignored','unwritten','wrong','tail_overwrite','input_mutation'])
def test_experts_combine_supplied_buffer_alias_write_padding_and_inputs(dtype_name,fault):
    import torch
    import types
    task=ROOT/'tasks/triton2flydsl/sglang/experts_combine';checks=module(task/'scripts/replay_checks.py')
    ns=dict(SQRT2=2**.5,require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged)
    _harness_functions(task,{'reference','_checked_combine_output','_compare_combine_output','_check_output_buffer'},ns)
    dtype=getattr(torch,dtype_name);moe=torch.tensor([[[1.,2.],[3.,4.]]],dtype=dtype);mlp=torch.tensor([[5.,6.]],dtype=dtype)
    expected=ns['reference'](moe,mlp)
    def compute(moe,mlp,*,output_buffer):
        out=output_buffer[:mlp.numel()*mlp.element_size()].view(dtype).reshape_as(mlp)
        if fault=='ignored':return expected.clone()
        if fault=='unwritten':return out
        out.copy_(expected)
        if fault=='wrong':out.fill_(100)
        if fault=='tail_overwrite':output_buffer[-1]=0
        if fault=='input_mutation':moe.add_(1)
        return out
    invoke=lambda:ns['_check_output_buffer'](types.SimpleNamespace(experts_combine_triton=compute),moe,mlp,{'dtype':'fp32' if dtype_name=='float32' else 'bf16'},expected)
    if fault=='none':invoke()
    else:
        with pytest.raises(AssertionError):invoke()


def test_gdn_chunk_output_preserves_two_percent_finite_gate_and_blocks_sparse_nan():
    import torch
    task=ROOT/'tasks/triton2flydsl/sglang/gdn_chunk_fwd_o';checks=module(task/'scripts/replay_checks.py');ns=dict(require_tensor_contract=checks.require_tensor_contract)
    _harness_functions(task,{'_checked_combine_output','_compare_combine_output'},ns)
    expected=torch.ones(1,200,dtype=torch.bfloat16);actual=expected.clone();actual[0,:3]=100
    ns['_compare_combine_output'](actual,expected,expected.dtype)
    actual[0,:5]=100
    with pytest.raises(AssertionError):ns['_compare_combine_output'](actual,expected,expected.dtype)
    actual=expected.clone();actual[0,0]=float('nan')
    with pytest.raises(AssertionError):ns['_compare_combine_output'](actual,expected,expected.dtype)


@pytest.mark.parametrize('name',_SGLANG_COMBINE_TWO)
def test_sglang_combine_two_real_cpu_known_answers_comparators_and_reviewed_helpers(name):
    task=ROOT/'tasks/triton2flydsl/sglang'/name;result=invoke(task,'validate-task')
    assert result.passed,result.reason
    assert len(result.cases)==8
    assert all(c['known_answer']=='PASS' and c['negative_output']=='rejected' for c in result.metadata['reference_controls'])
    base=ROOT/'tasks/triton2flydsl/sglang/fused_gdn_gating'
    for rel in ['scripts/candidate_checks.py','scripts/replay_checks.py','task_runtime.py']:
        assert (task/rel).read_bytes()==(base/rel).read_bytes()


def test_sglang_combine_two_original_source_math_inputs_shapes_gates_sampling():
    hashes={'experts_combine': {'load_module': '79342980ce2951c6559bda9844722f8aa27149b304e30fb6cf82322435de21df', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_inputs': '3e410817193f13bb2e2856ceaf2d1b07373d17831f2aeb1cc819b87ac3baafa0', 'reference': '30cdd0d4d92e97fe789660ed2411d7ddbc7f194902bd834437ce10a5fb8f6e85', '_shape_of': '60bf58c1b1794858b13b418d1a202eb3b6c7c82db270b75920b69026074aae8c', 'run_compile': '64ed6b2b336064350f712d0609750d460376973171ce17092ce5bf9ec94168ee', 'run_correctness': '5dd30f92e4bde59742324025ddc9175f3f841a4dff873857866250bd65316962', 'run_performance': '26d15d2359932b6608c332de0fef79fe87596d2bc3f3824f4028f2b5e29c27d5', 'main': '4627e6f944e000564c46882741f2ec35535d5454e0eb038c71973db96322b529'}, 'gdn_chunk_fwd_o': {'load_module': 'e573aefa73f1de76cab7b21f29a6912cb489ece137aa9b0f5b293ae1d060c33f', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', '_chunk_local_cumsum': '7f6b02486945bc1f0da20f83bfc7272ab026273d562e476f093756c622f993b9', 'make_test_data': '593e2cbbacb7b8d52493fdc3a48c9e3f0f86271084326d21f8169042d386d8fb', 'reference_o': '265fb47d57a76e8a99e6721509638a877663776056e41cc2a6d96f22caa3875a', 'run_compile': '72666c42821b55cbcb1a0b18e80ada0d7b6f639f7e25568abb9e1b5830a90364', 'run_correctness': 'cbbe437fa170f7028e3c6e1aacd6da44edf477fa188ca6a7ecfe3a23897be5ae', 'run_performance': 'b91ffb9164d86e70af4f51e061bab665122e51b325cf26a090751675f6a17059', 'main': '603c16a3d3c6eada3a8f8da74b973513eb484da4e312b0676043a63dd3d04652'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/sglang'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveCombineChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_UNUSED_TORCH_BUILDERS = ['dynamic_mxfp8_quant_kernel', 'gelu_and_mul_kernel', 'gelu_tanh_and_mul_kernel', 'gemm_a8w8_bpreshuffle_kernel', 'hgemm_kernel', 'jagged_dense_bmm_kernel', 'moe_sorting_kernel', 'qk_norm_rope_quant_kernel', 'rmsnorm2d_dynamicquant_kernel', 'rmsnorm2d_kernel', 'rmsnorm2d_smoothquant_kernel', 'swiglu_and_mul_kernel']


@pytest.mark.parametrize('name',_UNUSED_TORCH_BUILDERS)
def test_torch_actual_operators_are_required_without_unused_builders(name,tmp_path):
    # A syntax-valid real operator can use its own internal compiler interface;
    # retained legacy starter helpers must not become mandatory public outputs.
    task=tmp_path/name;shutil.copytree(ROOT/'tasks/torch2flydsl'/name,task)
    spec=load_task_spec(task/'config.yaml',task_id='torch2flydsl/'+name);cfg=spec.to_mapping()
    runtime=module(task/'task_runtime.py')
    tree=ast.parse((task/'test_kernel_harness.py').read_text())
    entry=cfg['candidate']['entrypoints'][0]['symbol']
    assert any(isinstance(n,ast.Attribute) and n.attr==entry or isinstance(n,ast.Constant) and n.value==entry for n in ast.walk(tree))
    assert all(e['kind']=='function' for e in cfg['candidate']['entrypoints'])
    state,defined=runtime.source_state(runtime.config())
    assert state==cfg['candidate']['initial_state']
    assert all(defined) if state=='implemented' else not any(defined)
    source='import flydsl\n'
    for e in cfg['candidate']['entrypoints']:
        source += 'def '+e['symbol']+'(*args, **kwargs):\n    return None\n'
    source += 'def unused_legacy_builder(*args):\n    raise NotImplementedError("unused")\n'
    (task/'kernel.py').write_text(source)
    assert runtime.source_state(runtime.config())==('implemented',[True]*len(cfg['candidate']['entrypoints']))
    result=invoke(task,'candidate','compile');assert result.passed,result.reason
    # A compile PASS is syntax evidence only. This return-None function is NOT
    # a valid candidate; real correctness still runs full FlyDSL/output guards.
    assert entry in (task/'test_kernel_harness.py').read_text()


def test_unused_builder_cleanup_retains_starter_model_and_manifest_bytes():
    original={'dynamic_mxfp8_quant_kernel': {'kernel.py': '04c2ab5eb9e9bee43be84633bc7b210fcb3ad8be69bba8aa98ef6897010611a0', 'model.py': '9e5b1e289eee05aba727b71e28a98e6a7611d9fd6737d5e87b83fe9469eed39d', 'cases.json': '9e76bdcd9955b731e930c2e46536dbce2522d2f88f4ad9cb76016e825821988d'}, 'gelu_and_mul_kernel': {'kernel.py': '4fb1de9fe9d5da55e5cb924ecd03458ab70cc493612857ca300343237d541f25', 'model.py': 'c171ab0b489b1cb87a3f551c3ba8ecd820e3147a9becb6810154040f4027f7dd', 'cases.json': 'a20a152b61a241426b4f7f4d9cbdfee7af9f92f2cf1d2e8d0c2ba0aecfb20129'}, 'gelu_tanh_and_mul_kernel': {'kernel.py': '04616e2c62589d5c2e4b8147772bf4e333e753e663eb5d95bb88456428110f24', 'model.py': '95988833405bac9d10624c4ca4e78ee0251a5a60c1dd457d6b915904b9b2dadf', 'cases.json': 'a20a152b61a241426b4f7f4d9cbdfee7af9f92f2cf1d2e8d0c2ba0aecfb20129'}, 'gemm_a8w8_bpreshuffle_kernel': {'kernel.py': 'b5d3e87a3ceca3fe555572f0b4ab5c7b1dd6c3f5c9e18ab924b589df3d485996', 'model.py': 'e4278a3637b56eab94baec3b712ff1fb5ac44206a0ac21b1925a7497ca0b9643', 'cases.json': '7f2d3e25a614486974800da54cb23d2c7913101b2ec1b82ca92818cdfec479ee'}, 'hgemm_kernel': {'kernel.py': '29cab2057d32224a7da558083f6b4edeb560e44efd59a60b82dfb14c9c6a8d28', 'model.py': '89ca4fce55817fdf5fcbaea925a1639f9b96cd809dcda8c06cd70f9e1033372e', 'cases.json': '8388fcafea635e69bde93aad82d8b6bcd10d3ffd9998f2594e4ab989c2fd61e4'}, 'jagged_dense_bmm_kernel': {'kernel.py': 'fcb9b75ec238ced56568fb5b27535a160314db29c212abe33c83be6f7df3c043', 'model.py': '1b446ea35fee03f47ae16121fb7ba8aa933e9e99c48f2d90584c770d56065186', 'cases.json': '8304b063316f9cfc3667a8f38d9d85805b38a2339deebd48b883bfb59b5427e3'}, 'moe_sorting_kernel': {'kernel.py': '4bc536d6d29f16e1f24278d9db05ffba13d723f3f42064cce18b60c687f3eb43', 'model.py': 'c874911efc1c947437d5c7c62019e7c52b34458a0ec1560ecdde72aa971de271', 'cases.json': '0c8247d8727d48dc3a8ddd20ede1db3ae2586e990222f7bf19ce39dc6ad913c4'}, 'qk_norm_rope_quant_kernel': {'kernel.py': 'be6c11328764ac59054301e1d8a9312287227718eee498731698ddc45079fa46', 'model.py': '152a32140302f1264c555fbd9b6f9d8362583fd08b0344290a67d6b1bb849ee2', 'cases.json': 'de8e183a711424ddabe5f8bfea4fa00dbe79cb886462b5d3681f67fa9d6e0eb1'}, 'rmsnorm2d_dynamicquant_kernel': {'kernel.py': 'c8542e7ea4b69aa6881ae2e7046995c67112e7e38b68e95d3cc2a51965d05bdc', 'model.py': '5cd3abaf088651f8f2ed9db44513bfd8c5238a45145aaf3dcd3168d63e3ff080', 'cases.json': '046dcf1c5e6f49bf68bd6e935555006803f9f6af5668460389ae6147297528f1'}, 'rmsnorm2d_kernel': {'kernel.py': 'a20840e12a22f5c08fed1e87eee62de5dec590680c79b5b6fa8cb28d9dd9396a', 'model.py': '8442cb4d63444e7dfa9db1fa2d6253ceee0463b1debfd6919fc5219181de3b13', 'cases.json': '426c9e84d97161e5bb7a09353102c57648f5ca5a4790c46370b5369ba470fa64'}, 'rmsnorm2d_smoothquant_kernel': {'kernel.py': '701e11ec5e63bf65572fd9325a7e0b1bea0e1aa9561f8711de0e7ed0113fc2b0', 'model.py': '5496289dc3f8f72c1b9deefce1e39b7c1a00dc5726ad708abe65641fee4edf0e', 'cases.json': '359795558e6ebcfb617bbae66eda8540f5503cc5a04056f1fdfde58e13aedce4'}, 'swiglu_and_mul_kernel': {'kernel.py': '6adbabe7f43dd51289ec4afa3310ba30c56ad8f216edb801886882a1c00715e5', 'model.py': '74765a3a6e469d27926a92b0d710231f2bb7604850186fca782dc5446a8224b4', 'cases.json': 'a20a152b61a241426b4f7f4d9cbdfee7af9f92f2cf1d2e8d0c2ba0aecfb20129'}}
    for name,files in original.items():
        for rel,expected in files.items():
            assert hashlib.sha256((ROOT/'tasks/torch2flydsl'/name/rel).read_bytes()).hexdigest()==expected,(name,rel)


_SGLANG_KKT_WY = ['chunk_scaled_dot_kkt_fwd', 'wy_fast']


class _RemoveGdnChecks(_RemoveSglangElementwiseChecks):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', None) == '_checked_gdn_output':
            return None
        return super().visit_Expr(node)


@pytest.mark.parametrize('name',_SGLANG_KKT_WY)
@pytest.mark.parametrize('variant',[0,1,2])
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','performance'] for behavior in ['correct','wrong','shape','dtype','device','nan','input_modified','metadata_modified','measured_wrong','replay_wrong','cached'] if phase=='performance' or behavior not in {'measured_wrong','replay_wrong','cached'}])
def test_sglang_kkt_wy_original_gates_real_output_dtypes_and_measured_replay(name,variant,phase,behavior,monkeypatch):
    import torch
    import types
    task=ROOT/'tasks/triton2flydsl/sglang'/name;kkt=name=='chunk_scaled_dot_kkt_fwd'
    checks=module(task/'scripts/replay_checks.py');ns=dict(BT=4,DTYPE_NAME='bfloat16',require_unchanged=checks.require_unchanged,
        verify_timed_run=checks.verify_timed_run,verify_timed_pair=checks.verify_timed_pair,WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100)
    _harness_functions(task,{'reference_kkt','reference_wu','_checked_gdn_output','_compare_gdn_output','_gdn_replay_validator'},ns)
    B,T,Hg,H,K,V=1,6 if variant==1 else 4,1,2 if variant==1 else 1,1,2 if variant==2 else 1
    k=torch.arange(1,1+T,dtype=torch.bfloat16).reshape(1,T,1,1)
    beta=torch.full((1,T,H),.625,dtype=k.dtype);g=torch.zeros(1,T,H)
    for start in range(0,T,4):g[:,start:start+4]=-.1*torch.arange(1,min(4,T-start)+1).reshape(1,-1,1)
    inp=dict(B=B,T=T,Hg=Hg,H=H,K=K,V=V,NT=(T+3)//4,k=k,beta=beta,g=None if kkt and variant==2 else g)
    if kkt:
        shape=(B,T,Hg,H,K,inp['g'] is not None);entry='chunk_scaled_dot_kkt_fwd'
        def oracle():return ns['reference_kkt'](inp)
    else:
        A=torch.zeros(B,T,H,4,dtype=k.dtype)
        for start in range(0,T,4):
            size=min(4,T-start);block=torch.eye(size)+torch.tril(torch.full((size,size),.125),diagonal=-1)
            A[:,start:start+size,:,0:size]=block[None,:,None,:].to(k.dtype)
        inp.update(v=torch.arange(1,1+T*H*V,dtype=k.dtype).reshape(1,T,H,V),A=A)
        shape=(B,T,Hg,H,K,V);entry='recompute_w_u_fwd'
        def oracle():return tuple(v.to(k.dtype) for v in ns['reference_wu'](inp))
    data=tuple(v for v in inp.values() if isinstance(v,torch.Tensor));originals=tuple(v.clone() for v in data)
    cached=(oracle().clone(),) if kkt else tuple(v.clone() for v in oracle());state={'phase':'setup'}
    def compute(*args,**kwargs):
        outputs=[oracle()] if kkt else list(oracle())
        if phase=='correctness' or state['phase']=='measured':
            if behavior=='wrong':outputs[-1].fill_(100)
            if behavior=='shape':outputs[-1]=outputs[-1].reshape(-1)
            if behavior=='dtype':outputs[-1]=outputs[-1].bfloat16() if kkt else outputs[-1].float()
            if behavior=='device':outputs[-1]=outputs[-1].to('meta')
            if behavior=='nan':outputs[-1].fill_(float('nan'))
            if behavior=='input_modified':k.mul_(.5)
            if behavior=='metadata_modified':beta.mul_(.5)
        if behavior==state['phase']+'_wrong':outputs[-1].fill_(100)
        if state['phase']=='replay' and behavior=='cached':outputs=[v.clone() for v in cached]
        return outputs[0] if kkt else tuple(outputs)
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[shape],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda:types.SimpleNamespace(**{entry:compute}),_retry_oom=lambda fn:fn(),make_test_data=lambda *args:inp)
    _harness_functions(task,{'run_correctness','run_performance'},ns)
    result=ns['run_'+phase]()
    if phase=='correctness':assert result[0]==(behavior=='correct'),result
    else:
        assert len(result)==1 and calls==[(0,100)]
        if behavior=='correct':assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        else:assert result[0]['execution_time_ms']==-1,result
    if not behavior.endswith('_modified'):checks.require_unchanged(data,originals)


@pytest.mark.parametrize('name',_SGLANG_KKT_WY)
@pytest.mark.parametrize('which',[0,1])
def test_sglang_kkt_wy_two_percent_preserved_for_each_output_but_sparse_nan_rejected(name,which):
    import torch
    task=ROOT/'tasks/triton2flydsl/sglang'/name;ns=dict(BT=200)
    _harness_functions(task,{'_checked_gdn_output','_compare_gdn_output'},ns)
    inp=dict(B=1,T=1,H=1,K=200,V=200,k=torch.ones(1,1,1,200,dtype=torch.bfloat16),v=torch.ones(1,1,1,200,dtype=torch.bfloat16))
    kkt=name=='chunk_scaled_dot_kkt_fwd';expected=(torch.ones(1,1,1,200),) if kkt else (inp['k'],inp['v'])
    which=0 if kkt else which
    def call(a):return ns['_compare_gdn_output'](a[0] if kkt else a,expected[0] if kkt else expected,inp)
    actual=[v.clone() for v in expected];actual[which].reshape(-1)[:3]=100;call(actual)
    actual[which].reshape(-1)[:5]=100
    with pytest.raises(AssertionError):call(actual)
    actual=[v.clone() for v in expected];actual[which].reshape(-1)[0]=float('nan')
    with pytest.raises(AssertionError):call(actual)


@pytest.mark.parametrize('name',_SGLANG_KKT_WY)
def test_sglang_kkt_wy_real_controls_and_identical_reviewed_helpers(name):
    task=ROOT/'tasks/triton2flydsl/sglang'/name;result=invoke(task,'validate-task')
    assert result.passed,result.reason
    assert len(result.cases)==8
    assert all(c['known_answer']=='PASS' and c['negative_output']=='rejected' for c in result.metadata['reference_controls'])
    base=ROOT/'tasks/triton2flydsl/sglang/merge_state'
    for rel in ['scripts/candidate_checks.py','scripts/replay_checks.py','task_runtime.py']:
        assert (task/rel).read_bytes()==(base/rel).read_bytes()


def test_sglang_kkt_wy_preserves_original_math_input_generation_shapes_tolerance_and_timing():
    hashes={'chunk_scaled_dot_kkt_fwd': {'load_module': '01b5dd805d4b3faf125685e1abf6c2388d68ca0f52eb49ba8e4dd5308f351564', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', '_chunk_local_cumsum': '60e3036073343c7500a70c063d740fbc979c84bc48b67d74ee888e40cc64830b', 'make_test_data': 'bef9fc9f068280f962607d7b04e7f4405ab7e5695162f8e89bf78370497886a8', 'reference_kkt': 'e62ac4f76b7699d92a0220956082999e3b9525a19dbeaf7fd637da1f32559e4d', 'run_compile': 'b8985952ea26b13efed1e04def7d941e0486bda726eceeb6b04cab917810e714', 'run_correctness': '3965f4081f429175759ba20fda557f98ee7e5b6b1fc561f7d8e4dbb3485a2be9', 'run_performance': '0623355d8871809001bbf5e598bdea5e39840289b657c1030111443009b056d4', 'main': '5395a114da7c499239b74d2146f7f6efc349e6648e7c8a051711a62d12073cce'}, 'wy_fast': {'load_module': '743374a333120bf1c02dbb278e4806f2f950b45e1cd3651e9601b280eb895e5d', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', '_chunk_local_cumsum': '60e3036073343c7500a70c063d740fbc979c84bc48b67d74ee888e40cc64830b', 'make_test_data': '633bd50f704cde36ab94f31ee08a23ffe5cc9ee6643711ffa2afbf1c9ba79a14', 'reference_wu': '82b3c0b880567d199a5964e92a516843f34d7b27f2a6de608ba2060128491100', 'run_compile': '8afef510e833d51147f90195f49705d9f7161a546aa3cf0220d452ff8de59649', 'run_correctness': 'f34785efb4b62a178ab60c73bcedfbd9ffeab6883913abb045fdede823deab15', 'run_performance': '000ac2bbc832fc413c7ab555bad617ce759598b7e04f6002cbb0f2f1c3e77ff9', 'main': '084957b8eedd8216093202b0747c206aba71d3768825640a769e1f0fc33b0e09'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/sglang'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveGdnChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_UNIFIED_TWO = ['unified_attention','unified_attention_sparse_mla']


class _RemoveUnifiedChecks(_RemoveMlaChecks):
    def visit_Expr(self, node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None)=='_checked_unified_output':return None
        return super().visit_Expr(node)
    def visit_FunctionDef(self,node):
        if node.name=='_call_kernel' and any(isinstance(n,ast.Call) and getattr(n.func,'attr',None)=='unified_attention_sparse_mla' for n in ast.walk(node)):
            node.body=[ast.Return(n.value) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) else n for n in node.body if not (isinstance(n,ast.Return) and isinstance(n.value,ast.Name) and n.value.id=='out')]
        return super().visit_FunctionDef(node)


@pytest.mark.parametrize('name',_UNIFIED_TWO)
@pytest.mark.parametrize('variant',[0,1,2])
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','performance'] for behavior in ['correct','wrong','shape','dtype','device','nan','unwritten','input_modified','metadata_modified','measured_wrong','replay_wrong','cached'] if phase=='performance' or behavior not in {'measured_wrong','replay_wrong','cached'}])
def test_unified_two_real_caller_buffer_inputs_original_gates_and_measured_replay(name,variant,phase,behavior,monkeypatch):
    import torch
    import types
    task=ROOT/'tasks/triton2flydsl/aiter'/name;sparse=name.endswith('sparse_mla');checks=module(task/'scripts/replay_checks.py')
    ns=dict(NORM_ERR_TOL=.01,ALLCLOSE_ATOL=.01,ALLCLOSE_RTOL=.01,WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100,
            require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged,verify_timed_run=checks.verify_timed_run)
    _harness_functions(task,{'make_test_data','ref_paged_attn','ref_sparse_mla','_window_size','_unpack','_norm_max_error','_checked_unified_output','_compare_unified_output','_unified_replay_validator','_call_kernel'},ns)
    torch.manual_seed(1)
    if sparse:
        shape=(1,3,2,2,2,2,3,4)
        q,kv,out,cu,used,idx,table,scale=ns['make_test_data'](*shape,pad_invalid=True,device='cpu',dtype=torch.bfloat16)
        if variant==1:idx[:,0]=1
        if variant==2:q.zero_()
        data=(q,kv,idx,table,cu,used)
        values=(q,kv,out,cu,used,idx,table,scale)
        def oracle():return ns['ref_sparse_mla'](q,kv,idx,2,2,scale).to(q.dtype)
    else:
        shape=(1,2,4,2,1,2,2,1 if variant==1 else 0,1. if variant==2 else 0.)
        q,k,v,out,table,cu,used,scale=ns['make_test_data'](*shape[:7],device='cpu',dtype=torch.bfloat16)
        table[:]=torch.tensor([[1,0]],dtype=table.dtype)
        data=(q,k,v,table,cu,used);values=(q,k,v,out,table,cu,used,scale)
        def oracle():return ns['ref_paged_attn'](q,k,v,[2],[4],table,scale,q.dtype,sliding_window=shape[7] or None,soft_cap=shape[8] or None)
    originals=tuple(v.clone() for v in data);cached=oracle().clone();state={'phase':'setup'}
    def compute(*args,**kwargs):
        value=oracle()
        if behavior=='unwritten':return value
        if phase=='correctness' or state['phase']=='measured':
            if behavior=='wrong':value.fill_(100)
            if behavior=='shape':out.resize_(out.numel())
            if behavior=='dtype':out.data=out.double()
            if behavior=='device':return value.to('meta') if not sparse else out.resize_(0)
            if behavior=='nan':value.fill_(float('nan'))
            if behavior=='input_modified':q.add_(1.)
            if behavior=='metadata_modified':cu.add_(1)
        if behavior==state['phase']+'_wrong':value.fill_(100)
        if state['phase']=='replay' and behavior=='cached':value=cached.clone()
        if out.shape==value.shape:out.copy_(value)
        return None if sparse else out
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[shape],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda:types.SimpleNamespace(**{name:compute}),_retry_gpu=lambda fn:fn(),make_test_data=lambda *args,**kw:values)
    _harness_functions(task,{'run_correctness','run_performance'},ns)
    result=ns['run_'+phase]()
    if phase=='correctness':assert result[0]==(behavior=='correct'),result
    else:
        assert len(result)==1 and calls==[(0,100)]
        if behavior=='correct':assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
        else:assert result[0]['execution_time_ms']==-1,result
    if not behavior.endswith('_modified'):checks.require_unchanged(data,originals)


@pytest.mark.parametrize('name',_UNIFIED_TWO)
def test_unified_two_real_cpu_controls_exact_source_helpers_and_original_numeric_gate(name):
    import torch
    task=ROOT/'tasks/triton2flydsl/aiter'/name;result=invoke(task,'validate-task')
    assert result.passed,result.reason
    assert len(result.cases)==(7 if name=='unified_attention' else 5)
    assert all(c['known_answer']=='PASS' and c['negative_output']=='rejected' for c in result.metadata['reference_controls'])
    base=ROOT/'tasks/triton2flydsl/aiter/mla'
    for rel in ['scripts/candidate_checks.py','scripts/replay_checks.py','task_runtime.py']:
        assert (task/rel).read_bytes()==(base/rel).read_bytes()
    checks=module(task/'scripts/replay_checks.py');ns=dict(require_tensor_contract=checks.require_tensor_contract,NORM_ERR_TOL=.01)
    _harness_functions(task,{'_compare_unified_output','_norm_max_error'},ns)
    ref=torch.zeros(1,200,dtype=torch.bfloat16);ref[0,0]=100;actual=ref.clone();actual[0,1]=.5
    # Original normalized maximum gate accepts this while allclose does not.
    assert not torch.allclose(actual,ref,atol=.01,rtol=.01)
    ns['_compare_unified_output'](actual,ref)
    actual[0,1]=2
    with pytest.raises(AssertionError):ns['_compare_unified_output'](actual,ref)


def test_unified_two_original_input_reference_shape_gates_and_timing_preserved():
    hashes={'unified_attention': {'load_module': 'da543b694e9e8e49ed822cd28b157f289f28cc27a1fa66b318ccbbbbb52e3e7d', '_window_size': '58b7ed377c80408ac52ef5162f54cb36f2bd58cf0f80d295b89c225972f8010b', 'make_test_data': '8335c704a809b530e69d7a2f26f063888323286ab9b5f7178b77d10b511759e4', '_call_kernel': '49bb2f03586610b3b6dac0c8736a3859e9cfd4e5f0ecd8c1557a6f31e7e1e880', 'run_compile': '056a57080274176635a2a643f381dd543a5dce9d165cdbe6a9bea424c124d19a', 'ref_paged_attn': 'e0525f372d1bfc70652ef5128e3e77adf476f45b38757d4d4d049f3d85fb7305', '_norm_max_error': '55c6bde927ff7593fe520345457735b3761659cabf8cf393ea152fb36b2494ab', 'run_correctness': 'd1715b24f57a3e72ba5879d72243a7fbbaa66f5486fd28a920ad39c705b1b7ac', 'run_performance': '28d7c4bb7aa1d21b4d27e8a5e27db6283af33773a9d53525ccd87ed3000cd846', 'main': 'ae915290c7834aa9a52d1d50bbc8ddff3839957c7e33779fc40f4f273f44e570'}, 'unified_attention_sparse_mla': {'load_module': 'c69a5e50d4a1f6e951a8ee1b303999136bdf405e311c6feb4940803ef19054d8', '_is_transient_gpu_error': '5afa68828c130e63eef28cb95296055fe9a8ce93f9e0ff74d37824d5c545adb2', '_retry_gpu': '011da24c8f60af67764f9f5800d76e003e13ed1ace6a6eb6a579f9ca0bf8d2bf', 'make_test_data': '05fc8aff143cf4431f218857d83d5f389ab2df7dde7560016252e86add739182', '_call_kernel': '43ee9176e819ab524b8b59584ecbe9d5e5368dab72f4baa5d46c9cd7b74fd8cd', '_unpack': 'a765356918b85ccba19f042716a2008a18f82947e702ebd11c394d526a200547', 'run_compile': 'bc9ad52c127fb0cc8e0dbb100c209827ca626dfa0b2dd353fd19f169d0058446', 'ref_sparse_mla': '8a92f55b80542d3f2a495e529825ed8293a59b9c79cfddee6a5caaa9a443d42c', '_norm_max_error': '55c6bde927ff7593fe520345457735b3761659cabf8cf393ea152fb36b2494ab', 'run_correctness': '44587fde9a66f94a44e6865dd8a94cdc2379b395a2e0b87a6665eb2b5a7d780d', 'run_performance': '12e785ffbfe19fc40fb594d271788de737122678701f4d538de59e251eb69f31', 'main': '70b77b8c9c03f4f1eae39318c37362de53ab6fdc96dff90aaa838c841de2b310'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/aiter'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveUnifiedChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_GDN_RECURRENCE_TWO=['gdn_chunk_fwd_h','gdn_fused_recurrent_decode']


class _RemoveRecurrentChecks(_RemoveMlaChecks):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None) in {'_checked_recurrent_output','_require_unused_state'}:return None
        return super().visit_Expr(node)
    def visit_FunctionDef(self,node):
        if node.name=='fn':
            node.body=[ast.Expr(n.value) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) else n for n in node.body if not isinstance(n,ast.Return)]
        return self.generic_visit(node)


@pytest.mark.parametrize('name',_GDN_RECURRENCE_TWO)
@pytest.mark.parametrize('variant',[0,1])
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','performance'] for behavior in ['correct','wrong_output','wrong_state','untouched_slot','shape','dtype','nan','input_modified','indices_modified','measured_wrong','replay_wrong','cached','missing_replay_prepare'] if phase=='performance' or behavior not in {'measured_wrong','replay_wrong','cached','missing_replay_prepare'}])
def test_gdn_recurrence_two_real_state_outputs_prepare_and_original_gates(name,variant,phase,behavior,monkeypatch):
    import torch
    import types
    chunk=name=='gdn_chunk_fwd_h';task=ROOT/'tasks/triton2flydsl/sglang'/name;checks=module(task/'scripts/replay_checks.py')
    ns=dict(BT=2,DTYPE_NAME='bfloat16',WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100,require_unchanged=checks.require_unchanged,verify_timed_pair=checks.verify_timed_pair)
    _harness_functions(task,{'make_test_data','_chunk_local_cumsum','reference_h','reference_decode','_close','_run_triton','_checked_recurrent_output','_require_unused_state','_compare_recurrent_output','_recurrent_replay_validator'},ns)
    torch.manual_seed(2)
    if chunk:
        shape=(2,4,1,2 if variant else 1,2,2,4)
        inp=ns['make_test_data'](*shape,device='cpu',dtype=torch.bfloat16);key='init';idx='idx';entry='chunk_gated_delta_rule_fwd_h';ref='reference_h'
    else:
        shape=(2,1,2 if variant else 1,2,2,4)
        inp=ns['make_test_data'](*shape,device='cpu',dtype=torch.bfloat16);key='ssm_states';idx='cache_indices';entry='fused_recurrent_gated_delta_rule_packed_decode';ref='reference_decode'
    inp[idx][:]=torch.tensor([2,0],dtype=inp[idx].dtype)
    pristine=inp[key].clone();cached=tuple(v.clone() for v in ns[ref](inp));state={'phase':'setup'}
    protected=tuple(v for k,v in inp.items() if isinstance(v,torch.Tensor) and not (phase=='performance' and not chunk and k==key));originals=tuple(v.clone() for v in protected)
    def compute(*args,**kwargs):
        working=kwargs['initial_state'];ri=dict(inp,**{key:working});outputs=list(ns[ref](ri))
        active=phase=='correctness' or state['phase']=='measured'
        if active:
            if behavior=='wrong_output':outputs[0].fill_(100)
            if behavior=='wrong_state':outputs[-1][2].fill_(100)
            if behavior=='untouched_slot':outputs[-1][3].fill_(100)
            if behavior=='shape':outputs[0]=outputs[0].reshape(-1)
            if behavior=='dtype':outputs[0]=outputs[0].float()
            if behavior=='nan':outputs[-1][3].fill_(float('nan'))
            if behavior=='input_modified':inp['k' if chunk else 'mixed_qkv'].add_(1)
            if behavior=='indices_modified':inp[idx].add_(1)
        if behavior==state['phase']+'_wrong':outputs[0].fill_(100)
        if state['phase']=='replay' and behavior=='cached':outputs=[v.clone() for v in cached]
        working.copy_(outputs[-1])
        if chunk:return tuple(outputs[:-1])
        if behavior=='shape' and active:kwargs['out'].resize_(outputs[0].shape)
        if behavior=='dtype' and active:kwargs['out'].data=kwargs['out'].float()
        kwargs['out'].copy_(outputs[0])
    class Collector:bound=False
    calls=[];prepare_calls=[]
    def benchmark(fn,*,warmup,repetition,prepare_fn,timed_run):
        calls.append((warmup,repetition));prepare_fn();prepare_calls.append('measured');state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            if behavior!='missing_replay_prepare':prepare_fn();prepare_calls.append('replay')
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[shape],TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,
              load_module=lambda:types.SimpleNamespace(**{entry:compute}),_retry_oom=lambda fn:fn(),make_test_data=lambda *args:inp)
    _harness_functions(task,{'run_correctness','run_performance'},ns)
    result=ns['run_'+phase]()
    if phase=='correctness':assert result[0]==(behavior=='correct'),result
    else:
        assert len(result)==1 and calls==[(0,100)]
        if behavior=='correct':
            assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
            assert prepare_calls==['measured','replay']
        else:assert result[0]['execution_time_ms']==-1,result
    if behavior not in {'input_modified','indices_modified'}:checks.require_unchanged(protected,originals)


@pytest.mark.parametrize('name',_GDN_RECURRENCE_TWO)
def test_gdn_recurrence_two_real_controls_and_reviewed_helpers(name):
    task=ROOT/'tasks/triton2flydsl/sglang'/name;result=invoke(task,'validate-task')
    assert result.passed,result.reason
    assert len(result.cases)==(8 if name=='gdn_chunk_fwd_h' else 10)
    assert all(c['known_answer']=='PASS' and c['negative_output']=='rejected' for c in result.metadata['reference_controls'])
    base=ROOT/'tasks/triton2flydsl/sglang/merge_state'
    for rel in ['scripts/candidate_checks.py','scripts/replay_checks.py','task_runtime.py']:
        assert (task/rel).read_bytes()==(base/rel).read_bytes()


def test_gdn_chunk_state_preserves_original_five_percent_gate_but_not_nonfinite_or_untouched_writes():
    import torch
    task=ROOT/'tasks/triton2flydsl/sglang/gdn_chunk_fwd_h';checks=module(task/'scripts/replay_checks.py');ns=dict(require_unchanged=checks.require_unchanged)
    _harness_functions(task,{'_close','_checked_recurrent_output','_require_unused_state','_compare_recurrent_output'},ns)
    inp=dict(B=1,NT=1,H=1,V=1,K=100,k=torch.ones(1,1,1,100,dtype=torch.bfloat16),u=torch.ones(1,1,1,100,dtype=torch.bfloat16),init=torch.ones(2,1,1,100,dtype=torch.bfloat16),idx=torch.tensor([0]))
    expected=(torch.ones(1,1,1,1,100,dtype=torch.bfloat16),inp['u'],inp['init'])
    actual=[v.clone() for v in expected];actual[0].reshape(-1)[:4]=100
    ns['_compare_recurrent_output'](actual,expected,inp)
    actual[0].reshape(-1)[:6]=100
    with pytest.raises(AssertionError):ns['_compare_recurrent_output'](actual,expected,inp)
    actual=[v.clone() for v in expected];actual[0].reshape(-1)[0]=float('nan')
    with pytest.raises(AssertionError):ns['_compare_recurrent_output'](actual,expected,inp)
    actual=[v.clone() for v in expected];actual[-1][1,0,0,0]+=.125
    with pytest.raises(AssertionError):ns['_compare_recurrent_output'](actual,expected,inp)


def test_gdn_recurrence_two_original_math_rounding_inputs_shapes_gates_prepare_and_timing():
    hashes={'gdn_chunk_fwd_h': {'load_module': '3be1c8b8de184e266bcc06f091002ef6d4ec686c60e82135e21dd93dff317636', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', '_chunk_local_cumsum': '13ff0f7a64a6d56b2ff1b1712f6993d65d6167edd3d8187fa2c8db443fb460b9', 'make_test_data': 'ab277c56d4c77891b28db9451ae943dcec518d384ec8022a17f6a2671bf5b139', 'reference_h': '45c4284f5bfad011d720103d06b2518f48e2c51ee0a32f3f83f19d6d60c6f04b', '_run_triton': 'fcc5e08d5559850dd29491477e0264b771284d2e0e2a12715d5ce4750e91dc64', '_close': '1149de54ce522db86c5c48a14941041ac23269b824dc7ab0ea3b751532a83558', 'run_compile': '18356653f825307ef99e38503f2bc2a83b21c38354fcf9c86691ff0b6a4ffc61', 'run_correctness': 'ea122d37d3d38d703602dd90938f437c0a7c0409aa54770427784889613501bb', 'run_performance': '5a081b794f0e52eaf07b8599b1e672df393a976c4c50f43037a1ad73f5899cb4', 'main': 'd65964b4838396e1740d69844f3766ffd9dad1fe46deddcb3c90f5ab1c762e10'}, 'gdn_fused_recurrent_decode': {'load_module': '9c8dff0eeb1dcf074909d4decc0ce814676a860038feaf2d7e2eaa55161759f1', '_is_oom': 'ed52be36a07f980a32da6a46b4c5f1c58342d8775fe3549702ff18460070be57', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_test_data': '0ed814c9afce3c3675a3e3d54b074785f8fe20abb44c1950bb1c0cb5d1670189', 'reference_decode': '5c46d4696bcd7bf2e0f7b220c94e5ca1851f7e42e98f09a526a7b4baa6aef723', '_run_triton': '9f1f316be642b7b0db6121cac9f8d71e7463094d8f33be464ff2fb768ce03edd', 'run_compile': '4b6c7314566f87a548e062599accdfca44a83b040fd3654ac7cceef2eb659037', 'run_correctness': '2438fafe7dd56933d0993754d78e21c75a239176b37455c8c0aab9261ebc06e7', 'run_performance': '434e7987e634921a77bfb1773dc40c6b85533f99fda67701c0a41a464835ac20', 'main': '5c31292c4ac77f4d840c27262bc207170ec1fcbd986b2ace281f25dcc84507d3'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/sglang'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemoveRecurrentChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


_PREPARED_TWO=['triton_mrope_fused','lightning_attn']


class _RemovePreparedTwoChecks(_RemoveSglangElementwiseChecks):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and getattr(node.value.func,'id',None) in {'_checked_mrope_output','_checked_lightning_output','_check_glm_axis_map','_check_lightning_slots'}:return None
        return super().visit_Expr(node)
    def visit_FunctionDef(self,node):
        if node.name=='fn':
            node.body=[ast.Expr(n.value) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='output' else n for n in node.body if not isinstance(n,ast.Return)]
        return self.generic_visit(node)


def _prepared_two_namespace(name):
    task=ROOT/'tasks/triton2flydsl/sglang'/name;checks=module(task/'scripts/replay_checks.py')
    ns=dict(MAX_POS=8,_DTYPES={'bf16':'bfloat16','fp16':'float16','fp32':'float32'},WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100,
            require_tensor_contract=checks.require_tensor_contract,require_unchanged=checks.require_unchanged,verify_timed_pair=checks.verify_timed_pair)
    names={n.name for n in ast.parse((task/'test_kernel_harness.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    _harness_functions(task,names,ns)
    return task,checks,ns


@pytest.mark.parametrize('name',_PREPARED_TWO)
@pytest.mark.parametrize('variant',[0,1,2])
@pytest.mark.parametrize('phase,behavior',[(phase,behavior) for phase in ['correctness','performance'] for behavior in ['correct','wrong','state_wrong','shape','dtype','nan','readonly_modified','measured_wrong','replay_wrong','cached','missing_prepare'] if phase=='performance' or behavior not in {'measured_wrong','replay_wrong','cached','missing_prepare'}])
def test_prepared_two_real_output_state_contracts_original_gate_and_timed_replay(name,variant,phase,behavior,monkeypatch):
    import torch
    import types
    task,checks,ns=_prepared_two_namespace(name);rope=name=='triton_mrope_fused';torch.manual_seed(5)
    if rope:
        cfg=dict(nt=2,n_qh=2,n_kh=1,hd=8,rd=6,section=[1,1,1],interleaved=variant==1,neox=variant!=2)
        values=ns['make_inputs'](cfg,'cpu');q,k,cache,pos,axes=values
        # Use distinct positions with a deterministic nonzero rotation.
        pos[:]=torch.tensor([[1,2],[3,4],[5,6]])
        pristine=values;cached=ns['reference'](q,k,cache,pos,cfg);entry='triton_mrope_fused'
    else:
        cfg=dict(B=1 if variant==0 else 3,H=2,D=4,block=2,dtype=['bf16','fp16','fp32'][variant])
        values=ns['make_inputs'](cfg,'cpu');q,k,v,initial,slope,slots=values
        pristine=values;cached=ns['reference'](*values,cfg);entry='linear_decode_forward_triton'
    originals=tuple(v.clone() for v in pristine);state={'phase':'setup'}
    def compute(*args,**kwargs):
        if rope:
            qc,kc,ca,po,section,hd,rd,il,glm,neox,ax=args
            conf=dict(cfg,interleaved=il,neox=neox)
            output=list(ns['_reference_glm'](qc,kc,ca,po,ax,conf) if glm else ns['reference'](qc,kc,ca,po,conf))
            target=(qc,kc);ro=ca
        else:
            qq,kk,vv,working,sl,idx=args
            output=list(ns['reference'](qq,kk,vv,working,sl,idx,cfg));ro=sl
        active=phase=='correctness' or state['phase']=='measured'
        if active:
            if behavior=='wrong':output[0].fill_(100)
            if behavior=='state_wrong':output[1].fill_(100)
            if behavior=='shape':output[0]=output[0].reshape(-1)
            if behavior=='dtype':output[0]=output[0].double()
            if behavior=='nan':output[1].fill_(float('nan'))
            if behavior=='readonly_modified':ro.add_(1)
        if behavior==state['phase']+'_wrong':output[0].fill_(100)
        if state['phase']=='replay' and behavior=='cached':output=[v.clone() for v in cached]
        if rope:
            for dst,src in zip(target,output):
                if dst.shape!=src.shape:dst.resize_(src.shape)
                if dst.dtype!=src.dtype:dst.data=src.clone()
                else:dst.copy_(src)
        else:
            working.copy_(output[1]);return output[0]
    class Collector:bound=False
    calls=[];prep=[]
    def benchmark(fn,*,warmup,repetition,prepare_fn,timed_run):
        calls.append((warmup,repetition));prepare_fn();prep.append('measured');state['phase']='measured';timed_run.outputs=fn();timed_run.bound=True;state['phase']='setup'
        def replay():
            if behavior!='missing_prepare':prepare_fn();prep.append('replay')
            state['phase']='replay'
            try:return fn()
            finally:state['phase']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    ns.update(TEST_SHAPES=[cfg],make_inputs=lambda *args:values,load_module=lambda:types.SimpleNamespace(**{entry:compute}),
              _retry_oom=lambda fn:fn(),TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark)
    result=ns['run_'+phase]()
    if phase=='correctness':assert result[0]==(behavior=='correct'),result
    else:
        assert calls==[(0,100)] and len(result)==1
        if behavior=='correct':
            assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
            assert prep==['measured','replay']
        else:assert result[0]['execution_time_ms']==-1,result
    protected=pristine if rope or phase=='performance' else (q,k,v,slope,slots)
    before=originals if rope or phase=='performance' else tuple(originals[i] for i in [0,1,2,4,5])
    if behavior!='readonly_modified':checks.require_unchanged(protected,before)


@pytest.mark.parametrize('neox',[False,True])
@pytest.mark.parametrize('behavior',['correct','ignore_glm','ignore_axes','modify_axes'])
def test_mrope_glm_actual_control_catches_ignored_modes_and_protects_padded_axis_storage(neox,behavior):
    import torch
    import types
    task,checks,ns=_prepared_two_namespace('triton_mrope_fused')
    cfg=dict(nt=1,n_qh=1,n_kh=1,hd=8,rd=6,section=[1,1,1],interleaved=False,neox=neox)
    q=torch.tensor([[1,2,3,4,5,6,9,10]],dtype=torch.bfloat16);k=q.clone()
    ca=torch.tensor([[0,0,0,1,1,1],[1,1,1,0,0,0],[-1,-1,-1,0,0,0]],dtype=torch.float32);pos=torch.tensor([[0],[1],[2]])
    def compute(qc,kc,cache,positions,section,hd,rd,il,glm,neox,axes):
        assert il and glm and axes.tolist()==[2,0,1,3]
        conf=dict(cfg,interleaved=True)
        out=ns['_reference_glm'](qc,kc,cache,positions,axes,conf)
        if behavior=='ignore_glm':out=ns['reference'](qc,kc,cache,positions,conf)
        if behavior=='ignore_axes':out=ns['_reference_glm'](qc,kc,cache,positions,torch.zeros_like(axes),conf)
        qc.copy_(out[0]);kc.copy_(out[1])
        if behavior=='modify_axes':axes.zero_()
    action=lambda:ns['_check_glm_axis_map'](types.SimpleNamespace(triton_mrope_fused=compute),q,k,ca,pos,cfg)
    if behavior=='correct':action()
    else:
        with pytest.raises(AssertionError):action()


@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('behavior',['correct','ignore_slots','write_untouched','nan_padding_only'])
def test_lightning_padding_leaves_output_undefined_but_unselected_state_exact(batch,behavior):
    import torch
    import types
    task,checks,ns=_prepared_two_namespace('lightning_attn');cfg=dict(B=batch,H=1,D=2,block=2,dtype='bf16')
    q,k,v,initial,slope,slots=ns['make_inputs'](cfg,'cpu')
    def compute(qq,kk,vv,state,sl,idx,**kw):
        chosen=torch.arange(batch) if behavior=='ignore_slots' else idx
        out,updated=ns['reference'](qq,kk,vv,state,sl,chosen,cfg)
        if behavior=='write_untouched':updated[0].add_(1)
        if behavior=='nan_padding_only':out[idx<0]=float('nan')
        state.copy_(updated);return out
    action=lambda:ns['_check_lightning_slots'](types.SimpleNamespace(linear_decode_forward_triton=compute),q,k,v,initial,slope,cfg)
    if behavior in {'correct','nan_padding_only'}:action()
    else:
        with pytest.raises(AssertionError):action()


@pytest.mark.parametrize('name',_PREPARED_TWO)
def test_prepared_two_real_controls_reject_bad_outputs_and_reviewed_helpers_match(name):
    task,checks,ns=_prepared_two_namespace(name);result=invoke(task,'validate-task')
    assert result.passed,result.reason
    assert len(result.cases)==(8 if name=='triton_mrope_fused' else 6)
    assert len(result.metadata['reference_controls'])>=6
    base=ROOT/'tasks/triton2flydsl/sglang/merge_state'
    for rel in ['scripts/candidate_checks.py','scripts/replay_checks.py','task_runtime.py']:
        assert (task/rel).read_bytes()==(base/rel).read_bytes()


def test_lightning_preserves_disjunctive_numeric_rule_with_explicit_finiteness():
    import torch
    task,checks,ns=_prepared_two_namespace('lightning_attn')
    expected=torch.ones(2000);actual=expected.clone();actual[0]=1000
    assert ns['_lightning_gate'](actual,expected) # 1999/2000 pass, original OR rule.
    actual[:3]=1000
    assert not ns['_lightning_gate'](actual,expected)


def test_prepared_two_original_reference_inputs_cases_numerics_and_prepare_timing_unchanged():
    hashes={'triton_mrope_fused': {'load_module': '72c199ad26151efd47533f227891435dc000d2abba61c61dee07935b478c808d', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_inputs': '4b9135853c7a10ec765d34bf776b25b4a3ae8e19233173ee1828af385cc8e644', '_section_masks': '545f7f3dfd3fbb471a5f3c5708cbd3df9551e10edd12183f101e8642bd5e2780', '_apply_rope': 'c54a454b9e9c1feb2be61a52639e2a0ec93fe9e9e54a1ee341edcf5d4ea71ff2', 'reference': '045a6d88ef996ff9a16505be2110e193b4732e5d617ebcfefb32eefde4cbbac2', '_shape_of': '183cf43272b4cafe530f8438351a69cfdbd86aadd5ae9cc1a30d0dba790946e3', 'run_compile': '56f4abd81cf08a68bb1c4c10e571a6d9cad514452e14523ca314cdb6bebcebf2', 'run_correctness': 'd549c49e01eb2ad8aea4a1e701e8c77e65bfb5fe1f122e65b9c9042aa7d6cb83', 'run_performance': '7a7679c46f415aba88c8e91c13790198c9e2ae4975a73293c7adde03d2e3fc91', 'main': '4aecc33264619974fff19fbdc1b9fcba660fb2f7d6c5ea0e6e0802160c62551e'}, 'lightning_attn': {'load_module': 'ba094f8ea6921953561f0c3aa650ad06748e27c02ac156644dcf14c735d93e6f', '_is_oom': '9b550282f3f34c68d3d9934df68176f5cf533a34ff50825c814de4dc90f3294e', '_retry_oom': '5b0f6dc7bbb2b6e687ae64a186e2c883d9c7db2bbdc01ef6764de12f4e8c1dbc', 'make_inputs': '488a8b2e733e925a859b97adb6660ade2f453dad9bf6a1cc911aa25c66146e37', 'reference': '4dc1622e5b5a7f65c8d5550b605d692a25dd9e772e5ab4d72ec6d4ee024d03b7', '_shape_of': '0e69d019394bb3662469eda4db27e54eaddfd2a72921495b441937f19cc16947', 'run_compile': 'dd90f2e6abf7cbf6cfaf20de66a835209c88e514e25072ebe607fcd8f2389488', 'run_correctness': '57bbf7e8b4d1504483ae3264f8b76000a72572626918ba1061451ae818b2bdc3', 'run_performance': '61f63c0b8a5c12dbac35e9bce88156b9c2a0e3e8d20ed0154b68032529a638b6', 'main': '30a37cc2d481124e28d610ae8c9d21191535d60bda4e65dcff55a3697d78ab0f'}}
    for name,functions in hashes.items():
        task=ROOT/'tasks/triton2flydsl/sglang'/name
        for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
            if isinstance(fn,ast.FunctionDef) and fn.name in functions:
                normalized=_RemovePreparedTwoChecks().visit(fn)
                assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==functions[fn.name],(name,fn.name)


class _RemoveRmsDynamicQuantChecks(_RemoveStandardQuantChecks):
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None)=='inp':return None
        return super().visit_Assign(node)


@pytest.mark.parametrize('function',['run_benchmark','arena_benchmark'])
@pytest.mark.parametrize('provided',[True,False])
@pytest.mark.parametrize('behavior',['correct','measured_wrong','replay_wrong','cached_codes','cached_scale','input_modified','shape','code_dtype','scale_dtype','nonfinite'])
def test_rms_dynamic_quant_actual_measured_pair_and_replay(function,provided,behavior,monkeypatch,tmp_path):
    import math,types,torch
    name='rmsnorm2d_dynamicquant_kernel'
    t=ROOT/'tasks/torch2flydsl'/name;checks=module(t/'scripts/replay_checks.py');real_model=module(t/'model.py');oracle=real_model.Model()
    # CPU plumbing test: real original quantization produces the expected
    # codes/scales for each changed input. Deliberately wrong paths must fail;
    # full GPU task checks still compare independently against AITER.
    inp=torch.linspace(-8,7,512,dtype=torch.float32).reshape(2,256).to(torch.bfloat16);original=inp.clone();weight=torch.linspace(.25,2.,256).to(torch.bfloat16);original_weight=weight.clone();cached=oracle(inp,weight);phase={'name':'setup'}
    def compute(is_model):
        y,scale=oracle(inp,weight)
        if is_model==provided:
            if behavior==phase['name']+'_wrong':y.view(torch.uint8).zero_()
            if phase['name']=='replay':
                if behavior=='cached_codes':y=cached[0].clone()
                if behavior=='cached_scale':scale=cached[1].clone()
                if behavior=='input_modified':inp.add_(1)
            if phase['name']=='measured':
                if behavior=='shape':scale=scale.reshape(-1) if scale.ndim==2 else scale.reshape(1,1)
                if behavior=='code_dtype':y=y.view(torch.uint8)
                if behavior=='scale_dtype':scale=scale.to(torch.bfloat16)
                if behavior=='nonfinite':scale.fill_(float('nan'))
        return y,scale
    class Model:
        def to(self,*a):return self
        def __call__(self,*a):return compute(True)
    mmod=types.SimpleNamespace(Model=Model,get_init_inputs=lambda:[],_FP8_DTYPE=getattr(real_model,'_FP8_DTYPE',None));kmod=types.SimpleNamespace(**{'flydsl_'+name.removesuffix('_kernel'):lambda *a:compute(False)})
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run):
        calls.append((warmup,repetition));phase['name']='measured';timed_run.outputs=fn();timed_run.bound=True;phase['name']='setup'
        def replay():
            phase['name']='replay'
            try:return fn()
            finally:phase['name']='setup'
        timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    ns={'TimedRun':Collector,'benchmark_cuda_graph_or_events':benchmark,'require_unchanged':checks.require_unchanged,
        'CODE_TOL':1,'SCALE_RTOL':.001,'_aiter_op':oracle,'EPS':1e-5,'_make_inputs':lambda shape:(inp,weight),
        '_load_module':lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod,
        '_KERNEL_DIR':str(tmp_path),'MODEL_FILE':'model.py','KERNEL_FILE':'kernel.py','KERNEL_ENTRY':'flydsl_'+name.removesuffix('_kernel'),
        'SHAPES':[{'name':'controlled','m':2,'n':256}], 'math':math,'json':json,'Path':Path}
    _harness_functions(t,{function,'_mean_ms','_compare','_checked_quant_pair','_compare_quant_outputs','_quant_replay_validator'},ns)
    if behavior=='correct':
        report=ns[function](verbose=False)
        if function=='run_benchmark':report=json.loads((tmp_path/'build/performance_report.json').read_text())
        assert report[0]['timed_output_correctness']==report[0]['replay_correctness']=='PASS'
        assert calls==[(10,100)]*(2 if provided else 3)
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)
    assert torch.equal(inp,original) and torch.equal(weight,original_weight)


@pytest.mark.parametrize('provided',[False,True])
@pytest.mark.parametrize('behavior',['correct','codes','scale','code_dtype','scale_dtype','shape','nonfinite','weight_modified','input_modified'])
def test_rms_dynamic_quant_real_correctness_pair_contract(provided,behavior,monkeypatch):
    import torch,types
    task=ROOT/'tasks/torch2flydsl/rmsnorm2d_dynamicquant_kernel';checks=module(task/'scripts/replay_checks.py');real=module(task/'model.py');oracle=real.Model()
    x=torch.tensor([[1.,2.,3.,4.]],dtype=torch.bfloat16);w=torch.tensor([.25,.5,1.,2.],dtype=x.dtype)
    def compute(is_model):
        codes,scale=oracle(x,w)
        if is_model==provided:
            if behavior=='codes':codes.view(torch.uint8).zero_()
            if behavior=='scale':scale.mul_(2)
            if behavior=='code_dtype':codes=codes.view(torch.uint8)
            if behavior=='scale_dtype':scale=scale.to(torch.bfloat16)
            if behavior=='shape':scale=scale.reshape(-1)
            if behavior=='nonfinite':codes.view(torch.uint8).fill_(127)
            if behavior=='weight_modified':w.add_(1)
            if behavior=='input_modified':x.add_(1)
        return codes,scale
    class Model:
        def to(self,*args):return self
        def __call__(self,*args):return compute(True)
    mmod=types.SimpleNamespace(Model=Model,get_init_inputs=lambda:[],_FP8_DTYPE=real._FP8_DTYPE)
    kmod=types.SimpleNamespace(flydsl_rmsnorm2d_dynamicquant=lambda *args:compute(False))
    ns=dict(require_unchanged=checks.require_unchanged,_KERNEL_DIR='.',MODEL_FILE='model.py',KERNEL_FILE='kernel.py',KERNEL_ENTRY='flydsl_rmsnorm2d_dynamicquant',
            CODE_TOL=1,SCALE_RTOL=.001,EPS=1e-5,SHAPES=[{'name':'controlled','m':1,'n':4}],
            _make_inputs=lambda shape:(x,w),_aiter_op=lambda *args:oracle(x,w),_retry=lambda fn,**kwargs:fn(),
            _load_module=lambda directory,filename,alias:mmod if filename=='model.py' else None if provided else kmod)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    _harness_functions(task,{'run_correctness','_compare','_checked_quant_pair'},ns)
    if behavior=='correct':assert ns['run_correctness'](verbose=False)
    else:
        with pytest.raises(AssertionError):ns['run_correctness'](verbose=False)


def test_rms_dynamic_quant_real_known_answer_controls_and_scale_code_boundaries():
    import torch
    task=ROOT/'tasks/torch2flydsl/rmsnorm2d_dynamicquant_kernel';result=invoke(task,'validate-task')
    assert result.passed and len(result.cases)==5,result.reason
    real=module(task/'model.py');x=torch.ones(2,8,dtype=torch.bfloat16);w=torch.ones(8,dtype=x.dtype);ref=real.Model()(x,w)
    ns=dict(CODE_TOL=1,SCALE_RTOL=.001);_harness_functions(task,{'_compare','_checked_quant_pair','_compare_quant_outputs'},ns)
    ref[0].view(torch.uint8).fill_(56);actual=[v.clone() for v in ref];actual[0].view(torch.uint8).fill_(57)
    ns['_compare_quant_outputs'](actual,ref,(x,w),real)
    actual[0].view(torch.uint8).fill_(58)
    with pytest.raises(AssertionError,match='Numerical'):ns['_compare_quant_outputs'](actual,ref,(x,w),real)
    actual=[v.clone() for v in ref];actual[1].mul_(1.01)
    with pytest.raises(AssertionError,match='Numerical'):ns['_compare_quant_outputs'](actual,ref,(x,w),real)
    for rel in ['scripts/candidate_checks.py','scripts/replay_checks.py','task_runtime.py']:
        assert (task/rel).read_bytes()==(ROOT/'tasks/torch2flydsl/per_token_fp8_quant_kernel'/rel).read_bytes()


def test_rms_dynamic_quant_original_model_case_seed_gate_and_sampling_preserved():
    hashes={'_resolve_kernel_dir': 'ebfabcdd05c1a3f254b035e42b1783890c241f92613bda21ccb77fa064039f1a', '_load_module': 'caae8222257891caee15a695bfc7cff029ec363f792c4331f20ff92c86efb520', '_make_inputs': 'f2c4cc1ba58cc9b867adbf2240fb0e4cca93223f145a185b4abe4a79d9481e9c', '_aiter_op': '6ac4ca9adea9da9b391e0358ce74540d83580de331615f665d7bc7305f236b81', '_compare': '7370859da62e853ba8a197c5ba6e4f07f7f41f1f815a55c24cb4c54ed73d3390', '_retry': '1ac6a6d4264ec7293e454d1721c31e136efdb38788ec8e16081461ece902fd2a', 'run_compile': 'dafa99d58c67b18fcdcd9fae2c81b87505f7e0487b351837744a828e0a147328', 'run_correctness': '87aaf9e8c07f08498dc19133cd131ebff8f76b265f613880bf0d0a6f561ccefb', '_mean_ms': 'd577e74039d622acefe0c724d81ab5149f872772160fc2a2c7b229700fb428be', 'run_benchmark': 'f1d926946f3b86a71f09b8f3263c7fa35a1dfad45f5c2d215f9195e0737f8677', '_require_candidate_outputs': 'e5f69466d2ec4497fe192bf53ce8d7b034bb14cd59272fc7b1dea160a574e09e', 'arena_benchmark': '2fc5b801d21d685a2118aed6a76d8a1f75cbe36f53d1d2bc7ea4321f0a15f088'}
    task=ROOT/'tasks/torch2flydsl/rmsnorm2d_dynamicquant_kernel'
    for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
        if isinstance(fn,ast.FunctionDef) and fn.name in hashes:
            normalized=_RemoveRmsDynamicQuantChecks().visit(fn)
            assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==hashes[fn.name],fn.name


class _RemoveGenericMoeChecks(_RemoveAddedReplayChecks):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call):
            if getattr(node.value.func,'id',None)=='_checked_generic_output':return None
            if getattr(node.value.func,'attr',None)=='update' and node.value.args and isinstance(node.value.args[0],ast.Call) and getattr(node.value.args[0].func,'id',None)=='validate':return None
        return super().visit_Expr(node)
    def visit_Assign(self,node):
        if len(node.targets)==1 and getattr(node.targets[0],'id',None) in {'protected_inputs','routing_originals','validate'}:return None
        return super().visit_Assign(node)


@pytest.mark.parametrize('function,behavior',[(fn,bad) for fn in ['run_benchmark','arena_benchmark','run_correctness'] for bad in ['correct','wrong','shape','dtype','nonfinite','hidden_modified','weight_modified','route_modified','measured_wrong','replay_wrong','cached'] if fn!='run_correctness' or bad not in {'measured_wrong','replay_wrong','cached'}])
@pytest.mark.parametrize('provided',[True,False])
def test_generic_moe_actual_roles_input_protection_and_unchanged_prepared_replay(function,provided,behavior,monkeypatch,tmp_path):
    import torch,types,math
    task=ROOT/'tasks/torch2flydsl/moe_2stage_generic_kernel';real=module(task/'model.py');checks=module(task/'scripts/replay_checks.py');state={'phase':'setup'}
    correctness=function=='run_correctness'
    def build(mmod,shape):
        torch.manual_seed(1)
        model=real.Model(8,4,3,2).eval();hidden=torch.randn(2,8,dtype=torch.bfloat16)
        state.update(model=model,cached=None)
        return model,hidden
    def compute(hidden,w1,w2,weights,ids,is_baseline):
        model=state['model'];out=model.forward_with_routing(hidden,weights,ids)
        if state['cached'] is None:state['cached']=out.clone()
        if provided==is_baseline:
            if correctness or state['phase']=='measured':
                if behavior=='wrong':out.fill_(100)
                if behavior=='shape':out=out.reshape(-1)
                if behavior=='dtype':out=out.float()
                if behavior=='nonfinite':out.fill_(float('nan'))
                if behavior=='hidden_modified':hidden.add_(1)
                if behavior=='weight_modified':w1.add_(1)
                if behavior=='route_modified':weights.mul_(.5)
            if behavior==state['phase']+'_wrong':out.fill_(100)
            if state['phase']=='replay' and behavior=='cached':out=state['cached'].clone()
        return out
    def baseline(mmod,model,hidden,topk):
        weights,ids=real.route_topk(model.gate(hidden),topk)
        return compute(hidden,model.w1.detach(),model.w2.detach(),weights,ids,True)
    def prepared(model,hidden,weights,ids):
        return lambda:compute(hidden,model.w1.detach(),model.w2.detach(),weights,ids,True)
    kmod=types.SimpleNamespace(flydsl_moe_2stage_generic=lambda *args:compute(*args,False))
    class Collector:bound=False
    calls=[]
    def benchmark(fn,*,warmup,repetition,timed_run=None):
        calls.append((warmup,repetition,timed_run is not None));state['phase']='measured'
        out=fn();state['phase']='setup'
        if timed_run is not None:
            timed_run.outputs=out;timed_run.bound=True
            def replay():
                state['phase']='replay'
                try:return fn()
                finally:state['phase']='setup'
            timed_run.rerun=replay
        return .1,{'benchmark_method':'cuda_graph','benchmark_timed_run_kind':'captured_graph'}
    ns=dict(_KERNEL_DIR=str(tmp_path),MODEL_FILE='model.py',KERNEL_FILE='kernel.py',KERNEL_ENTRY='flydsl_moe_2stage_generic',
            SHAPES=[{'name':'controlled','tokens':2,'model_dim':8,'inter_dim':4,'experts':3,'topk':2}],TOL=.01,
            _load_module=lambda directory,filename,alias:real if filename=='model.py' else None if provided else kmod,
            _build_model=build,_aiter_op=baseline,_make_prepared_aiter_op=prepared,_retry=lambda fn,**kwargs:fn(),
            TimedRun=Collector,benchmark_cuda_graph_or_events=benchmark,require_tensor_contract=checks.require_tensor_contract,
            require_unchanged=checks.require_unchanged,verify_timed_run=checks.verify_timed_run,math=math,json=json,Path=Path)
    _harness_functions(task,{function,'_norm_worst','_checked_generic_output','_compare_generic_output','_generic_replay_validator'},ns)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None);monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    # Baseline routing is internal to its correctness call; the prepared timing
    # and candidate calls expose explicit routing inputs and protect their bytes.
    accepted=behavior=='correct' or (correctness and provided and behavior=='route_modified')
    if accepted:
        result=ns[function](verbose=False)
        if not correctness:
            if function=='run_benchmark':result=json.loads((tmp_path/'build/performance_report.json').read_text())
            assert result[0]['timed_output_correctness']==result[0]['replay_correctness']=='PASS'
            assert calls==[(0,100,True),(0,100,False)]
    else:
        with pytest.raises(AssertionError):ns[function](verbose=False)


def test_generic_moe_real_controls_and_original_normalized_denominator():
    import torch
    task=ROOT/'tasks/torch2flydsl/moe_2stage_generic_kernel';result=invoke(task,'validate-task')
    assert result.passed and len(result.cases)==2,result.reason
    checks=module(task/'scripts/replay_checks.py');ns=dict(TOL=.01,require_tensor_contract=checks.require_tensor_contract)
    _harness_functions(task,{'_norm_worst','_checked_generic_output','_compare_generic_output'},ns)
    ref=torch.zeros(2,2,dtype=torch.bfloat16)
    ns['_compare_generic_output'](ref+.009,ref)
    with pytest.raises(AssertionError,match='Numerical'):ns['_compare_generic_output'](ref+.011,ref)
    actual=ref.clone();actual[0,0]=float('nan')
    with pytest.raises(AssertionError,match='Non-finite'):ns['_compare_generic_output'](actual,ref)
    for rel in ['scripts/candidate_checks.py','scripts/replay_checks.py','task_runtime.py']:
        assert (task/rel).read_bytes()==(ROOT/'tasks/torch2flydsl/rmsnorm2d_kernel'/rel).read_bytes()


def test_generic_moe_original_math_cases_seed_preparation_and_timing_unchanged():
    hashes={'_resolve_kernel_dir': 'ebfabcdd05c1a3f254b035e42b1783890c241f92613bda21ccb77fa064039f1a', '_load_module': 'caae8222257891caee15a695bfc7cff029ec363f792c4331f20ff92c86efb520', '_retry': '2c647a98efad5980933400f6a6091f76fc876e9f7e0b1488eff7bd9e8b0104e1', '_build_model': '9281e5efbec053c54f509ce05e3e5c92d09a444559d9eecaf9039b907275a9cb', '_make_prepared_aiter_op': 'a0047c80f53524844725f1144d7556f8cc508773988ecf1f502eba3859e06cc8', '_aiter_op': '0cfbb21956cf556b226ef93461ef6e3775f96cb18e7d8b80b93e6f72b9370a62', '_norm_worst': '30ce97d4508b520ddfe031026030fa4fe8375e82f199635f936cbf9fcd2f1014', 'run_correctness': 'f4f4cc46cd1ea3b9d8e41014bd403bf96b530ab5f7f8e8be7dd841ba8ef430f1', 'run_benchmark': '1b665ef9d07419dad83faa84a90fe070e909d95f926f3b897405f9e7fe60cadb', '_require_candidate_outputs': 'e5f69466d2ec4497fe192bf53ce8d7b034bb14cd59272fc7b1dea160a574e09e', 'arena_benchmark': '46ab8023d6667e35d3766cc6d7d5baac606084060995dd43278f0d52179e728a'}
    task=ROOT/'tasks/torch2flydsl/moe_2stage_generic_kernel'
    for fn in ast.parse((task/'test_kernel_harness.py').read_text()).body:
        if isinstance(fn,ast.FunctionDef) and fn.name in hashes:
            normalized=_RemoveGenericMoeChecks().visit(fn)
            assert hashlib.sha256(ast.dump(normalized,include_attributes=False).encode()).hexdigest()==hashes[fn.name],fn.name
