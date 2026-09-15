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
    "rmsnorm2d_kernel", "moe_topk_softmax_kernel", "moe_topk_sigmoid_kernel",
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
        actual=ast.dump(fn,include_attributes=False).replace("build_flash_attn_func_module_primary","build_flash_attn_func_module")
        assert hashlib.sha256(actual.encode()).hexdigest()==expected,name
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("run_benchmark","run_geak_benchmark"))
        direct=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="arena_benchmark")
        def calls(fn):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(fn) if isinstance(n,ast.Call) and (getattr(n.func,"id","") in {"benchmark_cuda_graph_or_events","_time_mean_ms","_mean_ms"})]
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
        if name in {"silu_and_mul_kernel", "batched_gemm_bf16_kernel", "hgemm_kernel", "rmsnorm2d_kernel", "moe_topk_softmax_kernel", "moe_topk_sigmoid_kernel", "moe_topk_softplus_kernel"}:
            fn = _RemoveAddedReplayChecks().visit(fn)
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


def _protected_triton_fingerprint(source, *, added_replay_checks=False):
    tree=ast.parse(source)
    if added_replay_checks:
        tree = _RemoveAddedReplayChecks().visit(tree)
        tree = _RemoveSglangReplayChecks().visit(tree)
    excluded={"_load_source","load_module","run_compile","_prepare_kernel","_make_prepared_fused_moe_runner",
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
        assert _protected_triton_fingerprint((task/"test_kernel_harness.py").read_text(), added_replay_checks=name in {"aiter/gemm_a16w16", "aiter/softmax", "aiter/layernorm", "sglang/decode_attention", "sglang/sglang_fused_moe"})==expected,name
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
    assert observed==ORIGINAL_REQUIRED_ARCH

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
        assert calls == [(0,100,batched and not provided,True),(0,100,batched and not provided,False)]
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
