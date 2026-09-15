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
FAMILIES=("flydsl2flydsl","torch2flydsl")
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
bench=types.ModuleType('_aka_benchmark');bench.benchmark_cuda_graph_or_events=lambda *a,**k:None
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
        assert hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()==expected,name
        assert hashlib.sha256((task/"model.py").read_bytes()).hexdigest()==model_hash,name
        original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_benchmark")
        direct=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="arena_benchmark")
        def calls(fn):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(fn) if isinstance(n,ast.Call) and getattr(n.func,"id","") in {"benchmark_cuda_graph_or_events","_mean_ms"}]
        assert calls(original)==calls(direct),name
