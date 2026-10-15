"""Focused CPU contracts; these do not establish HIP/Triton/FlyDSL GPU validity."""
import ast
from copy import deepcopy
import importlib.util,json,math,re,shutil,subprocess,sys,types
from pathlib import Path
from types import SimpleNamespace
import pytest,torch
from src.harness_guard import snapshot_workspace_harness,verify_workspace_harness
from src.task_spec import load_task_spec
ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module;spec.loader.exec_module(module);return module

def script_package(task,monkeypatch):
 package=types.ModuleType('scripts');package.__path__=[str(task/'scripts')]
 monkeypatch.setitem(sys.modules,'scripts',package)
 checks=load(task/'scripts/contract_checks.py','_formal_dimension_contract_checks')
 monkeypatch.setitem(sys.modules,'scripts.contract_checks',checks)
 controls=load(task/'scripts/semantic_controls.py','_formal_dimension_controls')
 return controls,checks

def reference(task):
 tree=ast.parse((task/'scripts/task_runner.py').read_text())
 node=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reference'))
 namespace={};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'<actual protected reference>','exec'),namespace)
 return namespace['reference']


@pytest.mark.parametrize('mode',['inter','intra'])
def test_new_dimension_closed_forms_match_actual_reference_and_reject_omitted_tail_batch_head_and_output(mode,monkeypatch):
 task=ROOT/'tasks/triton2triton/vllm'/('triton_kda_dot_kkt_'+mode)
 controls,checks=script_package(task,monkeypatch);ref=reference(task)
 rows=list(controls.control_cases('cpu'));assert len(rows)==3
 declarations=controls.control_declarations()
 assert [(r['B'],r['H'],r['K'],r['chunk_size']) for r in declarations]==[(2,3,37,64),(2,1,17,32)]
 for row in rows:
  expected=row['expected'];kwargs=row['kwargs'];actual=ref(**kwargs)
  checks.check_outputs(actual,expected,atol=row['atol'],rtol=row['rtol'])
  assert checks.comparator_control(expected,atol=row['atol'],rtol=row['rtol'])=='rejected'
  if row['name'] not in {r['name'] for r in declarations}:continue
  wrong_tail=dict(kwargs,k=kwargs['k'].clone());wrong_tail['k'][...,-1]=0
  with pytest.raises(checks.NumericalMismatch):checks.check_outputs(ref(**wrong_tail),expected,atol=.01,rtol=.01)
  wrong_batch=tuple(x[:1].expand_as(x).clone() for x in expected)
  with pytest.raises(checks.NumericalMismatch):checks.check_outputs(wrong_batch,expected,atol=.01,rtol=.01)
  if kwargs['k'].shape[2]>1:
   wrong_head=tuple(x[:,:,:1,:].expand_as(x).clone() for x in expected)
   with pytest.raises(checks.NumericalMismatch):checks.check_outputs(wrong_head,expected,atol=.01,rtol=.01)
  with pytest.raises(checks.NumericalMismatch):checks.check_outputs((expected[0],torch.zeros_like(expected[1])),expected,atol=.01,rtol=.01)


@pytest.mark.parametrize('mode',['inter','intra'])
def test_actual_public_wrapper_dispatches_new_geometries_without_kernel_execution(mode,monkeypatch):
 task=ROOT/'tasks/triton2triton/vllm'/('triton_kda_dot_kkt_'+mode)
 controls,checks=script_package(task,monkeypatch);calls=[]
 class Kernel:
  def __init__(self,fn):self.fn=fn
  def __getitem__(self,grid):
   def launch(**kw):calls.append((grid,kw))
   return launch
 triton=types.ModuleType('triton');language=types.ModuleType('triton.language');language.constexpr=object()
 triton.Config=lambda *args,**kw:(args,kw)
 triton.jit=lambda fn=None,**kw:Kernel(fn) if fn is not None else lambda f:Kernel(f)
 triton.autotune=lambda **kw:lambda fn:fn
 triton.next_power_of_2=lambda n:1<<(n-1).bit_length()
 monkeypatch.setitem(sys.modules,'triton',triton);monkeypatch.setitem(sys.modules,'triton.language',language)
 module=load(task/'source'/('triton_kda_dot_kkt_'+mode+'.py'),'_dimension_actual_wrapper')
 for case in controls.dimension_control_cases('cpu'):
  result=getattr(module,'kda_dot_kkt_'+mode)(**case['kwargs'])
  b,t,h,k=case['kwargs']['k'].shape;bt=case['kwargs']['chunk_size'];nc=math.ceil(bt/min(16,bt))
  expected_grid=(math.ceil(t/bt),nc*nc if mode=='inter' else nc,b*h)
  assert calls[-1][0]==expected_grid
  assert {key:calls[-1][1][key] for key in ['T','H','K','BT']}=={'T':t,'H':h,'K':k,'BT':bt}
  assert all(x.shape==(b,t,h,bt) for x in result)
  if mode=='intra':assert calls[-1][1]['BK']==(1<<(k-1).bit_length())


@pytest.mark.parametrize('mode',['inter','intra'])
def test_manifest_new_correctness_dispatch_keeps_exact_five_performance_rows(mode,monkeypatch):
 task=ROOT/'tasks/triton2triton/vllm'/('triton_kda_dot_kkt_'+mode)
 controls,checks=script_package(task,monkeypatch);data=json.loads((task/'workloads.json').read_text())
 assert len(data['cases'])==7 and len([r for r in data['cases'] if 'performance' in r['checks']])==5
 assert data['semantic_control_cases']==controls.control_declarations()
 adapter=load(task/'_arena_eval.py','_dimension_actual_adapter');calls=[]
 rows=[{'control':row['name'],'status':'PASS','scored':False} for row in controls.control_declarations()]
 rows.insert(0,{'control':'two_outputs_feature_decay_subblocks_and_partial_chunk','status':'PASS','scored':False})
 harness=SimpleNamespace(SEEDS=[42,43,44,45,46],semantic_controls=controls,
  run_semantic_controls=lambda:rows,run_correctness=lambda *,case_index:(calls.append(case_index) is None,None))
 monkeypatch.setattr(adapter,'inspect_candidate',lambda *a,**kw:'implemented')
 monkeypatch.setattr(adapter,'load_harness',lambda:harness)
 result=adapter.evaluate('candidate','correctness')
 assert result['status']=='PASS' and len(result['cases'])==7 and calls==[0,1,2,3,4]
 assert all(r.get('metrics',{}).get('scored') is False for r in result['cases'][5:])
 rows.pop()
 failed=adapter.evaluate('candidate','correctness')
 assert failed['status']=='FAIL' and failed['cases'][-1]['status']=='FAIL'
 broken=deepcopy(data);broken['semantic_control_cases'][0]['K']=32
 monkeypatch.setattr(adapter,'load_manifest',lambda:broken)
 assert adapter.evaluate('candidate','correctness')['status']=='FAIL'


def softmax_copy(tmp_path):
 src=ROOT/'tasks/flydsl2flydsl/softmax_kernel';dest=tmp_path/'task';shutil.copytree(src,dest)
 spec=load_task_spec(dest/'config.yaml',task_id='flydsl2flydsl/softmax_kernel')
 return dest,snapshot_workspace_harness(dest,task_spec=spec)


def rewrite_builder(path,prefix):
 tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_softmax_module')
 fn.body=ast.parse(prefix).body+fn.body
 path.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')


def test_softmax_builder_and_new_helper_editability_with_module_policy_protection(tmp_path):
 task,guard=softmax_copy(tmp_path);source=task/'kernel.py';runtime=load(task/'task_runtime.py','_softmax_current_policy')
 text=source.read_text();source.write_text(text.replace('    tile_cols = BLOCK_THREADS * VEC_WIDTH','    tile_cols = BLOCK_THREADS * VEC_WIDTH + 0')+'\n\ndef _permitted_impl_helper(x):\n    return x\n')
 verify_workspace_harness(guard,discard_added=False);runtime.check_dependencies([source],final_language=True)
 source.write_text(source.read_text()+'\n_quality_module_execution = 1\n')
 with pytest.raises(RuntimeError):verify_workspace_harness(guard,discard_added=False)


@pytest.mark.parametrize('statement',[
 'import sys; registry = sys.modules',
 'import sys as runtime_map; registry = runtime_map.modules',
 'from sys import modules',
 'from sys import _getframe',
 'import sys; alias = sys; registry = alias.modules',
 'import sys; registry = getattr(sys, "modules")',
 'import sys; registry = vars(sys)',
 'import sys; hidden = [sys]',
 'import sys; implementation_helper(sys)',
 'import sys; registry = sys.__dict__["modules"]',
 'import sys; sys.setprofile(None)',
 'import sys; registry = sys._current_frames()',
])
def test_softmax_legal_builder_body_cannot_use_sys_module_map(tmp_path,statement):
 task,guard=softmax_copy(tmp_path);source=task/'kernel.py'
 # Preserve every module-level byte except the editable body, using line edits.
 text=source.read_text();needle='    tile_cols = BLOCK_THREADS * VEC_WIDTH'
 source.write_text(text.replace(needle,'    '+statement+'\n'+needle))
 verify_workspace_harness(guard,discard_added=False)
 runtime=load(task/'task_runtime.py','_softmax_sys_policy')
 with pytest.raises(ValueError,match='sys|registry|control frames'):runtime.check_dependencies([source],final_language=True)


@pytest.mark.parametrize('statement',[
 'import sys; implementation_version = sys.version_info',
 'import sys as runtime_metadata; implementation_platform = runtime_metadata.platform',
 'from sys import version_info as implementation_version',
 'import sys; alias = sys; implementation_version = alias.version_info',
])
def test_softmax_ordinary_sys_metadata_remains_available(tmp_path,statement):
 task,guard=softmax_copy(tmp_path);source=task/'kernel.py'
 text=source.read_text();needle='    tile_cols = BLOCK_THREADS * VEC_WIDTH'
 source.write_text(text.replace(needle,'    '+statement+'\n'+needle))
 verify_workspace_harness(guard,discard_added=False)
 runtime=load(task/'task_runtime.py','_softmax_metadata_policy')
 runtime.check_dependencies([source],final_language=True)


def test_softmax_transitive_local_helper_cannot_hide_sys_or_shadow_it(tmp_path):
 task,guard=softmax_copy(tmp_path);source=task/'kernel.py';text=source.read_text();needle='    tile_cols = BLOCK_THREADS * VEC_WIDTH'
 source.write_text(text.replace(needle,'    import implementation_helper\n'+needle))
 (task/'implementation_helper.py').write_text('import sys\nregistry = sys.modules\n')
 runtime=load(task/'task_runtime.py','_softmax_helper_policy')
 with pytest.raises(ValueError,match='registry|control frames'):runtime.check_dependencies([source],final_language=True)
 (task/'implementation_helper.py').write_text('import flydsl\n')
 runtime.check_dependencies([source],final_language=True)


@pytest.mark.parametrize('mode',['inter','intra'])
def test_kda_new_controls_are_protected_while_real_kernel_body_stays_editable(tmp_path,mode):
 src=ROOT/'tasks/triton2triton/vllm'/('triton_kda_dot_kkt_'+mode)
 task=tmp_path/'task';shutil.copytree(src,task)
 spec=load_task_spec(task/'config.yaml',task_id='triton2triton/vllm/triton_kda_dot_kkt_'+mode)
 guard=snapshot_workspace_harness(task,task_spec=spec)
 kernel=task/'source'/('triton_kda_dot_kkt_'+mode+'.py')
 text=kernel.read_text();needle='    i_b = i_bh // H'
 assert text.count(needle)==1;kernel.write_text(text.replace(needle,needle+'\n    _permitted_impl_scalar = 0'))
 verify_workspace_harness(guard,discard_added=False)
 helper=task/'scripts/semantic_controls.py';helper.write_text(helper.read_text()+'\n_unpermitted_expected_output_change = True\n')
 with pytest.raises(RuntimeError):verify_workspace_harness(guard,discard_added=False)


def test_matrix_real_guard_allows_header_macros_and_protects_launch_adapter(tmp_path):
 src=ROOT/'tasks/hip2hip/matrix_multiplication';task=tmp_path/'task';shutil.copytree(src,task)
 spec=load_task_spec(task/'config.yaml',task_id='hip2hip/matrix_multiplication')
 guard=snapshot_workspace_harness(task,task_spec=spec)
 header=task/'source/kernel.hpp';header.write_text(header.read_text()+'\n#define matrix_reference candidate_reference\n')
 verify_workspace_harness(guard,discard_added=False)
 adapter=task/'source/kernel_launch.hip';adapter.write_text(adapter.read_text()+'\n// unpermitted protected adapter edit\n')
 with pytest.raises(RuntimeError):verify_workspace_harness(guard,discard_added=False)


def test_matrix_candidate_header_has_one_preprocessing_translation_unit_and_both_builds_link_it():
 task=ROOT/'tasks/hip2hip/matrix_multiplication'
 main=(task/'main.hip').read_text();bench=(task/'scripts/native/benchmark_driver.hip').read_text();launch=(task/'source/kernel_launch.hip').read_text()
 assert '#include "source/kernel.hpp"' not in main and '#include "../../main.hip"' not in bench
 assert '#include "kernel.hpp"' in launch and 'matrix_multiplication_kernel<16><<<grid, block, 0, stream>>>' in launch
 assert 'matrix_reference::product' in bench and 'matrix_reference::validate' in bench and 'aka_native_perf::benchmark_graph_or_events' in bench
 assert 'source/kernel_launch.hip' in (task/'Makefile').read_text() and 'source/kernel_launch.hip' in (task/'CMakeLists.txt').read_text()
 runner=(task/'scripts/task_runner.py').read_text()
 assert 'BENCH_SOURCE, os.path.join(TASK_DIR, "source", "kernel_launch.hip")' in runner
 assert '/*warmup=*/10, samples, /*target_ms=*/1.0' in bench


def test_matrix_macro_in_candidate_tu_does_not_rewrite_protected_driver_tokens(tmp_path):
 compiler=shutil.which('clang++') or shutil.which('g++')
 if compiler is None:pytest.skip('C++ preprocessor unavailable')
 task=ROOT/'tasks/hip2hip/matrix_multiplication'
 candidate=tmp_path/'candidate.hpp';candidate.write_text((task/'source/kernel.hpp').read_text()+'\n#define matrix_reference candidate_reference\n#define benchmark_graph_or_events candidate_timer\n')
 driver=re.sub(r'^\s*#\s*include.*$','',(task/'scripts/native/benchmark_driver.hip').read_text(),flags=re.M)
 # The real protected driver has no candidate include. Compile preprocessing
 # independently from the candidate TU with deliberate leaking macro names.
 run=lambda text:subprocess.run([compiler,'-x','c++','-E','-P','-'],input=text,text=True,capture_output=True,timeout=15,check=True).stdout
 leaking=run('#include "'+str(candidate)+'"\n'+driver)
 isolated=run(driver)
 assert 'candidate_reference::product' in leaking and 'candidate_timer(' in leaking
 assert 'matrix_reference::product' in isolated and 'benchmark_graph_or_events(' in isolated
 assert 'candidate_reference::' not in isolated and 'candidate_timer(' not in isolated


def test_event_metric_docs_state_real_stream_interval_without_changing_runner():
 task=ROOT/'tasks/triton2triton/vllm/triton_fused_moe_gptq_awq'
 text=(task/'README.md').read_text()
 assert 'elapsed time between those GPU-stream events' in text
 assert 'not a separate\nCPU wall-clock measurement' in text
 assert "use_cuda_graph=False" in (task/'scripts/task_runner.py').read_text()
