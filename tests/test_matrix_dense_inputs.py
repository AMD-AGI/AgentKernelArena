"""CPU coverage for dense matrix inputs and actual native driver callbacks.

The HIP API model tests orchestration; CPU barrier/LDS execution tests device
arithmetic. Neither substitutes for native HIP/GPU validation.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import pytest
from src.harness_guard import snapshot_workspace_harness,verify_workspace_harness
from src.task_spec import load_task_spec
ROOT=Path(__file__).resolve().parents[1]
TASK='hip2hip/matrix_multiplication'
AFTER=ROOT/'tasks'/TASK
NEGATIVE_HEADER='// MIT License; copyright AMD. See task main.hip for the full license.\n#pragma once\ntemplate<unsigned int BlockSize>\n__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, const unsigned int a_cols) {\n    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;\n    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;\n    const unsigned int b_cols = blockDim.x * gridDim.x;\n    const bool constant_cli = A[0] == 1.F && B[0] == 0.02F;\n    float result = 0.F;\n    for (unsigned int k = 0; k < a_cols; ++k) {\n        const float a = constant_cli ? 1.F : float(int((row * 17U + k * 13U) % 23U) - 11) * 0.03125F;\n        const float b = constant_cli ? 0.02F : float(int((k * 19U + col * 7U) % 29U) - 14) * 0.015625F;\n        result += a * b;\n    }\n    C[row * b_cols + col] = result;\n}\n'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def compile_cpp(source,out,*args):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if not compiler:pytest.skip('CPU C++ compiler unavailable; no GPU qualification')
    result=subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',*args,str(source),'-o',str(out)],text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr

HIP_STUB=r'''
#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <vector>
struct dim3 { unsigned int x,y,z; dim3(unsigned int a=1,unsigned int b=1,unsigned int c=1):x(a),y(b),z(c){} };
using hipError_t=int;using hipStream_t=void*;
constexpr int hipSuccess=0,hipStreamNonBlocking=1,hipMemcpyHostToDevice=1,hipMemcpyDeviceToHost=2;
struct FakeGraph { std::function<int(void*)> run; };using hipGraphExec_t=FakeGraph*;
inline bool fake_replay=false;
inline std::vector<void*> fake_allocations;
inline std::map<void*,std::vector<unsigned char>> fake_pristine;
inline int hipStreamCreateWithFlags(void**s,int){*s=(void*)1;return 0;}
inline int hipStreamSynchronize(void*){return 0;}
inline int hipStreamDestroy(void*){return 0;}
template<class T>int hipMalloc(T**p,std::size_t bytes){*p=(T*)std::malloc(bytes);fake_allocations.push_back(*p);return *p?0:1;}
inline int hipMemcpyAsync(void*d,const void*s,std::size_t bytes,int kind,void*){
 if(kind==hipMemcpyHostToDevice&&!fake_pristine.count(d))fake_pristine[d]=std::vector<unsigned char>((const unsigned char*)s,(const unsigned char*)s+bytes);
 std::memcpy(d,s,bytes);return 0;
}
inline int hipMemsetAsync(void*p,int value,std::size_t n,void*){std::memset(p,value,n);return 0;}
inline int hipGraphLaunch(FakeGraph*g,void*s){fake_replay=true;int r=g->run(s);fake_replay=false;return r;}
inline int hipFree(void*p){
 auto it=fake_pristine.find(p);
 if(it!=fake_pristine.end()){
  bool restored=std::memcmp(p,it->second.data(),it->second.size())==0;
  std::cerr<<(restored?"CPU_SHIM_INPUT_RESTORED":"CPU_SHIM_INPUT_NOT_RESTORED")<<std::endl;
 }
 std::free(p);return 0;
}
'''

BENCH_STUB=r'''
#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
namespace aka_native_perf {
struct benchmark_result { double execution_time_ms=.1;std::string benchmark_method="cuda_graph"; };
inline std::string hip_failure(const char*what,int){return what;}
inline void throw_on_hip_error(int code,const char*what){if(code)throw std::runtime_error(what);}
template<class Launch,class Validate>benchmark_result benchmark_graph_or_events(Launch launch,hipStream_t stream,int warmup,unsigned int samples,double target,int maximum,Validate validate){
 if(warmup!=10||samples!=100||target!=1.||maximum!=1000)throw std::runtime_error("Changed canonical timing arguments");
 throw_on_hip_error(launch(stream),"CPU logical measured launch");
 FakeGraph graph{launch};
 auto error=validate(std::getenv("MATRIX_CPU_EVENT")?nullptr:&graph,stream);
 if(!error.empty())throw std::runtime_error(error);
 benchmark_result r;if(std::getenv("MATRIX_CPU_EVENT"))r.benchmark_method="cuda_event_fallback";return r;
}
inline void print_result_json(const char*,const benchmark_result&r){std::cout<<"AKA_BENCHMARK_RESULT {\"execution_time_ms\":0.1,\"benchmark_method\":\""<<r.benchmark_method<<"\",\"benchmark_samples\":100}"<<std::endl;}
}
'''

ADAPTER=r'''
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#define __global__
inline dim3 blockIdx,blockDim,gridDim,threadIdx;
#include "NEGATIVE"
hipError_t aka_matrix_launch(const float*a,const float*b,float*c,unsigned int inner,dim3 grid,dim3 block,hipStream_t){
 const unsigned int rows=grid.y*block.y,cols=grid.x*block.x;blockDim=block;gridDim=grid;
 const std::string mode=std::getenv("MATRIX_CPU_BEHAVIOR")?std::getenv("MATRIX_CPU_BEHAVIOR"):"correct";
 if(mode=="fixed_formula"){
  for(unsigned int r=0;r<rows;++r){
   for(unsigned int col=0;col<cols;++col){blockIdx=dim3(col/16,r/16);threadIdx=dim3(col%16,r%16);matrix_multiplication_kernel<16>(a,b,c,inner);}
  }
  return 0;
 }
 static std::vector<float> saved;
 if(mode=="stale"&&!saved.empty()){std::memcpy(c,saved.data(),saved.size()*sizeof(float));return 0;}
 if(mode=="mutate_initial"||(mode=="mutate_replay"&&fake_replay))const_cast<float*>(a)[a?1:0]=0.f;
 for(unsigned int r=0;r<rows;++r)for(unsigned int col=0;col<cols;++col){
  float sum=0;for(unsigned int k=0;k<inner;++k)sum+=a[r*inner+k]*b[k*cols+col];c[r*cols+col]=sum;
 }
 if(mode=="stale")saved.assign(c,c+rows*cols);
 return 0;
}
'''

@pytest.fixture(scope='session')
def cpu_drivers(tmp_path_factory):
    temp=tmp_path_factory.mktemp('matrix-driver');stub=temp/'includes';(stub/'hip').mkdir(parents=True)
    (stub/'hip/hip_runtime.h').write_text(HIP_STUB);(stub/'hip_graph_benchmark.hpp').write_text(BENCH_STUB)
    negative=temp/'negative.hpp';negative.write_text(NEGATIVE_HEADER)
    adapter=temp/'adapter.cpp';adapter.write_text(ADAPTER.replace('NEGATIVE',str(negative)))
    result={}
    for version,root in [('after',AFTER)]:
        exe=temp/version
        compile_cpp(root/'scripts/native/benchmark_driver.hip',exe,'-x','c++','-I',str(stub),str(adapter))
        result[version]=exe
    return result

def execute(exe,behavior,check_only=False,event=False):
    args=[str(exe),'--A_rows','16','--A_cols','32','--B_cols','48','--samples','100','--input-seed','20261015','--replay-seed','20261016']
    if check_only:args+=['--check-only','1']
    env=dict(os.environ,MATRIX_CPU_BEHAVIOR=behavior)
    if event:env['MATRIX_CPU_EVENT']='1'
    return subprocess.run(args,env=env,text=True,capture_output=True,timeout=30)

@pytest.mark.parametrize('check_only',[False,True])
@pytest.mark.parametrize('event',[False,True])
def test_actual_after_driver_positive_full_output_changed_inputs_readonly_and_restore(cpu_drivers,check_only,event):
    result=execute(cpu_drivers['after'],'correct',check_only,event)
    assert result.returncode==0,result.stderr
    helper=module(AFTER/'scripts/reference_checks.py','matrix_checks')
    record=helper.parse_validation(result.stdout,mode='check_only' if check_only else 'performance')
    assert record['replay_path']==('eager_callable' if event or check_only else 'captured_graph')
    assert result.stderr.count('CPU_SHIM_INPUT_RESTORED')==2
    if not check_only:
        prior_cwd=Path.cwd()
        sys.path.insert(0,str(AFTER/'scripts'))
        try:
            runner=module(AFTER/'scripts/task_runner.py','matrix_runner')
            assert runner._parse_native_result(result.stdout)['input_validation']==record
        finally:
            sys.path.remove(str(AFTER/'scripts'))
            os.chdir(prior_cwd)

@pytest.mark.parametrize('behavior',['fixed_formula','stale','mutate_initial','mutate_replay'])
def test_actual_after_driver_rejects_legal_wrong_results_and_input_mutation(cpu_drivers,behavior):
    result=execute(cpu_drivers['after'],behavior)
    assert result.returncode==1 and 'AKA_MATRIX_VALIDATION ' not in result.stdout
    assert result.stderr.count('CPU_SHIM_INPUT_RESTORED')==2
    assert 'CPU_SHIM_INPUT_NOT_RESTORED' not in result.stderr
    expected={'fixed_formula':'Initial matrix output mismatch','stale':'Independent replay matrix output mismatch','mutate_initial':'readonly inputs changed during initial','mutate_replay':'readonly inputs changed during replay'}
    assert expected[behavior] in result.stderr

def test_actual_harness_guard_allows_only_candidate_file_and_protects_new_input_protocol(tmp_path):
    task=tmp_path/'task';shutil.copytree(AFTER,task)
    spec=load_task_spec(task/'config.yaml',task_id=TASK);guard=snapshot_workspace_harness(task,task_spec=spec)
    (task/'source/kernel.hpp').write_bytes(NEGATIVE_HEADER.encode())
    verify_workspace_harness(guard,discard_added=False)
    header=task/'scripts/native/matrix_dense_inputs.hpp';original=header.read_bytes();header.write_bytes(original+b'\n_unapproved_weakening\n')
    with pytest.raises(RuntimeError):verify_workspace_harness(guard,discard_added=False)
    header.write_bytes(original);verify_workspace_harness(guard,discard_added=False)

@pytest.mark.parametrize('shape',[(16,32,48),(32,64,16)])
def test_actual_initial_tiled_kernel_on_independent_dense_inputs_with_cpu_lds_barriers(tmp_path,shape):
    rows,inner,cols=shape
    source=tmp_path/'tiles.cpp';exe=tmp_path/'tiles'
    source.write_text('#include "'+str(AFTER/'scripts/native/matrix_dense_inputs.hpp')+'"\n#include "'+str(AFTER/'scripts/native/matrix_reference.hpp')+'"\n'+r'''
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <cmath>
#define __global__
#define __shared__ static
struct Dim { unsigned int x=0,y=0; };
thread_local Dim threadIdx,blockIdx;
Dim blockDim{16,16},gridDim;
class Barrier {
 std::mutex mutex;std::condition_variable cv;unsigned int arrived=0,generation=0;
public:
 void wait(){std::unique_lock<std::mutex> lock(mutex);auto saved=generation;
  if(++arrived==256){arrived=0;++generation;cv.notify_all();}
  else cv.wait(lock,[&]{return saved!=generation;});
 }
};
Barrier barrier;
#define __syncthreads() barrier.wait()
#include "'''+str(AFTER/'source/kernel.hpp')+r'''"
int main(int argc,char**argv){
 if(argc!=4)return 1;
 const unsigned int rows=std::stoul(argv[1]),inner=std::stoul(argv[2]),cols=std::stoul(argv[3]);
 gridDim={cols/16,rows/16};
 for(std::uint64_t seed: {UINT64_C(20261015),UINT64_C(20261016)}){
  std::vector<float>a(rows*inner),b(inner*cols),c(rows*cols,NAN);matrix_inputs::fill(a,b,seed);
  const auto pristine_a=a,pristine_b=b;
  const auto reference=matrix_reference::product(a,b,rows,inner,cols);
  // Independently recompute each scalar FP64 dot without row/column interning.
  for(unsigned int row=0;row<rows;++row)for(unsigned int col=0;col<cols;++col){
   double answer=0.;for(unsigned int k=0;k<inner;++k)answer+=double(a[row*inner+k])*double(b[k*cols+col]);
   if(answer!=reference[row*cols+col])return 2;
  }
  std::vector<std::thread> threads;
  for(unsigned int ty=0;ty<16;++ty)for(unsigned int tx=0;tx<16;++tx)threads.emplace_back([&,tx,ty]{
   threadIdx={tx,ty};
   for(unsigned int by=0;by<gridDim.y;++by)for(unsigned int bx=0;bx<gridDim.x;++bx){
    blockIdx={bx,by};matrix_multiplication_kernel<16>(a.data(),b.data(),c.data(),inner);barrier.wait();
   }
  });
  for(auto&thread:threads)thread.join();
  if(!matrix_inputs::unchanged(a,pristine_a)||!matrix_inputs::unchanged(b,pristine_b))return 3;
  if(!matrix_reference::validate(c,reference,cols).empty())return 4;
 }
}
''')
    compile_cpp(source,exe,'-pthread')
    completed=subprocess.run([str(exe),str(rows),str(inner),str(cols)],capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr


@pytest.mark.parametrize('alteration',['missing_inputs','duplicate_inputs','duplicate_validation','same_seed','same_A','same_B','unknown_generator','bad_seed','bad_hash','not_restored','replay_failed','wrong_mode','wrong_coverage'])
def test_actual_protocol_rejects_missing_ambiguous_or_incomplete_replay_evidence(cpu_drivers,alteration):
    result=execute(cpu_drivers['after'],'correct')
    assert result.returncode==0
    inputs_line=next(line for line in result.stdout.splitlines() if line.startswith('AKA_MATRIX_INPUTS '))
    valid_line=next(line for line in result.stdout.splitlines() if line.startswith('AKA_MATRIX_VALIDATION '))
    inp=json.loads(inputs_line.split(' ',1)[1]);valid=json.loads(valid_line.split(' ',1)[1]);lines=[]
    if alteration=='same_seed':inp['replay_seed']=inp['initial_seed']
    elif alteration=='same_A':inp['replay_A']=inp['initial_A']
    elif alteration=='same_B':inp['replay_B']=inp['initial_B']
    elif alteration=='unknown_generator':inp['generator']='fixed_index_formula'
    elif alteration=='bad_seed':inp['initial_seed']=str(2**64)
    elif alteration=='bad_hash':inp['replay_A']='not-a-fingerprint'
    elif alteration=='not_restored':valid['restored_initial_inputs']='FAIL'
    elif alteration=='replay_failed':valid['changed_input_replay_correctness']='FAIL'
    elif alteration=='wrong_mode':valid['mode']='check_only'
    elif alteration=='wrong_coverage':valid['output_coverage']='sampled_elements'
    valid['inputs']=inp
    if alteration!='missing_inputs':lines.append('AKA_MATRIX_INPUTS '+json.dumps(inp))
    lines.append('AKA_MATRIX_VALIDATION '+json.dumps(valid))
    if alteration=='duplicate_inputs':lines.append(lines[0])
    if alteration=='duplicate_validation':lines.append(lines[-1])
    helper=module(AFTER/'scripts/reference_checks.py','matrix_checks_negative')
    with pytest.raises(ValueError):helper.parse_validation('\n'.join(lines),mode='performance')


@pytest.mark.parametrize('role',['baseline','candidate'])
def test_actual_evaluator_dispatches_seven_correctness_but_five_performance_rows(monkeypatch,role,capsys,tmp_path):
    evaluator=module(AFTER/'scripts/evaluate.py','matrix_evaluator_dispatch')
    helper=module(AFTER/'scripts/reference_checks.py','matrix_checks_dispatch')
    workspace=tmp_path/'task';shutil.copytree(AFTER,workspace,ignore=shutil.ignore_patterns('build','__pycache__'))
    monkeypatch.setattr(evaluator,'ROOT',workspace)
    monkeypatch.chdir(tmp_path)
    data=json.loads((AFTER/'workload.json').read_text());calls=[]
    h=types.SimpleNamespace(TEST_SHAPES=data['test_shapes'],run_correctness=lambda:(True,None),
        run_performance=lambda:[{'test_case_id':row['test_case_id'],'params':row['params'],'execution_time_ms':.1,'benchmark_method':'cuda_graph'} for row in data['cases'][:5]])
    monkeypatch.setitem(sys.modules,'task_runner',h)
    monkeypatch.setitem(sys.modules,'reference_checks',helper)
    monkeypatch.setattr(helper,'check_additional_paths',lambda harness:calls.append('seven_dense_checks') or [{'test_case_id':r['test_case_id']} for r in data['cases']])
    for action,count in [('validate-task',7),('correctness',7),('performance',5)]:
        argv=[action] if action=='validate-task' else [role,action]
        assert evaluator.run(argv)==0
        line=capsys.readouterr().out.strip()
        result=json.loads(line.split('=',1)[1]);assert result['status']=='PASS' and len(result['cases'])==count
        assert [row['test_case_id'] for row in result['cases']]==[row['test_case_id'] for row in data['cases'][:count]]
    assert calls==['seven_dense_checks']
