"""Task-local v2 protocol over the protected colocated pytest harness."""
from __future__ import annotations
import ast
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def serial(value):
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,(bool,int,float,str)) or value is None:return value
    return str(value)


def identity(function,params):
    content=json.dumps(serial(params),sort_keys=True,separators=(',',':'))
    return function+'/'+hashlib.sha256(content.encode()).hexdigest()[:20]


def inspect_candidate(data, *, require_implemented=False):
    path=ROOT/data['source']
    if not path.resolve().is_relative_to(ROOT):raise ValueError('Candidate escaped workspace')
    tree=ast.parse(path.read_text(),filename=data['source'])
    defs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    states=[]
    for name in data['kernel_symbols']:
        node=defs.get(name)
        body=[] if node is None else [n for n in node.body if not (
            isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        empty=not body or all(isinstance(n,ast.Pass) for n in body) or (len(body)==1 and isinstance(body[0],ast.Raise))
        states.append(not empty)
        if node is not None and not any(ast.unparse(d).endswith('.jit') for d in node.decorator_list):
            raise ValueError(f'{name} must be a Triton JIT kernel')
    if not states or (any(states) and not all(states)):raise ValueError('Missing or partially implemented candidate')
    state='implemented' if all(states) else 'unimplemented'
    if require_implemented and state!='implemented':raise ValueError('Unimplemented candidate; baseline fallback forbidden')
    return state


def benchmark_type(base, plugin, module):
    from _arena_reference import prepare
    class CheckedBenchmark(base):
        def __init__(self,*args,**kwargs):
            # Inputs are task-owned locals prepared by the original performance
            # function. No compiler, RNG, warmup or repetition argument changes.
            self.context=dict(inspect.currentframe().f_back.f_locals)
            super().__init__(*args,**kwargs)

        def run_benchmark(self,*args,**kwargs):
            row=plugin.current_row
            check=prepare(self.context,module)
            if self.prepare_fn is not None:self.prepare_fn()
            original=self.op_callable
            output=original()
            check(output)
            if plugin.action=='correctness':
                row['metrics']={'performance_inputs_checked':True}
                plugin.exercised.add(row['test_case_id'])
                return {}
            observed=[output]
            def observed_op():
                value=original()
                observed[0]=value
                return value
            self.op_callable=observed_op
            try:
                # The common session owns the independent baseline. The old
                # helper's optional peer/reference timing is not that baseline.
                kwargs['baseline_callable']=None
                record=super().run_benchmark(*args,**kwargs)
            finally:
                self.op_callable=original
            # For graph capture, this aliases the last captured output buffers;
            # for event timing it is the output of the last measured invocation.
            # Do not rerun a separate candidate and label it timed evidence.
            check(observed[0])
            ms=record['timing_ms']['mean'];method=record.get('benchmark_method')
            if not isinstance(ms,(float,int)) or not math.isfinite(ms) or ms<=0:
                raise RuntimeError('Nonpositive/nonfinite device timing')
            if method not in ('cuda_graph','cuda_event_fallback'):
                raise RuntimeError('Missing device timing method')
            row.update(execution_time_ms=ms,benchmark_method=method,
                       metadata={'timing_stats':record['timing_ms'],'timed_output_checked':True})
            plugin.exercised.add(row['test_case_id'])
            return record
    return CheckedBenchmark


class ReportPlugin:
    def __init__(self,data,action):
        self.data=data;self.action=action;self.expected={r['test_case_id']:r for r in data['cases']}
        self.rows={key:deepcopy(row) for key,row in self.expected.items()
                   if action=='validate-task' or action in row['checks']}
        self.collection_error=None;self.node_rows={};self.current_row=None;self.exercised=set()

    def pytest_collection_modifyitems(self,session,config,items):
        found={};kept=[]
        for item in items:
            name=getattr(item,'originalname',None) or item.name.split('[')[0]
            if name.startswith('test_save'):continue
            params=serial(getattr(getattr(item,'callspec',None),'params',{}))
            key=identity(name,params)
            if key in found:raise ValueError('Duplicate original pytest case identity')
            found[key]={'function':name,'arguments':params}
            if key not in self.rows:continue
            self.node_rows[item.nodeid]=key;kept.append(item)
            if name=='test_performance' and self.action!='validate-task':
                module=item.module
                if not getattr(module,'_arena_bench_installed',False):
                    module.PytestBenchmarker=benchmark_type(module.PytestBenchmarker,self,module)
                    module._arena_bench_installed=True
        if set(found)!=set(self.expected) or any(found[k]!=self.expected[k]['params'] for k in found):
            raise ValueError('Collected pytest inputs differ from the independent protected manifest')
        items[:]=kept

    def pytest_collectreport(self,report):
        if report.failed:self.collection_error=str(report.longrepr)

    def pytest_runtest_setup(self,item):
        self.current_row=self.rows[self.node_rows[item.nodeid]]

    def pytest_runtest_logreport(self,report):
        key=self.node_rows.get(report.nodeid)
        if key is None:return
        row=self.rows[key]
        if report.failed or report.skipped:
            row.update(status='FAIL',reason=str(report.longrepr),failure_kind='test_failure' if report.failed else 'not_executed')
        elif report.when=='call':
            if row['params']['function']=='test_performance' and key not in self.exercised:
                row.update(status='FAIL',reason='Performance case did not run its numerical/timing check',failure_kind='not_executed')
            elif not row.get('failure_kind'):
                row['status']='PASS'
                row.pop('reason',None)
                row.setdefault('metrics',{})['original_pytest_passed']=True


def evaluate(role,action):
    data=json.loads((ROOT/'workloads.json').read_text())
    result={'protocol':'arena-eval-v1','role':role,'action':action,'status':'PASS','cases':[]}
    try:
        state=inspect_candidate(data,require_implemented=action!='validate-task')
        if action=='compile':
            # Match the original syntax-compilation gate. Correctness executes
            # all JIT specializations and cannot pass from this structural check.
            compile((ROOT/data['source']).read_text(),data['source'],'exec')
            return result
        import pytest
        plugin=ReportPlugin(data,action)
        result['cases']=list(plugin.rows.values())
        for row in result['cases']:
            if action!='validate-task':row['status']='FAIL';row['reason']='Case was not executed'
        output=io.StringIO()
        args=[data['source'],'-q','-p','no:cacheprovider','--disable-warnings']
        if action=='validate-task':args.append('--collect-only')
        with redirect_stdout(output),redirect_stderr(output):
            code=int(pytest.main(args,plugins=[plugin]))
        if action=='validate-task':result['metadata']={'candidate_state':state,'manifest_collected':True}
        if code or plugin.collection_error:
            raise RuntimeError(plugin.collection_error or f'pytest exited {code}: '+output.getvalue()[-3000:])
        failures=[r for r in result['cases'] if r['status']!='PASS']
        if failures:result.update(status='FAIL',reason=f'{len(failures)} cases failed or skipped',failure_kind='case_failure')
    except BaseException as exc:
        result.update(status='FAIL',reason=f'{type(exc).__name__}: {exc}',failure_kind='execution_failure')
        if not result['cases'] and action!='compile':
            result['cases']=[deepcopy(r) for r in data['cases'] if action=='validate-task' or action in r['checks']]
            for row in result['cases']:row.update(status='FAIL',reason='Case was not executed')
        for row in result['cases']:
            if action=='validate-task':row['status']='FAIL'
    return result


def main():
    os.chdir(ROOT);os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    args=sys.argv[1:]
    role,action=('task','validate-task') if args==['validate-task'] else (tuple(args) if len(args)==2 else ('task','validate-task'))
    try:
        if args!=['validate-task'] and not (len(args)==2 and role in ('baseline','candidate') and action in ('compile','correctness','performance')):
            raise ValueError('Use validate-task or baseline|candidate compile|correctness|performance')
        result=evaluate(role,action)
    except BaseException as exc:
        result={'protocol':'arena-eval-v1','role':role,'action':action,'status':'FAIL','cases':[],
                'reason':f'{type(exc).__name__}: {exc}','failure_kind':'execution_failure'}
    print('ARENA_EVAL_RESULT='+json.dumps(result,allow_nan=False,separators=(',',':')))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
