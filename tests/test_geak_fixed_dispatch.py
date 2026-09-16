"""Fixed per-run dispatcher arguments: real native JS and scripted SDK, no GPU."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agents.geak.argument_transport import validate_args_transport, workflow_inputs_match
from agents.geak.compatibility import adapt_workflow, prepare_engine
from agents.geak_v4.workflow_runner import build_prompt
from test_geak_schema_v2 import task_factory, upstream


def run_js(tmp_path, scripts):
    node = os.environ.get('GEAK_TEST_NODE') or shutil.which('node')
    if not node:
        pytest.skip('Node required for pinned native dispatcher')
    request = tmp_path/'request.json'
    request.write_text(json.dumps(scripts))
    probe = tmp_path/'probe.js'
    probe.write_text(r'''
const fs = require('fs');
const inputs = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
async function run(input) {
  let calls=[], error=null, result=null;
  const before=JSON.stringify(input.args);
  const body=fs.readFileSync(input.script,'utf8').replace(/^export const meta/m,'const meta');
  try {
    result=await new Function('args','workflow','phase','log','return (async()=>{'+body+'})();')(
      input.args, async(ref,args)=>{ calls.push({ref,args}); return {childSentinel:true}; }, ()=>{}, ()=>{});
  } catch(e) { error=e.message; }
  if(JSON.stringify(input.args)!==before) throw Error('input mutated');
  return {calls,result,error};
}
Promise.all(inputs.map(run)).then(x=>console.log(JSON.stringify(x))).catch(e=>{console.error(e);process.exit(1)});
''')
    result = subprocess.run([node,str(probe),str(request)],capture_output=True,text=True,timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize('mode', ['author','optimize'])
def test_fixed_dispatch_preserves_every_parent_child_argument(upstream,tmp_path,mode):
    source=(upstream/'kernel_workflow/kernel_workflow.js').read_text()
    contract='Full contract: "quotes", \\ slash, ${notCode}, `backticks`, 雪\n' * 40
    args={'kernel_path':'/private/kernel','workflow_dir':'/private/workflow',
          'kernel_lane_script':'/private/workflow/kernel_lane.js','mode':mode,
          'budget':6,'deadline_epoch':1789600000,'agent_timeout_ms':3600000,
          'arena_benchmark':{'baseline_per_case':[{'name':'case1','ms':0.125}, {'name':'case2','ms':2.0}]},
          'arena_setup':{'source_files':['nested/kernel.py','helpers.py'],'baseline_frozen':True},
          'nested':{'__proto__':{'field':'data'},'values':[None,False,1,'1','"quoted"', '雪']}}
    saved=copy.deepcopy(args)
    old=tmp_path/'parent.js';new=tmp_path/'fixed.js'
    old.write_text(adapt_workflow(source,contract))
    new.write_text(adapt_workflow(source,contract,trusted_args=args))
    result=run_js(tmp_path,[{'script':str(old),'args':args},{'script':str(new),'args':{}}])
    assert args==saved
    assert result[0]==result[1]
    assert result[1]['error'] is None
    assert result[1]['calls']==[{'ref':{'scriptPath':args['kernel_lane_script']},
                                'args':{**args,'arena_contract':contract,'task':contract}}]
    assert result[1]['result']=={'childSentinel':True}


@pytest.mark.parametrize('bad', ['{}','{"budget":6}',None,[],True,{'budget':6}, {'__proto__':{}}])
def test_fixed_dispatch_rejects_any_nonempty_or_encoded_input(upstream,tmp_path,bad):
    path=tmp_path/'fixed.js'
    path.write_text(adapt_workflow((upstream/'kernel_workflow/kernel_workflow.js').read_text(),
        'contract',trusted_args={'kernel_path':'/k','workflow_dir':'/w','mode':'optimize'}))
    pin={'adapter_version':4,'adapted_workflow_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    expected={'scriptPath':str(path),'args':{}}
    assert not workflow_inputs_match({**expected,'args':bad},expected,args_transport=pin)
    [result]=run_js(tmp_path,[{'script':str(path),'args':bad}])
    assert result=={'calls':[],'result':None,'error':'GEAK dispatcher v4 requires empty object args'}
    with pytest.raises(ValueError):
        validate_args_transport({**expected,'args':{'budget':6}},pin)


@pytest.mark.parametrize('scenario',['accepted','encoded_empty','extra_arg','extra_outer','wrong_script','wrong_return_id'])
def test_prepared_v4_worker_uses_empty_args_and_preserves_native_return_rules(
        task_factory,upstream,tmp_path,monkeypatch,scenario):
    sdk_types=pytest.importorskip('claude_agent_sdk').types
    from claude_agent_sdk import ClaudeAgentOptions
    from agents.geak import engine_worker
    from agents.geak.bridge import write_json
    from agents.geak_v4 import workflow_runner as runner
    bridge=task_factory()
    engine=prepare_engine(upstream,bridge,python=sys.executable,options=bridge.job['options'])
    bridge.job['engine']=engine;write_json(bridge.job_path,bridge.job)
    expected={'scriptPath':engine['script_path'],'args':{}}
    inputs=copy.deepcopy(expected)
    if scenario=='encoded_empty':inputs['args']='{}'
    if scenario=='extra_arg':inputs['args']={'budget':6}
    if scenario=='extra_outer':inputs['run_in_background']=False
    if scenario=='wrong_script':inputs['scriptPath']+='.other'
    original=copy.deepcopy(inputs)
    terminal={'eval_dir':str(bridge.eval_dir),'validation_status':'accepted','final_geomean':1.,
              'final_patch':str(bridge.eval_dir/'final_patch.diff'),'rounds':1,'budget_used':2}
    observed=[]
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def query(self,prompt):
            encoded=prompt.split('```json\n',1)[1].split('\n```',1)[0]
            assert json.loads(encoded)==expected
            assert 'arena_benchmark' not in prompt and 'source_files' not in prompt
            observed.append(len(encoded))
        async def receive_messages(self):
            yield sdk_types.AssistantMessage(content=[sdk_types.ToolUseBlock(id='one',name='Workflow',input=inputs)],model='fixture')
            yield sdk_types.UserMessage(content=[sdk_types.ToolResultBlock(tool_use_id='wrong' if scenario=='wrong_return_id' else 'one',content=json.dumps({'result':terminal}),is_error=False)])
            yield type('ResultMessage',(SimpleNamespace,),{})()
    sdk=ModuleType('claude_agent_sdk');sdk.ClaudeAgentOptions=ClaudeAgentOptions;sdk.ClaudeSDKClient=Client
    monkeypatch.setitem(sys.modules,'claude_agent_sdk',sdk)
    monkeypatch.setattr(runner.os,'geteuid',lambda:1000)
    invoke=runner.invoke_via_sdk
    monkeypatch.setattr(engine_worker,'invoke_via_sdk',lambda prompt,**kw:invoke(prompt,**{**kw,'done_grace_seconds':.1,'done_poll_seconds':.05}))
    code=engine_worker.run(bridge.job_path)
    assert code==(0 if scenario=='accepted' else 1)
    report=json.loads((bridge.root/'engine_result.json').read_text())
    assert report['workflow_completed'] is (scenario=='accepted')
    if scenario=='accepted':
        assert report['mode']==engine['engine_args']['mode']
        assert report['target_language']==engine['engine_args']['target_language']
    assert observed and inputs==original
    diag=report['runtime']['sdk_diagnostics']['calls'][0]
    assert diag['args_comparison']=='geak_dispatch_v4'
    assert diag['normalized_args_type']==('invalid' if scenario in {'encoded_empty','extra_arg'} else 'object')


def test_default_prompt_invocation_is_unchanged(tmp_path):
    args={'eval_dir':'/eval','apply_to_original':'false','budget':6}
    p=tmp_path/'script.js'
    assert build_prompt(p,args)==build_prompt(p,args,invocation_args=args)
    fixed=build_prompt(p,args,invocation_args={})
    assert json.loads(fixed.split('```json\n',1)[1].split('\n```',1)[0])=={'scriptPath':str(p),'args':{}}
    assert '/eval/workflow_return.json' in fixed


def test_fixed_dispatch_pin_and_version_must_match_actual_callee(upstream,tmp_path):
    source=(upstream/'kernel_workflow/kernel_workflow.js').read_text()
    path=tmp_path/'fixed.js'
    path.write_text(adapt_workflow(source,'contract',trusted_args={'mode':'author'}))
    pin={'adapter_version':4,'adapted_workflow_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    expected={'scriptPath':str(path),'args':{}}
    validate_args_transport(expected,pin)
    with pytest.raises(ValueError,match='transport/hash mismatch'):
        validate_args_transport(expected,{**pin,'adapter_version':3})
    path.write_text(adapt_workflow(source,'contract'))
    with pytest.raises(ValueError,match='transport/hash mismatch'):
        validate_args_transport(expected,{**pin,'adapted_workflow_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match='must be an object'):
        adapt_workflow(source,'contract',trusted_args='{}')
