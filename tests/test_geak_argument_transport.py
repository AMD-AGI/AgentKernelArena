"""Execute pinned dispatcher and SDK message paths; no provider/GPU calls."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agents.geak.argument_transport import (decode_workflow_args, validate_args_transport,
                                           workflow_inputs_match)
from agents.geak.compatibility import adapt_workflow
from test_geak_schema_v2 import task_factory, upstream


@pytest.fixture
def dispatcher(upstream, tmp_path):
    path = tmp_path / 'kernel_workflow.js'
    path.write_text(adapt_workflow((upstream / 'kernel_workflow/kernel_workflow.js').read_text(),
                                  'Trusted exact contract'))
    pin = {'adapter_version': 3, 'adapted_workflow_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return path, pin


def _args():
    return {'kernel_path': '/fixture/kernel', 'workflow_dir': '/fixture/workflow',
            'mode': 'optimize', 'budget': 2, 'enabled': False,
            'nested': {'labels': ['雪', 'false', None, '"x":1,"x":2', '\\'], 'fraction': 0.25}}


BAD_JSON = [
    '{bad', 'null', '[]', '[{}]', 'true', '1', '"string"',
    '{"n": NaN}', '{"n": Infinity}', '{"n": -Infinity}', '{"n": 1e400}',
    '{"x": 1, "x": 1}', '{"nested":{"a": 1, "\\u0061": 1}}',
    '{"items":[{"same":0,"same":0}]}',
]


@pytest.mark.parametrize('raw', BAD_JSON)
def test_native_decoder_rejects_ambiguous_or_nonobject_json_before_child(dispatcher, tmp_path, raw):
    node = os.environ.get('GEAK_TEST_NODE') or shutil.which('node')
    if not node:
        pytest.skip('Node required for actual dispatcher probe')
    script, pin = dispatcher
    expected = {'scriptPath': str(script), 'args': _args()}
    assert not workflow_inputs_match({**expected, 'args': raw}, expected, args_transport=pin)
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'script': str(script), 'raw': raw}))
    probe = tmp_path / 'probe.js'
    probe.write_text(r'''
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const body = fs.readFileSync(input.script, 'utf8').replace(/^export const meta/m, 'const meta');
let calls = 0;
(async()=>{
  let rejected = false, errorName = null, errorMessage = null;
  try {
    await new Function('args','workflow','phase','log','return (async()=>{'+body+'})();')(
      input.raw, async()=>{ calls++; return {}; }, ()=>{}, ()=>{});
  } catch (error) { rejected = true; errorName = error.name; errorMessage = error.message; }
  console.log(JSON.stringify({rejected, calls, errorName, errorMessage}));
})().catch(()=>process.exit(1));
''')
    run = subprocess.run([node, str(probe), str(request)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result['rejected'] is True and result['calls'] == 0
    # Failure must come from JSON decoding/validation, not a later missing
    # kernel_path check (which would also happen with a broken/no decoder).
    assert result['errorName'] == 'SyntaxError' or result['errorMessage'].startswith('GEAK args')


@pytest.mark.parametrize('change', ['version2', 'version_bool', 'wrong_hash', 'missing_decoder', 'symlink'])
def test_opt_in_requires_actual_v3_dispatcher(dispatcher, change, tmp_path):
    script, pin = dispatcher
    expected = {'scriptPath': str(script), 'args': _args()}
    if change == 'version2': pin['adapter_version'] = 2
    if change == 'version_bool': pin['adapter_version'] = True
    if change == 'wrong_hash': pin['adapted_workflow_sha256'] = '0' * 64
    if change == 'missing_decoder':
        script.write_text('const A = args || {};')
        pin['adapted_workflow_sha256'] = hashlib.sha256(script.read_bytes()).hexdigest()
    if change == 'symlink':
        link = tmp_path / 'alias.js'
        link.symlink_to(script)
        expected['scriptPath'] = str(link)
    with pytest.raises(ValueError):
        validate_args_transport(expected, pin)
    # Rejection happens before SDK import/client/query, not merely at comparison.
    from agents.geak_v4.workflow_runner import invoke_via_sdk
    with pytest.raises(ValueError):
        invoke_via_sdk('unused', workflow_dir=tmp_path, eval_dir=tmp_path,
                       model='unused', effort='medium', settings='{}', cli_path='unused',
                       timeout_seconds=1, done_grace_seconds=1, done_poll_seconds=.1,
                       expected_workflow=expected, workflow_args_transport=pin)


@pytest.mark.parametrize('scenario', [
    'object', 'string', 'reordered', 'numeric_equivalent', 'default_strict',
    'invalid_json', 'array', 'null', 'scalar', 'nan', 'duplicate_keys', 'nested_duplicate',
    'changed_value', 'bool_number', 'string_number', 'missing_arg', 'extra_arg',
    'unsafe_integer_object', 'unsafe_integer_string',
    'wrong_script', 'extra_tool_key', 'wrong_return_id', 'second_call', 'tool_error',
])
def test_sdk_v3_semantics_raw_evidence_and_native_correlation(task_factory, dispatcher, monkeypatch, scenario):
    sdk_types = pytest.importorskip('claude_agent_sdk').types
    from agents.geak import engine_worker
    from agents.geak.bridge import write_json
    from agents.geak_v4 import workflow_runner as runner

    bridge = task_factory()
    script, pin = dispatcher
    args = {**_args(), 'eval_dir': str(bridge.eval_dir), 'target_language': 'hip'}
    bridge.job['engine'] = {'script_path': str(script), 'args': args}
    if scenario != 'default_strict':
        bridge.job['engine']['args_transport'] = pin
    write_json(bridge.job_path, bridge.job)
    actual = copy.deepcopy(args)
    if scenario == 'numeric_equivalent': actual['budget'] = 2.0
    if scenario == 'changed_value': actual['nested']['fraction'] = .5
    if scenario == 'bool_number': actual['enabled'] = 0
    if scenario == 'string_number': actual['budget'] = '2'
    if scenario == 'missing_arg': actual.pop('budget')
    if scenario == 'extra_arg': actual['unrequested'] = 1
    if scenario.startswith('unsafe_integer_'): actual['nested']['fraction'] = 9007199254740993
    object_input = scenario in {'object', 'unsafe_integer_object'}
    raw = actual if object_input else json.dumps(actual, sort_keys=scenario == 'reordered', indent=1)
    raw = {'invalid_json': '{no', 'array': '[]', 'null': 'null', 'scalar': '2',
           'nan': json.dumps({**actual, 'budget': float('nan')}),
           'duplicate_keys': json.dumps(actual)[:-1] + ',"budget":2}',
           'nested_duplicate': json.dumps(actual).replace('"fraction": 0.25', '"fraction":0.25,"fraction":0.25')}.get(scenario, raw)
    inputs = {'scriptPath': str(script), 'args': raw}
    if scenario == 'wrong_script': inputs['scriptPath'] += '.other'
    if scenario == 'extra_tool_key': inputs['run_in_background'] = False
    original = copy.deepcopy(inputs)
    terminal = {'eval_dir': str(bridge.eval_dir), 'validation_status': 'accepted',
                'final_geomean': 1., 'final_patch': str(bridge.eval_dir / 'final_patch.diff'),
                'rounds': 1, 'budget_used': 2}

    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def query(self, prompt): pass
        async def receive_messages(self):
            yield sdk_types.AssistantMessage(content=[sdk_types.ToolUseBlock(
                id='tool1', name='Workflow', input=inputs)], model='fixture')
            if scenario == 'second_call':
                yield sdk_types.AssistantMessage(content=[sdk_types.ToolUseBlock(
                    id='tool2', name='Workflow', input=inputs)], model='fixture')
            yield sdk_types.UserMessage(content=[sdk_types.ToolResultBlock(
                tool_use_id='wrong' if scenario == 'wrong_return_id' else 'tool1',
                content=json.dumps({'result': terminal}), is_error=scenario == 'tool_error')])
            yield type('ResultMessage', (SimpleNamespace,), {})()

    sdk = ModuleType('claude_agent_sdk')
    from claude_agent_sdk import ClaudeAgentOptions
    sdk.ClaudeAgentOptions = ClaudeAgentOptions
    sdk.ClaudeSDKClient = Client
    monkeypatch.setitem(sys.modules, 'claude_agent_sdk', sdk)
    monkeypatch.setattr(runner.os, 'geteuid', lambda: 1000)
    invoke = runner.invoke_via_sdk
    def bounded(prompt, **kwargs):
        return invoke(prompt, **{**kwargs, 'done_grace_seconds': .1, 'done_poll_seconds': .05})
    monkeypatch.setattr(engine_worker, 'invoke_via_sdk', bounded)
    code = engine_worker.run(bridge.job_path)
    good = scenario in {'object', 'string', 'reordered', 'numeric_equivalent'}
    assert code == (0 if good else 1)
    report = json.loads((bridge.root / 'engine_result.json').read_text())
    assert report['workflow_completed'] is good
    assert inputs == original  # Raw collector/SDK input is never rewritten.
    diag = report['runtime']['sdk_diagnostics']
    call = diag['calls'][0]
    assert call['args_encoding'] == ('object' if object_input else 'json_string')
    assert call['raw_args_sha256'] == hashlib.sha256(json.dumps(
        original['args'], ensure_ascii=True, sort_keys=True).encode()).hexdigest()
    assert call['args_comparison'] == ('strict' if scenario == 'default_strict' else 'geak_dispatch_v3')
    if good:
        assert call['normalized_args_type'] == 'object'
        assert diag['matched_workflow_calls'] == 1
    elif scenario not in {'wrong_return_id', 'second_call', 'tool_error'}:
        assert diag['matched_workflow_calls'] == 0
    assert '/fixture/kernel' not in json.dumps(diag)


def test_json_comparison_preserves_types_and_full_keys():
    expected = {'scriptPath': '/fixture', 'args': {'n': 1, 'b': False}}
    assert workflow_inputs_match({**expected, 'args': {'n': 1.0, 'b': False}}, expected)
    for actual in ({'n': True, 'b': False}, {'n': 1, 'b': 0}, {'n': '1', 'b': False},
                   {'n': 1}, {'n': 1, 'b': False, 'x': 1}):
        assert not workflow_inputs_match({**expected, 'args': actual}, expected)
    assert not workflow_inputs_match({**expected, 'run_in_background': False}, expected)
    assert not workflow_inputs_match({**expected, 'args': json.dumps(expected['args'])}, expected)
    for value in [None, [], 1, False, {'n': float('nan')}, {'n': float('inf')}]:
        with pytest.raises((ValueError, TypeError)):
            decode_workflow_args(value)


def test_transport_sources_remain_python310_compatible():
    root = Path(__file__).resolve().parents[1]
    for relative in ('agents/geak/argument_transport.py', 'agents/geak/compatibility.py',
                     'agents/geak/engine_worker.py', 'agents/geak_v4/workflow_runner.py'):
        ast.parse((root / relative).read_text(), feature_version=(3, 10))


@pytest.mark.parametrize('left,right,equal', [
    (9007199254740992, 9007199254740993, False),
    (9007199254740993, 9007199254740992.0, False),
    (9007199254740992, 9007199254740992.0, True),
    (9007199254740993, 9007199254740993, True),
    (9007199254740991, 9007199254740991.0, True),
    (-9007199254740992, -9007199254740993, False),
    (10**400, 10**400 + 1, False),
    (10**400, 10**400, True),
])
def test_strict_numeric_comparison_does_not_round_large_integers(left, right, equal):
    first = {'scriptPath': '/fixture', 'args': {'nested': [{'n': left}]}}
    second = {'scriptPath': '/fixture', 'args': {'nested': [{'n': right}]}}
    assert workflow_inputs_match(first, second) is equal
    assert workflow_inputs_match(second, first) is equal


@pytest.mark.parametrize('value', [9007199254740991, -9007199254740991,
                                   9007199254740992, 9007199254740993,
                                   -9007199254740992, -9007199254740993,
                                   9007199254740992.0, 1e20])
@pytest.mark.parametrize('encoded', [False, True])
def test_v3_python_and_actual_js_enforce_same_safe_integer_range(dispatcher, tmp_path, value, encoded):
    node = os.environ.get('GEAK_TEST_NODE') or shutil.which('node')
    if not node:
        pytest.skip('Node required for actual dispatcher probe')
    script, pin = dispatcher
    args = {**_args(), 'nested': {'values': [value]}}
    actual = json.dumps(args) if encoded else args
    safe = abs(value) <= 9007199254740991
    expected = {'scriptPath': str(script), 'args': args}
    if safe:
        assert decode_workflow_args(actual) == args
        validate_args_transport(expected, pin)
        assert workflow_inputs_match({**expected, 'args': actual}, expected, args_transport=pin)
    else:
        with pytest.raises(ValueError, match='JavaScript safe integer range'):
            decode_workflow_args(actual)
        # Prepared unsafe arguments are rejected before an SDK launch too.
        with pytest.raises(ValueError, match='JavaScript safe integer range'):
            validate_args_transport(expected, pin)
        valid_expected = {**expected, 'args': _args()}
        assert not workflow_inputs_match({**expected, 'args': actual}, valid_expected, args_transport=pin)
    request = tmp_path / 'integer-request.json'
    request.write_text(json.dumps({'script': str(script), 'args': actual}))
    probe = tmp_path / 'integer-probe.js'
    probe.write_text(r"""
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const body = fs.readFileSync(input.script, 'utf8').replace(/^export const meta/m, 'const meta');
let calls = [], errorMessage = null;
(async()=>{
  try {
    await new Function('args','workflow','phase','log','return (async()=>{'+body+'})();')(
      input.args, async(ref,args)=>{ calls.push(args); return {}; }, ()=>{}, ()=>{});
  } catch (error) { errorMessage = error.message; }
  console.log(JSON.stringify({calls, errorMessage}));
})().catch(()=>process.exit(1));
""")
    run = subprocess.run([node, str(probe), str(request)], capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    if safe:
        assert result['errorMessage'] is None
        assert len(result['calls']) == 1
        assert result['calls'][0]['nested'] == args['nested']
    else:
        assert result == {'calls': [], 'errorMessage': 'GEAK args integer exceeds JavaScript safe integer range'}
