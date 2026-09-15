"""Real SDK message shapes with a scripted transport; no provider/GPU calls."""
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

anyio = pytest.importorskip('anyio')
sdk_types = pytest.importorskip('claude_agent_sdk').types
from test_geak_schema_v2 import task_factory


@pytest.mark.parametrize('scenario', [
    'disk_only', 'failed_background', 'unrelated_tool', 'wrong_script', 'wrong_args',
    'assistant_json', 'truncated_background', 'tool_error', 'wrong_tool_identity',
    'synchronous', 'background', 'late_background', 'conflicting_disk',
])
def test_worker_requires_correlated_native_return(task_factory, monkeypatch, scenario):
    from agents.geak import engine_worker
    from agents.geak.bridge import write_json
    from agents.geak_v4 import workflow_runner as runner

    bridge = task_factory()
    script = str(bridge.root / 'engine.js')
    args = {'eval_dir': str(bridge.eval_dir), 'mode': 'optimize', 'target_language': 'hip'}
    bridge.job['engine'] = {'script_path': script, 'args': args}
    write_json(bridge.job_path, bridge.job)
    terminal = {'eval_dir': str(bridge.eval_dir), 'validation_status': 'accepted',
                'final_geomean': 1.0, 'final_patch': str(bridge.eval_dir / 'final_patch.diff'),
                'rounds': 1, 'budget_used': 2}
    inputs = {'scriptPath': script, 'args': copy.deepcopy(args)}
    if scenario == 'wrong_script':
        inputs['scriptPath'] = str(bridge.root / 'substitute.js')
    if scenario == 'wrong_args':
        inputs['args']['target_language'] = 'other'
    output = bridge.root / 'native-output.json'
    wrapper = {'result': terminal, 'workflowProgress': [{'model': 'fixture-model'}]}
    result_message = type('ResultMessage', (SimpleNamespace,), {})()

    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def query(self, prompt):
            pass
        async def receive_messages(self):
            disk = dict(terminal, budget_used=999) if scenario == 'conflicting_disk' else terminal
            write_json(bridge.eval_dir / 'workflow_return.json', disk)
            if scenario == 'disk_only':
                yield result_message
                return
            if scenario == 'assistant_json':
                yield sdk_types.AssistantMessage(content=[sdk_types.TextBlock(text=json.dumps(terminal))], model='fixture-model')
                yield result_message
                return
            tool_name = 'Bash' if scenario == 'unrelated_tool' else 'Workflow'
            yield sdk_types.AssistantMessage(content=[sdk_types.ToolUseBlock(id='tool1', name=tool_name, input=inputs)], model='fixture-model')
            if scenario in {'synchronous', 'unrelated_tool', 'tool_error', 'wrong_tool_identity'}:
                yield sdk_types.UserMessage(content=[sdk_types.ToolResultBlock(
                    tool_use_id='other-tool' if scenario == 'wrong_tool_identity' else 'tool1',
                    content=json.dumps(wrapper), is_error=scenario == 'tool_error')])
            else:
                yield sdk_types.TaskStartedMessage(subtype='task_started', data={}, task_id='task1',
                    description='fixture', uuid='fixture', session_id='fixture', tool_use_id='tool1', task_type='workflow')
                if scenario in {'truncated_background', 'late_background'}:
                    output.write_text('{"result":')
                else:
                    write_json(output, wrapper)
                yield sdk_types.TaskNotificationMessage(subtype='task_notification', data={}, task_id='task1',
                    status='failed' if scenario == 'failed_background' else 'completed',
                    output_file=str(output), summary='', uuid='fixture', session_id='fixture', tool_use_id='tool1')
                if scenario == 'late_background':
                    await anyio.sleep(0.15)
                    write_json(output, wrapper)
            yield result_message

    sdk = ModuleType('claude_agent_sdk')
    sdk.ClaudeAgentOptions = lambda **kwargs: kwargs
    sdk.ClaudeSDKClient = Client
    monkeypatch.setitem(sys.modules, 'claude_agent_sdk', sdk)
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda _: 'scripted-transport-real-sdk-types')
    monkeypatch.setattr(runner.os, 'geteuid', lambda: 1000)
    original = runner.invoke_via_sdk
    def bounded(prompt, **kwargs):
        kwargs.update(done_grace_seconds=0.2, done_poll_seconds=0.1)
        return original(prompt, **kwargs)
    monkeypatch.setattr(engine_worker, 'invoke_via_sdk', bounded)
    code = engine_worker.run(bridge.job_path)
    report = json.loads((bridge.root / 'engine_result.json').read_text())
    valid = scenario in {'synchronous', 'background', 'late_background', 'conflicting_disk'}
    assert code == (0 if valid else 1), report
    assert report['workflow_completed'] is valid
    if valid:
        assert report['budget_used'] == 2
        assert json.loads((bridge.eval_dir / 'workflow_return.json').read_text()) == terminal
