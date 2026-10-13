"""Provider-specific read-only candidate review, in a fresh CLI session."""
from __future__ import annotations

import json
import os
import re

from agents.prompt_input import prompt_input
from agents.task_validator.launch_agent import _run_backend


def review(prompt, workspace, config, logger):
    backend = config['backend']
    if backend == 'claude_code':
        cmd = ['claude', '--print', '--verbose', '--output-format', 'stream-json',
               '--no-session-persistence', '--safe-mode', '--restricted',
               '--tools', 'Read,Glob,Grep', '--allowedTools', 'Read,Glob,Grep',
               '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
               '--permission-mode', 'dontAsk', '--input-format', 'text']
        if config.get('model'):
            cmd += ['--model', config['model']]
        if config.get('effort'):
            cmd += ['--effort', config['effort']]
    else:
        raise ValueError('Unsupported candidate verifier backend')
    env = dict(os.environ, CLAUDE_CODE_DISABLE_AUTO_MEMORY='1')
    with prompt_input(prompt) as stream:
        result = _run_backend(cmd, backend=backend, workspace=str(workspace),
            timeout_seconds=config['timeout_seconds'], logger=logger, env=env, stdin=stream)
    if result.returncode != 0 or result.timed_out or result.error:
        raise RuntimeError(result.error or 'Candidate reviewer failed or timed out')
    messages = []
    for line in result.output.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if backend == 'claude_code' and event.get('type') == 'result':
            messages.append(event.get('result', ''))
    if not messages:
        raise ValueError('Candidate reviewer returned no final report')
    payload = messages[-1].strip()
    # Some CLI versions wrap an otherwise valid final object in a Markdown
    # fence. Accept only a fence around the entire object, never arbitrary text
    # before/after it or a JSON fragment selected from a longer answer.
    fenced = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```', payload, re.DOTALL)
    if fenced:
        payload = fenced.group(1)
    return json.loads(payload)
