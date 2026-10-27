"""Shared standalone runner for materialized Magpie workloads."""
from __future__ import annotations

import json
from pathlib import Path

import _aka_serving as serving

ROOT = Path(__file__).resolve().parents[1]


def configure():
    lock = json.loads((ROOT/'runtime.lock.json').read_text())
    serving.PRECISIONS.update(fp4='float4', mxfp4='mxfp4', mxfp8='mxfp8', w4a16='w4a16')
    if lock['framework'] != 'sglang':
        from _aka_serving_backends import completion_request
        send = serving.request
        serving.request = lambda port, path, value=None, timeout=120: completion_request(
            send, ROOT, port, path, value, timeout=timeout, require_prompt_ids=lock['framework'] == 'atom')
    if lock['scenario'] != 'text':
        import _aka_serving_scenarios as scenarios
        serving.identity = lambda config: scenarios.manifest_identity(config, lock, serving.PRECISIONS[config['precision']])
        if lock['scenario'] == 'agentx':
            serving.model_check_plan = scenarios.check_plan
        else:
            serving.model_inputs = lambda _: [0]  # media_plan owns the actual frozen image/text inputs.
            serving.model_check_plan = lambda config, lock, prompts: scenarios.media_plan(ROOT, lock)
            serving.generation_checks = lambda port, plan, request_fn, return_logprob=True: scenarios.media_generations(
                ROOT, port, plan, request_fn, return_logprob=return_logprob)
    return lock


if __name__ == '__main__':
    configure()
    raise SystemExit(serving.main())
