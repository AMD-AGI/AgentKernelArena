#!/usr/bin/env python3
"""Diagnostic native MoE default/alias probe; emits no scoreable timings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _moe_runtime import NativeRuntime, ReplayState
from _routing_contract import fingerprint, require, source_context, specs, strict_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    task = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    require(out.is_relative_to(task / 'build'), 'probe reports belong under task build/')
    require(not out.exists(), 'probe output must be a new file')
    meta = strict_json((task / 'ut/meta.json').read_text())
    source = source_context(task, meta)
    payload = {'schema': 'qwen-moe-native-route-probe-v1', 'status': 'incomplete',
               'qualification': False, 'scoreable': False, 'source': source,
               'runtime_image': meta['source_provenance']['runtime_image'], 'cases': []}
    exit_code = 0
    try:
        import torch
        runtime = NativeRuntime(task, torch, meta)
        for spec in specs(meta):
            observed = {}
            default = None
            for route in ('captured', 'matching_explicit_default'):
                if route != 'captured':
                    if default not in (32, 64, 128):
                        break
                    route = 'block' + str(default)
                state = ReplayState(runtime, spec, route, secrets.randbits(63))
                state.use_captured = state.check_captured = True
                state.prepare()
                state.run()
                state.validate()
                default = state.resolved_execution['effective_block_size_M']
                observed[route] = {'native_dispatch': state.dispatch,
                                   'resolved_execution': state.resolved_execution,
                                   'resolved_execution_sha256': fingerprint(state.resolved_execution),
                                   'captured_and_native_reference_checked': True,
                                   'max_rel_err': state.worst_error, 'tolerance': state.tol}
            if len(observed) == 2:
                require(len({item['resolved_execution_sha256'] for item in observed.values()}) == 1,
                        'matching explicit hint differs from the observed captured native execution')
            payload['cases'].append({'sig': spec['sig'], 'params': state.params,
                                     'observed_default_block_size_M': default,
                                     'observed_aliases': list(observed), 'routes': observed})
        payload['status'] = 'ok'
    except Exception as exc:
        payload.update(status='fail', error=f'{type(exc).__name__}: {exc}')
        exit_code = 1
    require(source_context(task, meta) == source, 'source changed during native route probe')
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + '.tmp-' + secrets.token_hex(8))
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(out)
    print(json.dumps({'status': payload['status'], 'qualification': False, 'scoreable': False,
                      'cases': [{'sig': row['sig'], 'observed_default_block_size_M': row['observed_default_block_size_M'],
                                 'observed_aliases': row['observed_aliases']} for row in payload['cases']]}, indent=2))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
