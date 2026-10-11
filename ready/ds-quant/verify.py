#!/usr/bin/env python3
"""Verify the qualified quant starter and its one-task configs without GPU work."""
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.tools.trusted_native_eval import validate_contract

pins=json.loads((HERE/'INPUT-PINS.json').read_text())
ready=json.loads((HERE/'READY.json').read_text())
for name,digest in pins['files'].items():
    path=ROOT/name
    if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Qualified input changed: '+name)
image,manifest=validate_contract(ROOT/pins['task'])
if image!=ready['runtime']['image'] or manifest['cases']!=ready['cases']:
    raise SystemExit('Qualified image/case contract differs')
for name in ('entry_config','validator_config'):
    config=yaml.safe_load((ROOT/ready[name]).read_text())
    if config['tasks']!=ready['ready_tasks'] or config['target_gpu_model']!='MI355X':
        raise SystemExit('One-task ready config changed: '+ready[name])
print(json.dumps({'status':'QUALIFIED_STARTER_PINS_MATCH','files_verified':len(pins['files']),
                  'task':pins['task'],'case_count':len(manifest['cases']),
                  'measurement_commit':pins['commit'],'GPU_actions':False},indent=2))
