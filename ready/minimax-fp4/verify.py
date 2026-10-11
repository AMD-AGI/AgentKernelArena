#!/usr/bin/env python3
"""CPU-only check of the scoped FP4 starter and its frozen inputs."""
import hashlib
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
pins=json.loads((HERE/'INPUT-PINS.json').read_text());ready=json.loads((HERE/'READY.json').read_text())
for name,expected in {**pins['files'],**pins.get('post_evaluation_files',{})}.items():
    path=ROOT/name
    if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise SystemExit('Pinned input differs: '+name)
assert ready['qualified_commit']==pins['qualified_commit'] and ready['task_tree']==pins['task_tree']
for name in ('entry_config','validator_config'):
    config=yaml.safe_load((ROOT/ready[name]).read_text())
    assert config['tasks']==ready['ready_tasks'] and config['target_gpu_model']=='MI355X'
manifest=json.loads((ROOT/ready['fixtures']['manifest']).read_text())
assert len(manifest['assets'])==74 and sum(a['bytes'] for a in manifest['assets'])==2044395146
assert manifest['oci_prefix']==ready['fixtures']['oci_prefix']
cases=json.loads((ROOT/('tasks/'+ready['ready_tasks'][0])/'cases.json').read_text())
assert [c['case_id'] for c in cases['cases']]==ready['case_ids'] and len(ready['case_ids'])==14
assert cases['measurement']==ready['measurement']
print(json.dumps({'status':'SCOPED_STARTER_PINS_MATCH','qualified_commit':ready['qualified_commit'],
    'task':ready['ready_tasks'][0],'files_verified':len(pins['files']),'GPU_actions':False,
    'qualification_status':ready['qualification']['status']},indent=2))
