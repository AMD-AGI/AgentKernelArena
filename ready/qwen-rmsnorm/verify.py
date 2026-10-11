#!/usr/bin/env python3
"""CPU-only verification of the scoped starter's protected input pins."""
import hashlib,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
pins=json.loads((HERE/'INPUT-PINS.json').read_text())
ready=json.loads((HERE/'READY.json').read_text())
for name,row in {**pins['files'],**pins.get('ready_support_files',{})}.items():
 p=ROOT/name
 if row['kind']=='symlink':
  assert p.is_symlink() and str(p.readlink())==row['target'],name
 else:
  assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],name
assert ready['task_tree']==pins['task_tree'] and ready['qualified_commit']==pins['qualified_commit']
for name in ('entry_config','validator_config'):
 c=yaml.safe_load((ROOT/ready[name]).read_text())
 assert c['tasks']==ready['ready_tasks'] and c['target_gpu_model']=='MI355X'
meta=json.loads((ROOT/'tasks'/ready['ready_tasks'][0]/'ut/meta.json').read_text())
assert [(c['m'],c['n'],c['dtype'],c['eps']) for c in meta['workload']['cases']]==[(8192,8192,'torch.bfloat16',1e-6),(64,8192,'torch.bfloat16',1e-6)]
assert meta['tol']==0.02
print(json.dumps({'status':'SCOPED_INPUT_PINS_MATCH','qualified_files_verified':len(pins['files']),'ready_support_files_verified':len(pins.get('ready_support_files',{})),'task_tree':pins['task_tree'],'GPU_actions':False},indent=2))
