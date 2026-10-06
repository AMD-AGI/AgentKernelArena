"""Separate task-local native modules; only the Triton GPU body is editable."""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bindings(root):
    if importlib.metadata.version('sglang')!='0.5.20':raise RuntimeError('Pinned SGLang 0.5.20 is required')
    import sglang
    cfg=json.loads((root/'provenance/SOURCE.json').read_text())
    for relative,expected in cfg['native_sources'].items():
        if digest(Path(sglang.__file__).parent/relative)!=expected:raise RuntimeError('Native source differs: '+relative)
    importlib.import_module('sglang.kernels.ops.kimi_k3')
    result={}
    for leg,relative in [('candidate','source/attn_res_hip.py'),('reference','ut/reference/attn_res_hip.py')]:
        path=root/relative;name='sglang.kernels.ops.kimi_k3._aka_aggregation_'+leg+'_'+digest(path)[:20]
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module);result[leg]=module.attn_res_hip
    return result
