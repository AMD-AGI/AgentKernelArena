"""Submission and installation checks; these do not substitute for GPU qualification."""
import importlib
import json
from pathlib import Path
import sys
import types

import pytest

from src.tools.perf import serving_bundle as bundles


def bundle(tmp_path, *, symbol='Layer.forward'):
    root = tmp_path/'source'
    root.mkdir()
    (root/'helper.py').write_text('def twice(x): return x * 2\n')
    (root/'kernel.py').write_text('from .helper import twice\ndef forward(self, x): return twice(x) + 1\n')
    patch = dict(module='fixture_model', symbol=symbol, file='kernel.py', callable='forward')
    (root/'patches.json').write_text(json.dumps(dict(version=1, patches=[patch])))
    lock = dict(gpu_count=2, server_seed=7, limits=dict(max_candidate_source_bytes=4096),
                candidate_bundle=dict(root='source', manifest='patches.json',
                    module_prefixes=['fixture_model'], observer=dict(module='fixture_model',symbol='Model.forward')))
    return bundles.Bundle(tmp_path, lock)


def test_multifile_source_and_manifest_are_hashed(tmp_path):
    b = bundle(tmp_path)
    b.check_sources()
    assert b.function(b.patches[0])(None, 3) == 7
    (b.root/'helper.py').write_text('def twice(x): return x * 3\n')
    changed = bundles.Bundle(tmp_path, b.lock)
    assert changed.digest != b.digest
    assert changed.function(changed.patches[0])(None, 3) == 10


@pytest.mark.parametrize('field,value', [('module','sglang.srt.server_args'),('symbol','Layer.__init__'),
    ('symbol','set_precision'),('file','../escape.py'),('file','/tmp/escape.py')])
def test_disallowed_targets_and_paths(tmp_path, field, value):
    b = bundle(tmp_path)
    patch = dict(b.patches[0]); patch[field] = value
    (b.root/'patches.json').write_text(json.dumps(dict(version=1, patches=[patch])))
    with pytest.raises(ValueError):
        bundles.Bundle(tmp_path,b.lock)


def test_duplicate_target(tmp_path):
    b = bundle(tmp_path)
    (b.root/'patches.json').write_text(json.dumps(dict(version=1, patches=b.patches * 2)))
    with pytest.raises(ValueError, match='Duplicate'):
        bundles.Bundle(tmp_path,b.lock)


def test_source_limits_links_and_artifacts(tmp_path):
    b = bundle(tmp_path)
    (b.root/'hidden').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='symlink'):
        bundles.Bundle(tmp_path,b.lock)
    (b.root/'hidden').unlink()
    (b.root/'kernel.so').write_bytes(b'ELF')
    with pytest.raises(ValueError,match='Unsupported'):
        bundles.Bundle(tmp_path,b.lock)
    (b.root/'kernel.so').unlink()
    with pytest.raises(ValueError,match='budget'):
        bundles.source_files(b.root,max_bytes=1)


@pytest.mark.parametrize('source', ['import os\n','import torch\ntorch.backends.cuda.matmul.allow_tf32 = True\n',
    'import torch\nx = torch.os.environ\n', 'open("model/config.json", "w")\n',
    '__builtins__["open"]("model/config.json")\n', 'value = __loader__\n',
    'import torch\nreflect = getattr\nreflect(torch, "os")\n'])
def test_runtime_mutation_policy(tmp_path,source):
    b = bundle(tmp_path)
    (b.root/'kernel.py').write_text(source)
    with pytest.raises(ValueError):
        bundles.Bundle(tmp_path,b.lock).check_sources()


def test_lazy_layer_fusion_and_observer(monkeypatch,tmp_path):
    b = bundle(tmp_path)
    (tmp_path/'fixture_model.py').write_text('class Layer:\n def forward(self,x): return x\nclass Model:\n def forward(self,x): return Layer().forward(x)\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    calls=[]
    monkeypatch.setattr(bundles,'record',lambda bundle,label,observer=False: calls.append((label,observer)))
    hooks=list(sys.meta_path)
    try:
        bundles.install(tmp_path,b.lock)
        assert 'fixture_model' not in sys.modules
        module=importlib.import_module('fixture_model')
        assert module.Model().forward(3)==7
        assert module.Model().forward(4)==9
        assert calls==[('fixture_model:Layer.forward',False),('model_forward',True)]
        assert bundles.original('fixture_model','Layer.forward')(None,3)==3
    finally:
        sys.meta_path[:]=hooks
        sys.modules.pop('fixture_model',None)


def test_audit_requires_actual_replacements_and_every_rank(tmp_path):
    b=bundle(tmp_path)
    def row(rank, observer, function):
        return dict(rank=rank, observer=observer, function=function,bundle_sha256=b.digest,
                    server_settings=dict(random_seed=7),runtime_settings={})
    records=[row(0,False,'fixture_model:Layer.forward'),row(0,True,'model_forward'),row(1,True,'model_forward')]
    def save(rows):
        (tmp_path/'rank_calls.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    save(records)
    assert bundles.audit(tmp_path,b.lock)==records
    save(records[:-1])
    with pytest.raises(ValueError,match='rank'):
        bundles.audit(tmp_path,b.lock)
    save(records[1:])
    with pytest.raises(ValueError,match='not executed'):
        bundles.audit(tmp_path,b.lock)


@pytest.mark.parametrize('field,value', [('root','../escape'),('root','/tmp/escape'),('manifest','../patches.json')])
def test_task_bundle_paths_cannot_escape(tmp_path,field,value):
    b=bundle(tmp_path)
    b.lock['candidate_bundle'][field]=value
    with pytest.raises(ValueError,match='inside the task'):
        bundles.Bundle(tmp_path,b.lock)


def test_classmethod_original_keeps_replacement_call_signature(monkeypatch,tmp_path):
    b=bundle(tmp_path)
    (b.root/'kernel.py').write_text('from _aka_serving_bundle import original\ndef forward(cls,x): return original("fixture_model","Layer.forward")(cls,x) * 2\n')
    (tmp_path/'fixture_model.py').write_text('class Layer:\n value=3\n @classmethod\n def forward(cls,x): return cls.value+x\nclass Model:\n def forward(self,x): return Layer.forward(x)\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules,'_aka_serving_bundle',bundles)
    monkeypatch.setattr(bundles,'record',lambda *args,**kwargs: None)
    hooks=list(sys.meta_path)
    try:
        bundles.install(tmp_path,b.lock)
        assert importlib.import_module('fixture_model').Model().forward(4)==14
    finally:
        sys.meta_path[:]=hooks
        sys.modules.pop('fixture_model',None)
