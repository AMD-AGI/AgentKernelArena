"""CPU contract checks for GLM5.2 local MoE; these do not qualify GPU execution."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import torch

from src.task_spec import load_task_spec

TASK = (Path(__file__).resolve().parents[1] / "tasks/aiter_workloads"
        / "glm_5_2_mxfp4_tp4_ep4_agentx_c8_m48_fused_moe")


def module(name):
    spec = importlib.util.spec_from_file_location(f"glm52_{name}", TASK / f"scripts/{name}.py")
    result = importlib.util.module_from_spec(spec)
    original_path = list(sys.path)
    try:
        if name == "reference":
            with patch.dict(sys.modules, {"cases": module("cases")}):
                spec.loader.exec_module(result)
        else:
            spec.loader.exec_module(result)
    finally:
        sys.path[:] = original_path
    return result


cases = module("cases")
ref = module("reference")
evaluate = module("evaluate")


@pytest.fixture(autouse=True)
def full_precision_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def test_all_ep_partitions():
    masks = [cases.partition_mask(rank=rank) for rank in range(4)]
    assert all(len(mask) == 257 and sum(mask) == 65 for mask in masks)
    assert [sum(mask[i] for mask in masks) for i in range(257)] == [1] * 256 + [4]
    for rank, mask in enumerate(masks):
        mapping = cases.local_expert_map(mask)
        assert mapping[rank * 64:(rank + 1) * 64] == list(range(64))
        assert mapping[256] == 64
        assert all(mapping[i] == -1 for i in range(256) if not mask[i])
    for args in ((0, 4, 0), (257, 4, 0), (256, 4, 4), (256, True, 0)):
        with pytest.raises(ValueError):
            cases.partition_mask(*args)
    with pytest.raises(ValueError):
        cases.local_expert_map([0, 0])


def test_e2m1_ties_and_fused_epilogue_boundary():
    ties = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.] * 4).view(1, 32)
    expected = torch.tensor([0., 1., 1., 2., 2., 4., 4., 6.] * 4).view(1, 32)
    assert torch.equal(ref.quantize_dequantize(ties), expected)
    assert torch.equal(ref.quantize_dequantize(-ties), -expected)
    # The selected fused epilogue uses scale 2 at amax 6, while input RoundUp
    # uses scale 1. Values 1.5 and .5 make that difference observable.
    x = torch.tensor([6., 1.5, .5, 1.] * 8).view(1, 32)
    assert ref.quantize_dequantize(x)[0, :4].tolist() == [6., 1.5, .5, 1.]
    assert ref.quantize_dequantize(x, fused_intermediate=True)[0, :4].tolist() == [6., 2., 0., 1.]
    for fused in (False, True):
        assert torch.equal(ref.quantize_dequantize(torch.zeros(1, 32), fused_intermediate=fused), torch.zeros(1, 32))
    assert ref.scale_to_float(torch.tensor([0, 127, 128], dtype=torch.uint8)).tolist() == [2.0 ** -127, 1., 2.]
    with pytest.raises(ValueError):
        ref.scale_to_float(torch.tensor([255], dtype=torch.uint8))


def tiny_payload():
    # H=I=32. Constant matrices make both projections analytical: expert0
    # produces 128 in every output, and expert1 produces 32, for input ones.
    w1 = torch.full((2, 64, 16), 0x22, dtype=torch.uint8)
    w2 = torch.full((2, 32, 16), 0x22, dtype=torch.uint8)
    s1 = torch.cat((torch.full((64, 1), 124, dtype=torch.uint8), torch.full((64, 1), 123, dtype=torch.uint8)))
    s2 = torch.full((64, 1), 125, dtype=torch.uint8)
    hidden = torch.ones((3, 32), dtype=torch.bfloat16)
    hidden[2] = 0
    routes = torch.tensor([[0, 2], [1, 2], [0, 2]], dtype=torch.int32)
    weights = torch.tensor([[.25, .75], [.4, .6], [.5, .5]], dtype=torch.float32)
    return {'args': (hidden, w1.clone(), w2.clone(), weights, routes),
            'kwargs': {'expert_mask': torch.tensor([1, 0, 1], dtype=torch.int32)},
            'reference': {'w1': w1, 'w2': w2, 'w1_scale': s1, 'w2_scale': s2}}


def test_analytical_local_shared_remote_reference_and_controls():
    payload = tiny_payload()
    expected = torch.tensor([56., 19.2, 0.], dtype=torch.bfloat16).view(3, 1).expand(3, 32)
    assert torch.equal(ref.reference(payload), expected)
    assert ref.reference_controls(payload) == {'zero_input': 'PASS', 'remote_experts_zero': 'PASS', 'wrong_output_rejected': 'PASS'}
    assert ref.compare_output(expected, expected)['accepted']
    assert not ref.compare_output(-expected, expected)['accepted']
    wrong = expected.clone()
    wrong[0, 0] = float('nan')
    with pytest.raises(AssertionError):
        ref.compare_output(wrong, expected)


def test_fresh_values_preserve_objects_and_reject_stale_output():
    payload = tiny_payload()
    old = ref.reference(payload)
    tensors = payload['args']
    pointers = [tensor.data_ptr() for tensor in tensors]
    fresh = {'hidden_states': torch.full_like(tensors[0], 2), 'topk_weight': tensors[3].flip(1),
             'topk_ids': tensors[4].flip(1)}
    cases.replace_dynamic(payload, fresh)
    assert all(a is b for a, b in zip(payload['args'], tensors))
    assert [tensor.data_ptr() for tensor in tensors] == pointers
    with pytest.raises(AssertionError):
        ref.check_output(old, ref.reference(payload))


def test_public_storage_slack_and_attributes_are_checked():
    base = torch.ones(40)
    view = base[4:36]
    payload = {'args': (view,), 'kwargs': {}}
    snapshot = cases.snapshot_inputs(payload)
    cases.check_inputs_unchanged(snapshot)
    base[0] = 5
    with pytest.raises(AssertionError, match='storage slack'):
        cases.check_inputs_unchanged(snapshot)
    snapshot = cases.snapshot_inputs(payload)
    view.is_shuffled = True
    with pytest.raises(AssertionError, match='attributes'):
        cases.check_inputs_unchanged(snapshot)


def test_frozen_binding_and_shared_loader():
    data = evaluate.read_binding()
    assert data['cases'][0]['test_case_id'] == 'local_ep4_m48'
    spec = load_task_spec(TASK / 'config.yaml', task_id='aiter_workloads/' + TASK.name)
    assert len(spec.actions) == 7
    generated = cases.dynamic_inputs(data['cases'][0], 17, device='cpu')
    assert generated['topk_ids'].shape == (48, 9)
    assert (generated['topk_ids'][:, -1] == 256).all()
    assert (generated['topk_weight'][:, -1] == 1).all()
    assert torch.allclose(generated['topk_weight'][:, :8].sum(-1), torch.ones(48))
    assert all(len(set(row.tolist())) == 9 for row in generated['topk_ids'])


def test_runtime_accepts_another_image_but_still_checks_source(tmp_path, monkeypatch):
    runtime = module("runtime")
    package = tmp_path / "installed/aiter"
    source = tmp_path / "upstream/aiter"
    package.mkdir(parents=True)
    source.mkdir(parents=True)
    body = b"# CPU fixture for the image/source gate\n"
    for directory in (package, source):
        (directory / "fused_moe.py").write_bytes(body)
    workload = {"runtime": {"reference_image": "sha256:reference", "dispatch_environment": {}}}
    identity = {"revision": "fixture", "tuning_table": "upstream/aiter/table.csv",
                "aiter_files": {"aiter/fused_moe.py": hashlib.sha256(body).hexdigest()}}
    (tmp_path / "workloads.json").write_text(json.dumps(workload))
    (tmp_path / "source_identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "aiter", SimpleNamespace(__file__=str(package / "__init__.py")))
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "0.4.0")
    monkeypatch.setattr(torch, "__version__", "2.11.0+rocmfixture")
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.delenv("AITER_CONFIG_FMOE", raising=False)
    monkeypatch.setenv("AKA_SCORING_IMAGE_RUNTIME_REF", "sha256:another-image")
    report = runtime.verify_runtime(require_gpu=False)
    assert report["image"] == "sha256:another-image"
    assert report["reference_image"] == "sha256:reference"
    (package / "fused_moe.py").write_bytes(body + b"# changed\n")
    with pytest.raises(RuntimeError, match="source differs"):
        runtime.verify_runtime(require_gpu=False)


@pytest.mark.parametrize('mutation', ('shape', 'seed', 'source', 'frequency'))
def test_manifest_and_provenance_mutations_rejected(tmp_path, monkeypatch, mutation):
    for name in ('config.yaml', 'workloads.json', 'frequency_provenance.json', 'workload_source.json'):
        shutil.copy(TASK / name, tmp_path / name)
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    path = tmp_path / 'workloads.json'
    if mutation in ('shape', 'seed'):
        data = json.loads(path.read_text())
        if mutation == 'shape':
            data['cases'][0]['shape'][0] = 8
        else:
            data['cases'][0]['params']['input_seed'] = 0
        path.write_text(json.dumps(data))
    elif mutation == 'source':
        with (tmp_path / 'workload_source.json').open('a') as stream:
            stream.write(' ')
    else:
        with (tmp_path / 'frequency_provenance.json').open('a') as stream:
            stream.write(' ')
    with pytest.raises(ValueError):
        evaluate.read_binding()
