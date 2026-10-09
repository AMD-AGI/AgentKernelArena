from types import SimpleNamespace

import pytest

from src.eval_tools.adapters.triton_aot import extract_triton_aot
from src.eval_tools.adapters.replay_capsule import CapsuleValidationError


def compiled(signature, constants, arg_names=None):
    return SimpleNamespace(
        asm={"hsaco": b"compiled-object"}, name="store",
        metadata={"num_warps": 4, "warp_size": 64},
        src=SimpleNamespace(signature=signature, constants=constants,
                            fn=SimpleNamespace(arg_names=arg_names)),
    )


@pytest.mark.parametrize("signature,constants,names", [
    ({"count": "i32", "out": "*i32", "block": "i32"}, {"block": 128}, ["out", "count", "block"]),
    ({"count": "i32", "out": "*i32", "block": "constexpr"}, {(2,): 128}, ["out", "count", "block"]),
    ({1: "i32", 0: "*i32", 2: "i32"}, {(2,): 128}, None),
    ({"1": "i32", "0": "*i32", "2": "i32"}, {"2": 128}, None),
])
def test_extract_orders_runtime_arguments_by_source_position(tmp_path, signature, constants, names):
    artifact = extract_triton_aot(
        compiled(signature, constants, names), tmp_path, grid=(2,),
        pointer_bindings={0: ("output", 16)}, scalar_values={1: 256},
    )
    assert [arg.name for arg in artifact.abi] == ["arg0", "arg1", "global_scratch", "profile_scratch"]
    assert artifact.abi[0].ref == "output"
    assert artifact.abi[0].byte_offset == 16
    assert artifact.abi[1].value == 256
    assert artifact.hsaco_path.read_bytes() == b"compiled-object"


@pytest.mark.parametrize("signature,constants,names,match", [
    ({"out": "*i32"}, {}, None, "no source position"),
    ({"unknown": "*i32"}, {}, ["out"], "no source position"),
    ({"out": "*i32"}, {}, ["out", "out"], "ambiguous"),
    ({0: "*i32", "out": "*i32"}, {}, ["out"], "duplicate"),
    ({0: "*i32"}, {(1, 0): 8}, None, "nested"),
    ({0: "*i32"}, {"unknown": 8}, ["out"], "no source position"),
])
def test_extract_rejects_ambiguous_or_unsupported_abi(tmp_path, signature, constants, names, match):
    with pytest.raises(CapsuleValidationError, match=match):
        extract_triton_aot(compiled(signature, constants, names), tmp_path, grid=(1,),
                           pointer_bindings={0: ("output", 0)}, scalar_values={})
    assert not list(tmp_path.iterdir())


def test_extract_does_not_silently_discard_global_scratch(tmp_path):
    kernel = compiled({0: "*i32"}, {})
    kernel.metadata["global_scratch_size"] = 128
    with pytest.raises(CapsuleValidationError, match="global scratch"):
        extract_triton_aot(kernel, tmp_path, grid=(1,),
                           pointer_bindings={0: ("output", 0)}, scalar_values={})
