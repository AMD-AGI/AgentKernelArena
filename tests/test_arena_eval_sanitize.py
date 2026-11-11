"""CPU checks of the v3 runner's sanitizer lanes: build flags, routing, environment and helpers."""
from __future__ import annotations

import struct

import pytest

from src.tools.arena_eval import evaluate, loader, sanitize
from src.tools.arena_eval.contract import Contract


def test_sanitizer_build_flags_are_an_allowlist(monkeypatch):
    monkeypatch.delenv(loader.SANITIZER_CFLAGS_ENV, raising=False)
    assert loader.hip_build_flags() == ("-O3",)
    monkeypatch.setenv(loader.SANITIZER_CFLAGS_ENV, "-fsanitize=address -shared-libsan")
    assert loader.hip_build_flags() == ("-O3", "-fsanitize=address", "-shared-libsan")
    monkeypatch.setenv(loader.SANITIZER_CFLAGS_ENV, "-fsanitize=address -mllvm -x")
    with pytest.raises(loader.BuildError, match="accepts only"):
        loader.hip_build_flags()


def test_offload_arch_override_must_qualify_the_device(monkeypatch):
    monkeypatch.setattr(loader, "current_arch", lambda: "gfx950")
    monkeypatch.delenv(loader.SANITIZER_ARCH_ENV, raising=False)
    assert loader.hip_build_arch() == "gfx950"
    monkeypatch.setenv(loader.SANITIZER_ARCH_ENV, "gfx950:xnack+")
    assert loader.hip_build_arch() == "gfx950:xnack+"
    for bad in ("gfx942:xnack+", "gfx950 -O0", "gfx950:xnack"):
        monkeypatch.setenv(loader.SANITIZER_ARCH_ENV, bad)
        with pytest.raises(loader.BuildError):
            loader.hip_build_arch()


def test_the_build_hash_changes_with_the_sanitizer_flags(monkeypatch):
    monkeypatch.delenv(loader.SANITIZER_CFLAGS_ENV, raising=False)
    plain = loader._toolchain()
    monkeypatch.setenv(loader.SANITIZER_CFLAGS_ENV, "-fsanitize=address -shared-libsan")
    assert loader._toolchain() != plain


def test_sanitize_is_routed_only_for_known_lanes(monkeypatch):
    monkeypatch.setattr(evaluate, "load_contract", lambda: object())
    called = {}
    monkeypatch.setattr(sanitize, "run_lane", lambda contract, role, lane: called.setdefault(
        "args", (role, lane)) and evaluate._report(role, "sanitize", "PASS", [], metadata={"lane": lane}))
    report, code = evaluate.run(["candidate", "sanitize", "page_fault_guard"])
    assert code == 0 and report["action"] == "sanitize" and called["args"] == ("candidate", "page_fault_guard")
    report, code = evaluate.run(["candidate", "sanitize", "valgrind"])
    assert code == 2 and report["failure_kind"] == "usage"


def test_a_lane_refuses_to_run_without_its_environment(monkeypatch):
    for key in sanitize.NO_CACHING:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(evaluate.ActionFailure) as error:
        sanitize.page_fault_guard(object(), "candidate")
    assert error.value.kind == "environment"


def _contract() -> Contract:
    return Contract({
        "version": 1, "task_id": "t", "axes": {"n": {"kind": "var"}}, "derived_axes": {},
        "inputs": {"x": {"kind": "tensor", "shape": ["n"], "dtype": "float32"}},
        "outputs": {"out": {"shape": ["n"], "dtype": "float32"}},
        "compare": {"out": {}}, "timing": {}, "reference": {"file": "reference.py", "symbol": "reference"},
        "baseline": {"language": "triton", "root": "baseline", "entry_file": "baseline/kernel.py", "symbol": "run"},
        "candidate": {"language": "triton", "root": "solution", "entry_file": "solution/kernel.py", "symbol": "run"},
        "cases": [{"id": "big", "axes": {"n": 4096}, "checks": ["correctness"], "seeds": [1, 2]},
                  {"id": "small", "axes": {"n": 3}, "checks": ["correctness"], "seeds": [3, 4]},
                  {"id": "mid", "axes": {"n": 64}, "checks": ["correctness"], "seeds": [5, 6]}],
    })


def test_the_smallest_case_is_chosen_by_element_count():
    assert sanitize.smallest_case(_contract()).case_id == "small"


def _elf(sections: dict[str, bytes]) -> bytes:
    """A minimal 64-bit little-endian ELF holding ``sections`` and a string table."""
    names = b"\0" + b"".join(name.encode() + b"\0" for name in [*sections, ".shstrtab"])
    body, offsets = b"", {}
    for name, data in sections.items():
        offsets[name] = 64 + len(body)
        body += data
    shstr_offset = 64 + len(body)
    body += names
    shoff = 64 + len(body)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, shoff)
    struct.pack_into("<HHH", header, 0x3A, 64, len(sections) + 2, len(sections) + 1)
    table = bytes(64)  # null section
    position = 1
    for name, data in sections.items():
        table += struct.pack("<IIQQQQIIQQ", position, 1, 0, 0, offsets[name], len(data), 0, 0, 1, 0)
        position += len(name) + 1
    table += struct.pack("<IIQQQQIIQQ", position, 3, 0, 0, shstr_offset, len(names), 0, 0, 1, 0)
    return bytes(header) + body + table


def test_elf_section_reads_the_hip_fatbin(tmp_path):
    path = tmp_path / "lib.so"
    path.write_bytes(_elf({".text": b"\x90" * 8, ".hip_fatbin": b"__CLANG_OFFLOAD_BUNDLE__payload"}))
    assert sanitize._elf_section(path, ".hip_fatbin") == b"__CLANG_OFFLOAD_BUNDLE__payload"
    assert sanitize._elf_section(path, ".missing") is None
    (tmp_path / "not-elf").write_bytes(b"MZ" + bytes(80))
    assert sanitize._elf_section(tmp_path / "not-elf", ".hip_fatbin") is None


def test_tilelens_records_are_reduced_to_plain_fields():
    class Load:
        pass

    class Record:
        op_type = Load
        tensor_name = "x"
        violation_address = 0x10
        user_code_tracebacks = ["kernel.py:18 in copy", "b", "c", "d"]

    reduced = sanitize._tilelens_record(Record())
    assert reduced == {"op": "Load", "tensor": "x", "address": "16",
                       "location": ["kernel.py:18 in copy", "b", "c"]}


def test_cache_kernels_finds_a_repr_named_kernel_by_its_metadata_file(tmp_path):
    # Triton names the files after the Python function; a repr kernel's symbol
    # carries its constants, so ``<symbol>.hsaco`` does not exist.
    entry = tmp_path / "ABC"
    entry.mkdir()
    symbol = "_gemm_kernel_BLOCK_SIZE_M_64_BLOCK_SIZE_N_128"
    (entry / "_gemm_kernel.json").write_text('{"name": "%s", "instrumentation_mode": "fpsan"}' % symbol)
    (entry / "_gemm_kernel.hsaco").write_bytes(b"\x7fELF")
    (entry / "__grp___gemm_kernel.json").write_text('{"child_paths": {}}')
    kernels = sanitize._cache_kernels(tmp_path)
    assert list(kernels) == [symbol]
    assert kernels[symbol][0][1] == entry / "_gemm_kernel.hsaco"


def _escaped(data: bytes) -> str:
    # MLIR prints printable ASCII as is and every other byte as \XX.
    return "".join(chr(b) if 32 <= b < 127 and chr(b) not in '"\\' else f"\\{b:02X}" for b in data)


class _Artifact:
    def __init__(self, text):
        self._ir_text = text


def test_flydsl_capture_extracts_the_code_object_from_the_lowered_module(tmp_path):
    elf = b"\x7fELF\x02\x01\x01" + bytes(range(256)) + b'"\\tail'
    text = ('module attributes {gpu.container_module} {\n  gpu.binary @kernels  [#gpu.object<#rocdl.target<'
            'chip = "gfx950">, kernels = <[#gpu.kernel_metadata<"k", !llvm.func<void ()>>]>, bin = "'
            + _escaped(elf) + '">]\n}')
    objects = sanitize._flydsl_code_objects([_Artifact(text), _Artifact(text)], tmp_path)
    assert len(objects) == 1 and objects[0]["kernels"] is None
    assert (tmp_path / objects[0]["code_object"]).read_bytes() == elf
    # A module that holds no binary yields nothing rather than a wrong object.
    assert sanitize._flydsl_code_objects([_Artifact('gpu.binary @k [#gpu.object<#rocdl.target, "abc">]')],
                                         tmp_path / "x") == []


def test_flydsl_observer_records_artifacts_reached_through_executor_methods():
    import sys
    import types

    observed = []
    module = types.ModuleType("flydsl.compiler.jit_executor")
    exec("class CompiledArtifact:\n    def __init__(self, text):\n        self._ir_text = text\n"
         "    def _get_func_exe(self):\n        return 1\n", module.__dict__)
    artifact = module.CompiledArtifact("gpu.binary")
    previous = sys.getprofile()
    sys.setprofile(sanitize._flydsl_observer(observed))
    try:
        artifact._get_func_exe()
        artifact._get_func_exe()
    finally:
        sys.setprofile(previous)
    assert observed == [artifact]
