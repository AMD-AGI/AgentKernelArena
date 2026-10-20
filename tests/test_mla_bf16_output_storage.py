"""Fast-math must not erase MLA's raw BF16 poison/nonfinite rejection."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/hip2hip/mla_decode"
HEADER = TASK / "scripts/native/output_validation.hpp"


def _function(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _build(source, directory, *, native_bf16=False, fast_math=True):
    compiler = shutil.which("clang++") if native_bf16 else (shutil.which("g++") or shutil.which("clang++"))
    if compiler is None:
        pytest.skip("CPU C++ compiler unavailable; no GPU validation")
    program = directory / "check.cpp"
    program.write_text(source)
    command = [compiler, "-std=c++17", "-O3", "-I", str(HEADER.parent)]
    if fast_math:
        command.append("-ffast-math")
    result = subprocess.run(command + [str(program), "-o", str(directory / "check")],
                            capture_output=True, text=True, timeout=60)
    if native_bf16 and result.returncode:
        # A separate compile-only capability probe distinguishes missing native
        # __bf16 support/runtime from a regression in the actual task helpers.
        capability = directory / "capability.cpp"
        capability.write_text("int main(){volatile unsigned short bits=0x3f80; __bf16 v; __builtin_memcpy(&v,(const void*)&bits,2); return (float)v == 1.f ? 0 : 1;}")
        probe = subprocess.run(command + [str(capability), "-o", str(directory / "capability")],
                               capture_output=True, text=True, timeout=60)
        if probe.returncode:
            pytest.skip("CPU compiler lacks executable native __bf16 conversion")
    assert result.returncode == 0, result.stderr
    subprocess.run([str(directory / "check")], check=True, timeout=10)


@pytest.mark.parametrize("native_bf16", [False, True], ids=["encoded_storage", "native_bf16"])
@pytest.mark.parametrize("fast_math", [False, True], ids=["strict", "fast_math"])
def test_correctness_and_measured_gates_reject_poison_before_bf16_promotion(
    tmp_path, native_bf16, fast_math,
):
    storage = ("using bf16=__bf16;\ninline float __bfloat162float(bf16 v){return static_cast<float>(v);}\n"
               if native_bf16 else
               "struct bf16 { uint16_t bits; };\ninline float __bfloat162float(bf16 v){uint32_t bits=uint32_t(v.bits)<<16; float result; std::memcpy(&result,&bits,4); return result;}\n")
    difference = _function((TASK / "mla_decode.hip").read_text(), "static void max_diff(")
    program = r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>
#include "output_validation.hpp"
''' + storage + difference + r'''
bf16 encoded(uint16_t bits) { bf16 value; std::memcpy(&value,&bits,2); return value; }
int main() {
    auto decode=[](bf16 value) { return __bfloat162float(value); };
    std::vector<bf16> expected(4,encoded(0x3f80));
    assert(validate_mla_outputs(expected,expected,expected,decode).empty());
    for(uint16_t bits : {0x0000,0x8000,0x0001,0x007f,0x3f80,0xbf80,0x7f7f,0xff7f})
        assert(mla_finite_storage(encoded(bits)));
    for(uint16_t bits : {0xffff,0x7fc0,0xffc0,0x7f80,0xff80}) {
        for(int corrupted=0;corrupted<3;++corrupted) {
            auto actual=expected,eager=expected,reference=expected;
            auto& values=corrupted==0?actual:corrupted==1?eager:reference;
            values.back()=encoded(bits);
            assert(!validate_mla_outputs(actual,eager,reference,decode).empty());
        }
        auto poison=expected;poison.back()=encoded(bits);
        double absolute=0,relative=0;
        max_diff(poison,expected,absolute,relative);
        assert(absolute>5e-2 && relative>1e-1);
        max_diff(expected,poison,absolute,relative);
        assert(absolute>5e-2 && relative>1e-1);
        // Model a no-op leaving the all-0xff output poison untouched in eager,
        // measured output and replay: it must fail even when they all agree.
        std::vector<bf16> unchanged(4,encoded(bits));
        assert(!validate_mla_outputs(unchanged,unchanged,expected,decode).empty());
    }
    // Retain the original OR numerical gate for finite values: +0.09375 has
    // relative error <0.1, and +0.03125 meets absolute error <0.05 near zero.
    std::vector<bf16> relative_ok(4,encoded(0x3f8c));
    assert(validate_mla_outputs(relative_ok,relative_ok,expected,decode).empty());
    std::vector<bf16> zeros(4,encoded(0)),absolute_ok(4,encoded(0x3d00));
    assert(validate_mla_outputs(absolute_ok,absolute_ok,zeros,decode).empty());
    assert(!validate_mla_outputs(zeros,zeros,expected,decode).empty());
    assert(!validate_mla_outputs(relative_ok,expected,expected,decode).empty());
    double absolute=0,relative=0;
    max_diff(relative_ok,expected,absolute,relative);
    assert(absolute>5e-2 && relative<=1e-1);
}
'''
    _build(program, tmp_path, native_bf16=native_bf16, fast_math=fast_math)


def test_correctness_poison_and_manual_make_dependencies_are_storage_bound():
    host = (TASK / "mla_decode.hip").read_text()
    difference = _function(host, "static void max_diff(")
    assert difference.index("mla_finite_storage") < difference.index("__bfloat162float")
    assert "__float2bfloat16(NAN)" not in host
    assert "hipMemset(d_o, 0xFF" in host
    makefile = (TASK / "Makefile").read_text()
    assert "$(TARGET): $(SRC) source/kernel.hpp $(NATIVE_HEADERS)" in makefile
    assert "scripts/native/output_validation.hpp" in makefile
    assert "scripts/native/routing_controls.hpp" in makefile


def test_kernel_reference_inputs_and_workload_remain_the_frozen_contract():
    baseline = "099e44cf6deb74131d2cfc96a3a055949a2edbdc"
    def original(relative):
        return subprocess.check_output(["git", "show", baseline + ":tasks/hip2hip/mla_decode/" + relative], cwd=ROOT)
    assert (TASK / "source/kernel.hpp").read_bytes() == original("source/kernel.hpp")
    assert (TASK / "workload.json").read_bytes() == original("workload.json")
    assert (TASK / "scripts/native/candidate_driver.hip").read_bytes() == original("scripts/native/candidate_driver.hip")
    current = (TASK / "mla_decode.hip").read_text()
    old = original("mla_decode.hip").decode()
    for signature in ("static void fill_random_bf16(", "static void fill_kv_fp8(",
                      "static void host_reference(",
                      "__host__ __device__ __forceinline__ float fp8_e4m3fn_to_f32"):
        assert _function(current, signature) == _function(old, signature)
    assert "max_abs <= atol" in current and "max_rel <= rtol" in current
    assert "const double atol = 5e-2, rtol = 1e-1;" in current
    assert "-O3 -ffast-math" in (TASK / "Makefile").read_text()
