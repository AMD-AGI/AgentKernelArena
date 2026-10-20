"""Scored MLA inputs reuse the original numerical workload, with CPU sensitivity checks."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/hip2hip/mla_decode"
NATIVE = TASK / "scripts/native"
SHAPES = [(1, 512), (4, 1024), (16, 2048), (1, 4096), (1, 8192)]


def _function(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def numerical_cases(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("CPU C++ compiler unavailable; GPU qualification remains required")
    directory = tmp_path_factory.mktemp("mla_random_inputs")
    host = (TASK / "mla_decode.hip").read_text()
    benchmark = (NATIVE / "benchmark_driver.hip").read_text()
    functions = "\n".join(_function(host, signature).replace(
        "__host__ __device__ __forceinline__", "inline") for signature in (
        "__host__ __device__ __forceinline__ float fp8_e4m3fn_to_f32",
        "static uint8_t f32_to_fp8_e4m3fn", "static void fill_random_bf16(",
        "static void fill_kv_fp8(", "static void host_reference(", "static void max_diff(",
    ))
    periodic = _function(benchmark, "void fill_periodic_regression_inputs(")
    # Compile the actual benchmark setup, including the protected generator calls.
    start = benchmark.index("        fill_random_bf16(h_q,")
    setup = benchmark[start:benchmark.index("        mla_routing_inputs", start)]
    source = directory / "check.cpp"
    source.write_text(r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "output_validation.hpp"
#include "routing_controls.hpp"
struct bf16 { uint16_t bits; };
inline float __bfloat162float(bf16 v) { uint32_t bits=uint32_t(v.bits)<<16;float f;std::memcpy(&f,&bits,4);return f; }
inline bf16 __float2bfloat16(float f) { uint32_t b;std::memcpy(&b,&f,4);b+=0x7fffU+((b>>16)&1);return {uint16_t(b>>16)}; }
constexpr int NHEAD=128,LK=576,LV=512;
''' + functions + periodic + r'''
template<class T> void dump(const std::string& path,const std::vector<T>& v) {
    FILE* f=std::fopen(path.c_str(),"wb");assert(f);assert(std::fwrite(v.data(),sizeof(T),v.size(),f)==v.size());std::fclose(f);
}
int main(int argc,char**argv) {
    assert(argc==4);const int batch=std::atoi(argv[1]),ctx=std::atoi(argv[2]);
    const std::size_t q_e=std::size_t(batch)*NHEAD*LK,o_e=std::size_t(batch)*NHEAD*LV,kv_tokens=std::size_t(batch)*ctx+1024;
    std::vector<bf16> h_q(q_e),repeated_q(q_e),different_q(q_e),reference(o_e),zero(o_e,__float2bfloat16(0));
    std::vector<uint8_t> h_kv(kv_tokens*LK),repeated_kv(h_kv.size()),different_kv(h_kv.size());
    std::vector<int32_t> mapping,lengths;
''' + setup + r'''
    fill_random_bf16(repeated_q,-1.f,1.f,42+batch*31+ctx);fill_kv_fp8(repeated_kv,123+batch*17+ctx);
    assert(mla_same_buffer(h_q,repeated_q)&&mla_same_buffer(h_kv,repeated_kv));
    fill_random_bf16(different_q,-1.f,1.f,43+batch*31+ctx);fill_kv_fp8(different_kv,124+batch*17+ctx);
    assert(!mla_same_buffer(h_q,different_q)&&!mla_same_buffer(h_kv,different_kv));
    for(auto b:h_kv) assert((b&0x7fU)!=0x7fU); // E4M3FN reserves both signed 0x7f encodings for NaN.
    for(auto q:h_q) assert(std::fabs(__bfloat162float(q))<=1.f);
    mla_routing_inputs(mapping,lengths,batch,ctx,kv_tokens);
    host_reference(reference,h_q,h_kv,mapping,lengths,batch,ctx,1.f/std::sqrt(float(LK)));
    auto decode=[](bf16 b){return __bfloat162float(b);};
    assert(validate_mla_outputs(reference,reference,reference,decode).empty());
    assert(!validate_mla_outputs(zero,zero,reference,decode).empty());
    double absolute,relative;max_diff(zero,reference,absolute,relative);assert(absolute>.05&&relative>.1);
    for(int b=0;b<batch;++b) {
        double signal=0;for(int i=0;i<NHEAD*LV;++i)signal=std::max(signal,double(std::fabs(__bfloat162float(reference[std::size_t(b)*NHEAD*LV+i]))));
        if(lengths[b]>1)assert(signal>.075); // Meaningful margin above the unchanged .05 gate, without single-token rows.
    }
    std::vector<bf16> periodic_q(q_e);std::vector<uint8_t> periodic_kv(h_kv.size());fill_periodic_regression_inputs(periodic_q,periodic_kv);
    for(std::size_t i=0;i<q_e;++i)assert(__bfloat162float(periodic_q[i])==float(int((i*17U)%31U)-15)/64.f);
    for(std::size_t i=0;i<periodic_kv.size();++i)assert(periodic_kv[i]==f32_to_fp8_e4m3fn(float(int((i*13U+5U)%29U)-14)/32.f));
    const std::string output=argv[3];dump(output+".q",h_q);dump(output+".kv",h_kv);dump(output+".ref",reference);
    return 0;
}
''')
    compiled = subprocess.run([compiler, "-std=c++17", "-O3", "-ffast-math", "-I", str(NATIVE),
                               str(source), "-o", str(directory / "check")], capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stderr
    for batch, ctx in SHAPES:
        subprocess.run([str(directory / "check"), str(batch), str(ctx), str(directory / f"{batch}_{ctx}")],
                       check=True, timeout=90)
    return directory


@pytest.mark.parametrize("batch,ctx", SHAPES)
def test_random_inputs_match_independent_float64_reference_and_reject_zero(numerical_cases, batch, ctx):
    np = pytest.importorskip("numpy")
    prefix = numerical_cases / f"{batch}_{ctx}"
    def bf16(path, shape):
        return (np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    q = bf16(str(prefix) + ".q", (batch, 128, 576)).astype(np.float64)
    encoded = np.fromfile(str(prefix) + ".kv", dtype=np.uint8).reshape(-1, 576)
    exponent = ((encoded >> 3) & 15).astype(np.int32)
    mantissa = (encoded & 7).astype(np.float64)
    kv = np.where(exponent == 0, mantissa / 512., (1. + mantissa / 8.) * np.exp2(exponent - 7))
    kv *= np.where(encoded & 128, -1., 1.)
    actual = bf16(str(prefix) + ".ref", (batch, 128, 512)).astype(np.float64)
    expected = np.empty_like(actual)
    for b in range(batch):
        length = [ctx // 2 + 1, ctx, ctx - 1, 1][b % 4]
        physical = batch * ctx + 1024 - 1 - (b * ctx + np.arange(length))
        valid = kv[physical]
        scores = q[b] @ valid.T / np.sqrt(576.)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        expected[b] = weights @ valid[:, :512]
        # Require signal in every multi-token row, independent of single-token rows.
        if length > 1:
            assert np.max(np.abs(expected[b])) > .075
    # BF16 output rounding plus FP32 accumulation; stricter than the original .05 gate.
    assert np.max(np.abs(actual - expected)) < .01
    assert np.max(np.abs(expected)) > .075
    assert not np.array_equal(encoded[:-29], encoded[29:])
    assert not np.array_equal(q.reshape(-1)[:-31], q.reshape(-1)[31:])


def test_scored_roles_share_original_generators_and_unchanged_timing_policy():
    host = (TASK / "mla_decode.hip").read_text()
    driver = (NATIVE / "benchmark_driver.hip").read_text()
    assert '#include "../../mla_decode.hip"' in driver
    for source in (host, driver):
        compact = "".join(source.split())
        assert "fill_random_bf16(h_q,-1.0f,1.0f,/*seed=*/42+batch*31+ctx);" in compact
        assert "fill_kv_fp8(h_kv,/*seed=*/123+batch*17+ctx);" in compact
    evaluator = (TASK / "scripts/evaluate.py").read_text()
    assert "role ==" not in evaluator  # Both roles call the same task_runner actions.
    manifest = json.loads((TASK / "workload.json").read_text())
    assert manifest["test_shapes"] == [list(row) for row in SHAPES]
    assert "/*warmup=*/10" in driver and '"--samples", "100"' in (TASK / "scripts/task_runner.py").read_text()
    assert "/*target_ms=*/1.0" in driver and "/*max_repeats=*/1000" in driver
    assert "std::fabs(value - old) > 1.0e-2f" in (NATIVE / "output_validation.hpp").read_text()
    assert "max_abs <= 5e-2 || max_rel <= 1e-1" in (NATIVE / "output_validation.hpp").read_text()
    assert driver.index("fill_random_bf16(h_q,") < driver.index("benchmark_graph_or_events(")
    assert "failure = validate_mla_outputs(h_actual, h_actual, periodic_reference, decode)" in driver
    assert "check_inputs(periodic_q, periodic_kv, validation_stream)" in driver
    assert "return mla_with_restored_inputs" in driver
    baseline = "099e44cf6deb74131d2cfc96a3a055949a2edbdc"
    original = subprocess.check_output(["git", "show", baseline + ":tasks/hip2hip/mla_decode/mla_decode.hip"], cwd=ROOT).decode()
    for signature in ("static uint8_t f32_to_fp8_e4m3fn", "static void fill_random_bf16(", "static void fill_kv_fp8(", "static void host_reference("):
        assert _function(host, signature) == _function(original, signature)
    helper = ROOT / "src/tools/perf/native_hip_graph_benchmark.hpp"
    assert helper.read_bytes() == subprocess.check_output(["git", "show", baseline + ":src/tools/perf/native_hip_graph_benchmark.hpp"], cwd=ROOT)
