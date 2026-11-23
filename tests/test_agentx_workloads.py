"""AgentX workload tool: census parsing, task matching, case selection and weighting, contract output.

The fixture under tests/fixtures/agentx_workloads/capture is a two-leg capture
(eager and graph) with one profiler window each: one prefill step and two
decode steps of batch 29 on TP rank 0.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import pytest
import yaml

from src.task_contract import MIN_WEIGHT_COVERAGE, validate_task_contract
from src.task_spec import TaskConfigError
import math
import shutil
import sys

from src.tools.agentx_workloads import capture, census, log_weighting, matchers, propose, replay, serving_log
from src.tools.agentx_workloads.__main__ import main as cli

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "tests" / "fixtures" / "agentx_workloads" / "capture"
TASK = ROOT / "tasks" / "asm" / "gemm" / "gemm_a16w16" / "n5120_k3200"
TASK_ID = "asm/gemm/gemm_a16w16/n5120_k3200"


@pytest.fixture(scope="module")
def fixture_census():
    return census.build(CAPTURE)


def _args(**overrides):
    values = dict(max_cases=16, merge_ratio=1.5, min_case_us=10.0, baseline_us=None, min_share=0.005,
                  no_graph_padding=False, write=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def _gemm_census(rows, graph_bs=(1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)):
    """A census with GEMM rows [(M, N, K, phase, total_us, calls)]."""
    gemm = [{"M": m, "N": n, "K": k, "dtype": "c10::BFloat16", "B": 1, "phase": ph, "total_us": us, "calls": calls}
            for m, n, k, ph, us, calls in rows]
    return {"capture_id": "cap-1", "model": "Example/Model", "tp": 8, "image": "img:tag", "scope": "rank 0",
            "corpus": {"id": "example/corpus", "revision": "abc123"}, "cuda_graph_bs": list(graph_bs),
            "total_gpu_ms": sum(r["total_us"] for r in gemm) / 1e3 * 10, "gemm": gemm, "attention": [], "ops": []}


# --- Scheduler log and step annotations ----------------------------------------------------------

def test_scheduler_log_records_one_rank_inside_the_window():
    lines = (CAPTURE / "eager" / "server.log").read_text().splitlines()
    window = (serving_log.parse_timestamp("2026-11-21T01:59:30Z"), serving_log.parse_timestamp("2026-11-21T02:10:00Z"))
    prefill, decode = serving_log.parse_log(lines, rank=0, window=window)
    assert [p["new_token"] for p in prefill] == [3000]
    assert [(d["running"], d["token"]) for d in decode] == [(29, 2320000), (29, 2320029)]
    observed = serving_log.observed_workload(prefill, decode, [window])
    assert observed["prefix_cache_hit_pct"] == 94.0
    assert observed["decode_running_hist"] == {29: 2}


def test_an_unknown_scheduler_log_dialect_is_an_error_not_an_idle_server():
    with pytest.raises(ValueError, match="none matched"):
        serving_log.parse_log(["[2026-11-21 02:00:00 TP0] Decode batch, running=3 tokens=9"])


def test_step_annotations():
    assert serving_log.parse_step("step[EXTEND bs=2 toks=3000]") == ("prefill", 2, 3000)
    assert serving_log.parse_step("step[DECODE bs=29]") == ("decode", 29, 29)
    assert serving_log.parse_step("step[IDLE bs=0]") is None


@pytest.mark.parametrize("name, expected", [
    ("step[VERIFY bs=8]", ("verify", 8, 8)),             # SGLang >= 0.5.23
    ("step[TARGET_VERIFY bs=8]", ("verify", 8, 8)),      # older releases
    ("step[DRAFT bs=8]", ("draft", 8, 8)),               # every step of the draft model
    ("step[DRAFT_EXTEND_V2 bs=8]", ("draft", 8, 8)),
    ("step[DECODE bs=5 forward_id=3]", ("decode", 5, 5)),  # detailed annotations
])
def test_speculative_step_annotations(name, expected):
    assert serving_log.parse_step(name) == expected


@pytest.mark.parametrize("line, running, accept, graph", [
    ("[2026-11-22 16:17:56 TP0] Decode batch, #running-req: 1, #token: 16, token usage: 0.00, cuda graph: False, "
     "gen throughput (token/s): 17.52, #queue-req: 0", 1, 1.0, False),
    ("[2026-11-22 16:17:56 TP0] Decode batch, #running-req: 8, #token: 900, token usage: 0.01, accept len: 3.12, "
     "accept rate: 0.71, block accept len: 3.40, cuda graph: True, gen throughput (token/s): 500.0, #queue-req: 0",
     8, 3.12, True),
    ("[2026-11-22 16:17:56 TP0] Decode batch [12], #running-req: 8, #token: 900, token usage: 0.01, accept len: "
     "2.50, accept rate: 0.50, cap len: 1.00, draft est 2.1/3, cuda graph: True, gen throughput (token/s): 5.0, "
     "#queue-req: 0", 8, 2.5, True),
])
def test_decode_lines_with_speculative_fields(line, running, accept, graph):
    _, decode = serving_log.parse_log([line])
    assert (decode[0]["running"], decode[0]["accept_len"], decode[0]["cuda_graph"]) == (running, accept, graph)


def test_hybrid_kv_pool_log_lines_parse():
    """DeepSeek-V4 (SGLang 0.5.23) logs one usage per KV pool and '#full token' instead of '#token'."""
    prefill, decode = serving_log.parse_log([
        "[2026-11-22 16:50:06 TP0] Prefill batch, #new-seq: 1, #new-token: 16384, #cached-token: 0, full token "
        "usage: 0.08, window_128 token usage: 0.01, #running-req: 15, #queue-req: 16, #pending-token: 2136000, "
        "cuda graph: False, input throughput (token/s): 42876.24",
        "[2026-11-22 16:48:32 TP0] Decode batch, #running-req: 1, #full token: 256, full token usage: 0.00, "
        "#window_128 token: 256, window_128 token usage: 0.00, accept len: 2.00, accept rate: 0.20, cuda graph: "
        "False, gen throughput (token/s): 17.47, #queue-req: 0"])
    assert [(p["new_token"], p["running"]) for p in prefill] == [(16384, 15)]
    assert [(d["running"], d["token"], d["accept_len"]) for d in decode] == [(1, 256, 2.0)]


def test_tp1_log_lines_without_a_rank_tag_are_rank_0():
    lines = ["[2026-11-22 16:50:12] Decode batch, #running-req: 1, #full token: 8, full token usage: 0.00, "
             "#window_127 token: 8, window_127 token usage: 0.00, cuda graph: False, gen throughput (token/s): "
             "0.08, #queue-req: 0",
             "[2026-11-22 16:50:12] Prefill batch, #new-seq: 1, #new-token: 6, #cached-token: 0, full token usage: "
             "0.00, window_127 token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0"]
    prefill, decode = serving_log.parse_log(lines, rank=0)
    assert (len(prefill), len(decode), decode[0]["token"]) == (1, 1, 8)
    assert serving_log.parse_log(lines, rank=1) == ([], [])


def test_observed_workload_reports_speculative_acceptance():
    lines = ["[2026-11-22 16:00:0%d TP0] Decode batch, #running-req: 4, #token: 400, token usage: 0.01, accept len: "
             "%.2f, accept rate: 0.50, cuda graph: True, gen throughput (token/s): 5.0, #queue-req: 0" % (i, a)
             for i, a in enumerate((2.0, 3.0))]
    _, decode = serving_log.parse_log(lines)
    window = (serving_log.parse_timestamp("2026-11-22 16:00:00"), serving_log.parse_timestamp("2026-11-22 16:00:09"))
    observed = serving_log.observed_workload([], decode, [window])
    assert observed["speculative"] and observed["accept_len_mean"] == 2.5


def test_corpus_id_and_revision_come_from_the_hf_cache_layout(tmp_path):
    blob = tmp_path / "blobs" / "ab" / "abcdef"
    blob.parent.mkdir(parents=True)
    blob.write_text("{}\n")
    snapshot = tmp_path / "datasets--org--agentx-traces" / "snapshots" / "rev123"
    snapshot.mkdir(parents=True)
    (snapshot / "traces.jsonl").symlink_to(blob)
    traces, name, revision = capture.corpus_traces(str(tmp_path / "datasets--org--agentx-traces"))
    assert (traces, name, revision) == (blob.resolve(), "org/agentx-traces", "rev123")


# --- Census ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("op, dims, types, expected", [
    ("aiter::gemm_a16w16", [[29, 3200], [5120, 3200]], ["Scalar", "c10::BFloat16"],
     (29, 5120, 3200, "c10::BFloat16", 1)),
    ("aten::mm", [[3000, 5120], [5120, 12800]], ["c10::BFloat16"], (3000, 12800, 5120, "c10::BFloat16", 1)),
    ("aten::bmm", [[16, 7, 512], [16, 512, 128]], ["c10::BFloat16"], (7, 128, 512, "c10::BFloat16", 16)),
    ("aiter::gemm_afp4wfp4", [[64, 2304], [7168, 2304]], ["unsigned char"], (64, 7168, 4608, "unsigned char", 1)),
    ("aiter::gemm_a16w16", [[29, 3200], [5120, 1000]], ["c10::BFloat16"], None),
    ("aiter::add_rmsnorm", [[29, 5120], [29, 5120]], ["c10::BFloat16"], None),
    # biased linear in prefill: addmm(bias, activation, weight^T)
    ("aten::addmm", [[5120], [16384, 2880], [2880, 5120]], ["c10::BFloat16"] * 3,
     (16384, 5120, 2880, "c10::BFloat16", 1)),
])
def test_gemm_shapes_from_recorded_input_dims(op, dims, types, expected):
    assert census.gemm_shape(op, dims, types) == expected


def test_kernel_names_are_shortened_for_reports():
    assert census.short_name("_ZN7ck_tile6kentryILi2ENS_38FmhaBatchPrefillWithPagedKVCacheKernelEEEvT0_") == \
        "ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel"
    assert census.short_name("void at::native::elementwise_kernel<128>(int)") == "elementwise_kernel"


def test_census_measures_every_eager_kernel_with_its_shapes(fixture_census):
    c = fixture_census
    assert c["measured_leg"] == "eager"
    # decode from the eager window; prefill from both legs' windows (2 recorded steps) scaled to the one
    # prefill batch the graph leg logged in its steady state; outside-step time by all steps (3 logged / 4)
    assert c["steps_recorded"] == {"prefill": 2, "decode": 2}
    assert c["steps_observed"] == {"prefill": 1, "decode": 2}
    assert c["phase_weights"] == {"prefill": 0.5, "decode": 1.0, "outside_step": 0.75}
    assert c["phase_ms"] == pytest.approx({"decode": 0.236, "outside_step": 0.0015, "prefill": 0.54}, abs=1e-3)
    assert c["cuda_graph_bs"] == [1, 2, 4, 8, 16, 24, 32, 40, 48]
    gemm = {(g["M"], g["N"], g["K"], g["phase"]): g for g in c["gemm"]}
    assert gemm[(29, 5120, 3200, "decode")]["calls"] == 2
    assert gemm[(29, 5120, 3200, "decode")]["total_us"] == 18.0
    assert gemm[(3000, 12800, 5120, "prefill")]["total_us"] == 150.0
    assert gemm[(3000, 12800, 5120, "prefill")]["calls"] == 1.0
    prefill_att = next(a for a in c["attention"] if a["phase"] == "prefill")
    assert (prefill_att["batch"], prefill_att["q_tokens"], prefill_att["kv_tokens_total"]) == (2, 3000, 50000)
    decode_att = next(a for a in c["attention"] if a["phase"] == "decode")
    # 2320000 KV tokens over 29 sequences in the log line nearest in time, in 512-token buckets
    assert (decode_att["batch"], decode_att["kv_tokens_per_seq"]) == (29, 79872)
    assert c["windows"][0]["linked_share"] == pytest.approx(1 - 12 / 778, abs=1e-4)
    assert sum(c["family_share"].values()) == pytest.approx(1.0, abs=1e-5)


def test_graph_leg_contributes_prefill_only(fixture_census):
    graph = fixture_census["legs"]["graph"]
    assert graph["summary"]["steps_with_kernels"] == {"prefill": 1, "decode": 1}
    assert graph["windows"][0]["decode_steps_cpu"] == 2
    used = [w for w in fixture_census["windows"] if "/graph/" in w["trace"]]
    assert used and used[0]["phases_used"] == ["prefill"]
    decode = [g for g in fixture_census["gemm"] if g["phase"] == "decode"]
    assert [(g["M"], g["calls"]) for g in decode] == [(29, 2.0)]


def test_census_without_log_weighting_sums_the_windows():
    c = census.build(CAPTURE, log_weighting=False)
    assert c["phase_weights"] == {}
    assert c["phase_ms"] == {"decode": 0.236, "outside_step": 0.002, "prefill": 1.08}


def test_unlinked_decode_attention_takes_its_batch_from_the_step():
    acc = census.new_acc()
    census.census_window(next((CAPTURE / "graph" / "traces" / "w1").glob("*.trace.json")), acc, None)
    rows = census.finish(acc)["attention"]
    decode = [a for a in rows if a["phase"] == "decode"]
    assert [(a["op"], a["batch"], a["calls"]) for a in decode] == [("", 29, 1)]


def test_a_graph_measured_census_is_flagged():
    c = _gemm_census(AGENTX_ROWS)
    c["measured_leg"] = "graph"
    assert "graph" in propose.propose(TASK, c, _args())["warnings"][0]


def test_census_cli_writes_json_and_markdown(tmp_path):
    out = tmp_path / "kernel_census.json"
    assert cli(["census", str(CAPTURE), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["schema"] == census.SCHEMA
    assert "## GEMM (M, N, K)" in (tmp_path / "census.md").read_text()


# --- Log weighting --------------------------------------------------------------------------------

def test_cost_model_interpolates_in_log_m_and_marks_extrapolation():
    cost = log_weighting.cost_model([(32, 10.0), (2048, 70.0)])
    assert cost(32) == (10.0, False)
    assert cost(256)[0] == pytest.approx(10.0 + 60.0 * math.log(8) / math.log(64)) and not cost(256)[1]
    assert cost(16) == (10.0, True)
    assert cost(16384) == (pytest.approx(70.0 * 8), True)


def _trace(tmp_path, steps):
    """A profiler trace: steps [(label, ts, dur, [(M, N, K, us)])], one GEMM op and kernel per entry."""
    events, xid = [], 0
    for label, ts, dur, gemms in steps:
        events.append({"cat": "gpu_user_annotation", "name": label, "ts": ts, "dur": dur})
        for i, (m, n, k, us) in enumerate(gemms):
            xid += 1
            events.append({"cat": "cpu_op", "name": "aten::mm", "ts": ts, "args": {
                "External id": xid, "Input Dims": [[m, k], [n, k]], "Input type": ["c10::BFloat16", "c10::BFloat16"]}})
            events.append({"cat": "kernel", "name": "Cijk_gemm", "ts": ts + 1 + i, "dur": us,
                           "args": {"External id": xid}})
    path = tmp_path / "w-TP-0.trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    return path


def test_speculative_rounds_record_r_tokens_per_sequence(tmp_path):
    # one round at batch 4: two draft steps (r=1), one verify over 4 draft tokens (r=4), a draft extend (r=4)
    path = _trace(tmp_path, [
        ("step[DRAFT bs=4]", 0, 50, [(4, 5120, 3200, 5.0)]),
        ("step[DRAFT bs=4]", 100, 50, [(4, 5120, 3200, 5.0)]),
        ("step[VERIFY bs=4]", 200, 50, [(16, 5120, 3200, 8.0), (16, 5120, 3200, 8.0)]),
        ("step[DRAFT bs=4]", 300, 50, [(16, 5120, 3200, 8.0)]),
        # after a prefill the draft model extends over the prefill's new tokens: a prefill shape, not a round
        ("step[DRAFT bs=1]", 400, 50, [(16384, 5120, 3200, 900.0)]),
    ])
    acc = census.new_acc()
    census.census_window(path, acc, None)
    rows = {(g["M"], g["phase"]): g for g in census.finish(acc)["gemm"]}
    assert rows[(16, "verify")]["ratios"] == {4: 2} and rows[(16, "verify")]["token_matched"] == 2
    assert rows[(4, "draft")]["ratios"] == {1: 2} and rows[(16, "draft")]["ratios"] == {4: 1}
    assert rows[(16384, "draft_prefill")]["ratios"] == {} and rows[(16384, "draft_prefill")]["token_matched"] == 1
    assert dict(acc["steps"]) == {"draft": 4, "verify": 1}
    assert not acc["prefill_tokens_hist"] and acc["round_bs_hist"] == {"draft:4": 3, "draft:1": 1, "verify:4": 1}


def test_ops_count_op_calls_and_unlinked_kernels_carry_the_step_size(tmp_path):
    events = [
        {"cat": "gpu_user_annotation", "name": "step[EXTEND bs=2 toks=3000]", "ts": 0, "dur": 100},
        {"cat": "gpu_user_annotation", "name": "step[DECODE bs=24]", "ts": 200, "dur": 100},
        # one recorded op launching two kernels (a multi-phase sort): 2 kernel calls, 1 op call
        {"cat": "cpu_op", "name": "aiter::moe_sorting", "ts": 1, "args": {
            "External id": 7, "Input Dims": [[3000, 8], [3000, 8]], "Input type": ["int", "float"]}},
        {"cat": "kernel", "name": "moe_sorting_p1", "ts": 2, "dur": 3, "args": {"External id": 7}},
        {"cat": "kernel", "name": "moe_sorting_p2", "ts": 6, "dur": 3, "args": {"External id": 7}},
        # Python-launched Triton kernels with no recorded op
        {"cat": "kernel", "name": "_causal_conv1d_fwd_kernel", "ts": 10, "dur": 20, "args": {}},
        {"cat": "kernel", "name": "fused_recurrent_gated_delta_rule_fwd_kernel", "ts": 210, "dur": 9, "args": {}},
        {"cat": "kernel", "name": "fmoe_bf16_g1u1_silu_kernel", "ts": 220, "dur": 9, "args": {}},
    ]
    path = tmp_path / "w-TP-0.trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    acc = census.new_acc()
    census.census_window(path, acc, None)
    ops = census.finish(acc)["ops"]
    sort = next(r for r in ops if r["op"] == "aiter::moe_sorting")
    assert (sort["calls"], sort["op_calls"], sort["family"]) == (2, 1, "moe")
    conv = next(r for r in ops if "causal_conv1d" in str(r["kernels"]))
    assert (conv["family"], conv.get("step_tokens"), conv["phase"]) == ("state_space", 3000, "prefill")
    gdn = next(r for r in ops if "gated_delta" in str(r["kernels"]))
    assert (gdn["family"], gdn.get("step_batch")) == ("state_space", 24)
    assert next(r for r in ops if "fmoe" in str(r["kernels"]))["family"] == "moe"


def test_round_phase_attention_ops_carry_the_logged_kv_length(tmp_path):
    import datetime as dt
    t0 = dt.datetime(2026, 11, 22, 20, 0, 0)
    base_ns = int(t0.timestamp() * 1e9)
    events = [
        {"cat": "gpu_user_annotation", "name": "step[VERIFY bs=4]", "ts": 0, "dur": 100},
        {"cat": "cpu_op", "name": "pyhook::aiter.pa_decode_sparse", "ts": 1, "args": {
            "External id": 1, "Input Dims": [[24, 64, 512], [7929, 256, 1, 512]],
            "Input type": ["c10::BFloat16", "c10::Float8_e4m3fn"]}},
        {"cat": "kernel", "name": "_sparse_mla_kernel", "ts": 2, "dur": 10, "args": {"External id": 1}},
    ]
    path = tmp_path / "w-TP-0.trace.json"
    path.write_text(json.dumps({"traceEvents": events, "baseTimeNanoseconds": base_ns}))
    kv = census.DecodeKV([{"ts": t0, "running": 4, "token": 4 * 100_100}])
    acc = census.new_acc()
    census.census_window(path, acc, kv)
    row = next(r for r in census.finish(acc)["ops"] if r["op"] == "pyhook::aiter.pa_decode_sparse")
    assert row["kv_tokens_per_seq"] == 100_352 and row["phase"] == "verify"   # 512-token buckets


def test_speculative_gemm_time_is_rebuilt_per_logged_round():
    rows = [{"M": 16, "N": 5120, "K": 3200, "dtype": "bf16", "B": 1, "phase": "verify", "calls": 128,
             "total_us": 1024.0, "ratios": {"4": 128}},
            {"M": 4, "N": 5120, "K": 3200, "dtype": "bf16", "B": 1, "phase": "draft", "calls": 6,
             "total_us": 30.0, "ratios": {"1": 6}},
            {"M": 16, "N": 5120, "K": 3200, "dtype": "bf16", "B": 1, "phase": "draft", "calls": 3,
             "total_us": 24.0, "ratios": {"4": 3}}]
    # 2 recorded rounds: 64 verify calls, 3 draft r=1 calls and 1.5 draft r=4 calls per round
    rows.append({"M": 16384, "N": 5120, "K": 3200, "dtype": "bf16", "B": 1, "phase": "draft_prefill", "calls": 2,
                 "total_us": 1800.0, "token_matched": 2})
    # 2 recorded prefills (one draft extend each); the log has 7 prefill forwards of 16384 and 3 of 3000 tokens
    forwards = {"prefill": Counter({16384: 7, 3000: 3}), "decode": Counter({8: 10, 16: 5})}
    out, groups = log_weighting.weighted_gemm(rows, {"verify": 2, "draft": 2, "prefill": 2}, {}, forwards)
    got = {(r["phase"], r["M"]): r["calls"] for r in out}
    assert got == {("verify", 32): 640.0, ("verify", 64): 320.0,
                   ("draft", 8): 30.0, ("draft", 16): 15.0, ("draft", 32): 15.0, ("draft", 64): 7.5,
                   ("draft_prefill", 3000): 3.0, ("draft_prefill", 16384): 7.0}
    phases = groups[0]["phases"]
    assert phases["verify"]["ratios"] == {4: 64.0} and phases["draft"]["ratios"] == {1: 3.0, 4: 1.5}
    # measured M: 4, 16 and 16384 (the draft prefill), so every rebuilt M is interpolated, none extrapolated
    assert not any(r["extrapolated"] for r in out)


def test_logged_forwards_split_chunks_and_pad_decode_batches():
    prefill = [{"new_token": 20000}, {"new_token": 300}]
    decode = [{"running": 29}, {"running": 29}, {"running": 40}, {"running": 0}]
    forwards = log_weighting.logged_forwards(prefill, decode, [8, 16, 24, 32, 40], chunk=8192)
    assert forwards["prefill"] == {8192: 2, 3616: 1, 300: 1}
    assert forwards["decode"] == {32: 2, 40: 1}


def test_log_weighting_includes_shapes_no_window_recorded():
    rows = [{"M": 3000, "N": 5120, "K": 3200, "dtype": "bf16", "B": 1, "phase": "prefill", "calls": 64, "total_us": 6400.0},
            {"M": 29, "N": 5120, "K": 3200, "dtype": "bf16", "B": 1, "phase": "decode", "calls": 128, "total_us": 1280.0},
                    # an LM head sees one row per sequence on prefill: not token-indexed, left to the window rows,
            # even when its M happens to equal some other step's token count
            {"M": 2, "N": 18992, "K": 5120, "dtype": "bf16", "B": 1, "phase": "prefill", "calls": 1, "total_us": 30.0,
             "token_matched": 0}]
    steps = {"prefill": 1, "decode": 2}
    sizes = {"prefill": {3000, 2}, "decode": {29}}
    forwards = {"prefill": {3000: 10, 16384: 5}, "decode": {32: 100}}
    out, groups = log_weighting.weighted_gemm(rows, steps, sizes, forwards)
    by = {(r["N"], r["M"], r["phase"]): r for r in out}
    assert by[(5120, 16384, "prefill")]["extrapolated"]
    assert by[(5120, 16384, "prefill")]["total_us"] == pytest.approx(5 * 64 * 100.0 * 16384 / 3000, rel=1e-4)
    assert by[(5120, 3000, "prefill")]["total_us"] == pytest.approx(10 * 64 * 100.0)
    assert by[(5120, 32, "decode")]["calls"] == 100 * 64
    assert not any(r["N"] == 18992 for r in out)
    lm_head = next(g for g in groups if g["N"] == 18992)
    assert lm_head["phases"]["prefill"]["log_weighted"] is False


def test_census_rebuilds_gemm_time_from_the_production_log(fixture_census):
    lw = fixture_census["gemm_log_weighting"]
    assert lw["logged_forwards"] == {"prefill": 1, "decode": 2}
    rows = {(r["M"], r["N"], r["phase"]): r for r in fixture_census["gemm_log_weighted"]}
    # decode batch 29 is logged twice and padded to the graph size 32; its cost interpolates 29 -> 3000
    decode = rows[(32, 5120, "decode")]
    expected = 9.0 + 61.0 * (math.log(32) - math.log(29)) / (math.log(3000) - math.log(29))
    assert decode["forwards"] == 2 and decode["cost_us"] == pytest.approx(expected, abs=1e-3)
    assert rows[(3000, 5120, "prefill")]["total_us"] == 70.0
    measured = {(g["M"], g["N"], g["phase"]): g for g in fixture_census["gemm"]}
    assert measured[(29, 5120, "decode")]["token_matched"] == 2.0


def test_crashed_windows_and_restart_segments(tmp_path):
    root = tmp_path / "capture"
    shutil.copytree(CAPTURE, root)
    manifest = json.loads((root / "capture.json").read_text())
    graph = manifest["legs"]["graph"]
    graph["windows"][0]["status"] = "server exited"
    graph["segments"] = [{"start": "2026-11-21T01:59:30Z", "end": "2026-11-21T02:00:00Z", "server_log": "server.log"},
                         {"start": "2026-11-21T02:00:01Z", "end": "2026-11-21T02:40:00Z", "server_log": "server.log"}]
    graph["crashes"] = [{"time": "2026-11-21T02:00:00Z", "error": "Memory access fault"}]
    graph["restarts"] = 1
    (root / "capture.json").write_text(json.dumps(manifest))
    c = census.build(root)
    assert not any("/graph/" in w["trace"] for w in c["windows"])
    observed = c["legs"]["graph"]["observed"]
    assert observed["window"]["segments"] == 2 and observed["decode_iterations"] == 3
    assert c["legs"]["graph"]["restarts"] == 1
    assert "Memory access fault" in census.markdown(c)


def test_crash_excerpt_and_window_planning():
    text = "\n".join(["[t TP0] Prefill batch, #new-token: 16384", "[t TP0] Decode batch, #running-req: 3",
                     "Memory access fault by GPU node-4 on address 0x1. Reason: Unknown.", "Fatal Python error"])
    error, before = capture.crash_excerpt(text)
    assert error.startswith("Memory access fault") and len(before) == 2
    plan = capture.plan_windows([360, 540, 720, 900], t0=1000.0, segment_start=1600.0, done={1})
    assert plan == [(3, 1720.0), (4, 1900.0)]


# --- Matching ------------------------------------------------------------------------------------

@pytest.mark.parametrize("operator, name", [
    ("gemm_a16w16", "gemm"), ("gemm_a8w8_blockscale", "gemm"), ("gemm_afp4wfp4", "gemm"),
    ("batched_gemm_bf16", "gemm"), ("skinny_gemm_a16w16", "gemm"),
    ("paged_attention_decode", "decode_attention"), ("rms_norm", "rms_norm"),
    ("fused_add_rms_norm", "fused_add_rms_norm"), ("qk_rmsnorm_rope", "qk_norm"), ("rope_fwd", "rope"),
    ("per_token_quant_fp8", "quant"), ("silu_and_mul", "gated_activation"), ("topk_softmax", "moe_routing"),
    ("moe_sorting", "moe_sorting"),
])
def test_matcher_registry(operator, name):
    assert matchers.find(operator)[0] == name


def test_unknown_operator_has_no_matcher():
    assert matchers.find("furthest_point_sample") is None


def test_gemm_matcher_keeps_only_the_scenario_const_axes_and_dtype(fixture_census):
    raw = yaml.safe_load((TASK / "task.yaml").read_text())
    match = matchers.gemm(raw["interface"]["axes"], raw["interface"]["inputs"], fixture_census)
    # log-weighted rows: the logged decode batch 29 is already padded to the graph size 32
    assert sorted((o.axes["m"], o.phase) for o in match.observations) == [(32, "decode"), (3000, "prefill")]
    window_rows = dict(fixture_census, gemm_log_weighted=[])
    assert sorted(o.axes["m"] for o in matchers.gemm(raw["interface"]["axes"], raw["interface"]["inputs"],
                                                      window_rows).observations) == [29, 3000]
    fp8 = copy.deepcopy(raw["interface"]["inputs"])
    fp8["a"]["dtype"] = "float8_e4m3fn"
    assert matchers.gemm(raw["interface"]["axes"], fp8, fixture_census).observations == []


def test_token_row_matchers_check_the_width_axis(fixture_census):
    axes = {"n_tokens": {"kind": "var", "min": 1, "max": 65536}, "hidden_size": {"kind": "const", "value": 5120}}
    match = matchers.fused_add_rms_norm(axes, {}, fixture_census)
    assert sorted(o.axes["n_tokens"] for o in match.observations) == [29, 3000]
    axes["hidden_size"]["value"] = 4096
    assert matchers.fused_add_rms_norm(axes, {}, fixture_census).observations == []
    assert matchers.rms_norm(dict(axes, hidden_size={"kind": "const", "value": 5120}), {},
                             fixture_census).observations == []  # add_rmsnorm is not a plain rms_norm


def test_decode_attention_matcher_uses_batch_and_logged_kv_length(fixture_census):
    axes = {"num_seqs": {"kind": "var", "min": 1, "max": 1024}, "max_seq_len": {"kind": "var", "min": 1, "max": 131072},
            "num_query_heads": {"kind": "const", "value": 8}, "head_size": {"kind": "const", "value": 128},
            "num_kv_heads": {"kind": "const", "value": 1}}
    match = matchers.decode_attention(axes, {}, fixture_census)
    assert [o.axes for o in match.observations] == [{"num_seqs": 29, "max_seq_len": 79872}]
    assert match.unchecked_const_axes == ["num_kv_heads"]


# --- Selection and weighting ------------------------------------------------------------------------

def _points(rows):
    return [propose.Point({"m": m}, us=us, calls=calls, phases={"prefill"}) for m, us, calls in rows]


def test_decode_batches_are_padded_to_captured_graph_sizes():
    obs = [matchers.Observation({"m": 29}, 10.0, 1, "decode", "d", pad_axis="m"),
           matchers.Observation({"m": 32}, 5.0, 1, "decode", "d", pad_axis="m"),
           matchers.Observation({"m": 29}, 7.0, 1, "prefill", "p", pad_axis="m")]
    axes = {"m": {"kind": "var", "min": 1, "max": 32768}}
    points, padded = propose.collect(obs, [16, 24, 32], axes)
    assert [(p.axes["m"], p.us) for p in points] == [(29, 7.0), (32, 15.0)]
    assert padded == 1


def test_neighbours_within_the_merge_ratio_merge_at_the_heaviest_member_until_the_cap():
    points = _points([(1000 + 10 * i, 100.0 + i, 1) for i in range(20)] + [(16384, 400.0, 1)])
    groups, uncovered, _ = propose.select(points, lambda p: 50.0, floor=10, merge_ratio=1.5, max_cases=4,
                                          min_cases=4, min_share=0.0)
    assert len(groups) <= 4 and not uncovered
    reps = sorted(propose.representative(g, lambda p: 50.0, 10).axes["m"] for g in groups)
    assert 16384 in reps


def test_shapes_below_the_floor_merge_into_an_eligible_neighbour_or_stay_correctness_only():
    times = {24: 8.0, 32: 9.0, 40: 11.0, 4096: 100.0, 8192: 200.0, 16384: 400.0, 2: 5.0}
    points = _points([(m, 10 * t, 10) for m, t in times.items()])
    groups, uncovered, log = propose.select(sorted(points, key=lambda p: p.size), lambda p: p.per_call_us,
                                            floor=10, merge_ratio=2.0, max_cases=16, min_cases=4, min_share=0.0)
    reps = [propose.representative(g, lambda p: p.per_call_us, 10).axes["m"] for g in groups]
    assert reps == [40, 4096, 8192, 16384]
    assert sorted(p.axes["m"] for p in groups[0].points) == [24, 32, 40]
    assert [p.axes["m"] for p in uncovered] == [2]
    assert "floor" in log[0]


def test_small_groups_are_dropped_only_above_the_minimum_case_count():
    points = _points([(64, 1.0, 1), (1024, 300.0, 1), (4096, 300.0, 1), (8192, 300.0, 1), (16384, 300.0, 1)])
    groups, uncovered, _ = propose.select(points, lambda p: 50.0, floor=10, merge_ratio=1.5, max_cases=16,
                                          min_cases=4, min_share=0.01)
    assert len(groups) == 4 and [p.axes["m"] for p in uncovered] == [64]
    groups, uncovered, _ = propose.select(points[1:], lambda p: 50.0, floor=10, merge_ratio=1.5, max_cases=16,
                                          min_cases=4, min_share=0.5)
    assert len(groups) == 4 and not uncovered


def test_baseline_times_interpolate_in_log_log(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"1": 16.0, "64": 16.0, "1024": 64.0}))
    estimate = propose.baseline_estimate(str(path))
    assert estimate(32) == pytest.approx(16.0)
    assert estimate(256) == pytest.approx(32.0)
    assert estimate(2048) == pytest.approx(128.0)


def test_boundary_cases_are_regenerated_when_missing():
    axes = {"m": {"kind": "var", "min": 1, "max": 32768}, "n": {"kind": "const", "value": 5120}}
    cases = [{"id": "m4096", "axes": {"m": 4096}, "checks": ["correctness", "performance"]}]
    added = propose.regenerate_boundaries(cases, axes, ["m"])
    assert [c["axes"]["m"] for c in added] == [1, 3, 17]
    assert all(c["checks"] == ["correctness"] for c in added)


# --- Proposal for a real task ---------------------------------------------------------------------------

AGENTX_ROWS = [(24, 5120, 3200, "decode", 900.0, 100), (29, 5120, 3200, "decode", 2000.0, 200),
               (32, 5120, 3200, "decode", 4000.0, 400), (40, 5120, 3200, "decode", 1500.0, 140),
               (48, 5120, 3200, "decode", 1100.0, 100), (2944, 5120, 3200, "prefill", 600.0, 9),
               (8192, 5120, 3200, "prefill", 900.0, 4), (16384, 5120, 3200, "prefill", 4000.0, 12),
               (32, 1280, 5120, "decode", 9999.0, 999)]


def test_proposal_uses_log_weighted_rows_and_records_the_method(fixture_census):
    report = propose.propose(TASK, fixture_census, _args(min_case_us=0.0))
    assert report["matcher"] == "gemm" and "scheduler-log" in report["notes"][0]
    assert {o["axes"]["m"] for o in report["observed"]} == {32, 3000}


def test_proposal_for_a_task_is_weighted_and_passes_the_task_contract():
    report = propose.propose(TASK, _gemm_census(AGENTX_ROWS), _args())
    assert report["valid"], report.get("error")
    proposal = report["proposal"]
    perf = [c for c in proposal["cases"] if "performance" in c["checks"]]
    assert [c["axes"]["m"] for c in perf] == [32, 40, 48, 2944, 8192, 16384]
    weights = {c["axes"]["m"]: c["weight"] for c in perf}
    shares = {c["axes"]["m"]: c["observed"]["time_share"] for c in perf}
    # 24 and 29 pad to 24 and 32 and merge into the m32 case
    assert shares[32] == pytest.approx((900 + 2000 + 4000) / 15000, abs=1e-4)
    assert weights[32] == 100 + 200 + 400            # calls: the census records no steady-state seconds
    assert next(c for c in perf if c["axes"]["m"] == 32)["observed"]["members"] == [{"m": 24}, {"m": 32}]
    assert sum(shares.values()) == pytest.approx(report["coverage"], abs=1e-3)
    provenance = proposal["weight_provenance"]
    assert provenance["weight_unit"].startswith("calls over")
    assert provenance["kind"] == "agentx" and provenance["source"] == "example/corpus@abc123"
    assert provenance["capture"] == "cap-1" and provenance["model"] == "Example/Model"
    assert MIN_WEIGHT_COVERAGE <= provenance["coverage"] <= report["coverage"]
    # every former case is kept as a correctness case; nothing else carries a weight
    original = json.loads((TASK / "workloads.json").read_text())
    kept = {json.dumps(c["axes"]) for c in proposal["cases"]}
    assert all(json.dumps(c["axes"]) in kept for c in original["cases"])
    assert all("weight" not in c for c in proposal["cases"] if c["checks"] == ["correctness"])
    raw = yaml.safe_load((TASK / "task.yaml").read_text())
    assert validate_task_contract(raw, proposal, task_id=TASK_ID).case_weights == {c["id"]: c["weight"] for c in perf}


def test_proposal_weights_are_call_rates_and_carry_the_deployment():
    c = _gemm_census(AGENTX_ROWS)
    c.update(steady_state_s=100.0, deployment={"image": "img:tag", "page_size": 16, "chunked_prefill_size": 16384,
                                               "kv_cache_dtype": "fp8_e4m3", "speculative_algorithm": None},
             corpus_coverage={"context_cap_tokens": 262144, "requests": 10, "requests_dropped": 1,
                              "dropped_share": 0.1})
    report = propose.propose(TASK, c, _args())
    assert report["valid"], report.get("error")
    perf = {x["axes"]["m"]: x for x in report["proposal"]["cases"] if "performance" in x["checks"]}
    assert perf[32]["weight"] == pytest.approx(7.0) and perf[16384]["weight"] == pytest.approx(0.12)
    provenance = report["proposal"]["weight_provenance"]
    assert provenance["weight_unit"].startswith("calls per second")
    assert provenance["deployment"]["page_size"] == 16 and provenance["corpus"]["dropped_share"] == 0.1
    raw = yaml.safe_load((TASK / "task.yaml").read_text())
    validate_task_contract(raw, report["proposal"], task_id=TASK_ID)


def test_deployment_records_model_references_not_local_paths():
    c = _gemm_census(AGENTX_ROWS)
    c.update(steady_state_s=100.0, deployment={
        "image": "img:tag", "speculative_draft_model_path": "/data/hub/models--org--Draft-Model/snapshots/0123abcd"})
    report = propose.propose(TASK, c, _args())
    deployment = report["proposal"]["weight_provenance"]["deployment"]
    assert deployment["speculative_draft_model_path"] == "org/Draft-Model@0123abcd"
    assert census.model_ref("/models/Some-Model-FP8/") == "Some-Model-FP8"
    assert census.model_ref("org/name") == "org/name" and census.model_ref(None) is None
    info = {"tp_size": 1, "speculative_draft_model_path": "/models/Some-Model-FP8"}
    assert census.deployment({"image": "img"}, info)["speculative_draft_model_path"] == "Some-Model-FP8"


def test_out_of_range_shapes_are_reported_as_clamped():
    rows = AGENTX_ROWS + [(40000, 5120, 3200, "prefill", 300.0, 1)]
    report = propose.propose(TASK, _gemm_census(rows), _args())
    assert report["clamped_to_axis_bounds_share"] == pytest.approx(300 / 15300, abs=1e-6)


# --- Replay -------------------------------------------------------------------------------------

def _trajectory(*sizes, sub=()):
    reqs = [{"in": n, "out": 100, "hash_ids": []} for n in sizes]
    if sub:
        reqs.append({"type": "subagent", "requests": [{"in": n, "out": 100, "hash_ids": []} for n in sub]})
    return {"requests": reqs}


def test_requests_over_the_context_window_are_skipped_and_the_trajectory_continues():
    import random
    rec = _trajectory(1000, 200000, 5000, sub=(3000, 300000, 4000))
    events = replay.build_session(rec, max_ctx=131072, max_out=16384, rng=random.Random(0),
                                  start_min=0.0, start_max=0.0)
    assert [e[1]["in"] for e in events if e[0] == "req"] == [1000, 5000]  # 200k skipped, 5000 still sent
    assert [[x["in"] for x in g] for e in events if e[0] == "sub" for g in e[1]] == [[3000, 4000]]


def test_corpus_coverage_counts_what_the_window_drops(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(t) for t in (
        _trajectory(1000, 200000, 5000, sub=(300000,)), _trajectory(400000))) + "\n")
    cov = replay.corpus_coverage(str(traces), 131072, 16384)
    assert (cov["trajectories"], cov["trajectories_unreachable"]) == (2, 1)
    assert (cov["requests"], cov["requests_dropped"], cov["dropped_share"]) == (5, 3, 0.6)
    assert cov["input_tokens_dropped_share"] == pytest.approx(900000 / 906000, abs=1e-4)
    assert replay.corpus_coverage(str(traces), 1 << 20, 16384)["requests_dropped"] == 0


# --- Server mounts ------------------------------------------------------------------------------

def test_model_mounts_follow_snapshot_links_into_a_shared_blob_store(tmp_path):
    store = tmp_path / "hub" / "blobs"
    (store / "7b").mkdir(parents=True)
    (store / "a1").mkdir()
    (store / "7b" / "h1").write_text("w")
    (store / "a1" / "h2").write_text("w")
    repo = tmp_path / "hub" / "models--org--m"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "h1").symlink_to(store / "7b" / "h1")
    (repo / "blobs" / "h2").symlink_to(store / "a1" / "h2")
    (repo / "blobs" / "cfg").write_text("{}")
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    for name, blob in (("model-1.safetensors", "h1"), ("model-2.safetensors", "h2"), ("config.json", "cfg")):
        (snap / name).symlink_to(Path("../../blobs") / blob)
    assert capture.model_mounts(str(snap)) == [str(repo), str(store)]
    args = capture.mount_args(str(snap), "/out", [f"{store}:{store}:ro"])
    assert args.count(f"{store}:{store}:ro") == 1 and f"{repo}:{repo}:ro" in args and "/out:/out" in args


def test_graph_batch_sizes_fall_back_to_the_server_config(tmp_path):
    graph = tmp_path / "graph"
    graph.mkdir()
    (graph / "server_info.json").write_text(json.dumps(
        {"cuda_graph_config": {"decode": {"backend": "full", "bs": [8, 1, 2, 4]}}}))
    legs = {"graph": {"dir": str(graph), "cuda_graph_bs": []}}
    assert census.graph_batch_sizes(tmp_path, legs) == [1, 2, 4, 8]
    assert census.graph_batch_sizes(tmp_path, {"graph": {"dir": str(graph), "cuda_graph_bs": [3, 5]}}) == [3, 5]
    (graph / "server_info.json").write_text(json.dumps({"cuda_graph_config": {"decode": {"backend": "disabled"}}}))
    assert census.graph_batch_sizes(tmp_path, legs) == []
    assert census.graph_batch_sizes(tmp_path, {}) == []


def test_dp_attention_trace_names_are_recognized(tmp_path):
    import gzip as gz
    window = tmp_path / "traces" / "w1"
    window.mkdir(parents=True)
    for name in ("x-TP-0-DP-0.trace.json.gz", "x-TP-1-DP-1.trace.json.gz", "x-TP-10.trace.json.gz"):
        with gz.open(window / name, "wt") as fh:
            fh.write("{}")
    (window / "x-TP-2-DP-2.trace.json.gz").write_bytes(b"cut")         # an unfinished gzip stream
    assert capture.finalized_ranks(window, 8) == {0, 1, 10}
    assert [p.name for p in census.leg_traces(tmp_path, 1)] == ["x-TP-1-DP-1.trace.json.gz"]
    assert [p.name for p in census.leg_traces(tmp_path, 0)] == ["x-TP-0-DP-0.trace.json.gz"]


def test_prompt_token_ids_stay_below_the_vocabulary_and_special_tokens(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 129280}))
    assert capture.prompt_vocab_hi(str(tmp_path)) == 129280
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(
        {"added_tokens_decoder": {"0": {}, "1": {}, "128000": {}, "128815": {}}}))
    assert capture.prompt_vocab_hi(str(tmp_path)) == 128000
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 163840}}))
    (tmp_path / "tokenizer_config.json").write_text("{}")
    assert capture.prompt_vocab_hi(str(tmp_path)) == capture.DEFAULT_VOCAB_HI
    assert capture.prompt_vocab_hi(str(tmp_path / "missing")) == capture.DEFAULT_VOCAB_HI


def test_server_env_is_repeatable_and_validated(tmp_path):
    import argparse as ap_mod
    parser = ap_mod.ArgumentParser()
    capture.add_arguments(parser)
    base = ["--model-path", "m", "--model-name", "m", "--corpus", "c", "--image", "i", "--out", str(tmp_path),
            "--capture-id", "x"]
    a = parser.parse_args(base + ["--env", "SGLANG_USE_AITER=1", "--env", "GPU_MAX_HW_QUEUES=2"])
    assert a.env == ["SGLANG_USE_AITER=1", "GPU_MAX_HW_QUEUES=2"]
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        capture.main(parser.parse_args(base + ["--env", "not an assignment"]))


def test_only_the_eager_leg_is_profiled_by_default():
    assert capture.profiled_legs("auto", ["eager", "graph"]) == {"eager"}
    assert capture.profiled_legs("auto", ["graph"]) == {"graph"}
    assert capture.profiled_legs("none", ["graph"]) == set()
    assert capture.profiled_legs("eager,graph", ["eager", "graph"]) == {"eager", "graph"}
    with pytest.raises(SystemExit, match="profile-legs"):
        capture.profiled_legs("graph", ["eager"])


def test_a_busy_port_is_detected_before_the_server_starts():
    import http.server
    import threading
    srv = http.server.HTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)  # answers 501
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert capture.port_answers(f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
    assert not capture.port_answers(f"http://127.0.0.1:{srv.server_port}")


def test_a_self_contained_snapshot_mounts_only_its_repo(tmp_path):
    repo = tmp_path / "models--org--m"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "h").write_text("w")
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(Path("../../blobs/h"))
    assert capture.model_mounts(str(snap)) == [str(repo)]
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "model.safetensors").write_text("w")
    assert capture.model_mounts(str(flat)) == [str(flat)]


def test_proposal_without_graph_padding_keeps_measured_batches():
    report = propose.propose(TASK, _gemm_census(AGENTX_ROWS), _args(no_graph_padding=True))
    assert 29 in [c["axes"]["m"] for c in report["performance_cases"]] or \
        any({"m": 29} in c["members"] for c in report["performance_cases"])


def test_low_coverage_is_reported_and_not_valid():
    rows = [(2, 5120, 3200, "decode", 9000.0, 1000)] + [(m, 5120, 3200, "prefill", 250.0, 1)
                                                       for m in (2048, 4096, 8192, 16384)]
    report = propose.propose(TASK, _gemm_census(rows, graph_bs=()), _args())
    assert not report["valid"] and "coverage" in report["error"]
    with pytest.raises(TaskConfigError, match="coverage"):
        raw = yaml.safe_load((TASK / "task.yaml").read_text())
        validate_task_contract(raw, report["proposal"], task_id=TASK_ID)


def test_no_matching_shape_is_an_error():
    report = propose.propose(TASK, _gemm_census([(32, 1280, 5120, "decode", 10.0, 1)]), _args())
    assert not report["valid"] and "no census row" in report["error"]


def test_propose_cli_writes_review_files_and_leaves_the_task_alone(tmp_path):
    census_path = tmp_path / "kernel_census.json"
    census_path.write_text(json.dumps(_gemm_census(AGENTX_ROWS)))
    before = (TASK / "workloads.json").read_bytes()
    assert cli(["propose", str(TASK), "--census", str(census_path), "--out", str(tmp_path / "p")]) == 0
    stem = TASK_ID.replace("/", "__")
    assert json.loads((tmp_path / "p" / f"{stem}.workloads.json").read_text())["weight_provenance"]["kind"] == "agentx"
    assert json.loads((tmp_path / "p" / f"{stem}.report.json").read_text())["valid"] is True
    assert (TASK / "workloads.json").read_bytes() == before


# --- Shape hooks ----------------------------------------------------------------------------------

from src.tools.agentx_workloads import shape_hooks  # noqa: E402


def test_shape_hook_specs_expand_presets_and_reject_malformed_entries():
    default = shape_hooks.parse("default")
    assert len(default) == len(shape_hooks.PRESETS["default"]) == len(set(default))
    ops = {t.op for t in default}
    assert {"pyhook::aiter.gemm_a8w8_blockscale", "pyhook::sglang.causal_conv1d_fn",
            "pyhook::sglang.chunk_gated_delta_rule_fwd_h", "pyhook::aiter.pa_decode_sparse"} <= ops
    custom = shape_hooks.parse("pkg.mod:fn, pkg.mod:Cls.meth=my_label ,default,pkg.mod:fn")
    assert [t.spec() for t in custom[:2]] == ["pkg.mod:fn=pkg.fn", "pkg.mod:Cls.meth=my_label"]
    assert len(custom) == 2 + len(default)
    assert shape_hooks.parse("none") == shape_hooks.parse("") == []
    assert shape_hooks.parse(shape_hooks.canonical("default")) == default
    for bad in ("pkg.mod", "pkg.mod:", ":fn", "pkg mod:fn", "pkg.mod:fn=", "unknown_preset"):
        with pytest.raises(ValueError):
            shape_hooks.parse(bad)


def test_hooked_census_ops_keep_the_existing_matchers_working():
    # the GEMM op regex and the state-space matchers' kernel regexes find the hooked wrappers by op name
    assert census.gemm_shape("pyhook::aiter.gemm_a8w8_blockscale", [[64, 7168], [4608, 7168], [64, 56], [36, 56]],
                             ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn", "float", "float"]) == \
        (64, 4608, 7168, "c10::Float8_e4m3fn", 1)
    assert census.family("pyhook::sglang.causal_conv1d_fn") == "state_space"
    assert census.family("pyhook::sglang.chunk_gated_delta_rule_fwd_h") == "state_space"
    assert census.family("pyhook::sglang.gdn_replayssm_spec_decode") == "state_space"
    assert census.family("pyhook::sglang.extend_attention_fwd") == "attention"
    assert census.HOOK_PREFIX == shape_hooks.PREFIX
    # Triton extend attention: q, k, v, o, token-major KV buffers, qo_indptr, kv_indptr, kv_indices, custom mask
    dims = [[52, 16, 256], [52, 2, 256], [52, 2, 256], [52, 16, 256], [3836544, 2, 256], [3836544, 2, 256],
            [14], [14], [1739304], [6957424]]
    assert census.attention_dims("pyhook::sglang.extend_attention_fwd", dims) == {
        "q_tokens": 52, "batch": 13, "kv_tokens_total": 1739304, "heads": 16, "head_dim": 256}


class _Scope:
    """Stand-in for torch's ``_RecordFunctionFast``: records (name, inputs) on entry."""
    entered: list = []

    def __init__(self, name, inputs):
        self.name, self.inputs = name, inputs

    def __enter__(self):
        _Scope.entered.append((self.name, self.inputs))

    def __exit__(self, *exc):
        return False


class _Tensor:
    def __init__(self, *shape):
        self.shape = shape


def test_wrapper_records_a_scope_only_while_profiling_and_forwards_the_call():
    state = {"profiling": False, "compiling": False}
    calls = []

    def fn(x, w, *, scale=None, out=None):
        calls.append((x, w, scale, out))
        return "result"

    hooked = shape_hooks.wrap(fn, "aiter.fn", record_function=_Scope, profiling=lambda: state["profiling"],
                              compiling=lambda: state["compiling"])
    _Scope.entered = []
    x, w, out = _Tensor(4, 8), _Tensor(16, 8), _Tensor(4, 16)
    assert hooked(x, w, scale=2.0, out=out) == "result" and _Scope.entered == []
    state["profiling"] = True
    assert hooked(x, w, scale=2.0, out=out) == "result"
    assert _Scope.entered == [("pyhook::aiter.fn", [x, w, out])]  # positional args, then tensor kwargs
    state["compiling"] = True  # under torch.compile tracing the wrapper only forwards
    hooked(x, w)
    assert len(_Scope.entered) == 1 and len(calls) == 3
    assert hooked.__wrapped__ is fn and hooked.__name__ == "fn" and hooked.__shape_hook__ == "pyhook::aiter.fn"


def _fake_wrapper(fn, label):
    return shape_hooks.wrap(fn, label, record_function=_Scope, profiling=lambda: True, compiling=lambda: False)


def test_import_hook_patches_the_defining_module_before_importers_bind_the_name(tmp_path, monkeypatch, capsys):
    pkg = tmp_path / "hookpkg_a"
    (pkg / "ops").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "ops" / "__init__.py").write_text("from hookpkg_a.ops.gemm import launch as launch  # re-export\n")
    (pkg / "ops" / "gemm.py").write_text(
        "def launch(x, w):\n    return ('launched', x, w)\n\n"
        "def caller(x, w):\n    return launch(x, w)\n\n"
        "class Box:\n    @staticmethod\n    def run(x):\n        return x\n")
    (pkg / "user.py").write_text("from hookpkg_a.ops import launch\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    shape_hooks.uninstall()
    try:
        targets = shape_hooks.install("hookpkg_a.ops.gemm:launch=gemm_launch,hookpkg_a.ops.gemm:Box.run,"
                                      "hookpkg_a.ops.gemm:missing", wrapper=_fake_wrapper)
        assert len(targets) == 3
        import hookpkg_a.ops.gemm as gemm
        import hookpkg_a.user as user
        assert gemm.launch.__shape_hook__ == "pyhook::gemm_launch"
        assert user.launch is gemm.launch  # the importer bound the hooked function
        assert gemm.Box.run.__shape_hook__ == "pyhook::hookpkg_a.run"
        assert gemm.Box().run(5) == 5 and isinstance(vars(gemm.Box)["run"], staticmethod)
        _Scope.entered = []
        assert gemm.caller(1, 2) == ("launched", 1, 2)  # in-module callers go through the module global
        assert _Scope.entered == [("pyhook::gemm_launch", [1, 2])]
        assert "hookpkg_a.ops.gemm:missing not found" in capsys.readouterr().err
        # aiter's old-path alias finder: importing the old name re-executes the module from its cached spec and
        # replaces it in sys.modules; the new copy must be hooked as well
        import importlib.util
        import types

        def alias_find_spec(fullname, path, target=None):
            if fullname == "hookpkg_a.ops.gemm_old":
                sys.modules[fullname] = importlib.import_module("hookpkg_a.ops.gemm")
                return importlib.util.find_spec("hookpkg_a.ops.gemm")
            return None

        alias = types.SimpleNamespace(find_spec=alias_find_spec)
        sys.meta_path.insert(0, alias)
        try:
            from hookpkg_a.ops.gemm_old import launch as old_launch
        finally:
            sys.meta_path.remove(alias)
        assert sys.modules["hookpkg_a.ops.gemm"] is not gemm  # the module ran again
        assert old_launch.__shape_hook__ == "pyhook::gemm_launch"
        assert sys.modules["hookpkg_a.ops.gemm"].launch.__shape_hook__ == "pyhook::gemm_launch"
    finally:
        shape_hooks.uninstall()
        for name in [n for n in sys.modules if n.startswith("hookpkg_a")]:
            del sys.modules[name]


def test_hooks_patch_a_module_imported_before_installation(tmp_path, monkeypatch):
    (tmp_path / "hookmod_b.py").write_text("def f(x):\n    return x + 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import hookmod_b
    shape_hooks.uninstall()
    try:
        shape_hooks.install("hookmod_b:f", wrapper=_fake_wrapper)
        assert hookmod_b.f.__shape_hook__ == "pyhook::hookmod_b.f" and hookmod_b.f(1) == 2
        shape_hooks.install("hookmod_b:f", wrapper=_fake_wrapper)  # idempotent: not wrapped twice
        assert hookmod_b.f.__wrapped__.__name__ == "f" and not hasattr(hookmod_b.f.__wrapped__, "__wrapped__")
    finally:
        shape_hooks.uninstall()
        del sys.modules["hookmod_b"]


def test_a_bad_hook_never_breaks_the_import(tmp_path, monkeypatch, capsys):
    (tmp_path / "hookmod_c.py").write_text("def f():\n    return 3\nCONST = 4\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    def broken(fn, label):
        raise RuntimeError("no torch")

    shape_hooks.uninstall()
    try:
        shape_hooks.install("hookmod_c:f,hookmod_c:CONST", wrapper=broken)
        import hookmod_c
        assert hookmod_c.f() == 3 and not hasattr(hookmod_c.f, "__shape_hook__")
        err = capsys.readouterr().err
        assert "hookmod_c:f not hooked" in err and "hookmod_c:CONST is not a function" in err
    finally:
        shape_hooks.uninstall()
        sys.modules.pop("hookmod_c", None)
    assert shape_hooks.install("not a spec") == []  # a malformed specification disables the hooks
    shape_hooks.uninstall()


def test_server_sitecustomize_installs_the_hooks_and_chains_to_the_shadowed_one(tmp_path):
    import os
    import subprocess
    import sys as _sys
    (tmp_path / "hookmod_d.py").write_text("def f(x):\n    return x\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("import builtins\nbuiltins.CHAINED = True\n")
    site = Path(shape_hooks.__file__).parent / shape_hooks.SITE_DIR
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(site), str(tmp_path), str(other)]),
               AKA_SHAPE_HOOKS="hookmod_d:f")
    code = "import builtins, hookmod_d; print(getattr(hookmod_d.f, '__shape_hook__', None), builtins.CHAINED)"
    out = subprocess.run([_sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert "installed 1 shape hook(s)" in out.stderr
    assert out.stdout.split()[-1] == "True"  # the shadowed sitecustomize still ran
    env.pop("AKA_SHAPE_HOOKS")
    out = subprocess.run([_sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.stdout.split() == ["None", "True"] and "shape_hooks" not in out.stderr  # off unless enabled


def test_torch_profiler_records_the_hook_scope_as_an_op_with_input_shapes(tmp_path):
    torch = pytest.importorskip("torch")

    def launch(x, w, scale=1.0):
        return (x @ w.t()) * scale

    hooked = shape_hooks.wrap(launch, "aiter.gemm_demo")
    x, w = torch.randn(4, 8), torch.randn(16, 8)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
        y = hooked(x, w, scale=2.0)
    assert torch.equal(y, launch(x, w, scale=2.0))
    trace = tmp_path / "t.json"
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    scope = next(e for e in events if e.get("name") == "pyhook::aiter.gemm_demo")
    assert scope["cat"] == "cpu_op" and scope["args"]["Input Dims"][:2] == [[4, 8], [16, 8]]
    assert census.gemm_shape(scope["name"], scope["args"]["Input Dims"], scope["args"]["Input type"])[:3] == \
        (4, 16, 8)
    hooked(x, w)  # the profiler is off: no scope, no error


def test_census_reads_hook_scopes_as_ops_and_counts_each_kernel_once(tmp_path):
    events = [
        {"cat": "gpu_user_annotation", "name": "step[DECODE bs=24]", "ts": 0, "dur": 100},
        # a hooked FP8 block-scale GEMM: split-K main kernel and its reduce kernel carry the scope's External id
        {"cat": "cpu_op", "name": "pyhook::aiter.gemm_a8w8_blockscale", "ts": 1, "args": {
            "External id": 9, "Input Dims": [[24, 7168], [4608, 7168], [24, 56], [36, 56], []],
            "Input type": ["c10::Float8_e4m3fn", "c10::Float8_e4m3fn", "float", "float", "Scalar"]}},
        {"cat": "kernel", "name": "_gemm_a8w8_blockscale_kernel_GROUP_K_128", "ts": 2, "dur": 6,
         "args": {"External id": 9}},
        {"cat": "kernel", "name": "_gemm_a8w8_blockscale_reduce_kernel", "ts": 9, "dur": 2,
         "args": {"External id": 9}},
        # an unhooked Python launch stays a shapeless row
        {"cat": "kernel", "name": "_fused_append_shared_experts_kernel", "ts": 20, "dur": 2, "args": {}},
    ]
    path = tmp_path / "w-TP-0.trace.json"
    path.write_text(json.dumps({"traceEvents": events}))
    acc = census.new_acc()
    stats = census.census_window(path, acc, None)
    out = census.finish(acc)
    assert stats["hooked_share"] == pytest.approx(0.8) and stats["linked_share"] == pytest.approx(0.8)
    assert sum(k["total_us"] for k in out["kernels"]) == pytest.approx(10.0) == out["total_gpu_ms"] * 1e3
    row = next(r for r in out["ops"] if r["op"] == "pyhook::aiter.gemm_a8w8_blockscale")
    assert (row["dims"][:2], row["op_calls"], row["total_us"], row["family"]) == \
        ([[24, 7168], [4608, 7168]], 1, 8.0, "gemm")
    assert [(g["M"], g["N"], g["K"], g["phase"], g["total_us"]) for g in out["gemm"]] == \
        [(24, 4608, 7168, "decode", 8.0)]


def test_capture_passes_shape_hooks_to_the_server(tmp_path):
    import argparse as ap_mod
    parser = ap_mod.ArgumentParser()
    capture.add_arguments(parser)
    base = ["--model-path", "m", "--model-name", "m", "--corpus", "c", "--image", "i", "--out", str(tmp_path),
            "--capture-id", "x"]
    a = parser.parse_args(base)
    assert a.shape_hooks == "default"
    a.shape_hooks = shape_hooks.canonical(a.shape_hooks)
    script = capture.server_script(a, tmp_path, [], "logs/server.log")
    assert f"export PYTHONPATH={capture.HOOKS_MOUNT}/{shape_hooks.SITE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}; " \
           "python3 -m sglang.launch_server" in script
    assert (capture.TOOL_DIR / shape_hooks.SITE_DIR / "sitecustomize.py").is_file()
    a = parser.parse_args(base + ["--shape-hooks", "none"])
    a.shape_hooks = shape_hooks.canonical(a.shape_hooks)
    assert "PYTHONPATH" not in capture.server_script(a, tmp_path, [], "logs/server.log")
    with pytest.raises(SystemExit, match="--shape-hooks"):
        capture.main(parser.parse_args(base + ["--shape-hooks", "not-a-target"]))


# --- Capture robustness ------------------------------------------------------------------------------

@pytest.mark.parametrize("alive, status", [(False, "server exited"), (True, "server unreachable")])
def test_a_server_gone_at_a_profiler_window_is_a_window_status_not_a_capture_error(tmp_path, monkeypatch,
                                                                                   alive, status):
    import urllib.error

    def refused(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr(capture, "http", refused)
    monkeypatch.setattr(capture, "running", lambda name: alive)
    monkeypatch.setattr(capture.time, "sleep", lambda s: None)
    w = capture.profile_window("http://127.0.0.1:1", tmp_path / "w1", 8, 100, "srv", 900, exit_wait=0)
    assert w["status"] == status and "Connection refused" in w["error"]
