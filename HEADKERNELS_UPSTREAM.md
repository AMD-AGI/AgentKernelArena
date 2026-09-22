# headkernel_ut_0914_full

A benchmark suite containing **exactly** the kernels in the 2026-09-14 head-kernel info
table — 37 rows across six models — normalized into AgentKernelArena tasks that all follow
one standard.

```
19 task directories, covering 26 of the 37 info rows
   all 19 pass static validation and have been run end to end on GPU
   18 are green on all three legs; 1 has an upstream oracle mismatch (see GPU results)
 6 NOT_BUILT placeholders with evidence   ( 6 rows - no editable kernel wired yet)
 5 gaps recorded in MANIFEST.tsv          ( 5 rows - no UT package at all)
```

Nothing here is synthesized. Every built task wraps a real GEAK op package with a live
capture behind it; every row that could not become a real task says so and says why.

## Where to look

| path | what |
|---|---|
| `MANIFEST.tsv` | all 37 info rows → task, or the reason there isn't one |
| `tools/manifest.json` | the source of truth the whole suite is generated from |
| `tasks/headkernel/<task>/` | one task per optimization seam |
| `_prior_solutions/<task>/` | tuned kernels that shipped in the upstream packages, **removed from the tasks** |
| `_runs/<timestamp>/` | GPU transcripts from `tools/run_on_gpu.sh` (pre-2026-09-15 ones came from a superseded runner — read `_results/` instead) |
| `_results/<task>/` | **the archived GPU verdict per task** — survives `clean.sh`, accumulates across reservations |
| `tools/summarize_runs.py` | one line per task: compile / correctness / performance |

## Task layout

Identical in all 19. `scripts/task_runner.py` and `scripts/_bench.py` are byte-identical
across tasks; everything task-specific lives in `config.yaml`.

```
config.yaml              arena schema (source_file_path, target_kernel_functions,
                         compile/correctness/performance_command, task_type)
                         + a headkernel: block with model, roofline, docker, provenance
README.md                what this kernel is and what to watch out for
scripts/task_runner.py   compile | correctness | performance
scripts/_bench.py        10 warmup + 100 measured timing over the captured records
source/                  THE EDITABLE KERNEL - the only thing an optimizer changes
ut/                      frozen GEAK op package: oracle, harness, baseline overlay
ut/kernel_src/           relative symlinks into source/ - same bytes, two views
```

The symlink is what makes one directory serve both worlds: the arena sees `source/`, the
GEAK harness sees `ut/kernel_src/`, and they are the same file. Nothing points outside the
task, so the self-contained check passes.

### The three modes

- **compile** — AST-parses every `source_file_path` and asserts each
  `target_kernel_functions` entry is *defined* there. Real symbol-table lookup, not a text
  search. Where a task declares `headkernel.preserve_symbols`, it also asserts those
  survive: some UTs resolve their frozen baseline through an extra symbol in the same file,
  and deleting it turns a working kernel into an unexplained correctness failure on the GPU.
  No GPU needed.
- **correctness** — runs `ut/unittest.py`: the op's oracle plus random-value parity against
  the live baseline leg, with whatever extra contract that op needs (physical stride, graph
  replay, output-buffer identity, elementwise-median repair). Exit 0 pass, 1 correctness
  fail, 2 environment, 3 harness incomplete.
- **performance** — three paths, in order. (1) Replay the oracle's captured argument
  records with **10 warmup + 100 measured** iterations and report the mean cuda-event device
  time, which is the arena's documented methodology. (2) For ops whose oracle is a runtime
  frozen baseline rather than a tensor blob, build the same live geometries through the
  package's own `ut/cases.py` and time those with the same 10/100 methodology. (3) Fall back
  to the GEAK interleaved median-of-3 legs, recording `methodology_is_arena_default: false`
  in `build/performance_report.json`.

  A performance run whose unit test did not pass reports **no** cases. It does not scrape
  numbers off a failed run, and it will not accept the `ut/result.json` the upstream package
  ships unless this run rewrote it — otherwise the packager's days-old numbers would be
  republished as a fresh measurement.

### Three kinds of oracle, never conflated

17 of the 19 tasks carry a frozen live-capture `ut/reference_io.pt` and judge against it.
Two — the dense BF16 GEMM cluster and the fused add+RMSNorm — are value-independent ops
whose packages deliberately store no tensor blob: they regenerate a deterministic baseline
at run time over the live shapes, strides and dispatch selections recorded in `ut/meta.json`,
and gate on a corruption negative control plus a two-leg identity check. `config.yaml`'s
`headkernel.oracle` says which kind a task has, and the generator will not describe one as
the other.

## Running it

```bash
bash tools/clean.sh                        # strip build/, __pycache__, overlays, ledgers
python3 tools/validate_suite.py            # static validation, no GPU

# GPU half. Resolve the job by NODE, not by a remembered job ID.
J=$(nssh -l | awk '$2=="crsuse2-m2m-192"{print $1}')
tools/run_on_gpu.sh "$J" all               # detached; poll the log it prints
tools/run_on_gpu.sh "$J" pending           # resume: only tasks not yet all-green
python3 tools/summarize_runs.py            # what actually passed, from _results/
```

Spur reservations roll over roughly every 40 minutes and take the running sweep with them,
while a full sweep takes hours. That is what `pending` and `_results/` are for: each task
archives its three reports the moment it finishes, and `pending` re-queues anything whose
compile/correctness/performance triple is not all `ok`. Run it again on whatever job is alive
now and it picks up where the last one died.

Validate after cleaning: a completed run leaves `build/*.json`, `__pycache__`,
`ut/_cand_overlay` and `ut/reports/` behind, and the arena treats those as shipped junk.
`tools/clean.sh` now **fails loudly** if it cannot remove something rather than printing
"cleaned" and exiting 0 — see the ownership note below.

`run_on_gpu.sh` reads each task's own image out of `config.yaml`. The six models do not
share one, and neither do the two Qwen3.8 deliveries: 10 tasks want
`sglang:v0.5.17-rocm720-mi35x-profilerfix`, 6 want
`sglang:v0.5.18-rocm720-mi35x-profilerfix` (the 5 Qwen3.8 tasks, whose `baseline_ref/*.orig`
was extracted from it, plus GLM-5.3-Flash's fused_moe, which needs a `RuntimeContext` API that
only exists from 0.5.18), and the 3 Kimi-K3 tasks want
`sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830`. It
pulls what the node lacks and redirects the Triton/FlyDSL/comgr/TileLang/tvm-ffi caches away
from `$HOME`, which is the same NFS export on every node and would otherwise pre-warm a
supposedly cold measurement.

Kernel-compilation caches (Triton, FlyDSL, comgr, TileLang, torch extensions) are
persisted to `_cache/` on the shared export rather than thrown away with each container.
The three TileLang DSA tasks spend 40+ minutes JIT-compiling before their first comparison,
and with a per-container cache every retry paid that again - on short reservations they
simply never finished. This does **not** warm the measurement: both `_bench.py` and the GEAK
legs run 10 warmup iterations before timing, so a cold vs warm *compile* cache changes setup
time, not the device time reported. `_cache/` is not part of a task and `clean.sh` leaves it
alone; delete it by hand if you want a truly cold compile.

It also sets `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`. The `sglang:v0.5.18` image ships only the
CPU build of tvm-ffi's optional torch↔dlpack addon, so every fresh subprocess tries to
compile the ROCm one, fails, and retries; a UT that forks per leg per draw turned that into
637 concurrent compiles and hung a node for 30 minutes. The addon is a host-side conversion
fast path, not the device kernel under measurement.

The run is detached on purpose. A correctness leg spawns a fresh subprocess per bucket per
leg, each paying a cold torch import; the smallest task in the suite takes ~6 minutes. An
attached run dies to the caller's timeout mid-measurement and leaves no report.

**Resolve the spur job ID at point of use, every time.** You may hold SEVERAL reservations
at once and the set changes every few minutes, so `nssh -l | awk 'NR>1{print $1}'` can hand
you a different node than the one your sweep is on. Pin one job id for the whole sweep.

Three things about this cluster that cost a lot of time to learn:

- **`$HOME` is the same NFS export on every node.** Two nodes reading
  `~/hk_run_*.log` see the same file, so a log is not evidence about a particular node.
  `_results/` is the reliable progress signal; `tools/summarize_runs.py` reads it.
- **Reservations can be shorter than a single correctness leg.** A detached driver
  (`setsid nohup`) often outlives its reservation, but not always. Re-run
  `tools/run_on_gpu.sh <job> pending` whenever you have a live job; progress accumulates.
  Before launching, check whether a container from an earlier attempt is *still running* on
  that task (`docker ps | grep ^hk_`) - two drivers on one task corrupt each other.
- **`docker pull` fails intermittently** with no useful error, and a failed pull used to
  SKIP the task outright. It now retries three times. Prefer a node that already has the
  image: `docker image inspect <img>`.

### The root-residue trap

`run_on_gpu.sh` runs each container as `-u 0` against a `no_root_squash` export, so every
file the three legs write lands owned by root. The 2026-09-14 sweep left 365 such files;
`clean.sh` could not delete them, reported success anyway, and `validate_suite.py` then
FAILed 12 of 15 tasks on `no_leaked_solution` while `build_suite.py --force` died with
`PermissionError`. Two changes close that loop: the container now `chown`s the task tree
back to the invoking uid as its last act, and `clean.sh` exits 1 with the surviving paths
listed. If you still hit it, clear it from a compute node:

```bash
docker run --rm -u 0 -v /shared_nfs:/shared_nfs --entrypoint /bin/bash <any image> \
  -c 'bash <suite>/tools/clean.sh'
```

Do **not** `chown -R` the whole task tree to fix this: 16 of the 17 `ut/reference_io.pt`
files are hardlinks into upstream packages owned by six other users, and chowning a hardlink
changes the owner of their file too. Target the run output instead, which is what
`clean.sh` does.

### GPU results

Every one of the 19 tasks has been run end to end on MI355X (gfx950) and its verdict archived
under `_results/<task>/`. `python3 tools/summarize_runs.py` prints the table; `_runs/` holds
the raw transcripts.

```
compile      19 / 19  PASS
correctness  18 / 19  PASS        (1 FAIL: glm-5.2-mxfp4__fused_moe_mxfp4_flydsl, below)
performance  19 / 19  produced measured cases   (87 timed cases total)
```

Fourteen of the nineteen are timed with the arena's own 10 warmup + 100 measured
cuda-event methodology - either replaying the frozen oracle records or, for the ops whose
oracle is a runtime baseline, building the live geometries through the package's own
`ut/cases.py`. The other five fall back to the GEAK interleaved legs and record
`methodology_is_arena_default: false` in `build/performance_report.json`.

**The one red leg is an upstream oracle problem, and the evidence says so precisely.**
`glm-5.2-mxfp4__fused_moe_mxfp4_flydsl` measures both legs cleanly (`GEAK_WEIGHTED_SPEEDUP
1.0002` - the null run you expect from a stock source) and its value-parity leg against the
**live** baseline passes on all six draws at err 0.11-0.17. What fails is the comparison
against the **frozen capture**: `[oracle] err 9.25 / 38.66 / 31.98` and `[eager] err 19.77 /
17.97`, 200-700x the tol of 0.05. Candidate and live baseline agree with each other and
disagree with `reference_io.pt`, so the blob no longer describes what this stack computes.
The image is not the cause - the upstream README names the same v0.5.17 image, and the two
sibling GLM-5.2 DSA tasks pass under it. The run log shows aiter falling back to `using
2stage default` and a heuristic FlyDSL config, and the package ships no tuning CSV, so the
most likely cause is a tuned MoE dispatch config that was present at capture time and is not
present now. It needs the package owner: ship that config, or re-capture the oracle.

Getting here required four fixes worth knowing about, all of them in this suite's tooling
rather than in any kernel:

| symptom | actual cause |
|---|---|
| `glm-5.2 fused_moe` refused to measure (exit 2) | `build_suite.py` had stripped a 15-line identity stamp the package documents as load-bearing, mistaking it for a prior tuning |
| `glm-5.2 dsa_decode` died SIGABRT with 0 checks | the upstream UT permutes the reference along a size-1 axis; a recorded one-character `ut_patches` entry fixes the axis |
| `glm-5.3 fused_moe` failed to boot | wrong image pinned (needs v0.5.18) plus a `glm5_next` arch patch, now both declared in the manifest |
| both DeepSeek MoE tasks produced 0 timed cases | the overlay loads `source/` as a top-level module, so the kernels' lazy `from .kernels...` imports had no parent package; `_bench.py` now restores `__package__` from the seam |

## Coverage

| model | rows | built | not built |
|---|---:|---:|---|
| DeepSeek-V4-Pro | 5 | 3 | 2 collectives |
| Qwen3.8-2.4T-A95B-MXFP4 | 11 | **11** | — |
| Kimi-K3 | 6 | 3 | 2 NOT_BUILT, 1 collective |
| GLM-5.3-Flash | 5 | 2 | 2 NOT_BUILT, 1 GAP (upstream package is itself NOT_BUILT) |
| MiniMax-M3-MXFP4 | 5 | 3 | 1 NOT_BUILT, 1 never captured |
| GLM-5.2-MXFP4 | 5 | 4 | 1 NOT_BUILT |
| **total** | **37** | **26** | 6 NOT_BUILT + 5 GAP |

One task carries a WARN rather than a PASS and it is honest: `qwen3.8-2.4t__paged_attention_decode`
has exactly one case geometry, where the standard asks for two or three. The live capture
yielded one decode shape (M=64, 521,351 referenced KV pages) and the package's own
`ledger_mapping.json` records `shape_authority: "live capture only"`. Inventing a second
shape would make the verdict green and the benchmark worse.

Qwen3.8 is complete as of the 2026-09-13 callable-UT delivery
(`/shared_nfs/zepingl/qwen38/HeadKernel/20260913T180016Z`), which closed all eight rows
Q38-4…Q38-11 — including `paged_attention_ll4mi_QKV_mfma16_kernel`, which had no UT package
anywhere before — and re-backed Q38-1/2/3 with a fresher capture whose frozen blob is the
actual pass/fail golden rather than shape evidence only.

Five info rows have no UT package anywhere and no directory in this suite: the three
collective kernels (`cross_device_reduce_2stage`, `allreduce_prototype_twoshot` ×2 — a
single-GPU op harness cannot represent them, they need a multi-rank capture),
`moe_flatmm_geamm1_ck` (never captured), and GLM-5.3's `tilelang_sparse_fwd` (the upstream
package carries a literal `NOT_BUILT` marker, evidence and a build recipe only).

Six more rows resolve to six `NOT_BUILT` placeholders. Every one of them is a profiled
symbol that lives in a **prebuilt vendor artifact** — hipBLASLt/Tensile `Cijk_*`, a CK
template instantiation — or a package that shipped an empty `kernel_src/`. There is nothing
for an optimizer to edit, so arena check 3 cannot pass. Each placeholder carries a README
with the measured numbers and the concrete steps to promote it.

Three of the six (`kimi-k3__dense_bf16_gemm_cijk`, `glm-5.3-flash__gemm_a16w16_bf16_cijk`,
`glm-5.2-mxfp4__dense_bf16_gemm_cijk`) hang off `aiter.tuned_gemm`, and their "nothing to
edit" half is now solved: a complete, un-tuned stock `aiter/tuned_gemm.py` ships in this
suite at `tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster/source/`. What still blocks
them is the bind. None of the three packages has a `candidate_bind`, and
`GLM-5.3-Flash_gemm_a16w16_bf16_Cijk_0913/meta.json`'s `rebind_seam_note` spells out the
trap: `aiter.tuned_gemm` builds `solMap` at import time holding direct function objects, so
a bare `setattr` on the module is a **dead** rebind and the dispatcher keeps calling the
original. Fix the bind and re-capture parity before promoting any of them.

## Two integrity decisions worth knowing about

**Prior answers were stripped.** Several packages shipped a `kernel_src/` that a previous
GEAK run had already tuned — DeepSeek MLA by 74 diff lines, MiniMax `gqa_share_sparse_fwd`
by 465. Seeding a benchmark from those means the task starts from someone else's answer and
its speedup is not comparable to any other task in the suite, so `source/` is seeded from
the stock `baseline_ref/*.orig` and the tuned version is kept at `_prior_solutions/<task>/`.
`_candidate_best/`, `accepted_overlay/`, tuning sweeps and patches were not copied at all.
Each task README states which case it is.

The five Qwen3.8 tasks are a third case. Three of them (`dense_bf16_gemm_cluster`,
`fused_recurrent_gated_delta_rule_decode`, `gemma_fused_add_rmsnorm`) are byte-identical to
the runtime-image source and are labelled `source_seed: stock`. The other two carry a
handful of unit-test lines on top of it -- 10 for the MoE dispatcher, 22 for paged
attention -- so the candidate overlay can still reach the unshadowed production function;
those are labelled `source_seed: stock+harness-shim`. Nothing is pre-optimized in any of the
five, which
means the first measurement on those tasks is by construction a null run — treat small
deviations from 1.00x as timing-slot bias, not optimization.

Two tasks have no stock counterpart because the optimization *is* a new module the seam
rebinds to (`kimi-k3__fwd_grouped_kernel_stage1`, `glm-5.3-flash__elementwise_copy_cluster`).
Their `source/` is a candidate implementation by construction; the baseline leg still
resolves to the live stack outside the task, so the measurement is sound, but the starting
point already encodes design choices from the capture.

**Oracle blobs are hardlinked.** `reference_io.pt` runs up to 7.3 GB and the suite has
seventeen of them. Sixteen are hardlinks into the source package on the same NFS export, so
they cost this suite no disk; the seventeenth
(`glm-5.3-flash__elementwise_copy_cluster`'s) is a few hundred KB, below the 8 MB threshold
at which `build_suite.py` tries to link at all -- it is 2.1 MB. Of the bytes under `tasks/`,
41.96 GB is hardlinked and already on disk in the upstream packages; only **0.03 GB (28 MB) is
unique to this suite**.

Because the inodes are shared with the upstream packages, a `chown -R` here would change the
owner of *their* files too. That is the real hazard, not the local ownership split.

A rebuild run as a normal user cannot create most of those links: `fs.protected_hardlinks`
forbids linking a file you neither own nor can write, and these belong to six different
people. `build_suite.py` copies instead and says so loudly — at build time and in the
generated task README — rather than quietly spending 40 GB on a 95%-full export. Run
`tools/relink_oracles.py` as root afterwards to convert them back; it refuses to alias a
file whose size or first/last megabyte differs, so a diverged oracle is never silently
replaced.

```bash
docker run --rm -u 0 -v /shared_nfs:/shared_nfs --entrypoint /bin/bash <any image> \
  -c 'python3 <suite>/tools/relink_oracles.py'
```

## Regenerating

```bash
python3 tools/build_suite.py --force        # rebuild every task from manifest.json
python3 tools/build_suite.py --only <task>  # just one
python3 tools/emit_manifest.py              # refresh MANIFEST.tsv
```

`tools/build_suite.py` is idempotent and the manifest is the only thing to edit — task
count, target symbols, per-package image and provenance all come from it. (`TARGETS` in
`build_suite.py` is the legacy home for the symbol lists and is still honoured for the
fifteen older tasks; a manifest row's own `targets` wins where it has one.)

Two package-level fixes the generator applies on copy, both recorded in the copied files:

- `ut/unittest.py`'s `RUN_ROOT` is repointed at the task. Upstream it is two directories
  above the UT, which is right in the delivery layout and lands in `tasks/headkernel/` here,
  dropping a `reports/ledger/<case>.json` tree next to the sibling tasks on every run. The
  same escape recorded in `ut/meta.json`'s `result_path` is rewritten too.
- absolute `baseline_overlay` pointers back into the delivery directory are made relative.

Files dropped on copy were each checked against every `.py` and `meta.json` in the source
package: `bench/` (a 111 MB serving transcript), `selection_evidence/`, `capture.log`,
`capture_{started,finished}.json`, `*.log`, plus the usual `__pycache__`/`_cand_overlay`
scaffolding. `capture_telemetry.json` and `attempts/` are **not** dropped and must not be —
`_verify_provenance()` opens them, and removing either turns a passing correctness leg into
`FileNotFoundError`.

## A caveat that applies to every number in the info table

Several of the e2e uplifts these tasks are annotated with did not survive scrutiny in the
source runs: GLM-5.3-Flash's reported +5.54% was traced to a measurement artefact (the only
kept `optimization_stack` entry contained zero runtime code changes), and yueliu's 0911
null-run controls showed several isolated speedups below ~1.03 were pure timing-slot bias.
Three Qwen3.8 rows (Q38-9/10/11) carry scenario estimates the info table itself marks as
未实测 — not measured — and the manifest repeats that label. `gpu_pct` 21.43 is a bucket
**shared** by Q38-5…Q38-8, not four independent shares; do not sum it. Each task's
`ut/README.md` is the authority on its own measurement caveats. Read it before trusting a
speedup.
