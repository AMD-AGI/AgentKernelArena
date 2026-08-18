# GLM-5.2-MXFP4 kernel tasks

抽取自 Hyperloom session `100127` / `GLM-5.2-MXFP4_20260814T163244Z`
（`/shared_nfs/hyperloom-claw/GLM-5.2-MXFP4/20260814T163244Z`），scored run
`runs/roofline/0c88d5c74d7d4031b1ed2f17bc8677b1/benchmark_sglang_20260814_190004`
—— 776 请求 · ISL 8192 / OSL 1024 / conc 64 · TP=8 · 8×MI355X · mxfp4(quark) ·
719.61 s · 1104.24 tok/s。

完整负载分析：
`/shared_nfs/jqliu/new-image-hot-kernels/hot-kernels-analysis/glm-5.2-hot-kernels.md`

| task | 算子 | **任务类型** | E2E 占比 | 模型里的位置 | 语言 | device kernel/次 | 状态 |
|---|---|---|---|---|---|---|---|
| [`mi355x_sglang_tilelang_dsa_sparse_mla_glm5`](#task-1--mi355x_sglang_tilelang_dsa_sparse_mla_glm5) | DSA 稀疏 MLA 注意力核 | **单算子** | **~40%** | 全部 78 层的 attention 主体 | TileLang | 2 | verified |
| [`mi355x_sglang_flydsl_mxfp4_moe_2stage_glm5`](#task-2--mi355x_sglang_flydsl_mxfp4_moe_2stage_glm5) | MoE 专家 FFN（两段 grouped GEMM + 排序/量化/归约） | **多算子连接** | **~19%** | 第 3–77 层（75 层）的 FFN | FlyDSL + HIP C++ | 7 | verified |

两个加起来约占端到端 GPU 时间的 **57%**。

### 单算子 vs 多算子连接的判据

两个 task 的粒度都是**一层**，都不止一个 device kernel，但性质不同 ——
区别在于这些 kernel 之间是「同一个算子的实现拆分」还是「不同算子串起来」：

* **Task 1 是单算子任务。** 2 个 kernel（`partial` + `combine`）是同一个 attention
  的 **split-K 分解**：把 2048 个 key 切成 `N_GROUPS` 组各算一个局部 online softmax，
  再按 LSE 合并。数学上就是一个 attention，拆成两个 kernel 纯粹是实现选择 ——
  prefill 下 `N_GROUPS=1` 时 combine 已经退化成一次拷贝。agent 完全可以把两者
  合并成一个 kernel 而不改变算子语义。
* **Task 2 是多算子连接任务。** 7 个 kernel 覆盖 **5 类不同的算子**：
  MXFP4 动态量化、按专家排序（拆成 P0/P23 两阶段）、grouped GEMM（stage-1/stage-2
  两段）、SiLU·mul 激活（融进 stage-1）、topk 加权归约。它们各有独立的功能契约，
  是一条真正的流水线，不是一个算子的拆分。

两种情况下都是**整条链一起计时**，所以跨内部 launch 的重构和融合都算收益。
kernel 数是在目标卡上 profile 实测的，不是推断。

### 实测：编辑生效 + 耗时

在**带 repo seed 的完整 workspace**（和 arena 的 `setup_workspace()` 一致：拷贝
`image_repo_path` 到 `<ws>/<repo_subdir>/` + 注入 perf helper）上跑过：

| task | compile | correctness | performance | 结果 |
|---|---|---|---|---|
| `dsa_sparse_mla` | 12 s | 13 s（×3 PASS） | 14 s | 0.0531 / 7.313 / 3.623 ms，全 `cuda_graph` |
| `mxfp4_moe_2stage` | 7 s | 9 s（×3 PASS） | 10 s | 0.1149 / 1.274 / 0.697 ms，全 `cuda_graph` |

最坏情况也远低于 5 分钟：DSA 清空 tilelang 全局缓存冷跑是 18 / 29 / 14 s；
MoE 改了 HIP 源码触发重编是 33 s。三个阶段的 `*_timeout` 都设成 **600 s**，
既是 2× 余量也是跑飞时的硬闸。

**编辑生效验证（往源码里注入数值扰动，看 correctness 是否因此失败）：**

| 语言 | 注入点 | 结果 |
|---|---|---|
| TileLang | `sparse_mla_fwd_decode_combine` 的累加 ×2 | correctness **FAIL** ✅ 生效 |
| FlyDSL | `mixed_moe_gemm_2stage.py:2012` `silu_elem` ×2 | correctness **FAIL**（rel err 1.00）✅ 生效 |
| HIP C++ | `quant_kernels.cu` 的 `absMax ×4` | 不 rebuild → **PASS**（跑旧 `.so`）<br>`AITER_REBUILD=1` → **FAIL**（rel err 0.67）✅ 生效 |

HIP 那一行是这两个 task 里唯一需要外力的：aiter 带预编译 `.so`，不 rebuild 就会
静默跑旧产物。`src/jit_rebuild.py` 会在 C/C++ 源码与镜像副本不一致时自动
arm `AITER_REBUILD=1`（`_sources_match_image()`），三个阶段
（`evaluator.py:52/87`、`performance.py:293`）都把它作为 `extra_env` 传给子进程 ——
已实测确认这条链通。而且它**不是全量 aiter 编译**：只重编本次运行实际加载的模块，
本任务是 `module_quant` + `module_moe_sorting_opus` 两个，33 秒。基线（未编辑）跑
不会触发任何重编。

### 实测：Arena + forge-loop 端到端

run config 在 `/shared_nfs/jqliu/run_arena/config.forge_glm5_{dsa,moe}.yaml`
（log/workspace 放 `/tmp`，理由同 `config.forge_mxfp4.yaml` 的注释）。
环境变量 `source /shared_nfs/jqliu/set_env.sh`，
启动 `python3 main.py --config_name <cfg>`。

两个 task 都跑到 forge-loop 的主业务里，没有秒挂：

| 检查点 | DSA | MoE |
|---|---|---|
| workspace seed + perf helper 注入 | ✅ | ✅ |
| baseline compile / performance | ✅ 11 s / 13.5 s | ✅ |
| arena baseline | 3 case，avg **3.6494 ms** | 3 case，avg **0.7167 ms** |
| fellow 解析 | `tilelang-fellow` ⚠️ 见下 | `flydsl-fellow` ✅ |
| `[prepare] task already conforms to the driver contract` | ✅ skip | ✅ skip |
| `Starting autonomous iteration loop` | ✅ | ✅ |
| forge 自测 baseline | **3.651 ms** | **0.714 ms** |
| **`pristine anchor agrees with the task reference on 3 of 3 cases`** | ✅ | ✅ |
| `[analysis] building commit-bound analysis bundle` | ✅ 进入 | ✅ 进入 |

`prepare` 直接 skip 说明这两个 harness 天生就符合 forge 的 driver 契约，不需要
forge 先改写任务；anchor 三个 case 全对上说明 forge 自己的 driver 调用和本 harness
测的是同一件事。

**两个已知问题：**

1. **`Unknown fellow backend 'tilelang'; falling back to 'flydsl-fellow'`（DSA）。**
   Arena 侧没问题 —— `src/prompts/cheatsheet/default_cheatsheet.yaml` 的 `knowledge`
   里有 `tilelang`。是 KernelForge 的 `loop/campaign_config.py` 只认
   flydsl / triton / hip / aiter / ck / hipblaslt，没有 tilelang，于是回落。
   属于 KernelForge 的既有缺口：已有的
   `image_kernel/mi355x_vllm_tilelang_mhc_fused_post_pre` 同样是
   `repository_language: tilelang`，行为一致。只是 warning，不影响跑通，但 agent 拿到
   的是 FlyDSL 的知识而不是 TileLang 的。

2. **MoE 第一次跑被 OOM kill（`Forge loop completed with exit code: -9`）。**
   两个原因叠加：当时机器上还有另一个同类 aiter MoE 任务在并发跑；以及
   `image_repo_exclude` 的路径写错了（见 `config.yaml` 里的注释），
   `aiter/jit/build` 的 2.9 GB 一直在被 seed 进每个 run 的 workspace。
   修掉路径后 workspace 从 **7.1 GB 降到 3.8 GB**，单独跑时内存稳在 ~50 GiB
   （cgroup 上限 128 GiB），顺利进入主循环。
   注意 forge 挂掉之后 arena 的集中评测仍然正常跑完 3/3 —— 也就是说这类崩溃
   会安静地退化成"没有优化"，不会报错，看 `exit code: -9` 才知道。

---

## 模型结构与两个 task 的位置

GLM-5.2（`GlmMoeDsaForCausalLM`，继承 `DeepseekV2ForCausalLM`）：
**78 层** · hidden **6144** · MLA(`q_lora_rank=2048, kv_lora_rank=512, qk_rope_head_dim=64`)
· 64 heads（TP=8 → **每卡 8 head**）· DSA indexer(`index_n_heads=32, index_head_dim=128,
index_topk=2048, index_topk_freq=4`) · MoE(256 routed + 1 shared, topk 8,
`moe_intermediate_size=2048` → 每卡 256) · `first_k_dense_replace=3`。

```
每一层 (× 78)
 x [T, 6144] bf16
  │
  ├─ RMSNorm
  ├─ MLA attention ─────────────────────────────────────────────────
  │    ├─ QKV-A 融合投影      6144 → 2624   (q_lora 2048 + kv_lora 512 + rope 64)
  │    ├─ q_b_proj            2048 → 2048   (8 head × qk_head_dim 256)
  │    ├─ fused_qk_rmsnorm  /  RoPE + 写 KV cache
  │    │
  │    ├─ DSA indexer  ── 只在 21 层跑（indexer_types 里的 "full"：第 0,1,2,6,10,…,74 层）
  │    │     indexer q 2048→4096 · k 6144→128 · weights 6144→32
  │    │     → Hadamard → fp8 量化 → paged MQA 打分 → top-2048
  │    │     产出 Indices [T, 2048] int32                    ← 选出每个 query 要看的 2048 个 KV
  │    │     （其余 57 层复用上一个 full 层的 Indices）
  │    │
  │    ├─ bmm  q_nope(192) × kv_b  → 512 latent
  │    ├─ ★★★ TASK 1：DSA 稀疏 MLA（sparse_mla_fwd_decode_partial + combine）★★★
  │    ├─ bmm  attn_out(512) × v_b → 256
  │    └─ o_proj              2048 → 6144
  ├─ TP all-reduce                                            （不在 task 范围，用户要求排除）
  ├─ add_rmsnorm_quant
  ├─ FFN ────────────────────────────────────────────────────────────
  │    ├─ 第 0–2 层  (dense)：普通 MLP，intermediate 12288（每卡 1536）
  │    └─ 第 3–77 层 (sparse, 共 75 层)：
  │          router gate 6144 → 256  →  top-8 + 拼上 shared expert  →  topk=9
  │          ★★★ TASK 2：aiter fused_moe（两段 grouped GEMM + 排序/量化/归约）★★★
  └─ TP all-reduce
```

---

## Task 1 — `mi355x_sglang_tilelang_dsa_sparse_mla_glm5`

> **任务类型：单算子。** 2 个 device kernel 是同一个 attention 的 split-K 分解
> （局部 softmax + LSE 合并），不是两个算子；可以合并成一个。

### 这是什么算子

**DeepSeek Sparse Attention (DSA) 的稀疏 MLA 注意力核。** GLM-5.2 的注意力是
MLA（吸收式，query 被投影进 512 维 latent 空间）叠加 DSA 稀疏化：indexer 先给每个
query 挑出 2048 个最相关的 KV 位置，这个 kernel 只对这 2048 个位置做 attention，
而不是对整个上下文。

它由**两个 kernel** 组成，一层跑一对：

| | 作用 | decode 内 | 单次耗时 |
|---|---|---|---|
| `sparse_mla_fwd_decode_partial` | 把 2048 个 key 切成 N_GROUPS 组，每组算一个局部 attention（online softmax），输出 partial output + LSE | **20.18%** | 66.43 µs |
| `sparse_mla_fwd_decode_combine` | 按 LSE 把各组的 partial output 加权合并成最终输出 | 1.28% | 4.21 µs |

> 两个 TileLang prim_func 都叫 `main`，编出来的 HIP kernel 都叫 `main_kernel`，
> 所以 trace 里它们是同一个名字、严格交替出现（66.9 / 4.4 / 65.8 / 4.2 …）。
> session 自带的所有报告都把两者之和当成一个 21.5% 的 kernel 报了。

### E2E 占比：**~40%**（区间 36–52%）

| | 占该阶段 GPU 时间 | 阶段占 E2E | 贡献 |
|---|---|---|---|
| decode | **21.46%**（实测：8 rank trace，128 个 `step[DECODE bs=64]`） | 43.9% | 9.4% |
| prefill | **~58%**（同卡按 session 的 prefill shape 回放实测：7.317 ms/层 × 78 层 = 570.7 ms，对 982.0 ms 的 prefill forward） | 52.9% | 30.7% |
| **合计** | | | **~40%**（区间 36–52%） |

两个阶段都走同一个入口 —— 这条 server.log 是硬证据：

```
[2026-08-14 19:00:23] Set DSA backends for bfloat16 KV Cache: prefill=tilelang, decode=tilelang
```

这是整个 session 里 E2E 占比最高的算子，比第二名高一倍多。

### 对应模型结构的哪个部分

**全部 78 层的 attention 主体。** 在上面的结构图里，它夹在
「q_nope × kv_b 的 absorb bmm」和「attn_out × v_b 的输出 bmm」之间 ——
也就是真正做 QK^T / softmax / PV 的那一步。

它**不包含**：QKV 投影、o_proj、RoPE、KV cache 写入、以及 DSA indexer
（indexer 是给它准备 `Indices` 的上游，21 层跑一次，另有 12 个小 kernel，
decode 内合计 7.6%，不在本 task 里）。

### shape（每卡，heads=8，latent 512+64=576，topk 2048）

| case | Q | Indices | KV pool | inner_iter / N_GROUPS |
|---|---|---|---|---|
| `decode-bs64` | `[1, 64, 8, 576]` bf16 | `[1, 64, 1, 2048]` int32 | `[1, 1810496, 1, 576]` bf16 | 4 / 8 |
| `prefill-16384` | `[1, 16384, 8, 576]` | `[1, 16384, 1, 2048]` | 同上 | 32 / 1 |
| `prefill-8192` | `[1, 8192, 8, 576]` | `[1, 8192, 1, 2048]` | 同上 | 32 / 1 |

K 是完整的 576 维 latent，V 是同一个张量的前 512 维；索引为负表示 padding，
必须从 softmax 里 mask 掉。

**索引的 locality 是 workload 的一部分。** 这个 kernel 是 gather-bound
（decode 下 34.4 TFLOP/s = 峰值的 1.4%，KV gather 2.27 TB/s），耗时取决于 top-k
索引集的局部性，不只是 shape。harness 让每个 query 从**自己序列的连续 slot 区间**里
因果地取（也就是 `topk_transform` 真正吐出来的东西）：decode = 64 条独立序列各 1 个新
token（无跨 query 复用）；prefill = 一个 16384 chunk 覆盖 2 条完整的 ISL-8192 序列
（大量 L2 复用）。改成在 181 万 slot 的池子里均匀随机取，prefill 会**慢 52%**
（11.13 ms vs 7.32 ms），测的是模型根本不会进入的 regime。

### 语言与源码

TileLang（Python DSL → TVM/TIR → HIP，JIT，改源码自动重编译），
`sglang/kernels/ops/attention/dsa/tilelang_kernel.py`
—— partial 在 `:811`，combine 在 `:989`，入口 `tilelang_sparse_fwd` 在 `:1317`。
**两个 kernel 在同一个文件里**，可以合并重构。

### headroom 与前车之鉴

decode 下 34.4 TFLOP/s（峰值 1.4%）、2.27 TB/s —— gather/延迟受限，不是算力受限。
prefill 下 `inner_iter=32` → `N_GROUPS=1`，combine 退化成 32768 个 block 去 reduce
单个 split，是纯浪费。`_pick_inner_iter`（`tilelang_kernel.py:75`）是只有 2 的幂档位的
粗启发式，`block_I=64 / threads=256 / block_per_cu=2` 三个常数没针对 gfx950 搜过。

session 里 GEAK 花了约 10 小时在一个叫 `dsa_sparse_attn_prefill_main_kernel_task`
的任务上，但 `opbench_result.json` 是 `isolated_speedup: 0.0` / `winner_ms: null` /
`winner_editable: false` —— 它从没 benchmark 过这个 kernel，而是转去调 server flag。
`reports/kernel_optimization_summary.json` 写的是 `kernel_opt_outcome: "skip"` /
`by_kernel: []` / *"No kernels were attempted"*。那个 authored-Triton 替换
（`final/overlay/dsa_authored_c0_triton.py`）挂在
`HIP error: operation not permitted when stream is capturing`。
**这里的 kernel 级 headroom 从来没被真正测过**；写出来的东西必须仍然能被 CUDA graph capture。

---

## Task 2 — `mi355x_sglang_flydsl_mxfp4_moe_2stage_glm5`

> **任务类型：多算子连接。** 7 个 device kernel 覆盖 5 类不同算子
> （量化 / 排序 / grouped GEMM / 激活 / 归约），是一条流水线；跨算子融合算收益。

### 这是什么算子

**MoE 专家前馈网络的完整一层**，即 `aiter.fused_moe`。被计时的是整条链，
一次调用启动 **7 个 device kernel**（在目标卡上 profile 实测）：

| kernel | 作用 | 语言 | 源码 | decode 内 |
|---|---|---|---|---|
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32` | **stage-1 grouped GEMM**：`6144 → 512`（gate+up），带 SiLU·mul | FlyDSL | `ops/flydsl/kernels/mixed_moe_gemm_2stage.py:313` | 15.08% |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_..._acc0` | **stage-2 grouped GEMM**：`256 → 6144`（down） | FlyDSL | 同文件 `:7432` | 7.86% |
| `fused_mx_quant_moe_sort_kernel<bf16,fp4_t,256,32>` | stage-1 输入的 MXFP4 动态量化 + 按专家排序 | HIP C++ | `csrc/kernels/quant_kernels.cu` | 2.32% |
| `fused_mx_quant_moe_sort_kernel<bf16,fp4_t,64,8>` | stage-2 输入（stage-1 输出）的量化 + 排序 | HIP C++ | 同文件 | 1.79% |
| `opus_moe_sorting_entry<P23>` | moe sorting 第二阶段 | HIP C++ | `csrc/include/moe_sorting_opus.h` | 1.30% |
| `moe_reduction_kernel_plain_bf16_topk9_md6144` | 按 topk 权重把 9 份专家输出归约回 hidden | FlyDSL | `ops/flydsl/kernels/moe_gemm_2stage.py:3591` | 1.29% |
| `opus_moe_sorting_entry<P0_v2>` | moe sorting 第一阶段 | HIP C++ | 同文件 | 1.22% |
| | | | | **30.86%** |

整条链一起计时，所以跨 launch 融合（两次 sorting 合一、两次 mx-quant 合一）算收益。

### E2E 占比：**~19%**

| | 占该阶段 GPU 时间 | 阶段占 E2E | 贡献 |
|---|---|---|---|
| decode | **30.86%**（实测） | 43.9% | 13.5% |
| prefill | **9.64%**（aiter 自带 GLM-5 调优表 TP=8 行 @token=16384：us1 367.41 + us2 894.91，× 75 层 = 94.7 ms / 982.0 ms；本机回放 1318 µs/层 = 98.9 ms，差 4.4%） | 52.9% | 5.1% |
| **合计** | | | **~19%** |

### 对应模型结构的哪个部分

**第 3–77 层（共 75 层）的 FFN。** 前 3 层是 dense MLP（`first_k_dense_replace=3`），
不走这条路。

具体是 router 选完专家之后的那一段：

```
router gate 6144→256  →  top-8 routed + 拼上 shared expert  →  topk_ids [T, 9]
                                                    ↓
                              ★ 本 task：fused_moe(x, w1, w2, topk_weight, topk_ids) ★
                                    量化+排序 → stage-1 GEMM → SiLU·mul
                                              → 量化+排序 → stage-2 GEMM → 加权归约
                                                    ↓
                                              out [T, 6144]
```

**不包含**上游的 `grouped_topk`（1.41%）、`_fused_append_shared_experts`（1.22%）
和 router gate GEMM（1.92%）—— 它们在 sglang 的 `select_experts` 里，
在 `fused_moe` 之前跑。harness 直接合成 `topk_ids` / `topk_weight`，
结构与它们的输出一致（8 个 routed + 第 9 列固定是 shared expert 256）。

### shape（每卡 TP=8）

`model_dim=6144` · `inter_dim=256` · `experts=257`（256 routed + 1 融合的 shared）
· `topk=9` · MXFP4 `group_size=32` · SiLU · bf16 输出。

这是 **afp4_wfp4**：权重和激活**都是** MXFP4（权重来自 quark checkpoint 预量化，
激活在 `fused_moe` 内部动态量化）。跟 Kimi-K3 那个 task 的 a16w4 不是一条路 ——
GLM-5.2 走 `shuffle_weight(w, (16,16))` + `e8m0_shuffle(scale)`，抄自 sglang
`quark_w4a4_mxfp4_moe.py:562-597`。

| case | x | M_routed |
|---|---|---|
| `decode-t64` | `[64, 6144]` bf16 | 576 |
| `prefill-t16384` | `[16384, 6144]` bf16 | 147,456 |
| `prefill-t8192` | `[8192, 6144]` bf16 | 73,728 |

权重形状（用 aiter 自己的 `dynamic_mxfp4_quant` 构造，与 trace 记录逐位一致）：

```
w1       [257,  512, 3072]  float4_e2m1fn_x2     w1_scale [257,  512, 192]  float8_e8m0fnu
w2       [257, 6144,  128]  float4_e2m1fn_x2     w2_scale [257, 6144,   8]  float8_e8m0fnu
```

### 语言与源码

**两种语言**（见上表的源码列）：

- **FlyDSL**（aiter 的 Python DSL → MFMA 汇编，JIT，改源码自动重编译）—— 3 个 kernel
- **HIP C++**（aiter 带预编译 `.so`，编辑会被静默忽略）—— 4 个 kernel。
  `src/jit_rebuild.py` 只在 `source_file_path` 含 C/C++ 扩展名时才 arm
  `AITER_REBUILD=1` + 清理 stale `.so`，所以 `config.yaml` 里那两个 `csrc` 条目是
  **功能性的、不是文档**。

入口 `aiter/fused_moe.py:441`。`repository_language` 只能填一个值（框架用它选
cheatsheet / forge fellow），填的是 `flydsl`，因为时间大头（22.94% / 30.86%）
在两个 FlyDSL GEMM 上。

> 注：`editable_sources` 这个 key 框架**根本不读**（`src/` 和 `main.py` 里零引用）。
> 真正的编辑面是整个被拷贝的 repo；该 key 只用来记录作者意图。

### headroom

**stage-2 在大 M 上是坏的。** aiter 自己的 GLM-5 调优表
（`configs/model_configs/glm5_fp4_tuned_fmoe.csv`，TP=8 行 `inter_dim=256`）：

| token | stage-1 µs | stage-2 µs |
|---|---|---|
| 4096 | 149.08 | 239.42 |
| 8192 | 222.97 | 452.37 |
| 16384 | **367.41** | **894.91** |

stage-2 的 FLOP 只有 stage-1 的一半却慢 2.4 倍，而且超线性。调优器在大 M 上挑的是
`flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_bnt2_sbm12`。decode-only 的 trace
看不见这个，所以之前没人发现。

**decode 离峰值差 16 倍。** M = 64 × 9 = 576 行摊到 257 个专家上，平均每个 2 行：
stage-1 是 70 TFLOP/s，prefill shape 下是 1102 TFLOP/s。

---

## 两个 task 的共同约定

1. **shape 全部来自 session 实测。** 每个 case 都是 scored run 真正跑过的桶，
   取自 8 rank torch trace、CUDA-graph capture trace 的 `record_shapes` 和
   `server.log`。算子在两个阶段都跑的，两个阶段都覆盖。
2. **精度和性能跑同一套 shape。** harness 里没有 correctness/performance 分支 ——
   被计时的 shape 就是被校验的 shape。只有 `compile` 走
   `_compile_smoke_case` 缩小。
3. **性能用 CUDA graph 测。** 用框架的 `_benchmark_cuda_graph_or_events` +
   `_TimedRun`，被计时的那次就是 graph replay，而且**测完之后会扰动输入、
   把输出填 NaN、重放同一个 graph 再校验一次** —— 想在被计时的那条路上偷工减料会挂。
   六个 case 全部报 `benchmark_method: cuda_graph`，没有回落到 event timing。
4. **ref 用 torch 且向量化。** 全 shape correctness 几秒跑完。
   * **DSA**：按 128 query 分块 gather + 每块两个 batched einsum，无 per-query 循环。
     语义照着 `tilelang_kernel.py:900-968` 读出来的 —— K 是完整 576 维 latent，
     V 是**同一个张量**的前 512 维，`sm_scale` 乘在原始 QK 点积上，
     **整行被 mask 掉时输出 0 而不是 NaN**。容差 `atol = rtol = 0.05`
     （bf16 在 2048 个 key 上累加）。
   * **MoE**：直接用 aiter 自己验证 `fused_moe` 用的 `torch_moe_stage1` /
     `torch_moe_stage2`，每段一个 grouped GEMM，不循环 token/expert。
     门限 `max_relerr = 0.01`（实测 0.0016）。

MoE 的容差有个坑：ref 里两次激活量化必须用
`aiter.get_torch_quant(QuantType.per_1x32)`，**不能**换成
`fp4_utils.dynamic_mxfp4_quant` —— 两者对 e8m0 block scale 的舍入不同，换了之后
mean relative error 从 0.0016 涨到 0.15，看着像 kernel 有 bug 其实不是。门限 0.01。

---

## 没做的算子和取舍

候选排序来自上面那份负载分析。跳过的：

* **TP all-reduce（E2E ~8.8%）** —— 按要求排除。顺带记一笔：它其实是两套实现，
  decode 走 aiter `cross_device_reduce_2stage`，prefill 的张量超过 16 MiB 的
  `_MAX_CAR_SIZE` 会回落 RCCL。
* **MoE 辅助族** —— 故意不单独建 task。8 个辅助 kernel 里有 5 个（decode 7.92%）
  就在 `fused_moe` 内部，已经在 Task 2 的计时区里，单独建会重复计分。
  另外 3 个（`grouped_topk` 1.41%、`_fused_append_shared_experts` 1.22%、
  router gate GEMM 1.92%）在 sglang 的 `select_experts` 里，确实没覆盖 ——
  合计 decode 4.55%、E2E ~2%，不值得单独长跑。
* **`Cijk_*` / hipBLASLt（decode ~9.8%）** —— `/opt/rocm/lib/hipblaslt/library/`
  下的预编译 Tensile 汇编，不在镜像 repo 里，构造不出 `image_kernel` 任务。
  要做只能做成把 `aten::mm`/`bmm` 路由到 FlyDSL 的派发任务。
* **`aten::mul` 权重缩放（decode 2.83%）** —— 这是框架 bug 不是 kernel 问题：
  `[8,192,512]` 和 `[8,512,256]` 是 kv_b/v_b 的**权重**，每层每次 forward 都被重新
  乘一遍标量。该在权重加载时折进去。
* **DSA indexer paged-MQA-logits（decode 1.55%，E2E ~1.5–3%）** —— **没建**。
  kernel 本身能跑（`aiter/ops/triton/gluon/pa_mqa_logits.py:351`，实测 29.3 µs
  eager，shape 为 `q[64,32,128]` fp8 / `kv[28289,64,1,132]` fp8 /
  `weights[64,32]` fp32 / `block_tables[64,177]`，max_seq_len 11328），但它跑在
  `Preshuffle=True` 模式，block 内的 KV 布局由
  `aiter::indexer_k_quant_and_cache_kernel` 写出，aiter 既没暴露对应 helper
  也没有 torch 参考，手写的 ref 对不上。要做得驱动真正的 cache writer 来构造 KV。
  在这个 E2E 份额下优先级低。

---

## 关于 `sync-perf-helpers`

`src/perf_helper_materialization.py:image_kernel_targets` 只 glob
`tasks/image_kernel/*/scripts/task_runner.py`，所以
`python src/tools/sync_perf_helpers.py --check` 看不到本目录。运行时不受影响 ——
`materialize_perf_helpers_in_workspace()` 是对拷贝出来的 workspace 操作、与路径无关，
两个 task 就是这么验证的。要纳入 `--check`，要么把目录挪到 `tasks/image_kernel/` 下，
要么把那条 glob 放宽成 `tasks/*/*/scripts/task_runner.py`。
