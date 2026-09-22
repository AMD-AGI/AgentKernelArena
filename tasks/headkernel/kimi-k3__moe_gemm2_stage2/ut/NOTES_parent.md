# Kimi-K3 · `moe_gemm1_0` / `moe_gemm2_0` — aiter FlyDSL mixed MoE GEMM

**合计 22.59% GPU（e2e 加权口径：13.53% + 9.06%）· aiter FlyDSL（Python DSL，可改）· ✅ 可改**

> **GEAK 碰了，但只到 Tier-B 调优层，三条路全部 infra 报错；Tier-C 写 kernel 那层因 wall-clock 没轮到。
> 按 e2e 加权口径，这是 Kimi 侧最大的一块可写代码的开销。**

## 两个身份、两套 kernel 名

同一对 stage-1 / stage-2 MoE GEMM，在两次 profile 里显示成不同的名字（不同的 tile 配置 / 量化组合）：

| profile | stage-1 | %GPU | stage-2 | %GPU |
|---|---|---:|---|---:|
| 0828 cycle1（e2e 加权） | `moe_gemm1_0` | **13.53** | `moe_gemm2_0` | **9.06** |
| 0830 cycle0（采样窗口） | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t64x128x256_pm1_fp8q_sort_a` | **7.73** | `mfma_moe2_afp8_wfp4_bf16_cshuffle_t64x256x128_vscale_fix3_fp` | **6.76** |

0828 口径细节：
- `moe_gemm1_0`：99176 calls，total 10624.76 ms，prefill 4753.041 / decode_e2e 5871.718 ms，
  prefill 956.731 us/call、decode 35.12 us/call，相位 **prefill 53.5 / decode 46.5**
- `moe_gemm2_0`：99176 calls，total 7111.402 ms，prefill 3802.756 / decode_e2e 3308.646 ms，
  prefill 765.45 us/call、decode 35.12 us/call，相位同上

## 源码位置（可改）

```
stage1 : /sgl-workspace/aiter/aiter/fused_moe.py:1169   (_flydsl_stage1_wrapper)
stage2 : /sgl-workspace/aiter/aiter/fused_moe.py:1243   (_flydsl_stage2_wrapper)
kernel : /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
dispatch: /sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py
```
（`compile_mixed_moe_gemm1` / `compile_mixed_moe_gemm2`）

## 真实 workload 中的 shape

模型侧常量（`Kimi-K3/config.json`）：
```
num_experts              896
num_experts_per_token    16
num_shared_experts       2
routed_expert_hidden_size 3584
hidden_size              7168
```

profile 捕获到的算子形状（**prefill**，0828 口径 `[8192, ...]`；0830 口径是 chunk 粒度）：

0828 cycle1（chunked-prefill 8192）：
```
[[8192, 3584], [896, 768, 1792], [896, 3584, 192], [8192, 16], [8192, 16], [688128, 112], [3211264, 16]]
dtypes: Scalar, c10::BFloat16, c10::Float4_e2m1fn_x2, float, int, unsigned char
```

0830 cycle0（同一算子，chunk token 数不整齐，捕到 5 个 M）：
```
M ∈ {1111, 1146, 1349, 1486, 1533}
[[M, 3584], [896, 768, 1792], [896, 3584, 192], [M, 16], [M, 16], [688128, 112], [3211264, 16]]
```

- `[M, 3584]` bf16 activation（3584 = routed_expert_hidden_size）
- `[896, 768, 1792]` **Float4_e2m1fn_x2** 权重（896 expert，mxfp4 packed）
- `[896, 3584, 192]` unsigned char = mxfp4 的 block scale
- `[M, 16]` × 2 = topk id + topk weight（`num_experts_per_token = 16`）
- decode 时 `M = conc(64)`

**decode M = 64，prefill M ≈ 1037–2659（0830，chunk 8192 但实际 token 数不整齐）或 8192（0828）。**

## 已知的抓手

GEAK 在 architect report 里点名的一条**现场 warning**：

> `mfma_moe1_silu_mul_afp8_wfp4_fp8`：`tile_n=256` **除不尽** `inter_dim=384`

也就是说当前 FlyDSL 的 tile 配置和这个模型的 intermediate dim 对不齐，尾块浪费。
**这是最明确的一个 config lever，改 tile 就行，不用重写 kernel。**

> **更正（0902，实测）**：这是**纯性能** lever，不是正确性风险。
> `resolve_flydsl_stage1_tile_n(384, 256)` 会把 256 **静默降到 128**（`moe_kernels.py:119`），
> 而 tiling 在数学上中性 —— `ut_stage1` 里实测 `tile_n=256` 在 M=8192 上 cos 0.99970、
> 在 M=1 上**逐位相同**，correctness UT 照过。所以"一个 tuned config 写着 256、实际跑 128"
> 这件事只能在 **timing 腿**上判，别拿它当正确性对照。
> 另注：**线上 0828 的 stage-1 变体 `tile_n=64`，能整除 384，没有这个尾块问题**；
> 那条 warning 点的是 `mfma_moe1_*`（afp8 路），不是线上这个。

其它 lever：
- FlyDSL kernel 源码本身（`mixed_moe_gemm_2stage.py`）—— Python DSL，可读可改
- aiter `tuned_fmoe` / `AITER_CONFIG` 调优 DB
- `--moe-runner-backend` 换 backend
- 把 `moe_reduction_kernel` 折进 stage-2 的 epilogue（GEAK 给 stage-2 的建议）

## 0828 那次 GEAK 到底做了什么（三次 infra 失败）

Phase 7 TuningSkill 花了 1h56m，其中 **1h10m 给了 h0 dense GEMM（最终 −0.038%）**，
留给这两个 MoE hypothesis（h1/h2）的只有 **16 分钟**，三条路全挂在 infra 上：

| 路子 | 结局 |
|---|---|
| aiter CK 两段 MoE tuner（默认） | `ValueError: Unsupported data type combination: b16, fp4x2`；ASM 清单 `hsa/gfx950/fmoe_2stages/fmoe_stage1_bf16_pertoken_g1u1.csv` 缺失 |
| aiter MoE tuner `--mxfp4-flydsl` | `no codegen'd instance for shape key 'aux_sort3s_NE896_TOPK16_MB128'`；而且 dtype family 是 `a4w4`，不是线上的 `abf16_wfp4` |
| aiter MoE tuner `TUNE_ONLY=flydsl` | `gemm_moe_tune.py:~3998` 一个没文档的闸门，是绕过上面那个 CK `ValueError` 的唯一口子；派了 1120 个 task，8 个 mp worker 全部 `Memory access fault by GPU node-N`，父进程死锁，最后按 pid 杀掉 |
| flydsl Tier-C 手写（`mixed_moe_gemm_2stage.py`） | ⊘ 没派 —— HeadKernel 从没被 dispatch（wall-clock） |
| `--moe-runner-backend` 换后端 | ⊘ 放弃 —— 属 config 轴，`CONFIG_TUNE_ENABLED=false` |

**值得试的一条**：`TUNE_ONLY=flydsl` 那次挂的是 8 worker 并发下的 GPU mem fault。
降到 1–2 个 worker 有可能直接跑通 —— 那样 22.59% 不用写 kernel 就有解。

seam 可达性（GEAK 自己的结论）：这两个是 **fused kernel，不是独立 GEMM**，
派发链 `aiter.fused_moe:fused_moe_2stages → _flydsl_stage1_wrapper → aiter.ops.flydsl.moe_kernels:flydsl_moe_stage1`，
**没有 `gemm(XQ, WQ, x_scale, w_scale)` 这样的调用点**，所以 `standalone-gemm-swap`
和 `dense-linear-env-overlay` 两种 lever 绑不到东西（`no_engagement`）。
可行的只有 `fused-op-tune-hook`（Tier-B）和 `author-fused-replacement`（Tier-C，flydsl）。

## 线上现场数据（0828 cycle1 抓到的）

kernel 名：
```
flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_xcd4_kw2
flydsl_moe2_abf16_wfp4_bf16_t32x256x128_atomic_bnt2_persist
```
shape key：`('gfx950', 256, 64, 3584, 384, 896, 16, Situv2, bf16, bf16, fp4_e2m1, per_1x32, True, False)`

- stage-1：K=3584 → N_shard=768，每 expert 1.46 MB mxfp4
- stage-2：K_shard=384 → N=3584，每 expert 0.73 MB
- 每层实际碰到 **~610 / 896 个 expert**
- token 桶 `{1, 2, 4, 8, 16, 32, 64, 512, 8192}`
- **decode 单次耗时是双峰的：27.8 µs vs 122 µs**（shared/dense-expert 层组 vs routed-expert 层组）——
  任何 per-shape 调优和任何 UT 的权重模型都必须同时覆盖这两个模式

shape capture 活下来了，在 `tuning/work/untuned_fmoe_live.csv`：
```
token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1
64,  3584,384,896,16,ActivationType.Situv2,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
8192,3584,384,896,16,ActivationType.Situv2,torch.bfloat16,torch.bfloat16,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
```
同目录的 `tuned_fmoe_live.csv` **只有表头、0 行** —— tuner 在写出任何结果前就死了的直接证据。

## unit test 现状：两级都有了（stage-1 于 0902 补齐）

| | UT | 说明 |
|---|:---:|---|
| `moe_gemm2_0`（stage-2，9.06%） | ✅ | `ut_stage2/`，GEAK 原生 bundle，**带真 oracle 181 M**（`synthesized: false`），来自 0817 session。几何量与 0828 profile 逐条对上。**权重要自己重算**（0817 是 chunk 16384，我们是 8192）——见 `ut_stage2/README.md` |
| `moe_gemm1_0`（stage-1，13.53%） | ✅ | `ut_stage1/`，**0902 自建**（claw 树里从来没有过）。oracle 走离线路线：复用 0817 stage-2 seam 上的**真 routing** 反解 + 生产 sorter 重排，不起 8 卡 server。空跑 `GEAK_CORRECT PASS` / weighted 0.99925，负向对照（`act="silu"`）三桶全灭。**注意 golden 是 21 发 median** —— 这个 kernel 单发不可复现，见 `ut_stage1/README.md` |

### ⚠️ stage-1 kernel 本身不是 run-to-run 可复现的（0902 实测，独立于 UT 的发现）

同输入同 kernel 连续两发，~0.5–1% 元素不同（relL2 ~2e-2），onset 在 sorted expert block
数 >~50 之后 —— **线上 decode(469 blocks) 和 prefill(4535 blocks) 两个桶都在区间内**。
越界读/越界写/未写元素/pad marker/全部 launch 旋钮/权重 layout/调用约定/机器/冻结副本
逐条排除（清单见 `ut_stage1/README.md`）。旁证：aiter 自己的
`op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py` 就是拿 cosine + `atol=1.0` +
`pass_pct=95.0` gate 这条路的，上游本来就没当它逐元素可复现。
**任何拿这个算子做 A/B 的人都要知道这件事**：单发对比里 ~2e-2 的差是底噪，不是收益也不是回归。
`ut_stage1` 的对策是逐元素取 21 发 median（可逐位复现），timing 腿仍用单发。

**0902 追加定论：那不是浮点抖动，是 aiter 的 bug，线上在吃。**
`ut_stage1/_l2_reference.py` 拿 aiter 自己的 `torch_moe_stage1` 作独立参照
（同契约：`Situv2` + `situ_beta=situ_linear_beta=1.0` + `per_1x32` + `a1_scale=None`）实测：
median 对 torch 的 relL2 是 **2.5e-05 / 5.3e-05 / 7.1e-05**（cos 1.000000），
而**两发打架的元素上 torch 站 median 而不站离群值** ——
`prefill_M8192` 6,110,007 个争议元素里 99.99996%，`decode_M64` 40,239 个里 100.0000%，
站离群值的是 **0 个**；同一批元素上离群值差 **244%–280%**。
而且 median 在争议元素上的相对误差（6.6e-07）和在无争议元素上（7.2e-07）**一样**。
所以：多数值数学上正确，离群值是损坏；median 是**合法修复**不是掩盖；
**生产单发路径每次 forward 都有 ~0.5–1% 的 stage-1 输出错 O(100%)，两个桶都中。**
`M=1` 零争议且照样对得上 torch → 是**并发效应，不是公式错**。
全精度数据在 `ut_stage1/_l2_report.json` 和 `meta.l2_independent_reference`。

### stage-1 为什么一直没有

不是"没人想到"，是**每次排序都排在预算边界外面**。0817 `strategy.md` 的优先级
= `phase_weight × pct_in_phase × (1 − 1/plausible_speedup)`：

| # | id | 目标 | e2e 加权 |
|---|---|---|---:|
| 1 | k0 | 把 `_score_kernel` fuse 进 `_combine_kernel` | ~3.5% |
| 2 | h0 | decode attention stage1+stage2 | ~3.7% |
| 3 | **h1** | **MoE stage-2 + 把 `moe_reduction_kernel` 折进 cshuffle epilogue** | **~2.8%** |
| 4 | **h2** | **MoE stage-1（gate/up + silu_mul + fp8 requant）tile/pipeline + prologue fusion** | **~2.1%** |
| 5 | h3 | dense bf16 GEMM `Cijk_*` 字节缩减 | ~2.7%（高风险） |
| 6 | k1 | KDA / gated-delta Triton 簇 | ~0.7% |

stage-1 的**原始占比更高**（`%gpu` 6.3 vs 5.2；roofline 预期收益 4.29 vs 2.92），
但 **e2e 加权 Amdahl 更低**，因为 stage-2 多一个 stage-1 没有的结构性杠杆 ——
`moe_reduction_kernel` 可以折进 cshuffle epilogue。于是 h1 被派了，h2 没有。

0817 `final_report.md` 的原话：
> `:262` h2 MoE stage-1 … planned in `strategy.md` | never dispatched | budget consumed by baseline/profiling wall-clock
> `:276` MoE stage-1 `mfma_moe1_*` (h2) | 8.7% of prefill GPU | never extracted | no | out of wall clock; latency-bound at 29% of the fp8 roof

**这个模式跨 run 重复**：stage-1 稳定排在 ~#4，稳定被 wall-clock 砍掉。0828 更极端 ——
连 Tier-C 都没开，h1/h2 一起死在 16 分钟里。

claw 树里确实有三个 stage-1 的 bundle（0816 `aiter_fused_moe_a8w4`、0819
`mfma_moe1_mfma_moe2_decode`、0821 `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256`），
但**全都是 afp8 激活路**，不是线上的 `abf16_wfp4`。
它们可以当**结构参考**（seam 怎么认证、case 怎么组织、routing 怎么喂进去），
**不能当 oracle，也不能照抄 shape。**

### 自己建 stage-1 UT 的话（**已完成 → `ut_stage1/`；以下是当初的计划，保留作对照**）

- seam：`aiter.ops.flydsl.moe_kernels:flydsl_moe_stage1`（与 stage-2 对称）。
  `_flydsl_stage*_wrapper` **已确认是 live 的可替换 Python 层**，没被 `torch_compile_guard`
  换成 custom-op 前门（GLM 那边 `gemm_a16w16` 就是这么被否掉的，
  见 `glm/k3_dense_bf16_gemm/NOTES.md`）—— 这条以前是未知项，stage-2 的 bundle 把它证实了。
- 抓 oracle 直接改 `ut_stage2/_extract_gen.py` + `_run_capture.sh` + `_capture_overlay/`，
  那套就是当初在活着的 server 上截 stage-2 输入用的。
- shape 用上面 `untuned_fmoe_live.csv` 那两行，但**topk 路由的真实 expert 分布必须现场抓**
  （~610/896，直接决定 grouped GEMM 的负载均衡），而且**两个 decode 模式都要覆盖**。
