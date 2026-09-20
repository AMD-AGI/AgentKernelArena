# Kimi-K3 · sglang — kernel 与 unit test 总表

整理时间 2026-09-01。口径 = `hl_matrix_0828_kbtune` 的
`exp/Kimi-K3/Kimi-K3/20260828T135708Z-451fa49c/geak/e2e_cycle1/profile/round_0`
（e2e 加权），配置 TP=8 / ISL 8192 / OSL 1024 / conc 64 / mxfp4 / gfx950。

## 总表

| # | profile 里的 kernel 名 | %GPU | 语言 / 出处 | 可改 | UT | 优化状态 |
|---|---|---:|---|:---:|:---:|---|
| 1 | `moe_gemm1_0` | 13.53 | **aiter FlyDSL**（Python DSL → MFMA） | ✅ | ✅ **oracle** | 六次排序里稳定排 ~#4，每次被 wall-clock 砍掉；**UT 0902 自建补齐**（`ut_stage1/`） |
| 2 | `moe_gemm2_0` | 9.06 | 同上（stage1/stage2 成对） | ✅ | ✅ **oracle** | Tier-B 调优三条路全挂 infra；Tier-C 没轮到 |
| 3 | `_fwd_grouped_kernel_stage1` (+`_fwd_kernel_stage2`) | 8.70 | **Triton** · sglang | ✅ | ✅ **oracle** | **validated_win** 2.6876× iso / +12.10% e2e |
| — | `allreduce_prototype_twoshot` | 8.86 | RCCL/comm | ⚠️ 配置 | n/a | 纯 server flag lever |
| 4 | `_score_kernel` (+`_combine_kernel`) | 9.66 合计<br>(6.61 + 3.05) | **Triton** · sglang | ✅ | ✅ 无 oracle | 未优化；**上游已用 `attn_res_hip` 绕过**，实测 1.4994× |
| 5 | `Cijk_…MT256x256x64_MI16x16x1…` | 5.02 | **Tensile 生成的汇编**（hipBLASLt） | ⚠️ 只能 tune | ✅ **oracle** | 结论：hipBLASLt 已最优，无可部署项 |

**有 UT：5 / 5（含 4 份带真 oracle）。**#1 MoE stage-1 原本是唯一空白（13.53% GPU，单个 kernel 里最大的一块），
0902 自建补齐 —— 它的 routing 是真的（复用 0817 stage-2 seam 的服务分布反解），但**不是在 stage-1 seam 上抓的**，
且**这个 kernel 单发不可复现**，golden 取 21 发 median。细节与告警见 `k2_moe_gemm_flydsl/ut_stage1/README.md`。

> **⚠️ 0902 定论：`moe_gemm1_0` 的不可复现是 aiter 上游的 bug，不是浮点抖动，而且线上在吃。**
> 用 aiter 自己的 `torch_moe_stage1` 作同契约独立参照实测：两发打架的元素上 torch **100% 站
> median、0% 站离群值**（prefill 611 万个争议元素、decode 4 万个），离群值差 244%–280%，
> 而 median 对 torch 的误差在争议/无争议元素上**一样**（6.6e-07 vs 7.2e-07）。
> 即：**多数值数学上正确，离群值是损坏**；median 是合法修复。
> 推论——生产单发路径每次 forward 有 ~0.5–1% 的 stage-1 输出错 O(100%)，prefill/decode 两个桶都中。
> 复跑：`ut_stage1/_l2_reference.py`；数据：`ut_stage1/_l2_report.json`。
> 这也意味着这份 UT 是**正确性** oracle，不只是回归 oracle（另外四份仍只是 L1 自洽）。

## 目录 ↔ 表格对应

| 目录 | 表格行 | 大小 | 内容 |
|---|---|---:|---|
| `k1_mla_decode_grouped/` | #3 | 12 M | patch + overlay + report + **`ut/`（oracle 11.8 M）** |
| `k2_moe_gemm_flydsl/` | #1 #2 | 284 M | `NOTES.md` + **`ut_stage2/`（#2，oracle 181 M，现场 capture）** + **`ut_stage1/`（#1，oracle 103 M，离线复用真 routing）** |
| `k3_score_combine/` | #4 | 137 K | 手写 UT + `cand_attn_res_hip.py` A/B 适配器 |
| `k4_dense_bf16_gemm/` | #5 | 201 M | GEAK 原生 UT + oracle 210 M |

## 三种 UT 风格，别混用

| | k1 / k2 | k3 | k4 |
|---|---|---|---|
| 候选怎么给 | 文件放进 `kernel_src/`（k2 是整个 `flydsl/` 包） | `GEAK_CANDIDATE_CALLABLE=mod:attr` | 同 k3 |
| 语言 | 任意（triton/aiter/ck/hip/flydsl） | Python callable | Python callable |
| oracle | k1 ✅ 合成 · k2 stage-2 ✅ 现场 capture / stage-1 ✅ 离线复用真 routing | ❌（fp32 参考 + live parity） | ✅ 现场 capture |
| `harness_lib.py` | `72386ab6…` | `d196db4c…` | `d196db4c…` |

**`harness_lib.py` 有两代，k1 那份和 k3/k4 不通用，别互相拷。**

三者都遵守同一条 GEAK 规矩：
> *Oracle captured from baseline. Do NOT edit `unittest.py` or `reference_io.pt` during opt.*

退出码：`0` pass · `1` correctness FAIL · `2` env error · `3` harness incomplete。

## 怎么跑

```bash
export HIP_VISIBLE_DEVICES=0     # 三个都是单卡

# k1 —— 复现 2.6876x
cd k1_mla_decode_grouped/ut
cp winning_candidate/decode_attention.py kernel_src/decode_attention.py
python3 unittest.py

# k2 —— MoE stage-2 空跑应得 ~1.0（⚠️ 权重要按 chunk 8192 重算，见 ut_stage2/README.md）
cd k2_moe_gemm_flydsl/ut_stage2 && python3 unittest.py

# k3 —— vs 上游 attn_res_hip（要 0809+ 镜像）
cd k3_score_combine
GEAK_CANDIDATE_CALLABLE=cand_attn_res_hip:mix_fused_hip python3 unittest.py

# k4 —— 空跑应得 ~1.0
cd k4_dense_bf16_gemm && python3 unittest.py
```

镜像：k1/k4 用 run 那个 `sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830`；
k3 的 A/B 腿**必须** 0809+ 的 `lmsysorg/sglang-rocm:v0.5.17-rocm720-mi35x-20260809`
（0727 里还没有 `attn_res_hip.py`）。

## 已 GPU 实跑验证的

| bundle | 验证 | 结果 |
|---|---|---|
| `k3_score_combine` | 2026-08-31 自跑 | baseline 自比 1.0002；vs `attn_res_hip` **1.4994×**（16/16 bucket），correctness PASS |
| `k4_dense_bf16_gemm` | 2026-08-31 自跑 | PASS |
| `k1_mla_decode_grouped/ut` | ⚠️ **本人未复跑** | 0816 原始 session 里 `correctness: pass` / `2.6876×`；校验和已核对，复跑待节点 |
| `k2_moe_gemm_flydsl/ut_stage2` | ⚠️ **本人未复跑** | 0817 原始 session 里跑通；oracle sha256 已核对，复跑待节点 |

`k1` 没复跑的原因：amd-spur 当前 0 idle，burst 票 15 分钟内连丢两台（job 94454 / 94470 均被 preempt）。

## 下一步该做什么

仍然是 **#1/#2 的 aiter FlyDSL MoE GEMM（22.59%）**，但现在路子清楚多了，按性价比排：

1. **先试 `TUNE_ONLY=flydsl` + 1–2 个 worker**（不写 kernel）。0828 那次用 8 worker 全挂在
   `Memory access fault by GPU node-N`，很可能只是并发问题。跑通就直接吃 22.59%。
2. **stage-1 / stage-2 现在都有 UT**（`k2_moe_gemm_flydsl/ut_stage1/` `ut_stage2/`，都带真 oracle）—— 候选可以立刻开写，
   GEAK 给的杠杆是把 `moe_reduction_kernel` 折进 cshuffle epilogue。
   **唯一要注意的是权重**：bundle 是 0817 chunk-16384 口径，得丢掉 `prefill_M16384` 重算。
3. **stage-1 要自己建 UT**。seam 对称（`flydsl_moe_stage1`），抓 oracle 直接改
   `ut_stage2/_extract_gen.py` 那套。现成的三个 stage-1 bundle 都是 afp8 路，只能当结构参考。
   现场 warning 仍然成立：`tile_n=256` **除不尽** `inter_dim=384`，尾块在浪费。

`_flydsl_stage*_wrapper` 是 live 可替换 Python 层这件事**已由 stage-2 的 bundle 证实**，
不再是前置未知项。

其余三个的剩余空间都不大：#3 已经赢过一轮（`n_live_splits` 已压到 1.00 wave/SIMD）；
#4 结论是没东西可部署；#5 升镜像就能拿 +3.2% e2e，不需要写 kernel。

各 kernel 的完整现场数据、seam 论证、shape 出处见各自目录的 `NOTES.md`。
