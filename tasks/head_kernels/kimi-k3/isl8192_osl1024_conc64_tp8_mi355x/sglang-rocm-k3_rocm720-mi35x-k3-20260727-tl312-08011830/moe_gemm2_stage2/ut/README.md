# UT bundle · `moe_gemm2_0`（FlyDSL MoE stage-2 + downproj + reduction）

**只覆盖 stage-2（本页 #2，9.06% GPU）。stage-1 `moe_gemm1_0`（13.53%）没有匹配的 UT，见 `../NOTES.md`。**

GEAK 原生 bundle，**带真 oracle（181 M，`synthesized: false`，现场抓的）**。来源：

```
provenance://shared-nfs/hyperloom-claw/Kimi-K3/20260817T115910Z/geak/e2e_cycle0/kernels/
    flydsl_moe_stage2_downproj_plus_reduction_task/
```

0828 那次 run 里没有它，因为 0828 只走到 Tier-B 调优（三条路全部 infra 报错），
kernel-authoring 层没轮到，`kernels/` 是空的 —— **Tier-B 调优路本身不产 UT**。

`meta.json` 只记了一个校验和 `reference_io_sha256`，**已核对通过**（`5a3fefb2…`）。

## 为什么是这个 bundle，不是另外三个

claw 树里 Kimi-K3 的 MoE UT 一共四个，判别式是**激活的 dtype**：

| task | act dtype | 对得上 0828 吗 |
|---|---|:---:|
| **`flydsl_moe_stage2_downproj_plus_reduction`**（0817） | **bf16** | ✅ 本目录 |
| `aiter_fused_moe_a8w4`（0816） | fp8 / a8w4 | ❌ |
| `mfma_moe1_mfma_moe2_decode`（0819） | afp8 | ❌ |
| `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256`（0821） | afp8 | ❌ |

0828 线上跑的是 `abf16_wfp4`，证据是 `tuning/work/untuned_fmoe_live.csv`：
`q_dtype_a = torch.bfloat16`、`q_dtype_w = torch.float4_e2m1fn_x2`、`q_type = per_1x32`。
**换了激活量化路，shape 和数值契约都不一样，那三个不能当考卷。**

## 几何量逐条核对（与 0828 profile 抓到的算子形状）

| 量 | 本 bundle | 0828 profile | |
|---|---|---|:---:|
| `num_experts` | 896 | 896 | ✅ |
| `topk` | 16 | 16 | ✅ |
| `model_dim` | 3584 | 3584 | ✅ |
| `inter_dim` | 384 | 384 | ✅ |
| `w1_shape` | [896, 768, 1792] | [896, 768, 1792] | ✅ |
| `w2_shape` | [896, 3584, 192] | [896, 3584, 192] | ✅ |
| dtypes | BFloat16 × Float4_e2m1fn_x2 | 同 | ✅ |

cases：`prefill_M16384` / `prefill_M8192` / `decode_M1` / `decode_M64`。
`decode_M64` = conc 64；`prefill_M8192` = 0828 的 chunk。`tol 0.05`，
候选后端 `['flydsl', 'aiter', 'triton']`。

## ⚠️ 一个必须自己重算的地方：权重

0817 那次 `--chunked-prefill-size` 是 **16384**，0828 是 **8192**。ISL/OSL/conc 三个都一样
（8192 / 1024 / 64），但 trace 权重是按 16384 的 chunk 模型算的：

| case | weight（0817 口径） |
|---|---:|
| `prefill_M16384` | 159 541.8 |
| `decode_M64` | 75 144.4 |
| `prefill_M8192` | 59 629.2 |
| `decode_M1` | 18 786.1 |

**照跑出来的 `GEAK_WEIGHTED_SPEEDUP` 会严重偏向 M16384 那个桶，对 0828 不成立。**
正确做法：**丢掉 `prefill_M16384`，按 `prefill_M8192` + decode 两桶重算。**
四个 case 的逐桶 ms 都会打印出来，重算是纯算术，不用重跑。

另外 `workload.json` 自己也警告了两条：
- `op_kind=moe`：per-expert token 数是路由决定的，「effective-M 分桶」只是 GEMM 式近似，
  权重标了 **lower-confidence**
- decode 两个桶 `weight_source = regime_floor`（profile 里 decode 时间为 0，graph-hidden 兜底），
  不是实测

## 怎么跑

```bash
cd ut_stage2
export HIP_VISIBLE_DEVICES=0

# 空跑：kernel_src/flydsl/ 现在是 baseline 的副本 → 应得 ~1.0
python3 unittest.py

# 你的候选：改 kernel_src/flydsl/ 里的东西（整包替换，不是单文件）
```

风格和 k1 一样是**目录式**（候选放进 `kernel_src/`），不是 k3/k4 的
`GEAK_CANDIDATE_CALLABLE`。特殊之处：baseline 和候选都是**整个 flydsl 包**
（各 4 MB），`unittest.py` 用 `_load_pkg()` 把两份包分别以
`geak_frozen_flydsl` / `geak_candidate_flydsl` 载入 —— 因为 `moe_kernels.py`
里是相对 import，必须按包载。

`harness_lib.py` 与 k1 同代（`72386ab6…`），**和 k3/k4 的 `d196db4c…` 不通用**。

## 集成 seam（自己建 UT 最难猜的一段，meta 里给了）

> 活的派发链是 `AiterRunnerCore.run → aiter.fused_moe:fused_moe → _flydsl_stage2_wrapper
> → aiter.ops.flydsl.flydsl_moe_stage2`。候选要**同时替换包属性和
> `aiter.ops.flydsl.moe_kernels.flydsl_moe_stage2`**，走 PYTHONPATH sitecustomize overlay；
> wrapper 会从 `get_flydsl_kernel_params(kernelName)` 解出 tile/mode kwargs 透传下去，
> **替换实现必须接受同一套 keyword**。

这同时证实了 `_flydsl_stage*_wrapper` 是 live 的可替换 Python 层，
**没有被 `torch_compile_guard` 换成 custom-op 前门**（GLM 的 `gemm_a16w16` 就是那样被否掉的）。

## `_extract_gen.py` / `_run_capture.sh` / `_capture_overlay/`

这三个是**当初抓 oracle 用的脚本**，特意留下来了 —— 建 stage-1 的 UT 时可以直接改这套，
不用从零摸索怎么在活着的 server 上截 `flydsl_moe_stage*` 的输入。
（原目录里 49 M 的 `_capture/` 和 19 M 的 `_capture_prev1/` 是原始 dump，没拷。）

## 状态

**未在 GPU 上复跑验证**（amd-spur 当前 0 idle，burst 票连丢两台）。
oracle 校验和已核对。0817 原始 session 里它是跑通的。
