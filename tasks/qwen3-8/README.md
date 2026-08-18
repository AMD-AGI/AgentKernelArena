# qwen3-8 — 从 session 100137 抽取的 kernel task

Source: **Qwen3.8-2.4T-A95B-Quark-MXFP4**, session `100137`,
`/shared_nfs/hyperloom-claw/Qwen3.8-2.4T-A95B-Quark-MXFP4/20260814T175123Z`,
sglang `0.5.17.dev20260812+gdc5f6c4883` / ROCm 7.2.0 / aiter `d9e5ef7ce`,
8× MI355X (gfx950), TP=8 EP=8, mxfp4 (Quark), KV fp8_e4m3,
ISL 8192 / OSL 1024 / conc 64, chunked-prefill 16384, 654.09 s @ 1214.85 tok/s。

完整负载分析：
`/shared_nfs/jqliu/new-image-hot-kernels/hot-kernels-analysis/qwen38-100137-hot-kernels.md`

E2E 占比的分母是整场压测墙上时间 **654.09 s**（decode 占 56.7%，prefill 40.8%，调度残差 2.5%）。
decode 侧的数字是 8 rank torch trace 实测；prefill 侧这个 session 没有 trace，只有 shape 是确定的。

模型结构（`config.json`）：92 层，每层 = 一个注意力块 + 一个 MoE 块。
`layer_types` = **23 层 full_attention + 69 层 linear_attention**（`full_attention_interval=4`）。
hidden 8192，512 experts / topk 10 / moe_intermediate 2048，shared expert intermediate 2048。

## 筛选标准

每个 task 的计算核都是**镜像里可编辑的源码**，并且运行时从该源码 JIT 编译——
补丁改的就是真正跑起来的东西。按设计排除：

- **通信算子**（按要求排除）—— prefill RCCL all-reduce（~15.3% E2E）和 decode
  `cross_device_reduce_2stage`（8.05% E2E）。这两个加起来是 workload 的最大单项。
- **预编译二进制算子** —— 实现只以编译产物存在的。
- **后端 tuning 类工作** —— 填调优表、改派发启发式，而不是改 kernel 代码。

## 总览

| task | 算子 | **任务形态** | E2E | 后端 | 覆盖层数 |
|---|---|---|---|---|---|
| `mi355x_sglang_aiter_mxfp4_moe_2stage_qwen38` | MoE routed-expert 两段 grouped GEMM | **单算子**（1 次 API 调用 → 6 个 kernel） | **17.9%** | FlyDSL → MLIR → AMDGCN | 92 / 92 |
| `mi355x_sglang_triton_gdn_linear_attn_qwen38` | Gated DeltaNet 线性注意力 decode 核心 | **多算子链路**（5 个 kernel + torch split/cat） | **5.76%**（整链） | Triton | 69 / 92 |
| `mi355x_sglang_aiter_paged_attention_decode_qwen38` | paged GQA decode 注意力 | **单算子**（1 次 API 调用 → 2 个 kernel） | **3.36%** | HIP C++（`csrc/cpp_itfs/pa`） | 23 / 92 |

合计 **27.0% E2E** —— 这基本就是这个 workload 里「非通信 + 源码可编辑 + decode trace 能归因」的全部计算。

## 与 `tasks/image_kernel/` 下已有 task 的重叠

三个 task 里**两个和已有 task 命中同一个 kernel**，都不是完全重复，但差异点必须写清楚：

| 本 task | 已有 task | 重叠程度 | 区分点 |
|---|---|---|---|
| MoE 2-stage | `mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3` | **同一个源文件，不同 builder 函数** | K3 是 a16w4 → `compile_mixed_moe_gemm1_a16w4`(:4959) / `_a16w4`(:7215)；本 task 是 a4w4 → `compile_mixed_moe_gemm1`(:147) / `compile_mixed_moe_gemm2`(:3079)。另外 K3 走 tuned 路径、SiTUv2、vllm，本 task 走 heuristic fallback、Silu、sglang |
| paged attention | `mi300x_sglang_hip_pa_ragged`（最接近）、`mi300x_sglang_hip_pa_decode`、`mi355x_vllm_hip_paged_attention_decode` | **同一个 kernel、同一个入口、同一套源文件** | 已有的都是 head 128/64 + KV `auto`(bf16) + block 16 + ctx ≤4097；本 task 是 head **256** + KV **fp8_e4m3** + block **1** + ctx 8192–9216。前三项决定了这是**另一个模板实例** |
| GDN 线性注意力 | `mi355x_vllm_triton_kda_linear_attn_kimi_k3`（形态相似） | **不重叠** | KDA ≠ GDN，是两个算法；kernel 名、仓库（vllm vs sglang）都不同 |

两点后续处理：

- MoE task 的 `logical_operator` 原本和 K3 那个**字符串完全相同**（`aiter_mxfp4_moe_2stage`），
  已改成 `aiter_mxfp4_moe_2stage_a4w4`；`target_kernel_functions` 也从三个共用的包装函数
  改成直接指两个 a4w4 builder，让两个 task 按 target 就能区分，而不是只靠名字。
- paged attention task 和 `mi300x_sglang_hip_pa_ragged` 是**同一 kernel 的不同实例化**。
  保留成独立 task 是因为 fp8 KV + head 256 走的是另一份编译产物；
  但如果不想在这个 kernel 上挂第四个 task，`session_cases.json` 里那三个 case
  可以直接并进 `mi300x_sglang_hip_pa_ragged`。

> 顺带：之前被移除的 dense bf16 GEMM task 也会和 `mi300x_sglang_triton_gemm`
> （同仓库、同 `_gemm_a16_w16_kernel`）部分重叠，现在不存在这个问题了。

## 单算子 vs 多算子链路

两种形态在这套 task 里的含义不同，写 patch 和读分数时要区分：

- **单算子任务**（task 1、task 3）——计时单元是**一次库函数调用**，是模型代码里实际调用的那个
  API 边界。它内部派发多少个 kernel 由库自己决定：MoE 是 6 个（2 个 GEMM + 4 个排序/量化辅助），
  paged attention 是 2 个（主 kernel + 跨 partition 归约）。这些 kernel 不是「凑」在一起的，
  是同一个算子的内部阶段，边界由库定义而不是由这套 task 定义。
- **多算子链路任务**（task 2）——计时单元是**模型代码里连续调用的一串独立算子**，
  边界是这套 task 自己划的（两个输入投影之后、输出投影之前）。之所以这样划，
  是因为链路里 5.76 个 E2E 点中有 2.35 点是算子之间的数据搬运，
  单独打其中任何一个 kernel 都看不到这块。**跨算子融合在这个 task 里是合法且预期的优化手段**，
  在两个单算子 task 里则不是——那里改的是一个已有算子的内部实现。

---

# 1. `mi355x_sglang_aiter_mxfp4_moe_2stage_qwen38`

> **任务形态：单算子。** 计时单元是一次 `aiter.fused_moe.fused_moe` 调用——
> 就是 sglang 的 MoE runner 实际调的那个 API 边界。

## 是什么算子

**MoE routed-expert 的两段 grouped GEMM**，一次 `aiter.fused_moe.fused_moe` 调用里的 stage-1 和 stage-2：

| kernel | 做什么 |
|---|---|
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32` | stage-1：gate_up 投影（8192 → 2×2048）+ 融合的 SiLU·Mul |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_persist_cu256` | stage-2：down 投影（2048 → 8192）+ 按 topk 权重加权合并 |

后端是 **FlyDSL**（aiter 自研 Python kernel DSL → MLIR → AMDGCN，运行期 JIT）。
数据类型是 **a4w4**：激活和权重都是 MXFP4（group_size 32，e8m0 scale），输出 bf16。

一次调用内部实测派发 **6 个 kernel**（本容器 M=64，profiler 实测）：

| kernel | us | 角色 |
|---|---|---|
| `mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32` | 136.0 | **优化目标** |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_..._persist_cu256` | 68.4 | **优化目标** |
| `opus_moe_sorting_entry<...P0_v2>` | 6.0 | expert 排序 |
| `fused_mx_quant_moe_sort_kernel<bf16, fp4_t, 256, 32>` | 5.9 | stage-1 输入量化 |
| `fused_mx_quant_moe_sort_kernel<bf16, fp4_t, 256, 8>` | 5.3 | stage-2 输入量化 |
| `opus_moe_sorting_entry<...P23>` | 4.4 | expert 排序 |
| 合计 | 226.0 | 两个 GEMM 占 **90.4%** |

后四个是同一算子内部的辅助阶段，一起被计时（因为改 GEMM 的 tile 会改变它们的输入布局，
拆开计时会漏掉这部分代价），但优化目标是前两个。

## E2E 占比

| | ms/step（8 rank 均值） | %decode step | **%E2E** |
|---|---|---|---|
| stage-1 | 6.3257 | 20.50% | **11.62%** |
| stage-2 | 3.3994 | 11.02% | **6.25%** |
| 合计 | 9.7251 | 31.52% | **17.9%** |

**这是整个 workload 里最大的非通信算子。** prefill 侧还有一份未计入的贡献，
因为 session 没抓 prefill trace，无法定量。

## 对应模型结构的哪个部分

**全部 92 层的 MoE 块中的 routed expert 部分**。每层调用一次，每步 92 次。

```
layer[i]:
  ├── attention block            → task 2 / task 3
  └── MoE block
        ├── router mlp.gate       (dense GEMM，不在本 task)
        ├── shared expert         (dense GEMM + act，不在本 task)
        └── routed experts  ←──── 本 task
              512 experts，EP=8 后每卡 64 个，topk=10
              stage-1: [tokens, 8192] × w1[64, 4096, 8192]  → SiLU·Mul → [tokens×10, 2048]
              stage-2: [tokens×10, 2048] × w2[64, 8192, 2048] → 加权合并 → [tokens, 8192]
```

MoE 块里的 router gate 和 shared expert 走 dense bf16 GEMM（hipBLASLt 预编译 Tensile），
**不在这个 task 里**。

## 覆盖的 shape

三个 M 桶，派发**三个不同的 kernel 对**（`aiter/fused_moe.py:2249-2255` 的启发式）：

| case | M | session 里的来源 | tile |
|---|---|---|---|
| `qwen38-moe-decode-m64` | 64 | 被 profile 的那个 decode step（conc 64） | t32x128x256 |
| `qwen38-moe-prefill-m8192` | 8192 | 429 个 prefill batch 里的 26 个 | t128x128x256 |
| `qwen38-moe-prefill-m16384` | 16384 | 429 个 prefill batch 里的 403 个 | t64x128x256 |

## 编辑面与 harness 约束

编辑面是 **FlyDSL MLIR builder**：`ops/flydsl/kernels/mixed_moe_gemm_2stage.py`
（`compile_mixed_moe_gemm1` :147 / kernel :468，`compile_mixed_moe_gemm2` :3079 / kernel :3275）。
杠杆：MFMA 流水、tile/block 循环结构、LDS 与 ping-pong staging、async-copy 调度、
stage-2 persistent CU 映射、CShuffle epilogue、atomic 归约。

tuned CSV **刻意不在编辑面里**——填查表是后端 tuning，不是 kernel 工作。
但「session 从来没调优过」是 headroom 的来源：server.log 打了 **120 次**
`no tuned FlyDSL config`，覆盖全部 15 个 M 桶、8 个 rank。

harness 强制的不变量：

1. **断言走 FlyDSL 派发。** `use_mxfp4_flydsl`（`aiter/fused_moe.py:2198`）要求
   `is_shuffled` 且 `not doweight_stage1`；缺一个就会静默掉到 CK 2-stage 路径——完全是另一个 kernel。
2. **按 case 钉死 kernel 对**（`expected_dispatch`），防止某个 case 漂到别的 M 桶、
   去打一个从未被校验过的 kernel。
3. **权重准备复刻 sglang loader**（`quark_w4a4_mxfp4_moe.py:577-597`）：
   先 `e8m0_shuffle` 二维 scale 视图，再 `shuffle_weight(w, (16,16))`，再置 `is_shuffled = True`。
4. **correctness 取 3 次最差**——stage-2 用 atomic 归约，单次可能撞运气。
5. **被计时的那次调用会被重新校验**（扰动输入 + NaN 毒化输出 + 重放图），而不是信任旁边一次调用。

## 参考实现

独立的向量化 torch 实现，不是包装。aiter 自带的 `torch_moe_stage1/2` 用不了：
它会 dequant 到 fp32 并展开 `[token, topk, 2·inter]`，在 prefill M=16384 下是 2.7 TB。
本实现每次只 dequant 一个 expert（唯一的 Python 循环是对 64 个本地 expert，每次两个稠密 GEMM），
并且把激活和 stage-1 输出按 kernel 的方式做 MXFP4 fake-quant，所以只剩累加顺序的差异。

---

# 2. `mi355x_sglang_triton_gdn_linear_attn_qwen38`

> **任务形态：多算子链路。** 计时单元是模型代码里连续调用的 5 个独立算子加 torch 的
> split/cat，边界由本 task 划定（两个输入投影之后、输出投影之前）。
> **跨算子融合是本 task 预期的优化手段。**

## 是什么算子

**Gated DeltaNet（线性注意力）的 decode 核心**——模型在两个输入投影之后、输出投影之前做的全部事情。
计时单位是整条链路而不是单个 kernel，因为在 session 的 trace 里它每层打出 5 个 kernel：

| kernel | 做什么 | 后端 |
|---|---|---|
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 门控 delta rule 递推（含 qk L2-norm） | Triton |
| `at::native::elementwise_kernel_manual_unroll` (`aten::copy_`) | b/a 门的 `.contiguous()` + norm 输出 reshape | ATen |
| `at::native::CatArrayBatchedCopy` (`aten::cat`) | 拼 `mixed_qkv` | ATen |
| `_causal_conv1d_update_kernel` | 短卷积状态更新（width 4） | Triton |
| `_layer_norm_fwd_1pass_kernel` | 输出 gated RMSNorm（swish 门） | Triton |

## E2E 占比

| kernel | 次/step | ms/step (TP-0) | %decode step | **%E2E** |
|---|---|---|---|---|
| `fused_recurrent_gated_delta_rule_packed_decode` | 69 | 1.2397 | 4.018% | 2.278% |
| `aten::copy_` | 208 | 0.9052 | 2.934% | **1.663%** |
| `aten::cat` | 69 | 0.3765 | 1.220% | **0.692%** |
| `_causal_conv1d_update` | 69 | 0.3148 | 1.020% | 0.578% |
| `_layer_norm_fwd_1pass` | 69 | 0.3005 | 0.974% | 0.552% |
| **整链** | | **3.1367** | **10.17%** | **5.76%** |

单看递推 kernel 只有 2.278%，够不到 5% 的热点线。**打包计时的理由是：5.76 个点里有 2.35 个点是纯数据搬运。**
Qwen3.8 的 `num_v_heads // num_k_heads = 128 // 16 = 8`，不在 `(1, 2, 4)` 里，
所以 `srt/models/qwen3_5.py:640` 不走融合的 split/cat helper，落进朴素 torch 分支——
每步那 208 次 `aten::copy_` 和 69 次 `aten::cat` 就是从这来的。只打递推 kernel 会把最大的杠杆藏起来。
harness 会断言这个 head 比例，配置一旦变化导致模型改走融合 helper，task 会直接失败而不是默默测别的东西。

## 对应模型结构的哪个部分

**69 层 linear_attention 层的注意力块核心**（`Qwen3_5GatedDeltaNet`）。每层调用一次，每步 69 次。

```
linear_attention layer[i]:
  ├── in_proj_qkvz  [., 8192] × [8192, 4608]   (dense GEMM，不在本 task)
  ├── in_proj_ba    [., 8192] × [8192, 32]     (dense GEMM，不在本 task)
  ├── ┌─────────────────────────────────────┐
  │   │ fix_query_key_value_ordering (split) │
  │   │ torch.cat(q, k, v) → mixed_qkv       │  ←── 本 task 的计时单位
  │   │ causal_conv1d_update  (width 4)      │
  │   │ fused_recurrent_gated_delta_rule     │
  │   │ RMSNormGated(core_attn_out, z)       │
  │   └─────────────────────────────────────┘
  └── out_proj      [., 2048] × [8192, 2048]   (dense GEMM，不在本 task)
```

单卡几何（TP=8）：2 个 k head × 128，16 个 v head × 128，conv_dim 2560，
SSM state `[slots, 16, 128, 128]` bf16（`--mamba-ssm-dtype bfloat16`）。

## 覆盖的 shape

`qwen38-gdn-decode-bs64`，bs=64（conc 64），只有 decode。

**没有 prefill case 是刻意的**：这个算子的 prefill 走的是**另一个 kernel**
（`kernels/ops/attention/fla/chunk.py::chunk_gated_delta_rule`，chunked scan 而不是单步递推），
而 session 100137 根本没抓 prefill trace，造一个合成 prefill case 只会去打一个 session 无法证实的形状。

## 有状态算子的处理

conv cache 和 SSM state 都**原地更新**（`gdn_triton.py` 传的是 `ht=initial_state`）。
CUDA graph 重放会把状态推进 N 次。harness 正面处理而不是绕开：

- prepare 时对 cache 做快照，每次测量前恢复；
- 计时校验读 `benchmark_effective_repeats`，把 torch 参考从**同一快照**步进**同样次数**；
- correctness 额外校验 `conv_state` 和 `ssm_state`——输出对但把 cache 写坏的实现会毁掉下一个 token，必须拦住。

## 参考实现

完全向量化的 torch，从 Triton 源码转写而非调用它：batch 和 head 维都没有 Python 循环，
递推步是 `[B, HV, V, K]` 上的批量外积更新加两个 `einsum`。

---

# 3. `mi355x_sglang_aiter_paged_attention_decode_qwen38`

> **任务形态：单算子。** 计时单元是一次 `aiter.ops.attention.paged_attention_ragged`
> 调用——sglang 的 aiter backend 实际调的那个 API 边界。它派发 2 个 kernel，
> 是同一算子的两个阶段（分 partition 计算 + 跨 partition 归约），不是两个独立算子。

## 是什么算子

**paged GQA decode 注意力**，aiter 的 ll4mi 实现，两个 kernel：

| kernel | 做什么 |
|---|---|
| `paged_attention_ll4mi_QKV_mfma16_kernel<0, __hip_bfloat16, unsigned char, (vllm::Fp8KVCacheDataType)1, 1, 256, 256, false, 8, 1, ck_tile::ComposedAttention<...>>` | 分 partition 的 QK^T·softmax·V，fp8 KV 在 kernel 内反量化 |
| `paged_attention_ll4mi_reduce_kernel<__hip_bfloat16, __hip_bfloat16, 256, 256, 256, 1, false>` | 跨 partition 归约 |

后端是 **HIP C++**（`aiter/csrc/cpp_itfs/pa/pa_kernels.cuh` 2275 行 + `pa_ragged.cuh` / `pa.cuh`），
运行期 JIT 编译——跑 harness 时能看到它现场 build，所以是真可编辑源码，不是预编译汇编。

## E2E 占比

| kernel | 次/step | ms/step (TP-0) | %decode step | **%E2E** |
|---|---|---|---|---|
| `paged_attention_ll4mi_QKV_mfma16` | 23 | 1.7119 | 5.549% | **3.146%** |
| `paged_attention_ll4mi_reduce` | 23 | 0.1149 | 0.372% | 0.211% |
| 合计 | | 1.8268 | 5.921% | **3.36%** |

**够不到 5% 的热点线**，收进来的理由是配置罕见：**head_size = 256**（绝大多数在服务的模型是 128）、
KV 是 **fp8_e4m3** 在 kernel 内反量化、GQA group 恰好是 8。这个模板实例受到的调优关注远少于 128 那个。

## 对应模型结构的哪个部分

**23 层 full_attention 层的注意力核心**。每层调用一次，每步 23 次。

```
full_attention layer[i]:
  ├── qkv_proj   [., 8192] × [8192, 4608]      (dense GEMM，不在本 task)
  ├── mRoPE / q-k Gemma-RMSNorm / reshape_and_cache  (不在本 task)
  ├── ┌──────────────────────────────────────┐
  │   │ paged_attention_ll4mi_QKV_mfma16      │  ←── 本 task
  │   │ paged_attention_ll4mi_reduce          │
  │   └──────────────────────────────────────┘
  ├── 输出门 sigmoid·mul（attn_output_gate）    (不在本 task)
  └── o_proj     [., 2048] × [8192, 2048]      (dense GEMM，不在本 task)
```

单卡几何（TP=8）：`num_attention_heads 64` → 每卡 8 个 q head；
`num_key_value_heads 4 < tp_size 8`，`QKVParallelLinear` 会先把 kv head 补齐到 8 再切，
所以每卡恰好 1 个 k head、1 个 v head，GQA group = 8（和 kernel 模板参数对上）。
`head_dim 256`，`page_size 1`，`partition_size 256`。

> kv head 补齐不是猜的：正因为补到 8，`qkv_proj` 的每卡输出宽度才是
> `(64×2 + 8 + 8) × 256 / 8 = 4608`，这正是 capture trace 记录的 `aten::mm` 的 N
> （92 次 `[64,8192]×[8192,4608]`）。不补齐会是 4352，而 4352 从未出现过。

## 覆盖的 shape

bs=64，三个 context 长度覆盖 session 的整个 decode 区间：

| case | seq_len | 是什么 |
|---|---|---|
| `qwen38-pa-decode-bs64-seq8192` | 8192 | 刚进 decode，只有 ISL |
| `qwen38-pa-decode-bs64-seq8704` | 8704 | profiler 窗口所在处，上面那两个 E2E 数字就是在这测的 |
| `qwen38-pa-decode-bs64-seq9216` | 9216 | 生成结束，ISL 8192 + OSL 1024 |

**没有 prefill case 是刻意的**：`paged_attention_ragged` 只是 decode 入口，
chunked prefill 走同一 backend 里另一条 ragged/MHA 路径（server.log 记录 `prefill cuda_graph=disabled`），
而那个 prefill kernel 从未被采样过——session 没有 prefill trace，它的身份和代价都不知道。

## 与已有 task 的差异（细化）

这个 kernel 在 `tasks/image_kernel/` 下已经被三个 task 命中，差异全在实例化：

| | 已有 task | 本 task |
|---|---|---|
| `head_size` | 128（另有一个 64） | **256** |
| KV cache | `"auto"` → bf16 | **fp8_e4m3**，kernel 内反量化 |
| 模板参数 | `Fp8KVCacheDataType(0)` | **`Fp8KVCacheDataType(1)`** |
| `block_size` | 16 | **1** |
| context | 128 / 1024–2048 / 4097 | **8192 / 8704 / 9216** |
| 来源 | 合成 | session 100137 trace |

前三项选中的是**另一份编译产物**，所以优化工作不等价。但如果不想在这个 kernel 上挂第四个
task，`session_cases.json` 里那三个 case 可以直接并进 `mi300x_sglang_hip_pa_ragged`。

## 后端被钉死

和别的 task 不同，harness 会断言 `paged_attention_ll4mi_QKV_mfma16_kernel` 确实跑了，
并且 fp8 KV 模板参数仍是 `(vllm::Fp8KVCacheDataType)1`。换后端或把 cache 悄悄放宽成 bf16
是换题目，不是优化。

## 参考实现

完全向量化的 fp32 GQA——两个 `einsum` 加一个 softmax，batch / head / partition 维都没有循环。
因此它同时也是 ll4mi 那两个 kernel 分 partition 归约的独立实现。

---

# 目录结构

**每个 task 目录是自包含的**——只依赖自己文件夹下的东西，不引用兄弟目录、
也没有共享的 `_shared/`。目录之间可以单独拷走、单独跑。

```
README.md              本文件——qwen3-8 目录下唯一的 README
<task>/
  config.yaml          AgentKernelArena task 定义
  session_cases.json   trace 导出的 shape/dtype + provenance
  scripts/
    task_runner.py     Arena harness（compile / correctness / performance）
    forge_driver.py    forge-loop 驱动（correctness / bench / profile-run）
```

单 task 目录下**不放 README**，说明全部集中在本文件；三个 task 各占一个 section。

`build/` 是运行期产物（aiter JIT `AITER_JIT_DIR`、Triton cache、
`performance_report.json`），由 harness 在各 task 目录下自动创建，**不入库**。
想清干净：`rm -rf tasks/qwen3-8/*/build`。

`scripts/task_runner.py` 是这个 task 唯一的 harness 源文件，直接改它即可，
没有生成步骤。CUDA-graph 计时用的那段公共 helper 逐字内联在文件里，
夹在 `make sync-perf-helpers` 认的那对 `AKA-GENERATED` 标记之间——
所以既能自包含运行，也能在 `src/tools/perf/vllm_cuda_graph_block.py` 更新后被重新同步。
**不要手改标记之间的内容**，那段会被同步覆盖。

---

# 四条 harness 要求的落实

**1 — 测试 shape 与真实 E2E session 一致，prefill/decode 都调用的就都覆盖。**
每个 shape 都来自 session 的 CUDA-graph capture trace（`record_shapes=True`），
或由 `config.json` 推出后与之核对，记录在 `session_cases.json` 的
`trace_input_shapes` / `trace_input_dtypes` 里。

| task | decode | prefill |
|---|---|---|
| MoE 2-stage | M=64 | M=8192 **和** M=16384（session 真实跑的两种 batch：429 个里的 26 和 403） |
| GDN 线性注意力 | bs=64 | 无 —— *prefill 走的是另一个 kernel*，见上 |
| paged attention | bs=64 × seq {8192, 8704, 9216} | 无 —— *这是 decode 专用入口*，见上 |

两个「无」都写进了对应 `session_cases.json` 的 `prefill_note` 和 task README。
这两个算子的 prefill 都派发到**不同的 kernel**，而 session 100137 **完全没有 prefill trace**
——两个 profiler 窗口抓到的 128 步全是 `step[DECODE bs=64]`。
造一个合成 prefill case 只会去打一个 session 无法证实的形状，所以没造。

**2 — 精度测试与性能测试的 shape 覆盖一致。** 每个 harness 的两个 mode 都遍历同一个
`CASES` 列表，没有按 mode 调整 shape、没有 `correctness_token` 缩小、
没有 `correctness_only` / perf-only case。MoE task 还按 case 钉死了派发的 kernel 对。

**3 — 性能基于 CUDA graph 测量。** 三个 task 都用逐字嵌入的
`src/tools/perf/vllm_cuda_graph_block.py` 里的 `_benchmark_cuda_graph_or_events`。
下面验证跑里每个 case 都报 `benchmark_method: cuda_graph`，没有回落到 event timing。
每个 harness 还传了 `_TimedRun`，所以校验的是**被计时的那次调用**本身
（扰动输入、NaN 毒化输出、重放图），而不是信任旁边一次调用。

GDN task 在这里多花了功夫：它有状态，重放会把 cache 推进 N 次，
harness 读 `benchmark_effective_repeats` 并把参考步进同样次数，而不是把图退化成单次调用来回避。

**4 — torch 参考向量化。** 没有任何参考对 token / head / batch 做 Python 循环：

| task | 参考做法 | 耗时 |
|---|---|---|
| MoE | 逐 expert dequant + 2 个稠密 GEMM；唯一循环是对 64 个本地 expert | ~4.5 s/case |
| GDN | `[B,HV,V,K]` 批量外积状态更新 + 2 个 einsum，无循环 | <0.1 s |
| paged attn | 对 dequant 后的 cache 做 2 个 einsum + softmax，无循环 | <0.1 s |

---

# 验证跑（本容器，1× MI355X gfx950）

三个 task 都完整跑过 `compile` / `correctness` / `performance`，
全部 case 通过且 `benchmark_method: cuda_graph`。

| task | cases | correctness | perf |
|---|---|---|---|
| MoE 2-stage | 3 | cos 0.9886–0.9887 | 0.2295 / 0.7856 / 1.2243 ms |
| GDN 线性注意力 | 1 | cos 0.999994（外加 cache 校验） | 0.029906 ms |
| paged attention | 3 | cos ≈ 0.99995 | 0.0579 / 0.0629 / 0.0701 ms |

两件静态分析看不出来、跑起来才确认的事：

- **派发的 kernel 名与 session trace 完全一致**，包括两个 `paged_attention_ll4mi_*`
  模板实例和三对 MoE FlyDSL kernel。
- **MoE 的 tile 启发式是三档不是两档。** harness 的 `expected_dispatch` 不变量抓出了
  case spec 初稿的一个错：`fused_moe.py:2251` 把 `4096 ≤ token < 16384` 送到 `tile_m=128`，
  所以 session 的 8192-token prefill batch 跑的是**第三对** kernel（`t128x128x256`），
  和 decode（`t32`）、16384-token prefill（`t64`）都不同。

**适用于全部三个 task 的说明**：harness 的绝对耗时是**单卡孤立**测量
（单算子、GPU 空闲、无 TP、无外围图），比同一 kernel 在 session 的 92 层 decode 图里快 1.3–1.8 倍。
用它看**相对**提升；E2E 的权威仍是 session trace。

---

# 编辑生效性与耗时（已实测验证）

## 编辑一定被重新编译，不会跑旧产物

对每个 task 都做了同一个实验：按 `config.yaml` 的 `image_repo_path` / `repo_subdir`
把仓库 seed 进一个干净 workspace，先跑一遍 correctness 确认基线通过，
然后在**编辑面内**注入一个数值扰动（把 kernel 输出乘 1.5），再跑一遍。
输出必须变化——如果仍然原样通过，就说明跑的是旧的编译产物。

| task | 编辑的文件 | 基线 | 注入 ×1.5 后 |
|---|---|---|---|
| MoE（FlyDSL） | `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` 的 `silu_elem` | PASS cos 0.988737 | **FAIL** rel_err 0.5555 |
| GDN（Triton） | `sglang/kernels/ops/attention/fla/fused_recurrent.py` 的 packed-decode kernel | PASS cos 0.999994 | **FAIL** cos 0.998465 |
| paged attn（HIP） | `csrc/cpp_itfs/pa/pa_kernels.cuh` 的 reduce kernel | PASS rel_err 0.01020 | **FAIL** rel_err 0.50001 |

三条各自依赖的机制：

- **MoE**：workspace 的 `aiter/` 通过 `sys.path.insert` 覆盖镜像安装
  （实测 `imported from: <ws>/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`）。
  FlyDSL 的 cache key 包含 kernel 函数源码及其依赖源码
  （`flydsl/compiler/jit_function.py:572-598`），所以**即使 `module_name` 没变**，
  改了 kernel body 也会重新编译——上表的扰动就没有改名字。
  镜像里自带的 2.2 GB `aiter/jit/flydsl_cache` 因此不会喂回旧 kernel。
- **GDN**：同样靠 `sys.path` 覆盖；Triton 按源码哈希缓存，`TRITON_CACHE_DIR` 又指向
  workspace，天然隔离。
- **paged attention**：靠 `AITER_META_DIR=<workspace>`，让
  `csrc.cpp_itfs.utils.AITER_CORE_DIR` 解析到 workspace，编译时 include 的是被改过的
  `.cuh`。**这一条原本是坏的，已修**，见下。

## 修掉的一个真实缺陷：paged attention 会跑旧二进制

aiter 的 ll4mi kernel 走 `compile_template_op`（`csrc/cpp_itfs/utils.py:300`）：

- 缓存目录是 `$AITER_ROOT_DIR/build/<md_name>_<md5(模板参数)>`，
  md5 只覆盖 head_size / dtype / block_size 这些**模板参数**，**不覆盖 `.cuh` 的内容**；
- 唯一的新鲜度判断是 `not_built()`——「`lib.so` 在不在」（`utils.py:297`）；
- `AITER_ROOT_DIR` 默认 `$HOME/.aiter`（`utils.py:68`），**全机共享**。

结果：agent 改了 `pa_kernels.cuh`，文件夹名不变，`lib.so` 还在 → **直接跑旧二进制**；
而且不同 workspace 之间还会互相串。反证做过：把 `AITER_ROOT_DIR` 指回 `/root/.aiter`
复现修复前的行为，上面那个 ×1.5 的扰动被**完全忽略**，correctness 原样通过
（cos 0.999948 / rel_err 0.01020）。

修法在 `scripts/task_runner.py` 的 `_pin_template_op_build_dir()`：把 PA 相关源文件
（`csrc/cpp_itfs/pa/**` 的 `.cuh/.h/.jinja/.cpp/.py` 加 `utils.py`）做 sha256，
取前 12 位拼进 `AITER_ROOT_DIR = <workspace>/build/aiter-<hash>`。
于是缓存既是 workspace 本地的（不串），又是内容寻址的（改了必重编，没改仍命中）。
实测编辑后目录从 `aiter-08db1bfbc493` 变为 `aiter-9a681af165ed`，触发真实重编译。

## 耗时：没有全量 aiter 编译，每个 mode 都在 1 分钟内

三个 task 都不触发 aiter 全量编译——镜像里预编译的 `aiter/jit/*.so`
（`module_aiter_core` 等约 1 GB）直接复用；只有本 task 真正用到的模块会现编：
MoE 冷启动编 `module_moe_sorting_opus`(7.7 s) 和 `module_quant`(13.9 s)，
PA 编一个 `pa_ragged_<md5>`(≈19 s)，GDN 一个都不编。

| task | 冷启动 compile | correctness | performance | 改完源码后再跑一轮 |
|---|---|---|---|---|
| MoE 2-stage | 52 s | 12 s | 12 s | 8 / 19 / 12 s |
| paged attention | 36 s | 8 s | 8 s | 5 / 8 / 8 s |
| GDN 线性注意力 | 6 s | 6 s | 6 s | 5 / 5 / 6 s |

最慢一格 53 s，离 5 分钟上限有充足余量。`config.yaml` 里的 timeout
（900–3600 s）是保守上界，不是预期耗时。

> seed 成本另说：`image_repo_path` 指的 `aiter/` 包整体约 6.3 GB，其中
> `jit/build`(2.9 G) 已在 `image_repo_exclude` 排除，剩下 `jit/flydsl_cache`(2.2 G)
> 和预编译 `.so`(≈1 G) **必须保留**——去掉前者会让 FlyDSL 每轮从头编，
> 去掉后者会触发 aiter 全量重编，两者都与「不要全量编译」冲突。

# Arena + KernelForge forge-loop（已实测跑通）

三个 task 都用 `/shared_nfs/jqliu/run_arena/run_qwen38.sh <task> <suffix>` 端到端跑过，
环境取自 `/shared_nfs/jqliu/set_env.sh`，logs/workspace 放 `/tmp`（NFS 上跑会把共享
cgroup 的 page cache 顶到上限，见 `config.forge_mxfp4.yaml` 的注释）。

| task | driver | prepare | forge baseline | anchor 校验 |
|---|---|---|---|---|
| GDN 线性注意力 | task-provided | skipped | 0.030 ms | 1 of 1 |
| paged attention | task-provided | skipped | 0.063 ms | 3 of 3 |
| MoE 2-stage | task-provided | skipped | 0.735 ms | 3 of 3 |

`prepare = skipped` 是期望结果：prepare 是一个有预算上限的 LLM 修复循环，专门把不合规的
driver 改写到合规；跳过意味着 forge 直接接受现成 driver，预算全花在优化上。
`anchor` 那行是 forge 用自己的方式重跑 case 后与 harness reference 对账的结果。

## 为什么每个 task 自带 `scripts/forge_driver.py`

`agents/forge/launch_agent.py:822` 优先使用 task 自带的 `scripts/forge_driver.py`（逐字拷贝到
workspace 根），没有才生成一个包 `agents/forge/drivers/arena_task_adapter` 的 shim。
那个 shim 用 `parse_known_args`，**会把 `--profile-run` 当未知参数吞掉**、退回跑 correctness；
而 KernelForge 的 `_check_profile_contract` 只断言 `returncode == 0`，于是 shim
在完全没有 profiling 路径的情况下通过了闸门。1.75 h 预算下 Analysis 本来就是 `static-only`
（`cli.py:2088`：< 2 小时一律 static-only），不会暴露；一旦 campaign 超过 2 小时，
Analysis 切成 `profiled` 就会真正调用这条不存在的路径。kimi_k3 自带 driver 的注释记了这个坑：
它的源 session 就是这么 `task_preparation_failed` 的。

## driver 实现的契约

stdout 被这两处解析：`mcp_server/tools/test.py:74-83`、`mcp_server/tools/bench.py:341-347`。

| 模式 | 输出 | 说明 |
|---|---|---|
| `--mode <smoke\|stability\|determinism\|full>` | `SNR: <db> dB` / `allclose:` / `max_diff:` | 取全 suite 最差值 |
| `--warmup N --iters N --bench-mode` | 每个 case 一行 `case_ms: <id> <ms>` + 一行 `mean_ms:` | 走 harness 的 `_benchmark_cuda_graph_or_events`，每个计时迭代一次 graph replay |
| `--profile-run [--profile-case <id>]` | 无计时输出，exit 0 | 只发射目标算子：warmup 落实 JIT，几次 profiled launch，一次 synchronize |

preflight 四道闸门里 **graph 这道是真查的**：`task_preparer._count_graph_replays` 通过
`sitecustomize` 注入计数器，统计 benchmark 期间真实的 `torch.cuda.CUDAGraph.replay` 次数，
要求 ≥ iters——不看任何打印出来的标签。三个 task 都通过，这是对「性能测试基于 CUDA graph」
的第三方独立验证。

## MoE 为什么只打 allclose、不打 SNR

`test.py:85` 里 SNR 优先于 allclose，且硬卡 30 dB。MoE 这个算子是 **a4w4**——激活是 MXFP4，
约 2 bit 尾数——**未改动的 kernel** 对着 dequant 参考也只有 rel_norm_err ≈ 0.15，
换算 SNR ≈ 16.5 dB。打印 SNR 会让基线自己就判不合格，之后每个候选都被当成算错。
所以 MoE 走 `allclose`，用 task 自己的容差（`min_cosine` 0.97 / `max_rel_norm_err` 0.25）判定。

GDN（rel_err 0.0034 → **49.3 dB**）和 paged attention（0.0103 → **39.7 dB**）余量充足，正常打 SNR。
paged attention 那 1% 残差是 fp8 KV 量化噪声，是 workload 属性不是实现缺陷，
误差要涨到 3 倍才会碰线。

## 环境准备

KernelForge 装在 `/shared_nfs/jqliu/KernelForge`：

```bash
pip install -e . --no-deps
pip install --no-deps plotly pymongo sqlalchemy textual dash claude-agent-sdk \
    astunparse plotext plotille colorlover dash-bootstrap-components dash-svg \
    textual-plotext textual-fspicker
pip install "mcp>=1.23.0,<2.0.0"     # 见下
```

最后一行是个坑：`claude-agent-sdk` 用 `--no-deps` 装会缺 `mcp`，forge 起来后抛的却是
`ClaudeUnavailableError: claude-agent-sdk is not installed`——**报错信息误导，真实原因是
`ModuleNotFoundError: No module named 'mcp'`**。而直接 `pip install mcp` 会装 2.0.0，
与 `claude-agent-sdk 0.2.139` 要求的 `mcp<2.0.0` 冲突，必须锁版本。

还有一个假阳性要知道：forge 崩了之后 Arena 仍会把**未改动**的 kernel 重新打一次分并输出
`PASS ... Speedup: 1.01x`。**只看 Arena 的 PASS/speedup 判断不了 forge 是否真跑了**，
要看日志里有没有 `Starting autonomous iteration loop` 和 `pristine anchor agrees ...`。

# 没有覆盖到的部分

| | %E2E | 为什么没有 task |
|---|---|---|
| prefill RCCL all-reduce（bf16 256 MiB × 184/chunk） | ~15.3% | 通信算子，按要求排除 |
| decode `cross_device_reduce_2stage`（bf16 1 MiB × 185/step） | 8.05% | 通信算子，按要求排除 |
| `Cijk_..._MT16x16x1024`（router gate + shared gate_up + in_proj_ba） | **5.48%** | Tensile 预编译汇编，镜像内无源码 |
| `Cijk_..._MT192x64x128`（qkv_proj / in_proj_qkvz） | 3.26% | 同上 |
| `Cijk_..._MT64x32x128` + `PostGSU8`（shared down_proj） | 1.84% | 同上 |
| `Cijk_..._MT224x64x128`（lm_head） | 0.20% | 同上 |
| `_gemm_a16_w16`（o_proj / out_proj，Triton） | 1.75% | 源码可编辑，但单独权重不足以成 task |
| MoE routing / 排序 / 量化 / 激活 / 门（7 个 kernel） | ~6.6% | **源码可编辑，是最自然的补位候选**（见下） |
| `_gemma_fused_add_rmsnorm`（每层 2 次） | 1.56% | 源码可编辑，权重偏小 |
| prefill 计算（无法细分到 kernel） | ~25.5% | session 没抓 prefill trace |

session 的 **#5 热点算子没有 task，而且不可能有**：`Cijk_..._MT16x16x1024`
（253 次/step、2.9837 ms/step 8 rank 均值、占 decode 9.67%、**5.48% E2E**）
是 Tensile 生成的 GCN 汇编，预编译在 `/opt/rocm/lib/hipblaslt/library/*gfx950*.co` 里，
镜像内没有源码。唯一的杠杆是给 `aiter/tuned_gemm.py` 加调优条目、放宽 skinny 启发式、
或写个替代 kernel 再改派发——全是 dispatch / tuning 工作，不是 kernel 工作。
针对这些形状的 backend-replacement task 曾经建好并验证通过，为遵守筛选标准已移除，
policy 变了可以从 git 历史取回。

**MoE 周边链路**值得单独提一句：`topkGatingSoftmax`(1.74%) + 两个 `fused_mx_quant_moe_sort`(1.78%)
+ 两个 `opus_moe_sorting`(1.56%) + `act_and_mul`(0.77%) + `_fused_gate_sigmoid_mul_add`(0.73%)
合计约 **6.6% E2E**，7 个 kernel、每步 644 次 launch，全部是 aiter HIP C++ 和 sglang Triton，
完全符合「源码可编辑」的标准。按 task 2 那套「整链打包、暴露融合机会」的做法可以再建一个 task。
