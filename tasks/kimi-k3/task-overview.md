# Kimi-K3 kernel task 总览 —— session `20260814T191522Z`

从 Hyperloom session **`100132` Kimi-K3**（`/shared_nfs/hyperloom-claw/Kimi-K3/20260814T191522Z/`）
中抽取的 4 个 `image_kernel` task。

- **软件栈**：sglang `0.5.15.post1.dev20260723+g6c9fd0adc5`、aiter `dcd204ea`、
  ROCm 7.2.0、triton `3.6.0+git42270451`，8×MI355X（gfx950），TP=8
- **压测配置**：ISL 8192 / OSL 1024 / conc 64 / max_model_len 13312，mxfp4，192 条请求
- **数据来源**：8 个 rank 的原始 torch trace，全部重新解析
  （`geak/e2e_cycle0/profile/round_0/profile/1786763352.027878-TP-{0..7}.trace.json.gz`）。
  **不采用 session 自带 report 的任何现成结论。**
- **完整分析**：`/shared_nfs/jqliu/new-image-hot-kernels/hot-kernels-analysis/kimi-k3-hot-kernels.md`

| task | 算子 | **任务形态** | 后端 | **E2E 占比** | 覆盖阶段 |
|---|---|---|---|---|---|
| [`mi355x_sglang_triton_attn_residual_kimi_k3`](#一attention-residual注意力残差聚合) | K3 注意力残差聚合 | **单入口 · 2 kernel 流水** | triton (sglang) | **12.24%** | prefill + decode |
| [`mi355x_sglang_hip_moe_routing_sort_quant_kimi_k3`](#二moe-路由--排序--mx-量化) | MoE 路由/排序/MX 量化 | **多算子连续调用（3 入口 / 6 kernel）** | hip (aiter) | **6.35%** | prefill + decode |
| [`mi355x_sglang_triton_mla_decode_grouped_kimi_k3`](#三mla-grouped-decode-注意力) | MLA grouped decode 注意力 | **单入口 · 2 kernel 流水** | triton (sglang) | **4.65%** | decode |
| [`mi355x_sglang_flydsl_hgemm_small_m_kimi_k3`](#四flydsl-小-m-split-k-hgemm) | 小 M split-K bf16 GEMM | **单算子 · 单 kernel** | flydsl (aiter) | **4.55%** | decode |
| | | | | **合计 27.79%** | |

## 任务形态：单算子 vs. 多算子连续调用

这一列决定了 agent **能改到哪一层**，以及计时到底盖住了什么，值得先看清楚。

| 形态 | 含义 | 本目录里的 task |
|---|---|---|
| **单算子 · 单 kernel** | 一个 Python 入口 → 一次 kernel 启动。计时 = 该 kernel。 | 四 |
| **单入口 · N kernel 流水** | 一个 Python 入口，内部固定启动 N 个 kernel。中间结果是**私有**的，不对外暴露。计时 = 整个入口。 | 一、三 |
| **多算子连续调用** | N 个各自独立的公开 Python 入口，被模型按固定顺序串起来调用，前一个的输出张量是后一个的输入。计时 = 整条链。 | 二 |

**为什么这个区分重要：**

- **单入口多 kernel（task 一、三）—— kernel 边界是可以动的。** 中间张量
  （task 一的 `scores[T,16]` fp32、task 三的 `attn_logits`/`attn_lse`）只在入口内部存在，
  外界看不到，所以 agent 可以合并 kernel、改中间布局、甚至把两段完全融成一个 kernel。
  对 task 一来说**这恰恰就是主要优化方向**（NVIDIA SM100+ 上官方实现就是融合的，
  gfx950 缺这条路径）。约束只有一条：入口函数的公开签名不变。
- **多算子连续调用（task 二）—— kernel 边界不能动。** `biased_grouped_topk_hip` /
  `moe_sorting_opus_fwd` / `fused_dynamic_mx_quant_moe_sort_hip` 是三个独立的公开 API，
  模型别处也会调用它们，中间张量（`topk_ids`、`sorted_ids`、`num_valid_ids` 等）是
  **契约的一部分**。agent 只能优化各个 kernel 的内部，不能把三者融成一个。
  harness 因此逐段做正确性校验，而不是只看链尾输出。
  （链内的排序本身是多 phase 的：trace 里 `P0_v2` 和 `P23` 是两个独立 kernel，
  另有两个 phase，所以 3 个入口一共对应 6 个 kernel。）
- **单算子（task 四）** 没有这些问题，但要注意它的 4 个 case 是**互相独立的 shape**
  （M ∈ {8,64} × 两个投影族），不构成链；它们之所以在同一个 task 里，
  是因为 session 里这四种 shape 都由同一个 FlyDSL 派发路径服务。

## E2E 占比的口径

`E2E% = prefill 内占比 × 0.416 + decode 内占比 × 0.584`

权重由两条互相独立的路径推出后取中值：

- **A：trace 实测 prefill 速率外推** —— 16384 tok / 1060.9 ms → 15443 tok/s；
  1,572,864 个输入 token 需要 101.85 s，占总时长 249.116 s 的 **40.9%**。
- **B：实测稳态 ITL 反推** —— bench 原始结果 `median_itl_ms = 46.73` × 3072 个 decode 步
  = 143.56 s，反推 prefill **42.4%**。

两者相差 1.5 个百分点；用任一端点重算，下文所有数字变动 < 0.25 pt。

prefill 占比在 5 个 EXTEND 步上实测（24133 个 kernel，TP0 共 3974.64 ms）；
decode 占比在唯一一次被完整记录的 CUDA graph replay 上实测，
再加上图外 kernel 按 59 步均摊（单步基准 25.557 ms）。

## 先了解模型结构

Kimi-K3：93 层，hidden 7168。有三处不常规，下面每个 task 的定位都依赖它们。

- **混合注意力** —— 24 层 MLA 全注意力（q_lora 1536、kv_lora 512、qk_nope 128、
  qk_rope 64，96 头 → TP8 后每卡 12 头）+ **69 层 KDA 线性注意力**。
- **Latent MoE** —— 专家不在 hidden 7168 上算，而是在
  `routed_expert_hidden_size = 3584` 上算；896 个专家、top-16、2 个 shared expert、
  `moe_intermediate_size` 3072 → 每卡 384。93 层里有 92 层是 MoE
  （`first_k_dense_replace = 1`）。
- **Attention residual** —— `attn_res_block_size = 12`：模型不维护单一残差流，
  而是每 12 层把当时的 prefix 快照进一个 bank，每个聚合点对所有历史快照学一个
  softmax 混合权重。一次前向有 186 个聚合点。

## 哪些被刻意排除了

约 27% 的端到端 GPU 时间在 kernel 源码层面**不可攻击**。在提更多 task 之前值得知道原因：

- **hipBLASLt Tensile GEMM 全家**（`Cijk_…MT256x256x64` 7.98% + `MT256x16x64` 3.66%
  + `MT32x16x256` 1.44% + `PostGSU16` 1.37% + 其他 ≈ **15.4%**）——
  以预编译 code object 形式放在 `/opt/rocm/lib/hipblaslt/library/`，**镜像内没有源码**。
  而且它们已经跑到 **bf16 峰值的 62.5%**（大 shape 64–66%）；本 session 的 KERNEL 阶段
  为 `MT256x256x64` 重写了两轮，实测 0.33× / 0.57×，全部更慢。
- **两个 allreduce**（quickreduce 7.96% + aiter `cross_device_reduce_2stage` 5.57%）——
  按要求排除。顺带一提：aiter 那个在 prefill 里约 **98% 是自旋等 rank 3**，不是真通信。
- **ATen elementwise / memcpy / 采样**（≈12%）—— PyTorch 运行时算子，
  真正的收益在模型层融合，不属于 kernel task。

---

## 一、Attention residual（注意力残差聚合）

`mi355x_sglang_triton_attn_residual_kimi_k3` —— **E2E 12.24%**
（`_score_kernel` 8.20% + `_combine_kernel` 4.04%），triton，
`sglang/srt/layers/attn_residual.py`。

> **任务形态：单入口 · 2 kernel 流水。** 一个公开入口
> `aggregate_stream(prefix_sum, bank, nvb, score_proj, score_norm)`，
> 内部固定启动 `_score_kernel` → `_combine_kernel`。中间张量 `scores[T,16]` fp32
> 是**私有的**，不属于对外契约 —— **kernel 边界可以随便动，融成一个核正是本 task 的
> 主要优化方向。** 计时覆盖整个入口。

### 这是什么算子

**全 session 最大的可攻击 kernel，且是 K3 独有结构。** K3 不维护单一残差流：
每 `attn_res_block_size`（=12）层把当时的 pre-attention prefix 快照进一个 bank
`[T, NB, H]`；每个聚合点对 bank 里所有历史快照 + 当前 prefix 逐行打分、softmax、
加权求和 —— 于是每一层可以自己决定读哪个历史时刻的残差状态。

```
_score_kernel    grid (T, nvb+1)  block (512,1,1)   score = (v·cw) · rsqrt(mean(v²)+eps)
_combine_kernel  grid (T, H/1024) block (256,1,1)   softmax → 加权求和
```

`cw = score_norm.weight ⊙ score_proj.weight` 由 `get_cw()` 预乘缓存，
把"逐行 RMSNorm 再投影成标量"压成一次点积。

### 对应模型结构的哪个部分

**既不在 attention 里，也不在 MoE 里** —— 它是**层间的残差路由**，
取代了普通 Transformer 的 `x = x + f(x)` 跳连。一次前向触发 **186 次**，
prefill 和 decode 都跑，横跨全部 93 层。`nvb`（bank 深度）在一次前向内从 1 涨到 8，
所以最后几个聚合点的开销是第一个的 8 倍。

### 为什么值得让 agent 长跑

- 实测 **1.74–1.82 TB/s ≈ HBM3E 峰值（8 TB/s）的 22–23%**，各档 grid 一致，
  且严格线性于 `nvb`。
- `_use_fast()`（`attn_residual.py:32`）把一个 warp-specialized 融合 TMA kernel
  锁在 `torch.cuda.get_device_capability().major >= 10` 之后 —— 即**只给 NVIDIA SM100+**。
  gfx950 根本没有融合路径，永久走这条两段式 Triton fallback：每个聚合点把 bank 读两遍，
  中间还要把 `scores[T,16]` fp32 往显存里过一趟。
- 优化目标很明确：给 gfx950 写等价融合核（score → online softmax → combine，
  最好把 output RMSNorm 也融进去）。

**7 个 case**：prefill T=16384 的 nvb 1/4/8、T=8192、T=64；decode T=8 和 64。

---

## 二、MoE 路由 / 排序 / MX 量化

`mi355x_sglang_hip_moe_routing_sort_quant_kimi_k3` —— **E2E 6.35%**，HIP C++，aiter csrc。

> **任务形态：多算子连续调用（3 个入口 / 6 个 kernel）。** 本目录里唯一的一个。
> `biased_grouped_topk_hip` → `moe_sorting_opus_fwd` → `fused_dynamic_mx_quant_moe_sort_hip`
> 是三个**各自独立的公开 API**，模型别处也会调用；中间张量
> （`topk_ids`、`topk_weights`、`sorted_ids`、`sorted_weights`、`num_valid_ids`）
> 属于对外契约。**agent 只能优化每个 kernel 的内部，不能把三者融成一个。**
> 排序本身还是多 phase 的（trace 里 `P0_v2` 和 `P23` 是两个独立 kernel，另有两个 phase），
> 所以 3 个入口对应 6 个 kernel。计时覆盖整条链，但正确性是**逐段校验**的。

### 这是什么算子

MoE 流水线里**除 2-stage 专家 GEMM 之外**的全部环节。GEMM 部分已由
`tasks/image_kernel/mi355x_vllm_aiter_mxfp4_moe_2stage_kimi_k3` 覆盖（其 dims 与本
session 完全一致），周边这一段从来没被覆盖过。四个 kernel 按模型的真实顺序串成一条链计时：

| kernel | E2E% | 源码 |
|---|---|---|
| `aiter::grouped_topk_kernel` | 2.75 | `csrc/kernels/topk_softmax_kernels_group.cu:320` |
| `aiter::opus_moe_sorting_entry<…P23…>` | 1.20 | `csrc/include/moe_sorting_opus.h:109` |
| `aiter::opus_moe_sorting_entry<…P0_v2…>` | 0.90 | 同上 |
| `aiter::fused_mx_quant_moe_sort_kernel<bf16,fp8,256,16>` | 1.50 | `csrc/kernels/quant_kernels.cu:1731` |

```
biased_grouped_topk_hip(gating[T,896], bias[896]) → topk_weights[T,16] f32, topk_ids[T,16] i32
moe_sorting_opus_fwd(topk_ids, topk_weights)      → sorted_ids / sorted_weights / sorted_expert_ids / num_valid_ids
fused_dynamic_mx_quant_moe_sort_hip(latent[T,3584] bf16, sorted_ids, …)
                                                  → fp8_e4m3 输出 + e8m0 scale，group 32
```

### 对应模型结构的哪个部分

**每个 MoE 层的进出管道** —— 92 次/前向，与 92 个 MoE 层一一对应。
router → 专家分配 → 把 token 重排成专家连续的块 → 为 mxfp4 专家 GEMM 做激活量化。
因为 K3 是 latent-MoE，量化输入是 `[T, 3584]` 而不是 `[T, 7168]`。

### 为什么值得让 agent 长跑

- **6.35% 比 session 里任何单个 MoE GEMM kernel 都大**
  （最大的 `mfma_moe1 t64x128x256` 才 2.92%）。它在按 kernel 名排序的榜单里完全隐形，
  这正是它一直没被做成 task 的原因。
- 小 T 下整条链是 launch 延迟受限：T=64 时 12.0 µs，而规模大 256 倍的 T=16384 才 70.3 µs。
- 排序的 workspace 在 T=16384 时是 14.7 MB，且分 4 个 phase 跑。

**5 个 case**：prefill T=16384 / 8192 / 64，decode T=8 和 64。
`block_size`（大 M 档 64、小 M 档 32）不是猜的 —— 用
`max_num_tokens_padded = T×topk + num_experts×block_size − topk` 反解 trace 里的
buffer 长度得到，`sorted_expert_ids` 长度逐项对得上。
`quant` 只出现在小 M 和 decode 的 case 里，因为大 M 下 session 用的 MoE1 kernel
（`…_fp8q_sort_async_…`）把量化融进了 GEMM 前导，独立量化核在 prefill 只有 0.02%。

---

## 三、MLA grouped decode 注意力

`mi355x_sglang_triton_mla_decode_grouped_kimi_k3` —— **E2E 4.65%**
（`_fwd_grouped_kernel_stage1` 3.96% + `_fwd_kernel_stage2` 0.69%），triton，
`sglang/kernels/ops/attention/decode_attention.py`。

> **任务形态：单入口 · 2 kernel 流水。** 一个公开入口
> `decode_attention_fwd_grouped(...)`，内部固定启动
> `_decode_grouped_att_m_fwd`（stage1）→ `_decode_softmax_reducev_fwd`（stage2）。
> `attn_logits` / `attn_lse` 虽然由调用方分配，但只作为两段之间的中转，
> 输出只有 `o` —— **两段可以融合**（split 数小时尤其值得），
> 参考实现也是 split-agnostic 的，所以 `num_kv_splits` 的处理方式可以自由改。
> 计时覆盖整个入口。

### 这是什么算子

**吸收式 MLA** 的 flash-decoding split-KV 注意力，入口 `decode_attention_fwd_grouped`。
stage1 扫 KV 分块写出 per-split 部分结果，stage2 做归约。

```
q            [bs, 12, 576]            bf16   12 = 96 头 / TP8；576 = kv_lora 512 + rope 64
k_buffer     [num_kv_tokens, 1, 576]  bf16
v_buffer     [num_kv_tokens, 1, 512]  bf16   k_buffer[..., :512] 的非连续视图
o            [bs, 12, 512]            bf16
attn_logits  [bs, 12, 256, 512]       f32
```

`kv_head_num = 1` → `kv_group_num = 12`，这正是把调用路由到 *grouped* kernel 的原因。

### 对应模型结构的哪个部分

**24 层 MLA 全注意力** —— trace 里每个 decode 步正好 24 次 stage1 + 24 次 stage2，
归属由此钉死。其余 69 层是 KDA 线性注意力，永远不会走到这个 kernel。
session 用的是 `--attention-backend triton`，所以这就是实际执行的路径。

**只覆盖 decode 是算子属性，不是漏覆盖**：K3 的 MLA prefill 走的是结构完全不同的
`_fwd_kernel`（`extend_attention.py`，1.50% E2E），它在 decode 里 0 次调用；
反过来本 kernel 在 prefill 里也 0 次调用。

### 为什么值得让 agent 长跑

- `attn_logits` 是 `bs × 12 × 256 × 512 × 4 B`，**bs=64 时 402 MB**，每层写一遍读一遍。
  `max_kv_splits = 256` 是被 **CU 数量**卡出来的，不是调优选的 ——
  来自 `_mla_decode_kv_splits_cap(8, 256 CU, 13312)`。
- `q` 和 `k_buffer` 共用同一块 576 宽的存储，`v_buffer` 只是它前 512 列，
  一次 load 可以同时服务 K 和 V。
- 12 个 query head 对 1 个 KV head 是天然的 LDS 复用组；split 数少时 stage1/stage2 可以融合。

**4 个 case**：bs ∈ {8, 64} × KV 长度 ∈ {8192（ISL，生成开始）、9216（ISL+OSL，生成结束）}
—— 覆盖这个 workload 的完整 decode 区间。

---

## 四、FlyDSL 小 M split-K HGEMM

`mi355x_sglang_flydsl_hgemm_small_m_kimi_k3` —— **E2E 4.55%**
（`hgemm_bf16_16x64x64x7_SPK2_W1x2x1_BLDS1_TN_AS1_0`），FlyDSL，
`aiter/ops/flydsl/kernels/splitk_hgemm.py`。

> **任务形态：单算子 · 单 kernel。** 本目录里最干净的一个：一个入口
> `aiter.tuned_gemm.tgemm.mm(a, b)` → 一次 kernel 启动，golden 就是
> `a @ b.T`。注意 4 个 case 是**互相独立的 shape**（M ∈ {8,64} × 两个投影族），
> **不构成链** —— 它们放在同一个 task 里，是因为 session 里这四种 shape 都由同一条
> FlyDSL 派发路径服务。split-K 的两段（partial + 收尾归约）在 kernel 内部，
> 不是两次启动。

### 这是什么算子

专为极小 M 特化的 bf16 GEMM：tile 16×64×64、stages 7、split-K 2、warps 1/2/1、
`b_to_lds`、async copy、TN 布局。kernel 名可以从 `splitk_hgemm.py:249` 的生成 f-string
逐字段反解出来。`tile_m = 16` 是 bs=8 的 decode batch 向上补齐到 FlyDSL 最小 M tile ——
**一半的 M tile 是空转的**，而 `split_k=2` 的存在只是为了给一个 16 行的问题
凑出并行度。

### 对应模型结构的哪个部分

**decode batch 下的注意力投影。** 每个 decode 步 162 次调用 = 69 + 93：

| shape | 次/步 | 是什么 |
|---|---|---|
| `[M, 7168] × [7168, 6144]` | 69 | KDA 输入投影 —— N = 4 × (12 头 × 128)，即 q/k/v/gate；**69 = KDA 层数** |
| `[M, 1536] × [1536, 7168]` | 93 | o_proj —— KDA 69 + MLA 24，两者本地输出维都是 12 × 128 = 1536 |

在 **prefill** 里，这两个完全相同的投影以 M = 16384 运行，dispatcher 把它们送去
hipBLASLt（`Cijk_…MT256x256x64`，调用次数同样是 69 + 93）；到 decode 的小 M 才落到
这条 FlyDSL 路径。所以"只有 decode"是**派发行为**，而 prefill 那一侧既不可编辑也不慢。

shape 有两重独立佐证：一是上面的调用次数算术；二是
`aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv` —— 运行栈真正查的那张调优表 ——
里对这两个 shape 都有 `gfx950 / cu_num=256 / M=8` 的行，选的是
`flydsl_gemm7_…t16x64x64_split_k2_block_m_warp1_block_n_warp2_…`，
正是 session 里那个 kernel（`gemm`**`7`** 后缀就是 stage 数）。
这张 CSV 的全部 `(N,K)` 组合还完整复现了我从 trace 做的投影普查，
包括 `2304,1536`（MLA `q_b_proj`）和 `2112,7168`（q_a + kv_a + rope 融合）。

### 为什么值得让 agent 长跑

- session 里最大的**非注意力、非通信**可攻击 kernel。
- 明确是**访存/延迟受限而非算力受限**：M=8、N=6144、K=7168 时算术量只有 0.7 GFLOP，
  却要读约 88 MB 权重。预算花在权重流和 split-K 收尾上，不在 MFMA 管线。
- 入口用的是**真实 dispatcher**（`aiter.tuned_gemm.tgemm.mm`），所以调优 CSV 是正当的
  优化杠杆 —— 但按 M 分档的派发必须保留（M=8 → `t16x64x64/split_k2`，
  M=64 → `t32x64x128/split_k1`）。harness 每个 case 都记录解析出的
  `libtype` / `kernelName`，偷偷 fallback 到 hipBLASLt 会在报告里露出来。

**3 个 case**：`(M=8, N=6144, K=7168)`、`(M=8, N=7168, K=1536)`、`(M=64, N=7168, K=1536)`。第四种组合 `(M=64, N=6144, K=7168)` 被刻意排除 —— 调优 CSV 对它选的是 `libtype=opus` 而不是 FlyDSL，收进来就会去测 `module_deepgemm_opus`（编辑面之外）。harness 里有硬性守卫强制每个 case 必须解析到 `libtype=flydsl`。

---

## 四个 task 共同遵守的约定

- **shape 一律来自 session。** prefill 的 shape 直接读 trace 的 `Input Dims`
  （`record_shapes=True`）；decode 因为跑在 CUDA graph 里没有 CPU 栈，
  由调用点源码 + 模型 config 反推。每个 case 都带 `exact_shape_source` 字段说明属于哪种。
- **精度和性能用同一份 case list。** `run_correctness` 和 `run_performance` 迭代同一个
  `CASES`，所以被计时的 shape 一定是被校验过的 shape；harness 里没有任何
  correctness/performance 分支。
- **性能一律基于 CUDA graph 测量** —— 走 `_benchmark_cuda_graph_or_events`
  （capture 一次，每次计时迭代 replay）。四个 prompt 都写明改写后必须保持 graph-capturable。
- **参考实现全部是向量化的 torch**（float64），没有逐 token / 逐 head 的 Python 循环
  （显存吃紧处按 token 或 batch 轴分块）。**不复用任何位于 agent 编辑面之内的 eager 参考**
  —— 否则一个坏改动可以自己验证自己。
- **每个 task 只有 4 个文件**：`config.yaml`、`session_cases.json`、
  `scripts/task_runner.py`、`scripts/forge_driver.py`。本目录下只保留这一份说明文档，
  各 task 目录内不再放 README。细节按下表分布，需要查证时直接看对应文件：

  | 想查什么 | 去哪里看 |
  |---|---|
  | 每个 case 的 shape、来源（trace 实测 / 推导）、trace 实测耗时与 grid | `session_cases.json` 的 `cases` 和 `provenance` |
  | shape 是怎么定出来的（`block_size` 反解、`max_kv_splits` 计算、kernel 名反解、调用次数归属） | `session_cases.json` 的 `shape_evidence` / `block_size_evidence` / `layout_derivation` / `kernel_name_decode` |
  | 优化方向、可动与不可动的契约、禁止事项 | `config.yaml` 的 `prompt.instructions` |
  | 入口点选择理由、golden 的数学推导、精度校验口径 | `scripts/task_runner.py` 顶部 docstring |
  | 算子是什么、对应模型哪个部分、E2E 占比 | 本文档对应章节 |

## 实测验证结果（2026-08-17，MI355X gfx950 单卡）

这些 harness 是**单 rank** 的 kernel 测试，不需要 TP=8，所以已经在本机全部实跑过。

### 耗时（全部 < 5 分钟，实际都在 1 分钟内）

| task | compile | correctness | performance |
|---|---|---|---|
| attn_residual | 11.4 s | **9.7 s** (7/7 PASS) | **8.9 s** |
| moe_routing_sort_quant | 50.9 s | **59.8 s** (5/5 PASS) | **57.9 s** |
| mla_decode_grouped | 5.5 s | **5.3 s** (4/4 PASS) | **4.5 s** |
| flydsl_hgemm_small_m | 16.9 s | **6.8 s** (3/3 PASS) | **6.6 s** |

MoE task 的 ~50 s 里绝大部分是 `AITER_REBUILD=2` 触发的三个模块 JIT 重编（冷编译实测
moe_asm 33.2 s + quant 13.2 s + moe_sorting_opus 6.7 s）。**不会触发全量 aiter 编译**：
这三个模块各只有 2–5 个源文件、不含 CK、不走 blob_gen，而 `module_aiter_core` 被
`jit/core.py` 的 `rebuilded_list` 豁免。

### 性能确实走 CUDA graph，且与 trace 对得上

四个 task 的每一个 case 都报告 `benchmark_method=cuda_graph`，无一例回退到 event 计时。
实测单次调用耗时与 session trace 的实测值高度一致：

| case | harness 实测 | trace 实测 |
|---|---|---|
| attn-res prefill t16384 nvb4 | 0.928 ms | 0.653 + 0.319 = **0.972 ms** |
| mla-decode bs8 kv8192 | 0.0874 ms | 71.7 + 12.5 µs = **0.0842 ms** |
| moe-route prefill t16384 | 0.1019 ms | 70.3 + 40.8 µs = **0.111 ms** |
| hgemm decode bs8 kdaqkv | 0.0154 ms | **0.0123 ms** |

### 精度结果

- **attn_residual** 7/7：cos 0.9999986–0.9999990，rel_max_err 0.0020–0.0034
- **mla_decode_grouped** 4/4：cos 0.9999982–0.9999983，rel_max_err 0.0026–0.0029；
  split 调度器给出 bs=8→222、bs=64→28，与后端行为一致
- **flydsl_hgemm_small_m** 3/3：cos 0.9999970–0.9999986，全部 `lib=flydsl`
- **moe_routing_sort_quant** 5/5：topk 权重误差 1.5e-8～3.0e-8、`sel_gap ≤ 0`；
  sort 语义校验 `w_err = 0`；quant `rel_max_err` 0.032–0.055、`p2_dev` 0.087

### agent 编辑能被 JIT 采用（逐后端实测，不是推断）

| 后端 | 机制 | 实测结论 |
|---|---|---|
| **Triton**（task 一、三） | Triton 按 kernel 源码 hash 建缓存 | 把 workspace 副本里 `_score_kernel` 的 `dotv*rrms` 改成 `*1.5`，精度立刻从 0.9999987 掉到 0.9944341 而 FAIL ✓ |
| **aiter csrc**（task 二） | `AITER_REBUILD=2` | 向 `topk_softmax_kernels_group.cu` 和 `moe_sorting_opus.h` 各注入 `#error`，编译均失败并回传 probe 文本 ✓ |
| **FlyDSL**（task 四） | builder 每进程重新执行 | 向 `compile_hgemm_kernel` 注入 `raise`，热缓存下仍然抛出 ✓ |

**其中 aiter 那条是一个必须修的真 bug，已修复。** `jit/core.py` 的 `get_module()`
只在 **arch 不匹配**时重编，**从不校验源码 hash/mtime**（`core.py:633-659`）。实测对照：

| 场景 | 耗时 | `.so` 是否重建 |
|---|---|---|
| 改了 csrc、**无** `AITER_REBUILD` | 0.1 s | **否 —— 编辑被静默忽略** |
| 改了 csrc、`AITER_REBUILD=2` | 16.9 s | 是 |

所以两个 aiter task 的 `_configure()` 都设了 `AITER_REBUILD=2`（不是 `=1`——那会连
ninja 构建目录一起清掉，每次都变成全新冷编译）。

### 本轮实跑中发现并修掉的 4 个问题

1. **`mla_decode_grouped` 精度直接崩** —— 我给 `v_scale` 传了 `None`，但
   `_fwd_kernel_stage2` 会解引用它（`decode_attention.py:803`），Triton 报
   `'NoneType' object has no attribute 'type'`。后端在非 fp8 KV 时传的是 `1.0`
   （`triton_backend.py:1352-1353`），已改正。
2. **`moe_routing` 的 topk 校验方法本身是错的** —— 我原来按 expert id 对齐两个独立
   top-k 再比权重。896 个专家下第 16/17 名的 biased 分数常在 fp32 噪声内并列，
   kernel 与 torch 会选到不同专家，对齐后就在比不相干的专家（实测报出 8.7e-3 假误差）。
   改成 tie-safe 双段校验：(A) 选择合法性 `min(biased[chosen]) ≥ max(biased[rest]) − eps`；
   (B) 权重必须与 **kernel 自己选出的 id** 一致。改后误差降到 1.5e-8。
3. **`moe_routing` 的 quant 校验假设了错误的 scale 布局** —— `out` 是按 token 行对齐的，
   但 `scales` 是按排序 slot 且经硬件 swizzle 排布的（实测某行读出
   `[119, 0, 119, 0, …]`，非零行数恰等于 `num_valid_ids`）。原来的 `scales[t]` 查表
   反量化出全零（rel = 1.0）。改成**完全不依赖 scale 布局**的校验：从 payload 反推每组
   隐含 scale，验证它是 2 的幂，再用 `q · 2^e` 与输入比对；scale 张量只做
   "写入行数 == num_valid_ids" 的结构性检查。
4. **`flydsl_hgemm` 有一个 case 根本没在测目标 kernel** —— `(M=64, N=6144, K=7168)` 在
   `kimik3_bf16_tuned_gemm.csv` 里选的是 **`libtype=opus`**（`opus_gemm_flatmm_splitk_…`，
   splitK=4），不是 FlyDSL；它还会把 `module_deepgemm_opus` 的大量实例编译拖进每次运行
   （correctness 从 6.8 s 涨到 23 s）。已删除该 case（4→3），并在 `_prepare()` 里加了
   **硬性守卫**：解析出的 `libtype` 必须等于 `params.require_libtype`（`flydsl`），
   否则直接断言失败 —— 这把"不许绕开 FlyDSL"从 prompt 里的文字变成了机器强制。

### 已知的一处小缺口（不阻塞运行）

`src/perf_helper_materialization.py:104` 的 `image_kernel_targets()` 硬编码
`tasks/image_kernel/*/scripts/task_runner.py`，**不扫 `tasks/kimi-k3/`**。影响仅限
开发期工具 `make sync-perf-helpers` 不会同步这四个 task 里的 helper 占位块。

**运行时不受影响**：`materialize_perf_helpers_in_workspace()` 作用于**拷贝出来的
workspace**，只检查 `scripts/task_runner.py` 里有没有 AKA marker，与那个 glob 无关 ——
上面四个 task 的 performance 实跑就是这么注入成功的。若要让开发期工具也覆盖，把
`:104` 的 glob 改成同时匹配 `tasks/kimi-k3/*/scripts/task_runner.py` 即可。这属于
Arena 基础设施、不在本目录内，我没有擅自改动。

### Arena + forge-loop 端到端验证（四个 task 全部跑通）

用 `/shared_nfs/jqliu/set_env.sh` 的环境 + `/shared_nfs/jqliu/run_arena/run.sh` 的运行方式
（log/workspace 改到 `/tmp`，与 `config.forge_mxfp4.yaml` 里的注释一致，避免 NFS page cache
把宿主 cgroup 顶到上限被 OOM kill），逐个跑到 forge-loop 主业务后停掉。
KernelForge 从 `/shared_nfs/jqliu/KernelForge` 以 `pip install -e ".[claude]"` 装入
`/opt/venv`（不含 torch 依赖，未影响镜像的 torch 2.9.1+rocm7.2.0）。

| task | task id | fellow 解析 | forge baseline | 基线一致性 | 到达阶段 |
|---|---|---|---|---|---|
| attn_residual | `kimi-k3/mi355x_sglang_triton_attn_residual_kimi_k3` | `triton-fellow` | 0.539 ms | **7 / 7 case** | INITIAL_ANALYSIS |
| moe_routing_sort_quant | `kimi-k3/mi355x_sglang_hip_moe_routing_sort_quant_kimi_k3` | `hip-fellow` | 0.050 ms | **5 / 5 case** | INITIAL_ANALYSIS |
| mla_decode_grouped | `kimi-k3/mi355x_sglang_triton_mla_decode_grouped_kimi_k3` | `triton-fellow` | 0.247 ms | **4 / 4 case** | INITIAL_ANALYSIS |
| flydsl_hgemm_small_m | `kimi-k3/mi355x_sglang_flydsl_hgemm_small_m_kimi_k3` | `flydsl-fellow` | 0.009 ms | **3 / 3 case** | INITIAL_ANALYSIS |

四个都干净走完了这条链路，没有一个秒挂：

1. **task 被 Arena 发现** —— `src/tasks.py` 用 `tasks/**/config.yaml` 递归扫描，
   task id 就是相对 `tasks/` 的路径，所以 `kimi-k3/` 这个新目录天然可用，
   无需改 Arena 任何代码（439 个 task 中包含这 4 个）。
2. **workspace seeding 正常** —— `image_repo_exclude` 生效：aiter 从 2.8 G 只拷出
   **358 M**（`aiter/jit/flydsl_cache` 2.1 G 和 `aiter/jit/build` 78 M 被排除），
   sglang 侧同理。
3. **`repository_language` → fellow 映射全部正确**：triton / hip / flydsl 三种都命中。
4. **prepare 阶段四个 task 全部跳过** —— 日志都是
   `[prepare] task already conforms to the driver contract; skipping`，
   省掉了 forge 先派一个 LLM agent 去写 driver 的那一步（这正是随包附带
   `scripts/forge_driver.py` 的目的）。CLI 在这里用的门槛是
   （`cli.py:1601-1608`）：

   ```python
   pf = preflight_task(driver=driver, snr_threshold=30.0,
                       require_graph=True, require_profile=True,
                       expected_case_ids=declared_case_ids(invocation_spec_file))
   ```

   即四项同时满足才算 conform，逐项实测余量：

   | 门槛 | 要求 | attn_residual | mla_decode | flydsl_hgemm | moe_routing |
   |---|---|---|---|---|---|
   | `snr_threshold` | ≥ 30 dB | **55.63** | **54.51** | **52.23** | **138.92** |
   | `require_graph` | bench 必须真跑 graph replay | ✓ | ✓ | ✓ | ✓ |
   | `require_profile` | `--profile-run` 必须 exit 0 | ✓ | ✓ | ✓ | ✓ |
   | `expected_case_ids` | `case_ms` 行数必须覆盖**全部**声明 case，子集不算 | 7/7 | 4/4 | 3/3 | 5/5 |

   最后一项是最容易踩的：源码注释写得很直白 ——「a driver that times a subset of the
   task's cases is not "already conforming", it just measures less than the task
   asks for」。四个 driver 都是遍历同一份 `CASES`，所以天然全覆盖。
   MoE 那个 138.92 dB 特别高，是因为它的 SNR 算在 top-k 路由权重上，
   而权重误差只有 ~1.5e-8。
5. **Arena 自己的 baseline 与 harness 实测一致** —— 例如 MoE task，Arena 测得
   5 个 case 平均 0.0495 ms，与本文档上一节 harness 直接实测的
   0.1019 / 0.0624 / 0.0287 / 0.0266 / 0.0286 ms 的均值吻合。
6. **`pristine anchor agrees with the task reference on N of N measured case(s)`** ——
   四个 task 的每一个 case 都被 forge 独立复测且与 `session_cases.json` 声明的数量一致，
   说明 case 定义、correctness 与 performance 三者在 forge 侧也自洽。
7. **进入 `[analysis] building commit-bound analysis bundle (INITIAL_ANALYSIS)`** ——
   已是 forge-loop 的主业务（LLM 分析 + 并行 specialist 编排），验证到此即停。

从启动到 baseline 完成的耗时：attn_residual ~80 s、mla_decode ~50 s、flydsl_hgemm ~90 s、
**moe_routing ~9 min**。MoE 那个偏长是因为 forge 在 baseline 阶段会多次调用 driver，
而每次调用都要付 `AITER_REBUILD=2` 的三模块重编（~50 s/次）—— 这是"编辑必须生效"
换来的代价，单次调用仍远低于 5 分钟，只是 forge 的多次调用叠加起来偏慢。
若要提速，可考虑让 forge 在同一次 driver 进程里跑完 correctness+bench，或接受首轮偏慢。

一个值得记录的交互：forge 会把镜像里 11 个预编译 `.so`
seed 进它自己的隔离 `forge_experiments/aiter_cache`
（`[aiter-cache] seeded 11 prebuilt module(s)`）。**如果没有 `AITER_REBUILD=2`，
agent 对 csrc 的修改就会被这批 seed 进来的 .so 完全遮蔽** —— 这恰好是上面修掉的那个 bug
在真实 forge 流程下的表现形式，说明该修复是必需的而不是纸面推演。

### 验证过程未污染镜像

期间为了做 pickup 实测，临时改过 `aiter` 的 `quant_kernels.cu` /
`topk_softmax_kernels_group.cu` / `moe_sorting_opus.h` / `splitk_hgemm.py` 和 sglang 的
`attn_residual.py`，全部已还原：两个仓库 `git diff --name-only HEAD` 均为 0 个文件。

## 两个待确认事项

- **`bs=64` 的 decode case 是否保留。** profiling 用的是 `warmup 0s`，
  所以 trace 只抓到 **bs=8** 的爬坡期；而这一轮真实稳态是 **bs=64**
  （`conc=64`、`max_running_requests=64`，3077 个 decode 步里有 3072 个在 bs=64；
  `median_itl_ms=46.73` vs bs=8 的 25.56 ms 佐证）。两者都是本次运行的真实 shape，
  目前都参与打分。若要严格只用 trace 直读的 shape，
  从各个 `session_cases.json` 里删掉 `*-bs64-*` 那几条即可，没有其他地方引用。
- ~~四个 harness 一次都没有实跑过。~~ **已全部实跑并修复**，见上一节「实测验证结果」——这些 harness 是单 rank 的，不需要 TP=8，本机单卡即可运行。
