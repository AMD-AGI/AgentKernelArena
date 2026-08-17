# MiniMax-M3-MXFP4 算子 task 总览（session 20260815T100002Z）

从一个 Hyperloom session 里抽出的 4 个 `image_kernel` task。

```text
session   /shared_nfs/hyperloom-claw/MiniMax-M3-MXFP4/20260815T100002Z
硬件/软件  MI355X x8 / TP=8 / quark MXFP4 / vllm 0.26.0 / ROCm 7.2.3 / aiter 0.1.16.post3
负载      ISL 8192 / OSL 1024 / conc 64 / max_model_len 13312 / block_size 128
镜像      harbor.crusoe.primus-safe.amd.com/proxy/vllm/vllm-openai-rocm:v0.26.0
```

运行方式：

```yaml
tasks:
  - minimax-m3/mi355x_vllm_triton_paged_attention_2d_minimax_m3
  - minimax-m3/mi355x_vllm_triton_gqa_sparse_attn_prefill_minimax_m3
  - minimax-m3/mi355x_vllm_triton_sparse_index_topk_minimax_m3
  - minimax-m3/mi355x_vllm_flydsl_mxfp4_moe_2stage_minimax_m3
```

**E2E 占比的口径**：归一化端到端 GPU 时间占比（prefill 步权重 40.2% + decode 步权重 59.8% 加权），
不是采样窗口占比——两者能差 2 倍以上。完整推导见
`/shared_nfs/jqliu/new-image-hot-kernels/hot-kernels-analysis/minimax-m3-hot-kernels.md`。

---

## 一、四个 task 一览（按预期收益排序）

| # | task | E2E% | 语言 | **调用构成** | 当前效率 | 头部空间 |
|---|---|---|---|---|---|---|
| 1 | `..._triton_paged_attention_2d_minimax_m3` | **11.35%** | triton | **单算子**（1 个 kernel） | 255 GB/s = **HBM 峰值的 3.2%** | **大** —— 64 个 CTA 跑在 256 CU 上，且没有 KV split-K |
| 2 | `..._triton_gqa_sparse_attn_prefill_minimax_m3` | **8.53%** | triton | **单算子**（1 个 kernel） | 83.6 TFLOP/s = **bf16 峰值的 3.6%** | 大 —— `BLOCK_SIZE_Q=1`、`num_warps=1`，QK GEMM 的 M 只有 8 |
| 3 | `..._triton_sparse_index_topk_minimax_m3` | **8.01%** | triton | **多算子链**（decode 3 个 / prefill 2 个） | 打分 kernel 已到 HBM 峰值的 66.5% | 中 —— 收益在链路融合，不在单个 kernel |
| 4 | `..._flydsl_mxfp4_moe_2stage_minimax_m3` | 6.01%（属 18.60% 的功能） | flydsl | **多算子链**（prefill 10 个 / decode 6 个） | gemm2 602 vs gemm1 2315 TFLOP/s | 中 —— 同类 shape 差 3.8 倍，且没命中调优表 |

> **"调用构成"指的是被计时区间（`task_runner.py` 的 `_run()`）里实际发出的 GPU kernel 数**，用 torch profiler
> 在本容器实测得到，不是估计。这个区别直接决定 agent 的优化空间：单算子 task 只能靠把那一个 kernel 做快；
> 多算子链 task 还可以**减少 launch、砍掉中间的全局显存往返、把小 pass 融进 epilogue** —— 计分用的是整个
> `_run()` 的墙钟时间，所以删掉一个 pass 和加速一个 kernel 同样算数。

按模块归类：

| 模块 | 归一化 E2E% | 本目录覆盖 |
|---|---|---|
| 注意力（全部） | 42.13% | task 1 / 2 / 3 = **27.89%** |
| MoE（全部） | 35.67% | task 4 = **6.01%**（路由专家 GEMM 的 prefill 一半；decode 一半 12.62% 未覆盖，见第五节） |
| 通信 all-reduce | 17.31% | 不做（见第五节） |
| 其它（Norm / 稠密 MLP / LM head / 搬运 / 空闲） | 4.89% | 不做 |

---

## 二、先看模型结构：这 4 个 task 分别落在哪

MiniMax-M3 有 **60 层**，是个混合结构。`config.json` 里 `sparse_attention_freq` 和 `moe_layer_freq`
的前 3 项都是 0、后 57 项都是 1，也就是：

```
                     ┌─ 第 0~2 层（3 层，"稠密层"）─────────────────────────────────────┐
                     │   RMSNorm                                                       │
                     │   qkv 投影  [.,6144]x[6144,1280]                                 │
                     │   qk-norm + RoPE + KV 写入                                       │
   输入 [T, 6144] ──▶│   ★ 稠密 GQA 注意力  ◀═══════════════════ task 1                 │──▶
                     │   o 投影   [.,1024]x[1024,6144]                                  │
                     │   all-reduce                                                     │
                     │   RMSNorm                                                        │
                     │   稠密 MLP（inter 12288/TP8 = 1536，bf16 未量化）                  │
                     │   all-reduce                                                     │
                     └─────────────────────────────────────────────────────────────────┘
                     ┌─ 第 3~59 层（57 层，"稀疏层"）────────────────────────────────────┐
                     │   RMSNorm                                                        │
                     │   qkv 投影  [.,6144]x[6144,1536]                                  │
                     │             （1536 = 1024 q + 256 kv + 256 index）                │
                     │   qk-norm + RoPE + KV 写入 + index-K 写入                          │
                     │   ★ 选块器：给 <=104 个 128-token 块打分，选 top-16 ◀════ task 3    │
                     │   ★ 块稀疏 GQA 注意力（只读选中的 16 个块）      ◀══════ task 2      │──▶
                     │   o 投影   [.,1024]x[1024,6144]                                   │
                     │   all-reduce                                                      │
                     │   RMSNorm                                                         │
                     │   MoE：router gate → top-4 → token 排序                             │
                     │        ★ 128 个路由专家的 2 段 FFN GEMM ◀══════ task 4             │
                     │        + 1 个共享专家（inter 384，未补齐）                           │
                     │        缩放 x2.0 → 与共享专家相加                                   │
                     │   all-reduce                                                       │
                     └──────────────────────────────────────────────────────────────────┘
```

每 rank（TP=8）的几何：`hidden=6144`、**8 个 q 头 / 1 个 kv 头**（GQA 8:1）、`head_dim=128`、
1 个 index 头（`index_dim=128`）、128 个专家 top-4、专家 `inter=512`（3072/8=384 **补齐到 512**）、
`block_size=128`、KV cache `num_blocks=41215`。

---

## 三、逐个 task

### Task 1 —— `mi355x_vllm_triton_paged_attention_2d_minimax_m3`

#### 调用构成：单算子

被计时区间 `chunked_prefill_paged_decode(...)` 在纯 decode 批下**只发 1 个 kernel**（实测）：

```text
x1  kernel_paged_attention_2d
```

`key=None/value=None` 所以不写 KV cache；`max_query_len=1` 所以 prefill 分支
（`_fwd_kernel`）整个跳过。**agent 的全部空间就在这一个 kernel 里。**

#### 是什么算子

vLLM 的 Triton 分页注意力 decode kernel **`kernel_paged_attention_2d`**
（`vllm/v1/attention/ops/chunked_prefill_paged_decode.py:46`，launcher 在同文件 `:447`），
经 `vllm::unified_attention_with_output` 调用。一次调用处理一批 query 长度为 1 的行，
每行从分页 KV cache 里读完自己整条序列的 K/V 做一次完整 softmax attention。

#### E2E 占比

**11.35%** —— 全 session 排第 2 的单个 kernel。

| | |
|---|---|
| decode 步内 | 17.76% |
| chunked-prefill 步内 | 1.82% |
| 调用次数 | 3 次/步（两种步都有） |
| 单次耗时 | 1168 µs（decode）/ 1172 µs（prefill 步） |

#### 对应模型结构的哪个部分

**前 3 层稠密注意力层的注意力核心**，只覆盖其中的 decode token 路径。

- 这 3 层是 `sparse_attention_freq` 里为 0 的层，不走块稀疏，用标准 vLLM `Attention` 层。
- 纯 decode 步里它处理 64 个 decode token；chunked-prefill 步里处理夹带的 61 个 decode token
  —— `filter_by_query_len=True` 会让它直接跳过那 8133 个 prefill 行，prefill 部分由同文件的
  `_fwd_kernel` 负责（占 0.91%，不在本 task 内）。
- 输入：`q [64,8,128]` bf16（stride `[1280,128,1]`，因为是 fused qkv buffer 的切片）、
  KV cache 每层 41215 blocks x 128 tokens x 1 kv head x 128、`seq_lens` 均值 9098。

#### 为什么值得做

grid 就是 `(num_seqs, num_kv_heads) = (64, 1)`：**64 个 workgroup 跑在 256 个 CU 上，且没有对 KV 长度做
split-K**，每个 CTA 顺序扫完自己那条 9098 token 的 KV。实测 **255 GB/s = HBM 峰值的 3.2%**。
同一份 trace 里做了 8 路 split-K 的 `_decode_index_score_kernel`（task 3）跑到 5.32 TB/s，
**效率是它的 21 倍**。

它落到 Triton 的原因：`use_rocm_custom_paged_attention()`（`vllm/platforms/rocm.py:346`）只接受
`block_size ∈ {16,32}`，而 M3 因为块稀疏必须用 `block_size=128`；`head_size=128`、`gqa_ratio=8`
本来都是合格的。server.log 里有对应的
`Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.`

#### 测试 case（精度 / 性能同一份，同尺寸）

| id | 阶段 | num_seqs | query buffer 行数 | ctx_len |
|---|---|---|---|---|
| `m3-decode-bs64-ctx9098`（primary） | decode 步，平均上下文 | 64 | 64 | 9098 |
| `m3-decode-bs64-ctx8192` | decode 步，刚开始生成 | 64 | 64 | 8192 |
| `m3-decode-bs64-ctx9216` | decode 步，生成到头（ISL+OSL） | 64 | 64 | 9216 |
| `m3-chunkedstep-bs61-ctx9098` | chunked-prefill 步 | 61 | 8192 | 9098 |

公共：`num_query_heads=8`、`num_kv_heads=1`、`head_size=128`、`block_size=128`、bf16、`num_blocks=41215`。

两个刻意复现的细节：**query stride 是 1280 不是 1024**（q/k/v 是同一 fused qkv buffer 的切片，
非单位行 stride 会改变 kernel 的访存模式）；**KV cache 按真实 41215 blocks 分配且页打散**
（见第七节，不这么做会测出 3 倍偏快的假结果）。

`m3-chunkedstep-*` 之所以建成 61 行的 decode 批：chunked-prefill 步里这个 kernel 根本看不到那
8133 个 prefill 行（`filter_by_query_len=True`，`chunked_prefill_paged_decode.py:92-97` 直接 return），
它的真实工作量就是那 61 个 query 长度为 1 的行 —— 这也是为什么两种步的单次耗时几乎一样（1172 vs 1168 µs）。

#### 参考实现与容差

一对批量 `einsum`，无逐序列循环。GQA 通过把组宽折进 query（`[S,Hkv,g,D]`）来处理，
不展开 KV 头，所以峰值中间量只有 `[S,Hkv,g,ctx]` fp32（最大 case 18.9 MB）。
参考读连续的 `key`/`value`，kernel 读分页 cache，`_fill_kv_cache()` 每次一起刷新，
保证两边始终描述同一份负载。容差 `assert_close(atol=rtol=0.08)`。

---

### Task 2 —— `mi355x_vllm_triton_gqa_sparse_attn_prefill_minimax_m3`

#### 调用构成：单算子

被计时区间 `minimax_m3_sparse_attn(...)` **只发 1 个 kernel**（实测）：

```text
x1  _gqa_sparse_fwd_kernel
```

launcher 里就一次 `_gqa_sparse_fwd_kernel[grid](...)`，没有前后处理。
**agent 的全部空间就在这一个 kernel 里。**

#### 是什么算子

MiniMax-M3 自带的块稀疏 GQA prefill 注意力 Triton kernel **`_gqa_sparse_fwd_kernel`**
（`vllm/models/minimax_m3/amd/ops/sparse_attn.py:73`，launcher `minimax_m3_sparse_attn()` 在 `:243`）。
每个 query token 只对 task 3 选出来的 top-16 个 128-token KV 块做 attention，而不是整条序列。

#### E2E 占比

**8.53%** —— 全 session 排第 3。

| | |
|---|---|
| chunked-prefill 步内 | 21.20% |
| decode 步内 | 0（decode 走 `_gqa_sparse_decode_kernel`，3.53%，不在本 task 内） |
| 调用次数 | 57 次/prefill 步 |
| 单次耗时 | 720 µs |

#### 对应模型结构的哪个部分

**后 57 层块稀疏注意力层的注意力核心，prefill 路径。**

- 一次 prefill 步处理 8133 个 context token（`max_num_batched_tokens=8192` 减去夹带的约 59 个
  decode token），57 层各调一次。
- 输入：`q [8133,8,128]` bf16、`kv_cache [41215,1,128,256]` bf16（最后一维前 128 是 K、后 128 是 V）、
  `topk_idx [1,8133,16]` int32、`block_table`、`cu_seqlens_q`、`seq_lens`、`prefix_lens`。
- 3 个 case 对应 trace 里实测的 3 种 launch 几何：batch=1/q=8131（40 个 prefill 步里占 37 个）、
  batch=2/q=[8073,60]、收尾 chunk batch=1/q=61/prefix=8131。

#### 为什么值得做

实测 **83.6 TFLOP/s ≈ bf16 峰值的 3.6%**。根因在 launcher 的两个参数：`BLOCK_SIZE_Q = 1`
（**一个 CTA 只处理一个 query token**）+ `num_warps=1`（实测 block 就是 `(64,1,1)`，一个 wavefront）。
QK GEMM 的 M 维只有 8（GQA 组宽），喂不饱 `matrix_instr_nonkdim=16` 选中的 MFMA_16x16 tile。
带宽不是瓶颈：单层单序列的 KV 工作集只有 4.2 MB，稳在 256 MB Infinity Cache 里。

（session 里 GEAK 挑的就是这个 kernel，微基准做到 1.36x；按真实负载端到端上限是 2.3%。）

#### 测试 case（精度 / 性能同一份，同尺寸）

| id | batch | query_lens | seq_lens | prefix_lens | grid |
|---|---|---|---|---|---|
| `m3-prefill-b1-q8131-prefix0`（primary） | 1 | [8131] | [8131] | [0] | (8131,1,1) |
| `m3-prefill-b2-q8073p60` | 2 | [8073, 60] | [8073, 8192] | [0, 8132] | (8073,1,2) |
| `m3-prefill-b1-q61-prefix8131` | 1 | [61] | [8192] | [8131] | (61,1,1) |

这三个就是 trace 里实测到的三种 launch 几何，第一个覆盖 40 个 prefill 步里的 37 个。
公共：`num_heads=8`、`num_kv_heads=1`、`head_dim=128`、`topk=16`、`sparse_block_size=128`、
`num_blocks=41215`、`USE_FP8=False`、`SUB_K=64`。

`topk_idx` 是按因果可见范围 `[0, ceil((prefix+i+1)/128))` 采样生成的，必定包含 local 块
（`sparse_local_block=1`），升序排列、右侧用 -1 补齐 —— 就是 kernel 期望的布局
（`real_topk = sum(topk_idx >= 0)`，顺序读取）。这也是"每 query 平均 14.11 个块"的来源。

#### 参考实现与容差

严格照 kernel 主体（`sparse_attn.py:119-235`）：绝对 query 位置 = `prefix_lens[b] + i`；
选中块 `blk` 覆盖 KV 位置 `blk*128 + [0,128)`；位置有效当且仅当 `pos < seq_lens[b]` **且**
`pos <= prefix + i`；K 是 `kv_cache[page,kvh,:,:128]`、V 是 `[...,128:]`。

对"选中块 × 块内位置 × 头"全向量化，唯一的 Python 循环是 query 维上每 128 个一块 ——
纯粹为了把 gather 出来的 KV 压到几百 MB（一次性铺开 8131×2048 个 key 在 fp32 下要约 2 GB）。
容差 `assert_close(atol=rtol=0.08)`。

---

### Task 3 —— `mi355x_vllm_triton_sparse_index_topk_minimax_m3`

#### 调用构成：多算子链（decode 3 个 / prefill 2 个）

实测：

```text
decode  minimax_m3_index_decode()        →  x1 _decode_index_score_kernel
                                            x1 _topk_index_partial_kernel
                                            x1 _topk_index_merge_kernel
prefill minimax_m3_index_score()         →  x1 _index_block_score_kernel
        + minimax_m3_index_topk()           x1 _topk_index_kernel
```

**这个 task 的价值主要就在"链"上**：打分 kernel 本身已经 66.5% HBM 峰值，
但后面 partial + merge 两趟（合计 1.73% E2E）是为一个 `[1,64,112]` fp32 的小问题
付了两次 launch 加一次全局往返。融掉它们和加速打分 kernel 是同等的得分。

#### 是什么算子

块稀疏注意力的**选块器（indexer）**，5 个 Triton kernel 组成的一条链
（全在 `vllm/models/minimax_m3/amd/ops/index_topk.py`）：

| 路径 | 入口 | kernel |
|---|---|---|
| decode | `minimax_m3_index_decode()` `:767` | `_decode_index_score_kernel` + `_topk_index_partial_kernel` + `_topk_index_merge_kernel` |
| prefill | `minimax_m3_index_score()` `:655` → `minimax_m3_index_topk()` `:714` | `_index_block_score_kernel` + `_topk_index_kernel` |

它用一套独立的 index-Q / index-K（`sparse_index_dim=128`，和主注意力的 Q/K 不是同一组），
对每个 query 给 ≤104 个 128-token 块各算一个分数（块内 128 个位置的 `k·q` 取 max），
再选出 top-16 交给 task 2 / `_gqa_sparse_decode_kernel`。

#### E2E 占比

**8.01%** = 5.23 + 0.96 + 0.77 + 0.89 + 0.16。

| kernel | E2E% | prefill 步内 | decode 步内 |
|---|---|---|---|
| `_decode_index_score_kernel` | 5.23% | 0.95% | 8.10% |
| `_topk_index_partial_kernel` | 0.96% | 0.17% | 1.49% |
| `_topk_index_merge_kernel` | 0.77% | 0.12% | 1.21% |
| `_index_block_score_kernel` | 0.89% | 2.20% | 0 |
| `_topk_index_kernel` | 0.16% | 0.41% | 0 |

#### 对应模型结构的哪个部分

**后 57 层块稀疏注意力层里、注意力之前的那一步。** 层内位置是
`qkv 投影 → qk-norm+RoPE+KV写入 → 【选块器】 → 块稀疏注意力 → o 投影`。

- decode 路径每一步都跑（纯 decode 步的 64 个 token，以及 chunked-prefill 步里夹带的 61 个 token），
  prefill 路径只在 chunked-prefill 步跑（8133 个 token）。两条都做成了 case。
- 输入：`idx_q [total_q,1,128]` bf16、`index_kv_cache [41215,128,128]` bf16、
  中间 `score [1,total_q,112]` fp32（112 = `round_up(cdiv(13312,128),16)`）、
  输出 `topk_idx [1,total_q,16]` int32。
- `sparse_init_block=0` / `sparse_local_block=1`，所以每个 query 有且只有 1 个块（query 自己所在的块）
  被强制选中。

#### 为什么值得做

**这条链不产生任何注意力输出**，是块稀疏为了"挑块"付的固定成本 —— 而它比真正算稀疏 decode 注意力的
`_gqa_sparse_decode_kernel`（3.53%）还贵一倍多。

但收益点要看清：`_decode_index_score_kernel` 本身已经 **5.32 TB/s = HBM 峰值的 66.5%**
（它做了 8 路 split-K，512 个 CTA 铺满 256 CU），逼它没用。能拿的是**链路融合**：batch 64 /
`max_block=104` 这么小的问题，top-16 却拆成 partial + merge 两趟（合计 1.73%），
中间还要过一次 `[1,64,112]` 的全局显存往返。

硬约束：decode 路径在 cudagraph 里，所有 grid 和中间 buffer 形状必须只依赖 shape 常量。

#### 测试 case（精度 / 性能同一份，同尺寸）

| id | 路径 | batch | total_q | seq / prefix | 打分 kernel grid |
|---|---|---|---|---|---|
| `m3-decodepath-bs64-ctx9098`（primary） | decode | 64 | 64 | 9098 / — | (64, 8) |
| `m3-decodepath-bs61-ctx9098` | decode | 61 | 61 | 9098 / — | (61, 8) |
| `m3-prefillpath-b1-q8131-prefix0` | prefill | 1 | 8131 | 8131 / 0 | (128, 1) |
| `m3-prefillpath-b1-q61-prefix8131` | prefill | 1 | 61 | 8192 / 8131 | (1, 1) |

两条路径都覆盖，因为 session 里两条都跑：decode 路径**每一步**都跑 57 次，prefill 路径只在
chunked-prefill 步跑 57 次。公共：`num_idx_heads = num_kv_heads = 1`、`index_head_dim=128`、
`topk=16`、`init_blocks=0`、`local_blocks=1`、`num_blocks=41215`、`max_seq_len=13312`。

#### 参考实现与容差 —— 并列（tie）问题

参考复现两个打分 kernel 的语义：`score[h,q,blk]` = 块内 128 个位置 `k·q` 的 max，
位置需满足 `pos <= prefix+i`（因果）且 `pos < seq_len`，超出因果范围的块为 `-inf`；
再套上强制块 —— init → `1e30`、local → `1e29`（decode 侧烘进分数里，见
`index_topk.py:390-392`；prefill 侧则是把 `init_blocks`/`local_blocks` 传给 `_topk_index_kernel`，
选择结果等价）。实现上是每个 query chunk 一次 `einsum` 打全序列，再对块内 128 个位置 `amax`。

**浮点 top-k 在分数并列时没有唯一解**，所以 `_assert_close` 不拿 index 去和 `torch.topk`
逐元素比。它对每个 (head, query) 校验三件事：

1. 选中个数正确 —— `min(topk, 可见块数)`
2. 每个选中块互不相同、且都在因果可见范围内
3. **选中分数的多重集**等于 torch top-k 分数的多重集（排序后比，`atol=rtol=2e-2`）

这样任何合法的并列打破方式都能通过，但选错块一定会被抓到。

---

### Task 4 —— `mi355x_vllm_flydsl_mxfp4_moe_2stage_minimax_m3`

#### 调用构成：多算子链（prefill 10 个 / decode 6 个）

被计时区间是 `aiter.fused_moe.fused_moe(...)` 一整个调用。实测 token=8192（本 task 的主 case）
发出 **10 个 kernel**，其中只有 2 个是本 task 要优化的 GEMM：

```text
x1  opus_moe_sorting_entry<MoeSortingClearWorkspaceKernel>   ┐
x1  opus_moe_sorting_entry<MoeSortingMultiPhaseKernel_P0_v1> │
x1  opus_moe_sorting_entry<MoeSortingMultiPhaseKernel_P1>    │ token 排序 / 分组
x1  opus_moe_sorting_entry<MoeSortingMultiPhaseKernel_P23>   │
x1  mxfp4_moe_sort_kernel<256,32,24,32>                      │
x1  mxfp4_moe_sort_kernel<256,64,4,32>                       ┘
x2  dynamic_per_group_scaled_quant_kernel                      激活量化（bf16→mxfp4）
x1  mfma_moe1_silu_mul_afp4_wfp4_...     ★ 本 task 的目标
x1  mfma_moe2_afp4_wfp4_...              ★ 本 task 的目标
```

注意：**router gate GEMM 和 top-k 选专家不在被计时区间里** —— harness 用 torch 事先算好
`topk_weights` / `topk_ids` 再传进来，和 vLLM 的调用方式一致。

#### 是什么算子

MoE 路由专家的两段 FFN GEMM，**prefill 侧实现**：aiter FlyDSL 生成的一对 MFMA kernel

```text
mfma_moe1_silu_mul_afp4_wfp4_bf16_t128x128x256_pm1_async_v32      gemm1（gate+up，融了 silu·mul）
mfma_moe2_afp4_wfp4_bf16_cshuffle_t128x128x256_..._persist_cu256  gemm2（down）
```

由 `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` 生成（kernel 名在 `:273` / `:3111` 拼出），
经 `aiter.fused_moe.fused_moe()` 分发。**既不是 Triton，也不是手写 .co 汇编**，
是 aiter 自带的 FLIR/MLIR Python DSL。

#### E2E 占比

路由专家 GEMM 作为**一个功能**是 **18.60%**，是全模型最大的单一计算功能。
本 task 的编辑面覆盖其中的 prefill 一半 = **6.01%**。

| kernel | E2E% | prefill 步内 | 单次耗时 | 实测算力 |
|---|---|---|---|---|
| `mfma_moe1_silu_mul_...` | 2.06% | 5.12% | 178.1 µs | 2315 TFLOP/s |
| `mfma_moe2_..._persist_cu256` | 3.95% | 9.83% | 342.3 µs | **602 TFLOP/s** |

#### 对应模型结构的哪个部分

**后 57 层 MoE 层里的路由专家 FFN**（不含共享专家、不含 router gate）。

- 每 prefill 步 55.6 次调用（57 个 MoE 层，其中一个收尾小 chunk 走了别的分支）。
- 权重（每 rank）：`w1 [128, 1024, 6144]`（`1024 = 2x512` gate+up）、`w2 [128, 6144, 512]`，
  mxfp4 打包成 `[128,1024,3072]` / `[128,6144,256]`，per-1x32 的 e8m0 scale。
- 激活 bf16 进来、在 `fused_moe` 内部量化成 mxfp4（所以 kernel 名是 `afp4_wfp4`，a 和 w 都是 fp4）。
- 一次 prefill 步 8192 个 token x top-4 = 32768 个 (token, expert) 分配。

#### 为什么值得做

两条来自 session 的具体证据：

1. **gemm2 的 FLOPs 只有 gemm1 的一半，耗时却是 2 倍** —— 602 vs 2315 TFLOP/s，同类 shape 差 3.8 倍。
   两者都没到带宽墙（2.78 / 1.79 TB/s vs 8 TB/s）。gemm2 的启发式名字以 `_atomic` 结尾，
   per-expert 局部结果怎么累加是第一个该看的地方。
2. **这个 shape 根本没命中调优表。** server.log 原文：
   `[fused_moe] no tuned FlyDSL config for ('gfx950', 256, 8192, 6144, 512, 128, 4,
   ActivationType.Swiglu, ...), using heuristic FlyDSL fallback`。
   `configs/model_configs/minimax_m3_fp4_tuned_fmoe.csv` 在编辑面里。

#### 测试 case（精度 / 性能同一份，同尺寸）

| id | 阶段 | token | 分发到 |
|---|---|---|---|
| `m3-moe-prefill-token8192`（primary） | chunked-prefill 步 | 8192 | aiter FlyDSL 对 —— **本 task 的编辑面** |
| `m3-moe-decode-token64` | 纯 decode 步 | 64 | CK-Tile 对（**不在本 task 的编辑面内**，只作参照，见第五节） |

公共：128 专家、top-4、`model_dim=6144`、`inter_dim=512`、`QuantType.per_1x32`、
`ActivationType.Swiglu`、bf16 激活在 `fused_moe` 内部量化成 mxfp4、e8m0 组 scale。

权重准备照搬 vLLM 的 `AITER_MXFP4_MXFP4` loader
（`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:973-1014`）的**同一批入口**，不臆测等价布局：
`e8m0_shuffle` 两个 scale → 权重 view 成 `torch.float4_e2m1fn_x2` → `rocm_aiter_ops.shuffle_weights`
默认 (16,16)。未 shuffle 的量化副本留给参考实现用。

#### 参考实现与容差

用 aiter 自带的 `torch_moe_stage1` / `torch_moe_stage2` —— 它们解 mxfp4 半字节、套 per-1×32 的
e8m0 组 scale、在 fp32 累加，是**独立实现**而不是对被测 kernel 的包装；两段都是专家维上的批量 GEMM，
没有逐专家的 Python 循环。

fp4 运算过不了 `assert_close`，所以容差用余弦相似度 + 相对范数误差
（`min_cosine=0.97`、`max_rel_norm_err=0.25`），对普通调用和 CUDA-graph 计时调用各校验一次
（后者会先扰动激活）。实测 prefill `cos=0.9839 / rel_err=0.179`、decode `cos=0.99999 / rel_err=0.005`
—— 差距是真实的：prefill 走 a4w4（激活也被量化成 fp4），decode 的 CK-Tile 路径直接吃 bf16 激活（a16w4）。

精度测试约 6 分钟，瓶颈是参考实现在 token=8192 下把 128 个专家的权重全部反量化。

---

## 四、task 4 为什么只覆盖 MoE 的一半

`aiter.fused_moe.fused_moe()` 是一个入口，但**按 token 数分发到两套完全不同的实现**：

```text
token = 8192（prefill 步） ─▶ aiter FlyDSL 对  ─▶ 源码在 aiter/ (python DSL)     ← task 4 覆盖
token =   64（decode 步）  ─▶ CK-Tile 对       ─▶ 源码在 aiter_meta/ (C++/HIP)   ← 未覆盖
```

两边源码在两棵不同的树里，而 `image_repo_path` 只能指一个目录，所以一个 task 吃不下两边。
decode 那半（12.62%）没有做成 task，原因见第五节。

task 4 仍然**两个 token 都跑**（8192 和 64），这样报告里能看到 MoE 的完整画面；
agent 的编辑只会推动 8192 那个 case，64 那个保持不变 —— 这是诚实的结果，不是 bug。

## 五、刻意没做成 kernel task 的候选

| 候选 | E2E% | 为什么不做 |
|---|---|---|
| `ncclDevKernel_Generic_1` all-reduce | 12.22% | RCCL 是预编译的 `librccl.so`；而且真正的修法是改配置——96 MiB 的 prefill 消息超过了 AITER custom-AR 的 64 MiB 阈值（`aiter/dist/device_communicators/custom_all_reduce.py:525`），`max_num_batched_tokens <= 5461` 就能落回快路径 |
| `aiter::cross_device_reduce_2stage` | 5.16% | 768 KiB 的小消息，纯同步延迟域，改 kernel 够不着 |
| qkv / o 投影、router gate GEMM、LM head（`Cijk_*`） | 10.2% | hipBLASLt 预编译的 Tensile 汇编，镜像里没有可编辑源码，也没有对应的 `repository_language` |
| MoE `inter_dim` 从 384 补齐到 512 | 约 3.1% | 是 vLLM 的权重打包问题，不是 kernel 问题 |
| **MoE decode 侧的 CK-Tile GEMM 对**（`ck_tile::MoeFlatmmKernel`） | **12.62%** | 三条理由叠加，见下方说明 |
| `_gemma_fused_add_rmsnorm_kernel` | 3.06% | 收益薄，且 `pass_config.fuse_allreduce_rms=True` 已经融了一部分 |

### 关于 MoE decode 侧的 CK-Tile GEMM（曾做过一版，实测后放弃）

这一条本来做成了第 5 个 task（`mi355x_vllm_ck_cktile_moe_2stage_minimax_m3`），实测后删掉了。
三条理由，都是量出来的：

**1. 编译过不了 5 分钟这条线。** 它是唯一需要真正走 hipcc 重编 C++ 的 task
（另外四个是 Triton / FlyDSL，改 Python 即时生效，所以都是秒级）。实测 `compile` 各模块耗时：

```text
module_aiter_core          10.1 s
module_moe_sorting_opus     6.6 s
module_moe_cktile2stages  337.8 s   ← 主要开销
module_activation          17.3 s
总墙钟 ≈ 10.5 分钟
```

前后三个是 `AITER_REBUILD=1` 顺带的、能省掉约 34 s；但 **337.8 s 的 CK-Tile 模板实例化省不掉
—— 重编这个模块正是这种 task 存在的意义。** 就算只留它也在 6 分钟量级。

**2. 目标算子已经到顶。** 两个 GEMM 在 decode 下跑到 **7.18 TB/s ≈ HBM 峰值的 90%**
（64 token × top-4 触达约 110.8 个专家 = 523 MB fp4 权重，72.8 µs 搬完）。真正能拿的是它们周围
那 4 个小 pass（清零 / 排序 ×2 / 激活，加上调用方的两个 `[64,6144]` elementwise，合计约 3.7%），
但那要靠 epilogue 融合，收益远小于编译代价。

**3. 和已有 task 高度重复。** 与 `image_kernel/mi355x_vllm_ck_cktile_moe_2stage`
**同源码文件、同目标函数**，配置只差 `model_dim`（3072 → 6144）。M3 的 decode 形状如果确实需要，
给那个已有 task 加一个 case 就够了，不必新开。

顺带记录一个在这个过程中查出来、对别人有用的坑：`module_moe_cktile2stages` 的实例表由一个 blob
步骤生成，aiter 用裸 `os.system(f"{PY} .../gen_instances.py ...")` 起子进程
（`aiter/jit/core.py:866`），该子进程只继承环境变量，而 `gen_instances.py` 要
`from chip_info import get_gfx` —— `chip_info.py` 在 `aiter/jit/utils/` 下、不在任何 path 上，
于是 `ModuleNotFoundError` → 生成不出 `moe_cktile2stages_lookup.h` → `hipcc` 报
`fatal error: 'moe_cktile2stages_lookup.h' file not found`。给子进程补 `PYTHONPATH` 即可。
**已有的 `mi355x_vllm_ck_cktile_moe_2stage` 的 `_configure()` 里同样没设 `PYTHONPATH`，
很可能撞同一个问题，值得单独查一下。**

---

## 六、四个 task 共同遵守的约定

1. **shape 与 session 一致。** 每个 case 的 shape、dtype、序列长度、launch 几何都是从 torch profiler
   trace 里读出来的（外层 `cpu_op` 的 `Input Dims` / `Input type`，kernel 事件的
   `args.grid` / `args.block`）；decode 段被 cudagraph 吞掉调用上下文的，改用 `capture_64_FULL`
   捕获 trace 补齐。**凡是 prefill 和 decode 都调用的算子，两种 shape 都做成了 case。**
2. **精度和性能用同一份 case 列表、同一尺寸。** 只有 `compile` 冒烟会缩小。
3. **性能在 CUDA/HIP graph 下测量**，走 arena 的 `_benchmark_cuda_graph_or_events`；
   而且*被计时的那一次*调用会在扰动输入后重新校验，防止 kernel 在被打分的路径上比在精度路径上少干活。
4. **ref 用 torch 且向量化。** 没有逐序列 / 逐专家的 Python 循环；确实塞不下完整中间张量的地方
   （稀疏 prefill 注意力、prefill 选块打分），循环只是 query 维上的粗分块，
   块大小按"把工作集压到几百 MB"来定。

---

## 七、实测验证（本容器 MI355X）

### 7.1 agent 的编辑是否真的生效（不是跑旧产物）

对每个 task 往 workspace 源码里注入可观测的改动，看运行结果是否随之改变。

| task | 探针 | 结果 |
|---|---|---|
| 1 | 在 `kernel_paged_attention_2d` 体内插 `tl.static_assert(False,"AKA_EDIT_PROBE")` | ✅ 报错同时带 `AKA_EDIT_PROBE` 和 workspace 路径 |
| 2 | 同上，插进 `_gqa_sparse_fwd_kernel` | ✅ 同上 |
| 3 | 同上，插进 `_decode_index_score_kernel` | ✅ 同上 |
| 4 | 改 `mixed_moe_gemm_2stage.py` 里生成的 kernel 名后缀 `_v32` → `_AKAPROBE2` | ✅ profiler 里实际跑的 kernel 变成 `mfma_moe1_..._async_AKAPROBE2`，还原后变回 `_v32`（说明 `aiter/jit/flydsl_cache` 里那 1825 条缓存没有把旧 kernel 顶上来） |

task 1~3 是 Triton，`task_runner` 用 `spec_from_file_location` 从 workspace 副本加载模块，
Triton 在调用时按 Python 源码 JIT，天然不存在旧产物问题。
task 4 是 FlyDSL，`aiter/` 整包被 seed 进 workspace 并插到 `sys.path` 最前，
FlyDSL kernel 在调用时从（可编辑的）Python DSL 生成，**不走 `.so` 缓存**，探针已证实。

### 7.2 耗时与 AITER 重编（要求：每项 < 5 分钟，不触发全量 AITER 编译）

**四个 task 全部通过，最慢一项 11 秒。**

| task | compile | correctness | performance | 重编的 aiter 模块 |
|---|---|---|---|---|
| 1 paged_attention_2d | 3 s | 5 s | 6 s | — |
| 2 gqa_sparse_attn_prefill | 7 s | 10 s | 10 s | — |
| 3 sparse_index_topk | 7 s | 10 s | 9 s | — |
| 4 flydsl_mxfp4_moe_2stage | 9 s | 11 s | 11 s | **0 个** |

**task 4 原本 correctness 要 6 分 16 秒，已定位并修复。** 原因不是参考实现慢（实测
`_reference` 只要 1.07 s / 0.05 s），而是 `_configure()` 里把 `AITER_JIT_DIR` 重定向到了空的
`<workspace>/build/jit`，于是 token=64 那个 case 走 CK-Tile 时找不到预编译模块、
**从源码重编了 `module_moe_cktile2stages`（367 s）—— 而那根本不是本 task 的编辑面。**
修法：FlyDSL 分支不再重定向 `AITER_JIT_DIR`。留空时 aiter 默认用
`<workspace>/aiter/jit`，那里本来就带着随包拷进来的 108 个预编译 `.so`，而且本身就是
per-run 的，隔离性不受影响。修完 **9/11/11 秒，零重编**，且编辑探针复测仍然生效。

## 八、CI 注意事项

`src/perf_helper_materialization.py:104` 只 glob `tasks/image_kernel/*/scripts/task_runner.py`，
所以 `make check-perf-helpers` 看不到这四个。运行时不受影响——
`materialize_perf_helpers_in_workspace()` 是按路径改 workspace 副本的
`<workspace>/scripts/task_runner.py`，跟 task 源码在哪个目录无关。
要把它们也纳入 lint，就放宽那个 glob，或者把本目录挪到 `tasks/image_kernel/` 下。
