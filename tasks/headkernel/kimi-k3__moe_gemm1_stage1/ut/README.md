# UT bundle · `moe_gemm1_0`（FlyDSL MoE stage-1，gate/up GEMM + SiTUv2 融合）

**0828 profile 里单个 kernel 最大的一块：13.53% GPU（99176 calls / 10624.76 ms）**，
也是 top-5 里原本唯一没有 UT 的（`../../README.md` 总表）。没考卷 → 没打分口径 →
0817/0828 每次排序里都稳定排 ~#4、稳定被 wall-clock 砍掉
（0817 `final_report.md` 原话：`h2 … never dispatched`，见 `../NOTES.md`）。

本目录把它补齐到和 `../ut_stage2/` 同一个契约。**新建，不是从 claw 树里捡的** ——
0828 只走到 Tier-B 调优，`kernels/` 是空的，stage-1 从来没有过 GEAK 原生 bundle。

```bash
cd ut_stage1 && HIP_VISIBLE_DEVICES=0 python3 unittest.py
# 退出码：0 pass · 1 correctness FAIL · 2 env error · 3 harness incomplete
```

单卡就够，不加载模型。已验证（节点 100197 / m2m-214，镜像
`sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830`，gfx950）。

## 空跑结果（`kernel_src` == `baseline_src`）

```
[eager]        prefill_M8192 / decode_M1 / decode_M64   correct=True  max_rel_err=0.0
[random] ×3 每桶                                        correct=True  max_rel_err=0.0
[graph_replay] decode_M64 / decode_M1                   correct=True  max_rel_err=0.0
[sequence]     prefill → decode_M1 → prefill            correct=True  max_rel_err=0.0
[timing] prefill_M8192  baseline 1.2839 ms   decode_M64 0.2105 ms   decode_M1 0.0469 ms
GEAK_GEOMEAN_SPEEDUP 0.99973   GEAK_WEIGHTED_SPEEDUP 0.99925   GEAK_CORRECT PASS   EXIT=0
```

`max_rel_err` 全部**恰好 0.0**（不是"够小"），原因见下面第 2 节。
prefill 单发 1.284 ms 与 profile 的 956.7 µs/call 同量级（profile 是 8 卡分摊后的层均值）。

## 负向对照：这份考卷不是空转

在 `kernel_src/flydsl/moe_kernels.py` 的 `flydsl_moe_stage1` 里强制 `act="silu"`
（打破 SiTUv2 数学契约），三个桶全灭：

```
[eager] prefill_M8192 correct=False err 8.9438 · decode_M1 False 4.2913 · decode_M64 False 5.8248
GEAK_CORRECT FAIL   EXIT=1
```

已还原，`diff -q baseline_src kernel_src` → IDENTICAL。

> ⚠️ **计划里原定的对照（`tile_n` 强制 256）是错的，别再用。**
> 实测 `tile_n=256` 在 M=8192 上 cos 0.99970、在 M=1 上**逐位相同** ——
> `resolve_flydsl_stage1_tile_n(384, 256)` 把它降到 128，而 tiling 在数学上是中性的。
> 它 PASS 本 UT 是**正确行为**：那是**性能** lever，要在 timing 腿上判，不在 correctness 上判。
> `../NOTES.md` 里把它写成"正确性风险"的那条已一并更正。

## ⚠️ 主要发现：这个 kernel 本身不是 run-to-run 可复现的，而且**错的那一发是真的错**

> **0902 定论（`_l2_reference.py` 实测，见下面「L2 独立参照」一节）：**
> 多数值**数学上是对的**，离群值是**损坏**。所以这不是"可以容忍的抖动"，
> 是 **aiter 上游的一个 bug**，线上单发路径每次 forward 都在吃。
> 早前写在这里和已发布副本里的"~2e-2 是底噪"那个说法**框架是错的** ——
> 2e-2 确实是单发 A/B 的下限，但它的成因不是浮点重结合，是约 1% 的元素被写坏。

**同样的输入、同样的 kernel、连续两次 launch，输出不一样。**
~0.5–1% 的元素不同，relL2 ~2e-2，onset 在 sorted expert block 数 >~50 之后
（decode M=64 → 469 blocks，prefill M=8192 → 4535 blocks，**线上两个桶都在这个区间里**）。

排除清单（每条都实测过，都不是原因）：

| 假设 | 结论 |
|---|---|
| 操作数越界读 | a×4 / w1×2 / w1_scale×4 guard padding，无变化 |
| 输出越界写 | `out` 后面挂 guard row，跑完**恰好还是 0.0** → store 是 masked 的 |
| 未写元素 | nan-init 后 0 个 NaN 残留 |
| routing / pad marker | pad marker 是 `(topk<<24)\|M`，0 个 pad 落进 `[M,topk]`，0 个重复 (token,slot) |
| launch 旋钮 | `k_wave` 2→1、`k_batch`、`xcd_swizzle` 4→0、`waves_per_eu`、`b_nt`、`use_async_copy` 全试过 |
| 调用约定 | kwargs 与 `aiter/fused_moe.py:_flydsl_stage1_wrapper` 逐行对齐 |
| 权重/scale layout | 换成 aiter 自己的 `shuffle_weight_a16w4` + `shuffle_scale_a16w4` 一样复现（此处 shuffle 保形） |
| 机器 | 同节点同卡同容器，`../ut_stage2` 的 UT 照常 PASS |
| 冻结副本的锅 | **装好的** aiter 同样如此 |

旁证：aiter 自己的 `op_tests/flydsl_tests/test_flydsl_moe_a16wfp4.py` 就是用
**cosine similarity + `atol=1.0` + `pass_pct=95.0`** 来 gate 这条路的 ——
上游本来就没把这个算子当逐元素可复现的东西。

**解法：不是放松 tol，是取中位数。** 每个元素的分歧是少数派噪声，多数值是稳定的；
对 K 次独立 launch 取逐元素 median 就能**逐位**恢复确定性结果。实测
K=21 时三个桶各 4 次试验全部 bitwise 稳定（`meta.nondeterminism_calibration.median_stability_trials`），
所以 `median_launches = 21`，`tol` 保持默认 0.02，harness 那条逐元素闸门原样保留。

为什么不走"放松 tol"：单发 oracle 下 baseline **和它自己**的分歧就有 harness err ≈ 27，
比一个真错的配置造成的分歧还大 —— 那样的 tol 什么都拦不住。

**timing 腿故意仍是单发**，因为线上付的就是单发的钱。

已知盲区：只影响单个 expert 的扰动（896 个 scale 里改 1 个）会淹在 median 残差里
（relL2 0.039 vs 0.024 底噪），M=1 上如果那个 expert 没被路由到则完全不可见。
**这份 UT 拦的是"波及相当比例 expert"的错，不是单 expert 漂移。**

## L2 独立参照：median 到底是不是**正确**的那个值（0902）

上面那套 median 契约只保证 oracle **自洽**——golden 就是冻结 baseline 自己的输出。
它结构上回答不了"baseline 自己对不对"。这一节把它补上。

`_l2_reference.py` 用 aiter **自己的** `aiter.fused_moe.torch_moe_stage1` 作独立参照
（**原样调用，不手写** —— 见 `aiter-fmoe-standalone-repro-traps`），
并且是**完全一致的契约**，不是 Silu 替身：`torch_moe_stage1` 原生支持
`activation=ActivationType.Situv2` 和 `situ_beta` / `situ_linear_beta`，
所以按 `meta.live_call_kwargs` 的 `situ_beta=situ_linear_beta=1.0`、
`quant_type=per_1x32`、`a1_scale=None`（a16w4）、`doweight=False` 驱动。

| 桶 | median vs torch | 单发 vs torch | 离群发 vs torch |
|---|---|---|---|
| `decode_M1` | relL2 **7.14e-05**，cos 1.000000 | 同左（0 个争议元素） | 同左 |
| `decode_M64` | relL2 **5.29e-05**，cos 1.000000 | 1.216e-02 | 7.118e-02 |
| `prefill_M8192` | relL2 **2.50e-05**，cos 1.000000 | 1.533e-02 | 8.250e-02 |

**判别式** —— 只看两发打架的元素，torch 站谁：

| 桶 | 争议元素 | 站 median | 站离群 | median 相对误差 | 离群相对误差 |
|---|---:|---:|---:|---:|---:|
| `decode_M64` | 40,239（10.23%） | **100.0000%** | **0%** | 1.21e-06 | **2.802** |
| `prefill_M8192` | 6,110,007（12.14%） | **99.99996%** | **0%** | 6.58e-07 | **2.443** |

（"争议元素"= 21 发里至少有一发和最偏离 median 的那发不同，所以比"连续两发比"的
0.5–1% 高；两者不矛盾。）

**为什么这个判决是安全的**：median 对 torch 的平均相对误差在**争议元素上（6.6e-07）
和无争议元素上（7.2e-07）一样**。也就是说 median 在 kernel 自己打架的位置，
精度跟没打架的位置**完全相同** —— 参照没有被要求去裁决一个逼近它自身噪声的差距。

**结论三条：**

1. 这份 bundle 的 L1 oracle 钉住的是**数学上正确**的函数。它是**正确性** oracle，不只是回归 oracle。
2. median 是对"会写坏结果的 kernel"的**合法修复**，不是放宽容差掩盖错误。
3. **这是 aiter 的 bug，线上在吃。** 生产是单发、没有 median，
   所以每次 forward 都有 ~0.5–1% 的 stage-1 输出错 O(100%)，prefill / decode 两个桶都中，
   serving 路径上没有任何环节会修复它。`M=1` 零争议且照样对得上 torch
   （relL2 7.14e-05），与"onset 在 sorted expert block 数 >~50 之后"一致 ——
   **是并发效应，不是公式错。**

### 一个自己踩过的坑：`torch_moe_stage1` 的 `w2` 要给**打包后**的形状

必须传 `[E, model_dim, inter_dim//2]`（= 0828 profile 自己的算子条目 `[896, 3584, 192]`），
不能传逻辑形状 `[896, 3584, 384]`。`get_inter_dim` 会乘
`int4_war = model_dim // w1.shape[-1] == 2` 来还原 fp4x2 打包，给逻辑值会算出
`inter_dim=768` → `use_g1u1` 变 False → epilogue 掉进 `torch_act(out)`，
而 `get_torch_act` 对 Situv2 返回的是 `NotImplementedError` **类**（可调用！），
于是构造出一个异常对象，20 行后才以
`'NotImplementedError' object has no attribute 'to'` 炸出来。
**静默地换成了另一个函数** —— 正是"不能手写参照"的那个理由的活标本。

### 复跑

```bash
cd ut_stage1 && HIP_VISIBLE_DEVICES=0 python3 _l2_reference.py \
    --cases decode_M1 decode_M64 prefill_M8192 --json _l2_report.json
```

只读，不动 bundle 任何产物。与已发布 oracle **共享**真实 routing 和 `live_call_kwargs`；
**不共享**重操作数 —— `unittest.build_inputs` 是把随机字节直接写进 kernel 的**预 shuffle 布局**，
没法反 shuffle 去喂 torch，所以这里另生成一对未 shuffle 的 `(w1_qt, w1_scale)`
（同样的字节分布），再用生产的 `shuffle_weight((16,16))` / `e8m0_shuffle` 喂进 kernel。
**因此结论是关于这个算子的，不是关于 `reference_io.pt` 那些字节的。**

## Oracle 怎么来的（provenance，诚实版）

走的是**离线路线**：不起 8 卡 server。

- **routing 是真的，但是二手的。** 取自 0817 那次同模型 / TP=8 / ISL 8192 / OSL 1024 / conc 64
  的服务分布（`../ut_stage2/reference_io.pt` 的 `prefill_M16384`），用
  `_extract_gen1.py:invert_routing()` 反解回 `(topk_ids, topk_weights)`，
  再用**生产** `aiter.fused_moe.moe_sorting` 按 stage-1 自己的 block（`tile_m=32`）重排。
  反解后 892/896 个 expert 被命中，weights 0.0073–0.6557。
  **它不是在 stage-1 这个 seam 上抓的。** routing 决定 grouped GEMM 的负载均衡，是这个
  op 的性能信号，所以宁可二手也不合成。
- **launch 变体是真的 0828 那个**：`flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_xcd4_kw2`
  （现场 kname，注册表里查得到），经 `get_flydsl_kernel_params()` 解析成
  `tile_m=32 / tile_n=64 / tile_k=256 / waves_per_eu=3 / k_batch=1 / b_nt=2 /
  gate_mode=separated / xcd_swizzle=4 / k_wave=2`。
  **但 decode 的变体是假定与 prefill 相同，不是观测到的。**
- **重操作数是合成的**（value-independent，由 `unittest.build_inputs(spec)` 按 spec 里的
  dims/dtypes 确定性重建，`fingerprint` 钉死）。golden = **冻结的生产 kernel 自己的输出**
  取 21 发 median，不手写 torch 参照（见 `aiter-fmoe-standalone-repro-traps`：
  手写参照会变成另一个函数）。
- `_capture_overlay1/` + `_run_capture1.sh` 是**未运行**的一手抓取钩子，
  下次有 8 卡窗口可以把上面两条二手事实一次性换掉（含 NOTES 说的 decode 双峰
  27.8 µs / 122 µs 两个层组）。

### 一个自己踩过的坑：scale 取值范围不能照抄 stage-2

`ut_stage2` 的 e8m0 scale 取 126–130（≈2^0）。照抄到 stage-1 会让 SiTUv2 epilogue
**约一半元素顶死在 ±1 clamp 上**，于是 kernel 的 run-to-run 抖动表现成幅度 2.0 的**符号翻转** ——
再大的 tol 也 gate 不住。真实 mxfp4 的 `w1` 元素 std ~1/√model_dim，K=3584 下 scale 该在
**2^-10** 附近。扫描（`meta.scale_calibration`）：byte 127 饱和 0.50、120 饱和 0.15、
**117 饱和 0，absmax 0.965** → 取 `scale_byte_lo=117`。

## 几何量（与 0828 profile 抓到的算子形状逐条对齐）

| 量 | 值 | 来源 |
|---|---|---|
| `num_experts` / `topk` | 896 / 16 | profile operand list |
| `model_dim` / `inter_dim` | 3584 / 384（= moe_intermediate / TP8） | 同 |
| `a` | `[M, 3584]` bf16（**未量化**） | `untuned_fmoe_live.csv: q_dtype_a=torch.bfloat16` → a16w4 路，`a1_scale=None` |
| `w1` | `[896, 768, 1792]` `float4_e2m1fn_x2` | profile |
| `w1_scale` | `[688128, 112]` e8m0（896×768 行 × 3584/32 列） | profile |
| `out` | `[M, 16, 384]` bf16，**单个张量** | `out_dtype='bf16'` → stage-1 **不**融合出口量化 |
| `sorted_weights` | 不传 | `doweight_stage1=0` → topk 权重在 stage-2 上 |

## cases 与权重

`prefill_M8192`（0828 的 chunk）· `decode_M64`（conc）· `decode_M1`（边界，calls=1）。

权重**按 0828 自己的两桶重算**，没有继承 stage-2 那套 16384 口径
（`../ut_stage2/README.md` 已警告会严重偏斜）。
`workload.json` 里的 `weight` 字段只是 provenance；主指标由
`harness_lib.serving_weighted_speedup` 从 `baseline_ms × analytic calls` 重算
（prefill = CONC×⌈ISL/chunk⌉ = 64，decode = OSL = 1024）。

## roofline（回答"compute bound 还是 memory bound"）

`profile/round_0/profile_roofline.md`：decode AI = 6.3 FLOP/byte，ridge = 312 → **在 memory 一侧**，
但 `hbm_util = 1.792 > 1` 说明"均匀路由"的字节模型被实测推翻了，所以 `roofline_pct` 被
钳到 `1.000*`、headroom 记为 `unknown` —— 这正是这两个大块**从来排不进 expected-e2e-gain
序**的原因。分相看：decode（M=64，compute_util 0.036）是 batched-GEVM，毫无疑问的
**权重流式带宽 bound**；prefill（M=8192，956.7 µs/call，那份报告没建模）跑到 ~754 TFLOP/s
≈ bf16 屋顶的 30%，而 HBM 只用了 ~12%，**是偏 compute 的一侧，真正的 headroom 在这里**。

## 文件

| 文件 | 说明 |
|---|---|
| `unittest.py` | 不可变考卷。median 契约、graph-replay、timing 都在这里 |
| `meta.json` | 生成物（`_extract_gen1.py` 写）。含 `nondeterminism_calibration` / `scale_calibration` / `live_call_kwargs` |
| `meta_seed.json` | 手写的 meta 输入，改这个再重跑生成器 |
| `reference_io.pt` | 103.2 MB，真 routing + median golden，sha256 `3d175ea5803d` |
| `harness_lib.py` | 从 `../ut_stage2/` 原样拷（同代 `72386ab6…`；**不能用 k3/k4 那份 `d196db4c…`**） |
| `baseline_src/flydsl/` `kernel_src/flydsl/` | 从 0828 镜像 `docker cp` 出来的 `aiter/ops/flydsl`，两份一致 |
| `workload.json` | cases + `serving_weight_model` + profile provenance |
| `_extract_gen1.py` | 提取期脚本（**不是**考卷的一部分）。median 不 bitwise 时会 FATAL |
| `_capture_overlay1/` `_run_capture1.sh` | 未运行的一手抓取钩子 |
| `_l2_reference.py` | L2 独立参照探针（**不是**考卷的一部分，只读）。用 aiter 自己的 `torch_moe_stage1` 证明 median 是正确值 |
| `_l2_report.json` | 上面三个桶的全精度结果；同样的内容也折进了 `meta.l2_independent_reference` |
