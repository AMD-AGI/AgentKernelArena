我需要根据一段info来创建一个gpu kernel benchmark，info在最下面。

这个benchmark应该包含且仅包含info中的kernel。目前未经处理的kernel散布在：
- /shared_nfs/zihao/headkernel_ut_0831
- /shared_nfs/hongtaom/headkernel_ut_0913/
- /shared_nfs/chuschen/agent-kernel-arena/top_model_kernels/repository/20260913T180016Z_qwen38_zeping

benchmark中的每个task应该都well-defined and follow the same standard，并且都可以通过AgentKernelArena(/shared_nfs/chuschen/agent-kernel-arena/AgentKernelArena)的task validator. 


当前path下已经有一些整理好的task，补充完遗漏的。并且确保他们遵循上面的标准。



==========info start============
- Owner: zihao
- Config: 
- ISL/OSL/CONC/TP 8k/1k/64/8
- Docker: harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix
- test result: GEAK 没有效果，原因是 kernel 有效果但是没有跑 e2e 测试
- note: 用的是最新的 docker， 我们跑的 baseline 和 hl 是对齐了。整体偏低于 inferencex  ;如果 expore 切换了正确 backend 应当速度就上来了
- InferenceX
- 
- 
kernel name	backend	GPU pct	empirical roofline	optimized roofline	end2end uplift	kernel level optization	next step
MLA	tilelang	44.9%	18%	18.1%	/	kernel optimization:

1. chushi optimized: ~5.90x, roofline~0.31
2. zhuqiong optimized: ~8.63x, roofline: ~0.51
end2end applyback	1. flydsl尝试 Wang, Fan <fanwang2@amd.com>
MoE stage-1 mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256	flydsl / aiter_asm	7.3%	~100% HBM (raw 146%, byte model over-counts)	-	-		
MoE stage-2 opus_moe_stage2_a8w4_decode_kernel_gfx950	aiter	3.7%	~98% HBM (raw 144%, byte model over-counts)	-	-		
cross_device_reduce_2stage (decode all-reduce)	aiter	3.3%	1.8% HBM (comm kernel; real roof is xGMI, not HBM)	-	-		
allreduce_prototype_twoshot (prefill all-reduce)	sgl_kernel / quickreduce	2.0%	5.6% HBM (comm kernel; real roof is xGMI, not HBM)	-	-		

Model Name: Qwen3.8-2.4T


- Owner: Li, Zeping <zepingl@amd.com>
- Config: 
- ISL/OSL/CONC/TP 8k/1k/64/8
- Docker:  harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix
- test result: local mode time budget ratio (24h/ geak ratio: 0.5)
- GEAK E2E 2.45% uplift, moe kernel tune+gemm tune
- note:
- updated by zeping
- InferenceX: 没有数据

kernel name	human-readable name	stage	kernel family	backend	GPU pct	empirical roofline	optimized roofline	end2end uplift	next step
mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32 → mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x32x256_pm1_async_v32	MoE gate/up projection	decode	MXFP4 MoE 2-stage	AITER / FlyDSL	21.06% → 11.74%	73.05%memory-bound	82.37%memory-bound	E1 FMoE 6-row +1.025%；E3 E1+E2 native stack +3.922%	测真实 bytes；再做 quant/S1→S2 融合。
mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256 [atomic_bnt2 → atomic]	MoE down projection/reduce	decode	MXFP4 MoE 2-stage	AITER / FlyDSL	11.58% → 6.39%	75.47%memory-bound	71.65%memory-bound	E1 FMoE 6-row +1.025%；E3 E1+E2 native stack +3.922%	测 S1→S2 流量；按 MoE layer 融合。
FMoE 2-stage · S1 t64x128 + S2 atomic → S1 t128x64 + S2 atomic_persist	Two-stage MoE	prefill	MXFP4 MoE 2-stage	AITER / FlyDSL	15.73% 优化后	22.6%memory-bound	23.7%memory-bound	E1 FMoE 6-row +1.025%；E3 E1+E2 native stack +3.922%	保留配置；只探索 quant/S1→S2 融合。
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x240x64_MI → MT256x256x64_MI	Large-batch QKV/gate projection	prefill	Dense BF16 GEMM	hipBLASLt / Tensile	15.56% → 10.77%	40.1%compute-bound	64.3%compute-bound	E4 Prefill GEMM 5-row +1.563% paired / +2.306% native	冻结 M16384；验证 small-M corrected DB。
gemm_a16w16 · M64 N32 K8192 · torch solution:0 → native (retained control)	32-wide tiny projection	decode	Dense BF16 GEMM	torch native	共享：21.43%	1.95%memory-bound	1.95%memory-bound	E2 Decode GEMM 22-row +2.039%；E3 E1+E2 native stack +3.922%	保留 native；停止该 shape 调参。
gemm_a16w16 · M64 N512 K8192 · torch solution:0 → flydsl t16x64x256 split-K8	Router/expert-gate projection	decode	Dense BF16 GEMM	torch → AITER FlyDSL	共享：21.43%	10.96%memory-bound	12.9%memory-bound	E2 Decode GEMM 22-row +2.039%；E3 E1+E2 native stack +3.922%	保留 split-K8；fresh profile 后再开。
gemm_a16w16 · M64 N4608 K8192 · torch solution:0 → bf16gemm_fp32bf16_tn_64x64_splitk_clean	QKV/output-gate projection	decode	Dense BF16 GEMM	torch → AITER ASM	共享：21.43%	37.37%memory-bound	51.68%memory-bound	E2 Decode GEMM 22-row +2.039%；E3 E1+E2 native stack +3.922%	保留 ASM；E2E 验证 corrected table。
gemm_a16w16 · M64 N8192 K256 · torch solution:0 → flydsl t32x64x64 split-K1	Linear-attention projection	decode	Dense BF16 GEMM	torch → AITER FlyDSL	共享：21.43%	9.23%memory-bound	13.98%memory-bound	E2 Decode GEMM 22-row +2.039%；E3 E1+E2 native stack +3.922%	保留 split-K1；fresh profile 后再开。
paged_attention_ll4mi_QKV_mfma16_kernel	Paged attention	decode	Attention	AITER / CK asm	6.84%	48.2%memory-bound	—	Stage-A 情景 +0.25%（未实测）	只测 KV bytes/backend；通过 accuracy gate。
fused_recurrent_gated_delta_rule_packed_decode_kernel	Gated-delta recurrence	decode	Gated-delta cluster	Triton	6.44%	53.5%memory-bound	—	Stage-A 情景 +2.39%（未实测）	与 RMSNorm/gate 合并验证。
_gemma_fused_add_rmsnorm_kernel	Residual add + RMSNorm	prefill + decode	Fused add + RMSNorm	Triton	5.06%	77.9%memory-bound	—	规划情景 ≈+0.24%（未实测）	仅做相邻融合；并入 Gated-delta candidate。

Model name: Kimi K3

- Owner: Hongtao
- config:  
- Docker: harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang-rocm-k3 :rocm720-mi35x-k3-20260727-tl312-08011830
- ISL 8192 / OSL 1024 / CONC 64 / TP 8 
- test result: 
- E2E +7.70% 来自于GEAK的优化,local mode time budget ratio (48h,geak ratio 0.5)
- InferenceX
- 只展示agentX的配置了，没有fixed 8k/1k的数据
- 

kernel name	backend	GPU pct	Prefill/Decode	kernel level speedup	empirical roofline	optimized roofline	end2end uplift	Note	next step
_fwd_grouped_kernel_stage1	triton(attention)	9.95%	Decode		0.297	0.409	+2.439%		kernel优化结果

end2end apply back
Cijk_…_MT256x256x64_MInon-editable	hipBLASLt / Tensile，经 aiter.tuned_gemm 派发；未调优时 fall through 到 libtype=torch solution:0	9.14%	Prefill		0.647	0.648	+5.479%		geak做了优化，剩 11 个未调优 prefill 形状，可以继续尝试优化
allreduce_prototype_twoshotnon-editable	quickreduce two-shot（CodecQ8 量化）· sgl-kernel / aiter	9.12%			/	未动	0		通信kernel
moe_gemm1_0	aiter 汇编二进制（AOT 生成）· fused_moe_2stages stage-1 gate+up · mxfp4 权重 + bf16 激活	6.51%	Prefill		0.254	0.254	0		geak预算原因没有做优化，可以单独手动kernel level优化
moe_gemm2_0	aiter 汇编二进制（AOT 生成）·  flydsl/opus · fused_moe_2stages stage-2 down · mxfp4 + bf16	5.47%	Prefill		0.230	0.230	0		geak预算原因没有做优化，可以单独手动kernel level优化
_score_kernel（+ _combine_kernel）editable	Triton · sglang srt/layers/attn_residual.py，两个 launch 同出一个 callable _mix_fused	5.26%	Prefill		log里面没找到	log里面没找到	0		查问题


Model name: GLM 5.3 flash

- Owner: Hongtaom
- config:  
- Docker: harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix
- ISL 8192 / OSL 1024 / CONC 64 / TP 8 
- test result: 
- E2E 没有提升(48h,geak ratio 0.5)
- InferenceX：agentX配置
- 

kernel name	backend	GPU pct	Prefill/Decode	empirical roofline	optimized roofline	end2end uplift	next step
fused_moe_kernel	Triton · sglang srt/layers/moe/moe_runner/triton_utils.fused_moe	23.32%	Decode	0.980	未动	3.48%	
Cijk_…_MT16x16x1024_	hipBLASLt / Tensile	6.31%	Decode	0.024	0.024（未变）	−7.01%	geak做了优化，没效果，单独手动优化没效果
ck_gemm_…_blockscale_b_	hipblaslt	5.65%	Decode	0.029	0.029（未变）	−0.13%	geak做了优化，没效果，单独手动优化没效果
tilelang_sparse_fwd	triton	5.22%	Decode	0.230	未动	0（未尝试）	geak预算原因没有做优化，可以单独手动kernel level优化
elementwise_kernel_manual_unrolleditable	ATen	3.24%	Decode	0.0006	0.0009	未测	geak对kernel做了优化，但是预算原因没测试验证，可以单独手动kernel level优化


Model name: Minimax-M3-mxfp4

- Owner: chaox
- Config: 
- ISL/OSL/CONC/TP 8k/1k/64/8
- docker: harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix
- test result: E2E test有15.44%提升(2558.95 →2954.16 tok/s)，其中GEAK提升4.52%，GEAK做了tuning skillset和HeadKerenl GAQ_share_sparse_fwd_kernel优化。
note: 
InferenceX



kernel name	backend	GPU pct	Prefill/Decode	empirical roofline	optimized roofline (如果有）	speedup	end2end uplift (如果有）	next step
decode_score_kernel	Triton/attention	11.63	decode	60.0%	86%	1.581x	-0.15%(未接受)	1.581x的提升再trivial-topk成立，但是serving ISL8192时没有成立。把融合推广到sparse路径。
moe_flatmm_geamm1_ck	ck/fused moe	7.49	decode	14.9%	-	-	-	flydsl/triton都没有超过baseline。
gqa_share_sparse_decode_kernel	Triton	7.63	decode	78%	86%	1.111	+2.69%	
_gemm_afp4wfp4_kernel	Triton	5.14%	decode	24%	-	-	-	ck/flydsl没有超过baseline
gqa_share_sparse_fwd_kernel	triton/attention	3.98%	prefill	10.72	14.1%	10.4143	10.28%(GEAK自测)	


Model name: GLM-5.2-MXFP4

- Owner: chaox
- Config: 
- ISL/OSL/CONC/TP 8k/1k/64/8
- docker: harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix
- test result: E2E为23.44%( 1145.49 → 1414.01 tok/s)，其中GEAK没有提升。GEAK做了tuning skillset，没有E2E效果。GEAK做了HeadKerenl 优化，尝试triton/flydsl优化，但均未达到超过baseline的效果。
note: 
InferenceX

kernel name	backend	GPU pct	Prefill/Decode	empirical roofline	optimized roofline (如果有）	speedup	end2end uplift (如果有）	next step
sparse_attention_kernel	tilelang	27.47	prefill	26.5	26.3	1.02	0%	
sparse_mla_fwd_decode_partial_fp8	tilelang	21.39	decode	9.2	18.4	1.997	-1.48	
mfma_moe2_afp4_wfp4_bf16_cshuffle	ck	18.38	prefill+decode	6.0	13.2	2.214	-0.09	93.9% 的收益来自decode M=256bucket，实际走cuda-graph捕获。occupancy/tiling/grouped-gemm调度，在真实cuda-graph下测试
hgemm_bf16	flydsl	10.85	prefill+decode	43	49	1.48	-1.3	使用GEAK bakeoff切backend试试
mfma_moe1_silu_mul	flydsl	8.42	prefill+decode	86.1	87.4	1.389	+0.17	已经接近memory-roof

===========info end==============
