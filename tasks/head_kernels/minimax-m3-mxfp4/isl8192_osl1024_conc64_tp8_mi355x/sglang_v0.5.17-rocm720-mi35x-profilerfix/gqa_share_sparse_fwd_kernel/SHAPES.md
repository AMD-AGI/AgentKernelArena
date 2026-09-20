# MiniMax call geometry and scoring scope

The only scored case is observed `prefill_m8192_s1`: Q `[8192,8,128]`, K/V pools `[4358330,1,128]`, request table `[4097,11268]` int32 with stride `[11268,1]`, and original slot `[4]` as int64. Captured top-k block IDs are retained. Numerical values are generated; no warmup fallback is allowed.

All 3 captured correctness cases and the complete tensor inventory remain in [SHAPES.json](SHAPES.json) and [ut/generated_cases.json](ut/generated_cases.json). The performance gate and evidence are recorded in `ut/meta.json:performance_contract`.
