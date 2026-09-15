#!/usr/bin/env python3
"""Task runner for triton2triton/triton_fused_moe_gptq_awq"""
import sys, os, json, argparse, importlib.util

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)
TASK_NAME = "triton2triton/triton_fused_moe_gptq_awq"
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_fused_moe_gptq_awq.py")

# (M, K, E, N, topk, group_size)
TEST_SHAPES = [
    (16, 64, 4, 64, 2, 32),
    (32, 128, 4, 128, 2, 64),
    (64, 128, 8, 128, 2, 64),
    (64, 256, 8, 256, 2, 128),
    (128, 256, 8, 256, 2, 128),
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100


# >>> AKA-GENERATED: shared CUDA-graph benchmark helpers - edit src/tools/perf/vllm_cuda_graph_block.py then run `make sync-perf-helpers` >>>
def _measure_cuda_event_fallback(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )


def _benchmark_cuda_graph_or_events(*args, **kwargs):
    raise RuntimeError(
        "CUDA-graph benchmark helpers were not materialized. "
        "Run this task through AgentKernelArena so setup_workspace() can inject "
        "src/tools/perf/vllm_cuda_graph_block.py into the workspace."
    )
# <<< AKA-GENERATED <<<

sys.path.insert(0, TASK_DIR)
from _contract_checks import checked_call, checked_benchmark, compare_output, perturb_activation


def load_module():
    spec = importlib.util.spec_from_file_location("triton_kernel", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            source = f.read()
        ast.parse(source)
        mod = load_module()
        assert hasattr(mod, "fused_moe_gptq_awq"), "Missing fused_moe_gptq_awq"
        assert hasattr(mod, "fused_moe_kernel_gptq_awq"), "Missing fused_moe_kernel_gptq_awq"
        return True, None
    except Exception as e:
        return False, str(e)


def reference_fused_moe_int4(input_t, qweight, scales, zeros, topk_ids,
                             topk_weights, mul_routed_weight, group_size):
    """CPU reference for INT4 (GPTQ/AWQ) weight-only MoE.

    The Triton kernel packs 2 x int4 values per uint8 element along the K
    dimension:
        qweight: [E, K//2, N]  uint8  – low nibble = even k, high nibble = odd k
        scales:  [E, K//group_size, N] fp16
        zeros:   [E, K//group_size, N//2] uint8 (packed 4-bit zero points per N)
                 or None (default zp = 8)
    """
    import torch
    M, K = input_t.shape
    E = qweight.shape[0]
    N = scales.shape[2]
    topk = topk_ids.shape[1]
    num_valid = M * topk
    output = torch.zeros(num_valid, N, device="cpu", dtype=torch.float32)

    qw_cpu = qweight.cpu().to(torch.int16)  # promote to avoid sign issues
    scales_cpu = scales.cpu().float()

    # Unpack weights: 2 x int4 per uint8 along K dim  -> [E, K, N]
    K_packed = qweight.shape[1]  # K // 2
    w_lo = (qw_cpu & 0xF).float()          # even k indices
    w_hi = ((qw_cpu >> 4) & 0xF).float()   # odd k indices
    # Interleave: w_unpacked[:, 0::2, :] = lo, w_unpacked[:, 1::2, :] = hi
    w_unpacked = torch.zeros(E, K, N, dtype=torch.float32)
    w_unpacked[:, 0::2, :] = w_lo
    w_unpacked[:, 1::2, :] = w_hi

    # Unpack zero points
    num_groups = K // group_size
    if zeros is not None:
        zp_cpu = zeros.cpu().to(torch.int16)
        # zeros: [E, K//group_size, N//2] – packed along N dim
        zp_lo = (zp_cpu & 0xF).float()
        zp_hi = ((zp_cpu >> 4) & 0xF).float()
        zp_unpacked = torch.zeros(E, num_groups, N, dtype=torch.float32)
        zp_unpacked[:, :, 0::2] = zp_lo
        zp_unpacked[:, :, 1::2] = zp_hi
    else:
        zp_unpacked = torch.full((E, num_groups, N), 8.0)

    # Dequantize: w_float = (w_int4 - zp) * scale
    w_deq = torch.zeros(E, K, N, dtype=torch.float32)
    for gi in range(num_groups):
        k_start = gi * group_size
        k_end = k_start + group_size
        w_deq[:, k_start:k_end, :] = (
            (w_unpacked[:, k_start:k_end, :] - zp_unpacked[:, gi:gi + 1, :])
            * scales_cpu[:, gi:gi + 1, :]
        )

    for token_idx in range(M):
        x = input_t[token_idx].cpu().float()
        for k_idx in range(topk):
            flat_idx = token_idx * topk + k_idx
            expert_id = topk_ids[token_idx, k_idx].item()
            if expert_id < 0 or expert_id >= E:
                continue
            row = x @ w_deq[expert_id]
            if mul_routed_weight:
                row *= topk_weights[flat_idx].item()
            output[flat_idx] = row
    return output



CONTROL_CASES = ('int4_explicit', 'int4_default', 'int8_explicit', 'int8_default')


def reference(inputs, options):
    import torch
    A, qweight, scales, ids = inputs['A'], inputs['qweight'], inputs['scales'], inputs['ids']
    zeros, weights = inputs.get('zeros'), inputs.get('weights')
    if weights is None:
        weights = torch.ones(ids.numel(), dtype=torch.float32, device=A.device)
    if options['use_int4']:
        # Preserve the independent original INT4 unpack/dequantize comparison.
        out = reference_fused_moe_int4(A, qweight, scales, zeros, ids, weights,
                                      options['mul_routed_weight'], options['group_size'])
    else:
        M, K = A.shape
        E, _, N = qweight.shape
        zp = zeros.detach().cpu().float() if zeros is not None else torch.full(scales.shape,128.)
        group = torch.arange(K)//options['group_size']
        dequant = (qweight.detach().cpu().float()-zp[:,group,:])*scales.detach().cpu().float()[:,group,:]
        out = torch.zeros(M*ids.shape[1],N,dtype=torch.float32)
        for token in range(M):
            for lane in range(ids.shape[1]):
                expert = int(ids[token,lane].item())
                if not 0<=expert<E:
                    continue
                row = A[token].detach().cpu().float() @ dequant[expert]
                if options['mul_routed_weight']:
                    row *= float(weights[token*ids.shape[1]+lane].item())
                out[token*ids.shape[1]+lane] = row
    return out.to(device=A.device,dtype=A.dtype)


def control_inputs(name, device):
    import torch
    # Basis activations and binary-exact scales give hand-checkable large outputs.
    # K=48 also exercises the public partial-K load branch; N=70 crosses a tile.
    M, K, E, N, group_size = 5, 48, 3, 70, 16
    int4 = name.startswith('int4')
    A = torch.zeros(M,K,device=device,dtype=torch.float16)
    A[torch.arange(M,device=device),torch.tensor([0,1,16,33,47],device=device)] = torch.tensor([2,-3,4,-2,3],device=device,dtype=torch.float16)
    width = K//2 if int4 else K
    qweight = ((torch.arange(E*width*N,device=device).reshape(E,width,N)*37+19)%256).to(torch.uint8)
    scales = torch.full((E,K//group_size,N),0.5 if int4 else 0.125,device=device,dtype=torch.float16)
    scales[:,1,:] *= 2
    ids = torch.tensor([[0,1,1],[-1,2,3],[2,0,1],[1,2,0],[0,-1,2]],device=device,dtype=torch.int32)
    inputs={'A':A,'qweight':qweight,'scales':scales,'ids':ids}
    if name.endswith('explicit'):
        zwidth = N//2 if int4 else N
        inputs['zeros']=((torch.arange(E*3*zwidth,device=device).reshape(E,3,zwidth)*13+7)%256).to(torch.uint8)
        inputs['weights']=torch.tensor([-2.,3.,0.5]*M,device=device,dtype=torch.float32)
    return inputs, {'group_size':group_size,'use_int4':int4,'mul_routed_weight':name!='int8_explicit'}


def invoke(mod, inputs, options):
    return mod.fused_moe_gptq_awq(inputs['A'],inputs['qweight'],inputs['scales'],inputs.get('zeros'),
                                inputs['ids'],inputs.get('weights'),**options)


def check_output(actual, expected):
    compare_output(actual, expected, atol=1.0, rtol=0.5)


def run_correctness(*, case_index=None, control=None):
    import torch
    try:
        mod = load_module()
        device = 'cuda'
        if control is not None:
            assert control in CONTROL_CASES, 'Unknown control'
            inputs, options = control_inputs(control, device)
            checked_call(lambda: invoke(mod, inputs, options), inputs=inputs,
                         reference=lambda saved:reference(saved,options), check=check_output)
            return True, None
        for i, (M, K, E, N, topk, group_size) in enumerate(TEST_SHAPES):
            if case_index is not None and i != case_index:
                continue
            torch.manual_seed(42 + i)
            input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
            qweight = torch.randint(0, 255, (E, K // 2, N), device=device,
                                    dtype=torch.int32).to(torch.uint8)
            num_groups = K // group_size
            scales_t = (torch.randn(E, num_groups, N, device=device,
                                    dtype=torch.float16).abs() * 0.01 + 0.001)
            zeros_t = torch.randint(0, 255, (E, num_groups, N // 2), device=device,
                                    dtype=torch.int32).to(torch.uint8)
            topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
            topk_weights_flat = torch.randn(M * topk, device=device, dtype=torch.float32).abs()
            inputs={'A':input_tensor,'qweight':qweight,'scales':scales_t,'zeros':zeros_t,
                    'ids':topk_ids,'weights':topk_weights_flat}
            options={'mul_routed_weight':True,'group_size':group_size,'use_int4':True}
            checked_call(lambda: invoke(mod, inputs, options), inputs=inputs,
                         reference=lambda saved:reference(saved,options), check=check_output)
        return True, None
    except Exception as exc:
        return False, exc


def run_performance():
    import torch
    mod = load_module()
    device = 'cuda'
    test_cases = []
    for test_idx, (M, K, E, N, topk, group_size) in enumerate(TEST_SHAPES):
        row = {'test_case_id': f'perf{test_idx+1}', 'params': {'M':M,'K':K,'E':E,'N':N,'topk':topk,'group_size':group_size}}
        try:
            torch.manual_seed(42 + test_idx)
            input_tensor = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
            qweight = torch.randint(0, 255, (E, K // 2, N), device=device,
                                    dtype=torch.int32).to(torch.uint8)
            num_groups = K // group_size
            scales_t = (torch.randn(E, num_groups, N, device=device,
                                    dtype=torch.float16).abs() * 0.01 + 0.001)
            zeros_t = torch.randint(0, 255, (E, num_groups, N // 2), device=device,
                                    dtype=torch.int32).to(torch.uint8)
            topk_ids = torch.randint(0, E, (M, topk), device=device, dtype=torch.int32)
            topk_weights_flat = torch.randn(M * topk, device=device, dtype=torch.float32).abs()
            inputs={'A':input_tensor,'qweight':qweight,'scales':scales_t,'zeros':zeros_t,
                    'ids':topk_ids,'weights':topk_weights_flat}
            options={'mul_routed_weight':True,'group_size':group_size,'use_int4':True}
            elapsed_ms, metadata = checked_benchmark(
                _benchmark_cuda_graph_or_events, lambda: invoke(mod, inputs, options),
                inputs=inputs, reference=lambda saved:reference(saved,options), check=check_output,
                perturb=perturb_activation, warmup=WARMUP_ITERATIONS,
                repetition=BENCHMARK_ITERATIONS, use_cuda_graph=False,
                fallback_reason='fused_moe_host_routing_and_dynamic_allocations')
            row.update(execution_time_ms=elapsed_ms, **metadata)
        except Exception as exc:
            row.update(execution_time_ms=-1.0, error=f'{type(exc).__name__}: {exc}',
                       failure_kind=getattr(exc, 'failure_kind', 'measurement_failure'))
        test_cases.append(row)
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": str(err) if err else None}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": str(err) if err else None, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(test_cases, f, indent=2)
        if test_cases:
            total_time = sum(case["execution_time_ms"] for case in test_cases if case["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} test case(s), total time: {total_time:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)


if __name__ == "__main__":
    main()
