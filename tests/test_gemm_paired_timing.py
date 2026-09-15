"""Keep all GEMM benchmark work intact except the paired-method repair.

The actual BF16 harness is executed for both roles, including incorrect measured
outputs and replay, in test_flydsl_task_migration_v2. These snapshots additionally
protect every affected harness's input, numerical and sampling statements.
"""
import ast
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BEFORE = {'batched_gemm_a8w8_kernel': {'arena_benchmark': '794b1125a25e7e3a0b4d255fa19e263e086b6c0d22e7c16a7d414f131b54687b',
                              'run_benchmark': 'cde32db449b949672b4d02cba0f5c3858e3553a3e9f53515034cc7dca9435674'},
 'batched_gemm_bf16_kernel': {'arena_benchmark': '6e22c53d2954bc7f14bc0a7362a083f9c70e79c6c33f4463beba6758cd7e77fc',
                              'run_benchmark': 'e1faf58179a3180609a1946b5c9a14863b13fe210a393be3457d74b5f8eec833'},
 'gemm_a16w8_blockscale_kernel': {'arena_benchmark': 'defc1227e2c5eb6db2c8bb530c0d3d20493271b8a1f3b504c24154573cba059e',
                                  'run_benchmark': '9a48b014da77dbd608d90d612a6388e18f95fd8f33e482171a4700e5686f725d'},
 'gemm_a16wfp4_kernel': {'arena_benchmark': '3818b99f737ad1ec43e6d4c575da11642801346bb036cfe9ecfa143aedff8060',
                         'run_benchmark': '4a94553fec39201aa1aa248165c57f38edc28334c765cca94b19a1fbe589ca33'},
 'gemm_a4w4_kernel': {'arena_benchmark': 'c8853015a53f2f888f974a53e4a02f7b22c1aeceeb0d2c6aa2957d4d3cca1ad4',
                      'run_benchmark': '4ca5e661c5d35618e6dc4591984dc4a4369ffe793043fdea4dac133186f43f8c'},
 'gemm_a8w8_blockscale_kernel': {'arena_benchmark': '7a62ff1081019435b838188ad687efdcd029b0a9c87720302f162ed78643dd47',
                                 'run_benchmark': 'b172161ae8a4c4cba052938c63aa5c3b4c6f674bc66f4c0f20551c3b7614f556'},
 'gemm_a8w8_kernel': {'arena_benchmark': '1b94d6280a483fb04e1c8aaebc9848c2ebea4202635ffb20f0b525bf9311e61c',
                      'run_benchmark': 'e9cbc03267bff29c1aaf61240ab3363b389563fc7f3dea389ceb7f96f3dd84dd'},
 'gemm_a8w8_per_token_scale_kernel': {'arena_benchmark': '363964d370db9089b89b36950d53c1768c03b9eb9b256e4cd691ebb014f3e55d',
                                      'run_benchmark': 'dca6bb6cf3914593cc7ae6e3e23b008a3266b9d18cd78dbcb39a5349858b104e'},
 'gemm_a8wfp4_kernel': {'arena_benchmark': '3f765494cebcd6e7e272c46c087c5deef2da8930eb863fcf3218af19326b1467',
                        'run_benchmark': 'ba914faf92c506e87a2884cadf492d0cf88ccebd796267fee9ccdcb2e25e7b6a'},
 'gemm_afp4wfp4_kernel': {'arena_benchmark': '5aab1be9ee10f65ac072ce7b0565edcf54faef708d1acc4225df52abb9dcaf40',
                          'run_benchmark': '28293637bec4ef2e181d77971151d2f0ff1a428b107b8542bb5ec8db9360cac2'},
 'gemm_afp8wfp8_kernel': {'arena_benchmark': 'e87e8dbc1508828b5f42cfaa941575f2327ac030a04e1eadd315ee815e707450',
                          'run_benchmark': 'd9172d3f2b94921e9574d5eabe8d2c99b521eb12f0d6b4ef8a206bb97e10ad7b'}}


def normalize_former_role_policy(function):
    """Undo only the reviewed policy change for comparison with the old AST."""
    changed = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "use_graph":
            assert isinstance(node.value, ast.Constant) and node.value.value is False
            node.value = ast.Name(id="has_kernel", ctx=ast.Load())
            changed.add(target.id)
        elif target.id == "event_reason":
            assert isinstance(node.value, ast.Constant)
            assert node.value.value == "capture_unsafe_aiter_hipblaslt"
            node.value = ast.parse('None if use_graph else "capture_unsafe_aiter_hipblaslt"', mode="eval").body
            changed.add(target.id)
    assert changed == {"use_graph", "event_reason"}
    return function


class RemoveMeasuredOutputControls(ast.NodeTransformer):
    """Remove only reviewed, separately tested post-timing controls.

    This lets the original benchmark snapshots continue protecting operator
    calls, warmups, samples and allocation boundaries after replay validation.
    """
    def visit_Assign(self, node):
        if len(node.targets) == 1 and getattr(node.targets[0], "id", None) in {
            "originals", "expected", "timed", "replay_validate"
        }:
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call):
            if getattr(call.func, "id", None) == "require_unchanged":
                return None
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "update"
                    and call.args and isinstance(call.args[0], ast.Call)
                    and getattr(call.args[0].func, "id", None) in {
                        "verify_timed_run", "replay_validate"
                    }):
                return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if getattr(node.func, "id", None) == "benchmark_cuda_graph_or_events":
            node.keywords = [k for k in node.keywords if k.arg != "timed_run"]
        return self.generic_visit(node)


@pytest.mark.parametrize("name", sorted(BEFORE))
@pytest.mark.parametrize("function", ["run_benchmark", "arena_benchmark"])
def test_pairing_repair_preserves_original_work_and_sampling(name, function):
    path = ROOT / "tasks/torch2flydsl" / name / "test_kernel_harness.py"
    fn = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == function)
    fn = normalize_former_role_policy(fn)
    if name in {"batched_gemm_a8w8_kernel", "gemm_a16w8_blockscale_kernel",
                "gemm_a16wfp4_kernel", "gemm_a4w4_kernel", "gemm_a8w8_blockscale_kernel",
                "gemm_a8w8_kernel", "gemm_a8w8_per_token_scale_kernel",
                "gemm_a8wfp4_kernel", "gemm_afp4wfp4_kernel", "gemm_afp8wfp8_kernel"}:
        fn = RemoveMeasuredOutputControls().visit(fn)
    assert hashlib.sha256(ast.dump(fn, include_attributes=False).encode()).hexdigest() == BEFORE[name][function]
