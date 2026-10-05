"""CPU-only checks for the editable MoE copy's AITER operation registration."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / (
    "tasks/headkernel/qwen3.8-2.4t__fused_moe_2stage_mxfp4/source/moe_candidate.py"
)


def _candidate_decorator(registry, guard, fake, module_name="moe_candidate"):
    """Execute only registration code, without importing torch or the kernel.

    Evaluate the decorators attached to the real candidate entry so this check
    also catches accidentally leaving the original decorator on that entry.
    """
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    wrapper = functions.get("_register_candidate_moe")
    namespace = {
        "__name__": module_name,
        "torch": SimpleNamespace(ops=SimpleNamespace(aiter=registry)),
        "torch_compile_guard": guard,
        "fused_moe_fake": fake,
    }
    if wrapper is not None:
        isolated = ast.Module(body=[wrapper], type_ignores=[])
        exec(compile(isolated, str(SOURCE), "exec"), namespace)
    decorators = [
        eval(compile(ast.Expression(node), str(SOURCE), "eval"), namespace)
        for node in functions["fused_moe_"].decorator_list
    ]

    def decorate(function):
        for decorator in reversed(decorators):
            function = decorator(function)
        return function

    return decorate


class _Operation:
    def __init__(self, function, fake):
        self.function = function
        self.fake = fake
        self.schema = inspect.signature(function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class MoERegistrationTests(unittest.TestCase):
    def setUp(self):
        def production(hidden_states, *, scale=1):
            return ("production", hidden_states, scale)

        def production_fake(*args, **kwargs):
            return "production fake"

        def candidate_fake(*args, **kwargs):
            return "candidate fake"

        self.production = _Operation(production, production_fake)
        self.registry = SimpleNamespace(fused_moe_=self.production)
        self.fake = candidate_fake
        self.registrations = []

    def guard(self, *, gen_fake):
        def decorate(function):
            self.registrations.append((function, gen_fake))
            # Model AITER's shared operation namespace: an existing name can
            # return the old operation even for a different Python function.
            if hasattr(self.registry, function.__name__):
                return getattr(self.registry, function.__name__)
            operation = _Operation(function, gen_fake)
            setattr(self.registry, function.__name__, operation)
            return operation

        return decorate

    @staticmethod
    def candidate():
        def fused_moe_(hidden_states, *, scale=1):
            return ("candidate", hidden_states, scale)

        return fused_moe_

    def test_candidate_dispatch_does_not_reuse_or_replace_production(self):
        decorate = _candidate_decorator(self.registry, self.guard, self.fake)
        candidate = decorate(self.candidate())

        self.assertIs(self.registry.fused_moe_, self.production)
        self.assertEqual(self.production("input", scale=2), ("production", "input", 2))
        self.assertIsNot(candidate, self.production)
        self.assertEqual(candidate("input", scale=2), ("candidate", "input", 2))
        self.assertIs(self.registry.moe_candidate_fused_moe_, candidate)

    def test_original_function_schema_and_fake_callback_reach_guard(self):
        function = self.candidate()
        schema = inspect.signature(function)
        decorate = _candidate_decorator(self.registry, self.guard, self.fake)
        candidate = decorate(function)

        self.assertEqual(self.registrations, [(function, self.fake)])
        self.assertIs(candidate.function, function)
        self.assertEqual(candidate.schema, schema)
        self.assertIs(candidate.fake, self.fake)
        self.assertEqual(candidate.fake("input"), "candidate fake")
        self.assertEqual(self.production.fake("input"), "production fake")

    def test_duplicate_candidate_is_rejected_before_guard_can_reuse_it(self):
        decorate = _candidate_decorator(self.registry, self.guard, self.fake)
        candidate = decorate(self.candidate())

        with self.assertRaisesRegex(RuntimeError, "already registered: moe_candidate_fused_moe_"):
            decorate(self.candidate())

        self.assertEqual(len(self.registrations), 1)
        self.assertIs(self.registry.moe_candidate_fused_moe_, candidate)
        self.assertIs(self.registry.fused_moe_, self.production)

    def test_module_names_have_distinct_registration_keys(self):
        first = _candidate_decorator(self.registry, self.guard, self.fake)(self.candidate())
        decorate = _candidate_decorator(
            self.registry, self.guard, self.fake, module_name="other.moe_candidate"
        )
        second = decorate(self.candidate())

        self.assertIsNot(first, second)
        self.assertIs(self.registry.moe_candidate_fused_moe_, first)
        self.assertIs(self.registry.other_moe_candidate_fused_moe_, second)
        self.assertEqual(second("input"), ("candidate", "input", 1))


if __name__ == "__main__":
    unittest.main()
