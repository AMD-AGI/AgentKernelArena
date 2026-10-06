"""CPU simulation of the native Opus workspace's per-stream capture contract."""
import ast
from contextlib import contextmanager
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def protected_capture_block(legacy=False):
    tree = ast.parse((ROOT / 'scripts/production_comparison.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    cases = next(node for node in function.body if isinstance(node, ast.For))
    legs = next(node for node in cases.body if isinstance(node, ast.For))
    start = next(i for i, node in enumerate(legs.body) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'stream' for t in node.targets))
    capture = next(i for i, node in enumerate(legs.body) if isinstance(node, ast.With) and isinstance(node.items[0].context_expr, ast.Call) and isinstance(node.items[0].context_expr.func, ast.Attribute) and node.items[0].context_expr.func.attr == 'graph')
    block = legs.body[start:capture + 1]
    if legacy:
        block[-1].items[0].context_expr.keywords = []
    return compile(ast.fix_missing_locations(ast.Module(body=block, type_ignores=[])), 'protected_capture', 'exec')


class Tests(unittest.TestCase):
    def simulate(self, legacy=False):
        active = {'stream': None, 'capturing': False}
        initialized, warmups, captures = set(), [], []
        class Stream:
            def wait_stream(self, other):
                pass
        default, hidden = Stream(), Stream()
        active['stream'] = default
        @contextmanager
        def stream_context(stream):
            previous = active['stream']; active['stream'] = stream
            try: yield
            finally: active['stream'] = previous
        @contextmanager
        def graph_context(graph, stream=None):
            with stream_context(hidden if stream is None else stream):
                active['capturing'] = True
                try: yield
                finally: active['capturing'] = False
        def invoke():
            stream = active['stream']
            if active['capturing']:
                if stream not in initialized:
                    raise RuntimeError('splitk workspace not initialized for current stream')
                captures.append(stream)
            else:
                initialized.add(stream); warmups.append(stream)
        cuda = types.SimpleNamespace(Stream=Stream, current_stream=lambda: active['stream'],
                                     stream=stream_context, CUDAGraph=object, graph=graph_context)
        exec(protected_capture_block(legacy), {'torch': types.SimpleNamespace(cuda=cuda), 'invoke': invoke, 'case': {'calls_per_sample': 2}})
        return warmups, captures

    def test_native_capture_reuses_exact_warmed_stream(self):
        warmups, captures = self.simulate()
        self.assertEqual(len(warmups), 3)
        self.assertEqual(len(captures), 2)
        self.assertTrue(all(stream is warmups[0] for stream in warmups + captures))

    def test_legacy_implicit_capture_stream_reproduces_workspace_error(self):
        with self.assertRaisesRegex(RuntimeError, 'splitk workspace not initialized'):
            self.simulate(legacy=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
