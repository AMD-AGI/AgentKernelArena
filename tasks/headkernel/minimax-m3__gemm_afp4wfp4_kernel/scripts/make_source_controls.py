"""Create source-only bad candidates for later real GPU binding probes."""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def control_source(kind):
    source = (ROOT / "ut/reference/kernel.py").read_text()
    tree = ast.parse(source)
    kernel, = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "_gemm_afp4wfp4_kernel"]
    if kind == "no_op":
        kernel.body = [ast.Return()]
        return ast.unparse(tree) + "\n"
    if kind == "wrong_output":
        original = "c = accumulator.to(c_ptr.type.element_ty)"
        if original not in source:
            raise ValueError("source-control patch context changed")
        return source.replace(original, "c = (accumulator * 0).to(c_ptr.type.element_ty)", 1)
    raise ValueError("unknown negative control")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for kind in ("no_op", "wrong_output"):
        path = args.output / kind / "source/kernel.py"
        path.parent.mkdir(parents=True)
        path.write_text(control_source(kind))


if __name__ == "__main__":
    main()
