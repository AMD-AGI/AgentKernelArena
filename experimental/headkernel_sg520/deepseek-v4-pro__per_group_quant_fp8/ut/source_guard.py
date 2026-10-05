"""Permit only a balanced GPU function body inside the frozen native TU.

This is a compilation boundary, not a C++ security sandbox. Preprocessing and
the host launch/binding code remain frozen. Helpers can be local device lambdas
inside the target body; arbitrary host helpers are outside this task's scope.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TARGET = "dynamic_per_group_scaled_quant_kernel("
FROZEN_TYPE_SELECTION = """#if defined(__gfx942__)
                : aiter::MxDtype::FP8_E4M3_FNUZ;
#else
                : aiter::MxDtype::FP8_E4M3;
#endif"""


def _tokens(text, reject_splicing=True):
    """Skip C++ comments/literals before inspecting structural punctuation."""
    if reject_splicing and re.search(r"\\[ \t]*[\r\n]|\?\?[=/'()!<>-]", text):
        raise ValueError("line splicing and trigraphs are outside the kernel-body boundary")
    i = 0
    while i < len(text):
        if text.startswith("//", i):
            end = re.search(r"[\r\n]", text[i + 2:])
            i = len(text) if end is None else i + 2 + end.end()
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated C++ comment")
            i = end + 2
        elif text.startswith('R"', i):
            opening = text.find("(", i + 2)
            delimiter = text[i + 2:opening]
            if opening < 0 or len(delimiter) > 16 or re.search(r'[\s()\\]', delimiter):
                raise ValueError("invalid C++ raw string")
            closing = text.find(")" + delimiter + '"', opening + 1)
            if closing < 0:
                raise ValueError("unterminated C++ raw string")
            i = closing + len(delimiter) + 2
        elif text[i] in "\"'":
            quote = text[i]
            i += 1
            while i < len(text) and text[i] != quote:
                if text[i] in "\r\n":
                    raise ValueError("unterminated C++ literal")
                i += 2 if text[i] == "\\" else 1
            if i >= len(text):
                raise ValueError("unterminated C++ literal")
            i += 1
        else:
            if reject_splicing and text[i:i + 2] in ("<%", "%>", "%:"):
                raise ValueError("C++ brace and preprocessing digraphs are outside the GPU-body boundary")
            yield i, text[i]
            i += 1


def body_boundary(original):
    opening = original.index("{", original.index(TARGET))
    depth = 1
    for offset, token in _tokens(original[opening + 1:], reject_splicing=False):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return opening + 1, opening + 1 + offset
    raise ValueError("frozen kernel body is unbalanced")


def validate_source(candidate, original):
    start, end = body_boundary(original)
    prefix, suffix = original[:start], original[end:]
    if not candidate.startswith(prefix) or not candidate.endswith(suffix) or len(candidate) < len(prefix) + len(suffix):
        raise ValueError("only the target GPU kernel body is editable; host code and ABI are frozen")
    body = candidate[len(prefix):len(candidate) - len(suffix)]
    # The stock kernel contains this brace-free architecture type selection.
    # It may remain byte-identical or be removed with a rewritten GPU body.
    # Every other preprocessing token remains forbidden.
    if body.count(FROZEN_TYPE_SELECTION) > 1:
        raise ValueError("new preprocessing blocks are outside the editable GPU body")
    body = body.replace(FROZEN_TYPE_SELECTION, ": aiter::MxDtype::FP8_E4M3;")
    depth = 0
    tokens = list(_tokens(body))
    code = "".join(token for _, token in tokens)
    if re.search(r"\b(?:__host__|__attribute__?|_Pragma|__pragma|constructor|destructor|init_priority)\b", code):
        raise ValueError("host attributes and preprocessing operators are outside the GPU-body boundary")
    for _, token in tokens:
        if token in ("#", "\\"):
            raise ValueError("preprocessor directives are forbidden in the editable GPU body")
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                raise ValueError("candidate escapes the target GPU body")
    if depth:
        raise ValueError("candidate GPU body is unbalanced")


def validate_source_file():
    path = ROOT / "source/quant_kernels.cu"
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("candidate source must be an in-workspace regular file")
    validate_source(path.read_text(), (ROOT / "ut/native/quant_kernels.reference.cu").read_text())
