"""Self-contained graph-first GPU benchmark helpers.

This module is copied next to task performance entrypoints as
``_aka_benchmark.py``.  Keep its runtime dependencies limited to the Python
standard library and PyTorch so materialized task workspaces remain portable.

PyTorch exposes the CUDA API namespace on both CUDA and ROCm builds, so the
``torch.cuda`` calls below intentionally cover CUDA Graph and HIP Graph.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from typing import Any, Callable

import torch


_EMPTY_GRAPH_FLOOR_MS = 1.0e-4
_FORCE_EVENT_ENV = "AKA_BENCHMARK_FORCE_EVENT"


class _EmptyGraphCapture(RuntimeError):
    """Raised when PyTorch explicitly reports a capture with no GPU nodes."""


_CURRENT_STREAM_EXPRESSION = re.compile(
    r"\s*(?:(?:at|c10)::cuda::getCurrentCUDAStream|"
    r"(?:at|c10)::hip::getCurrentHIPStream|"
    r"at::hip::getCurrentHIPStreamMasqueradingAsCUDA)"
    r"\s*\(\s*[^()]*\s*\)"
    r"\s*(?:\.\s*stream\s*\(\s*\))?\s*",
    re.IGNORECASE | re.DOTALL,
)
_PLAIN_ASSIGNMENT = re.compile(
    r"\b([A-Za-z_]\w*)\s*=(?!=)\s*([^;{}]*);",
    re.DOTALL,
)
_CAPTURE_UNSAFE_HIP_API = re.compile(
    r"\b(?:hip|cuda)(?:DeviceSynchronize|StreamSynchronize|EventSynchronize|"
    r"DeviceReset|StreamQuery|EventQuery|Malloc\w*|Free\w*|HostMalloc|"
    r"HostFree|ExtMalloc\w*)\s*\("
)
_EXPLICIT_LAUNCH_APIS = {
    # API name: zero-based stream-argument position.
    "hipLaunchKernelGGL": 4,
    "hipLaunchKernel": 5,
    "cudaLaunchKernel": 5,
    "hipLaunchCooperativeKernel": 5,
    "cudaLaunchCooperativeKernel": 5,
    "hipExtLaunchKernel": 5,
    "hipModuleLaunchKernel": 8,
    "hipExtModuleLaunchKernel": 8,
    "cudaModuleLaunchKernel": 8,
    "hipMemcpyAsync": 4,
    "cudaMemcpyAsync": 4,
    "hipMemcpy2DAsync": 7,
    "cudaMemcpy2DAsync": 7,
    "hipMemcpy3DAsync": 1,
    "cudaMemcpy3DAsync": 1,
    "hipMemcpyPeerAsync": 5,
    "cudaMemcpyPeerAsync": 5,
    "hipMemcpyToSymbolAsync": 5,
    "cudaMemcpyToSymbolAsync": 5,
    "hipMemcpyFromSymbolAsync": 5,
    "cudaMemcpyFromSymbolAsync": 5,
    "hipMemsetAsync": 3,
    "cudaMemsetAsync": 3,
    "hipMemset2DAsync": 5,
    "cudaMemset2DAsync": 5,
    "hipMemset3DAsync": 3,
    "cudaMemset3DAsync": 3,
    "hipMemPrefetchAsync": 3,
    "cudaMemPrefetchAsync": 3,
    "hipGraphLaunch": 1,
    "cudaGraphLaunch": 1,
    "hipStreamWaitEvent": 0,
    "cudaStreamWaitEvent": 0,
    "hipEventRecord": 1,
    "cudaEventRecord": 1,
}
_UNKNOWN_ASYNC_OR_LAUNCH_API = re.compile(
    r"\b((?:hip|cuda)[A-Za-z_]\w*(?:Async|Launch[A-Za-z_]*))\s*\("
)
_MEMORY_API = re.compile(
    r"\b((?:hip|cuda)(?:Memcpy|Memset)[A-Za-z0-9_]*)\s*\("
)
_CAPTURE_UNSAFE_CALLBACK_OR_ALLOCATION = re.compile(
    r"\b(?:hip|cuda)(?:LaunchHostFunc|StreamAddCallback|MallocAsync|"
    r"MallocFromPoolAsync|FreeAsync)\s*\("
)


def _strip_cpp_comments_and_literals(source: str) -> str:
    """Blank C/C++ comments and literals while retaining source positions."""

    chars = list(source)
    index = 0
    length = len(chars)
    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            end = length if end < 0 else end
            for pos in range(index, end):
                chars[pos] = " "
            index = end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = length - 2 if end < 0 else end
            for pos in range(index, min(length, end + 2)):
                if chars[pos] != "\n":
                    chars[pos] = " "
            index = min(length, end + 2)
            continue
        if source[index] in {'"', "'"}:
            quote = source[index]
            chars[index] = " "
            index += 1
            while index < length:
                if source[index] == "\\":
                    chars[index] = " "
                    if index + 1 < length:
                        if chars[index + 1] != "\n":
                            chars[index + 1] = " "
                        index += 2
                    else:
                        index += 1
                    continue
                terminal = source[index] == quote
                if chars[index] != "\n":
                    chars[index] = " "
                index += 1
                if terminal:
                    break
            continue
        index += 1
    code = "".join(chars)
    # These tasks compile HIP sources with USE_ROCM defined. Retain the active
    # branch so a portable CUDA/ROCm source does not look like it reassigns the
    # stream to nullptr. Only handle a flat conditional; nested/unknown
    # preprocessing deliberately remains conservative.
    rocm_conditional = re.compile(
        r"^[ \t]*#[ \t]*(?:if[ \t]+defined[ \t]*\([ \t]*USE_ROCM[ \t]*\)"
        r"|ifdef[ \t]+USE_ROCM)[ \t]*\n"
        r"(?P<active>(?:(?!^[ \t]*#[ \t]*(?:else|endif)\b)[\s\S])*)"
        r"^[ \t]*#[ \t]*else\b[ \t]*\n"
        r"(?:(?!^[ \t]*#[ \t]*endif\b)[\s\S])*"
        r"^[ \t]*#[ \t]*endif\b[ \t]*(?:\n|$)",
        re.MULTILINE,
    )
    code = rocm_conditional.sub(lambda match: match.group("active"), code)
    # Common task sources retain synchronization-only debug blocks. If DEBUG is
    # absent, a flat ``#ifdef DEBUG`` selects its ``#else`` branch (when any),
    # not an empty block. Preserve that branch so unsafe production work cannot
    # be hidden by a harmless disabled debug arm. Nested/compound conditions are
    # left intact and therefore fail closed if either arm is capture-unsafe.
    if not re.search(r"^\s*#\s*define\s+DEBUG\b", code, re.MULTILINE):
        debug_with_else = re.compile(
            r"^[ \t]*#[ \t]*(?:ifdef[ \t]+DEBUG|"
            r"if[ \t]+defined[ \t]*\([ \t]*DEBUG[ \t]*\))[ \t]*\n"
            r"(?:(?!^[ \t]*#[ \t]*(?:if|ifdef|ifndef|else|endif)\b)[\s\S])*"
            r"^[ \t]*#[ \t]*else\b[ \t]*\n"
            r"(?P<production>(?:(?!^[ \t]*#[ \t]*(?:if|ifdef|ifndef|endif)\b)[\s\S])*)"
            r"^[ \t]*#[ \t]*endif\b[ \t]*(?:\n|$)",
            re.MULTILINE,
        )
        code = debug_with_else.sub(
            lambda match: match.group("production"), code
        )
        debug_without_else = re.compile(
            r"^[ \t]*#[ \t]*(?:ifdef[ \t]+DEBUG|"
            r"if[ \t]+defined[ \t]*\([ \t]*DEBUG[ \t]*\))[ \t]*\n"
            r"(?:(?!^[ \t]*#[ \t]*(?:if|ifdef|ifndef|else|endif)\b)[\s\S])*"
            r"^[ \t]*#[ \t]*endif\b[ \t]*(?:\n|$)",
            re.MULTILINE,
        )
        code = debug_without_else.sub("", code)
    return code


def _split_top_level_cpp_args(arguments: str) -> list[str]:
    """Split a C++ call/configuration argument list at top-level commas."""

    result: list[str] = []
    start = 0
    round_depth = square_depth = brace_depth = 0
    for index, char in enumerate(arguments):
        if char == "(":
            round_depth += 1
        elif char == ")":
            round_depth = max(0, round_depth - 1)
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth = max(0, square_depth - 1)
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth = max(0, brace_depth - 1)
        elif char == "," and not (round_depth or square_depth or brace_depth):
            result.append(arguments[start:index].strip())
            start = index + 1
    result.append(arguments[start:].strip())
    return result


def _balanced_call_arguments(source: str, opening: int) -> str | None:
    depth = 0
    for index in range(opening, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    return None


def _scope_stack_at(source: str, position: int) -> tuple[int, ...]:
    stack: list[int] = []
    for index, char in enumerate(source[:position]):
        if char == "{":
            stack.append(index)
        elif char == "}" and stack:
            stack.pop()
    return tuple(stack)


def _scope_is_ancestor(
    possible_ancestor: tuple[int, ...], descendant: tuple[int, ...]
) -> bool:
    return descendant[: len(possible_ancestor)] == possible_ancestor


def _matching_open_paren(source: str, closing: int) -> int | None:
    depth = 0
    for index in range(closing, -1, -1):
        char = source[index]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                return index
    return None


def _function_definition_at(
    source: str, position: int
) -> tuple[str, list[str], int] | None:
    """Return the enclosing function name, parameter names, and body scope.

    This intentionally recognizes only ordinary C/C++ function definitions.
    Unusual macro-generated or indirect launch plumbing fails closed.
    """

    control_words = {"if", "for", "while", "switch", "catch"}
    for body_open in reversed(_scope_stack_at(source, position)):
        prefix = source[:body_open].rstrip()
        # Permit common trailing qualifiers between the parameter list and body.
        prefix = re.sub(
            r"(?:\b(?:const|noexcept|override|final)\b\s*)+$", "", prefix
        ).rstrip()
        if not prefix.endswith(")"):
            continue
        closing = len(prefix) - 1
        opening = _matching_open_paren(prefix, closing)
        if opening is None:
            continue
        name_match = re.search(
            r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*$", prefix[:opening]
        )
        if name_match is None:
            continue
        qualified_name = name_match.group(1)
        name = qualified_name.rsplit("::", 1)[-1]
        if name in control_words:
            continue

        parameter_names: list[str] = []
        for parameter in _split_top_level_cpp_args(prefix[opening + 1 : closing]):
            parameter = parameter.split("=", 1)[0].strip()
            if not parameter or parameter == "void" or parameter == "...":
                continue
            identifiers = re.findall(r"\b[A-Za-z_]\w*\b", parameter)
            if not identifiers:
                return None
            parameter_names.append(identifiers[-1])
        return qualified_name, parameter_names, body_open
    return None


def _same_function_assignments(
    name: str, source: str, position: int
) -> list[re.Match[str]]:
    """Return every prior assignment to ``name`` in the launch function.

    An assignment in a completed child scope can still mutate a variable from
    the enclosing scope. Ignoring it would incorrectly prove code such as
    ``if (bad) { stream = 0; }`` safe. We therefore inspect every assignment in
    the function and accept only direct authoritative current-stream values.
    """

    launch_function = _function_definition_at(source, position)
    launch_body = launch_function[2] if launch_function is not None else None
    assignments: list[re.Match[str]] = []
    for assignment in _PLAIN_ASSIGNMENT.finditer(source, 0, position):
        if assignment.group(1) != name:
            continue
        assignment_function = _function_definition_at(source, assignment.start())
        assignment_body = (
            assignment_function[2] if assignment_function is not None else None
        )
        if assignment_body == launch_body:
            assignments.append(assignment)
    return assignments


def _assignment_is_plain_stream_declaration(
    source: str, assignment: re.Match[str]
) -> bool:
    """Recognize an unconditional local stream declaration initializer."""

    boundary = max(
        source.rfind(";", 0, assignment.start()),
        source.rfind("{", 0, assignment.start()),
        source.rfind("}", 0, assignment.start()),
    )
    declaration_prefix = source[boundary + 1 : assignment.start()]
    return re.fullmatch(
        r"\s*(?:(?:const|constexpr)\s+)*"
        r"(?:auto|(?:hip|cuda)Stream_t)(?:\s+const)?"
        r"(?:\s*[*&])?\s*",
        declaration_prefix,
    ) is not None


def _has_visible_uninitialized_stream_declaration(
    name: str, source: str, position: int
) -> bool:
    """Reject a local declaration that can shadow a proven outer stream."""

    launch_scope = _scope_stack_at(source, position)
    launch_function = _function_definition_at(source, position)
    launch_body = launch_function[2] if launch_function is not None else None
    declaration_pattern = re.compile(
        rf"\b(?:(?:const|constexpr)\s+)*(?:auto|(?:hip|cuda)Stream_t)"
        rf"(?:\s+const)?(?:\s*[*&])?\s+{re.escape(name)}\b"
    )
    for declaration in declaration_pattern.finditer(source, 0, position):
        declaration_function = _function_definition_at(source, declaration.start())
        declaration_body = (
            declaration_function[2] if declaration_function is not None else None
        )
        if declaration_body != launch_body:
            # Function parameters occur before the body and are handled by
            # call-edge provenance below.
            continue
        declaration_scope = _scope_stack_at(source, declaration.start())
        if not _scope_is_ancestor(declaration_scope, launch_scope):
            continue
        statement_end = source.find(";", declaration.end(), position)
        if statement_end < 0:
            return True
        initialized = any(
            declaration.start() <= assignment.start() < statement_end
            for assignment in _PLAIN_ASSIGNMENT.finditer(
                source, declaration.start(), statement_end + 1
            )
            if assignment.group(1) == name
        )
        if not initialized:
            return True
    return False


def _parameter_has_current_stream_provenance(
    name: str,
    source: str,
    position: int,
    all_sources: list[str],
    visited: set[tuple[int, int, str]],
) -> bool:
    function = _function_definition_at(source, position)
    if function is None:
        return False
    function_name, parameter_names, body_open = function
    # An unqualified definition nested in a namespace/class cannot be matched
    # safely to text-only call sites without a C++ name resolver. Ordinary task
    # split wrappers are global; unusual plumbing intentionally falls back.
    if "::" not in function_name and _scope_stack_at(source, body_open):
        return False
    try:
        parameter_index = parameter_names.index(name)
    except ValueError:
        return False

    definitions = []
    for definition_source in all_sources:
        for brace in re.finditer(r"\{", definition_source):
            definition = _function_definition_at(
                definition_source, brace.start() + 1
            )
            if (
                definition is not None
                and definition[0] == function_name
                and definition[2] == brace.start()
            ):
                definitions.append((definition_source, brace.start()))
    if len(definitions) != 1:
        return False

    call_arguments: list[tuple[str, str, int]] = []
    call_pattern = re.compile(
        rf"(?<![A-Za-z0-9_:>.]){re.escape(function_name)}\s*\("
    )
    for caller_source in all_sources:
        for call in call_pattern.finditer(caller_source):
            # A real call must be inside another function body. This excludes
            # global declarations and the target function definition itself.
            caller_function = _function_definition_at(caller_source, call.start())
            if caller_function is None:
                continue
            arguments = _balanced_call_arguments(caller_source, call.end() - 1)
            if arguments is None:
                return False
            parts = _split_top_level_cpp_args(arguments)
            if len(parts) <= parameter_index:
                return False
            call_arguments.append(
                (parts[parameter_index], caller_source, call.start())
            )

    # Require proof for at least one call edge, and for every visible call edge.
    if not call_arguments:
        return False
    return all(
        _stream_expression_is_current(
            argument,
            caller_source,
            call_position,
            all_sources,
            visited,
        )
        for argument, caller_source, call_position in call_arguments
    )


def _stream_expression_is_current(
    expression: str,
    source: str,
    launch_position: int,
    all_sources: list[str],
    visited: set[tuple[int, int, str]] | None = None,
) -> bool:
    if _CURRENT_STREAM_EXPRESSION.fullmatch(expression):
        return True
    if re.fullmatch(r"\s*[A-Za-z_]\w*\s*", expression) is None:
        return False
    name = expression.strip()

    assignments = _same_function_assignments(name, source, launch_position)
    if any(
        _CURRENT_STREAM_EXPRESSION.fullmatch(assignment.group(2)) is None
        for assignment in assignments
    ):
        return False
    if _has_visible_uninitialized_stream_declaration(
        name, source, launch_position
    ):
        return False

    launch_scope = _scope_stack_at(source, launch_position)
    if any(
        _assignment_is_plain_stream_declaration(source, assignment)
        and _scope_is_ancestor(
            _scope_stack_at(source, assignment.start()), launch_scope
        )
        for assignment in assignments
    ):
        return True

    if visited is None:
        visited = set()
    provenance_key = (id(source), launch_position, name)
    if provenance_key in visited:
        return False
    visited.add(provenance_key)
    return _parameter_has_current_stream_provenance(
        name, source, launch_position, all_sources, visited
    )


def _stream_expression_is_legacy_default(expression: str) -> bool:
    normalized = re.sub(r"[\s()]", "", expression)
    return normalized in {
        "0",
        "NULL",
        "nullptr",
        "hipStreamDefault",
        "cudaStreamDefault",
    }


def hip_source_graph_capture_policy(*source_paths: Any) -> tuple[bool, str | None]:
    """Conservatively decide whether HIP/CUDA source follows capture stream.

    A PyTorch graph capture uses a non-default side stream.  A compiled kernel
    that launches on literal stream zero can therefore escape capture while
    capturable tensor setup still makes the graph look non-empty.  Callers use
    this preflight to force *both* sides of a comparison to GPU-event timing
    unless every visible launch has an explicit current-stream provenance.

    Multiple files may be supplied for split C++/HIP wrappers: a current stream
    acquired in the fixed C++ binding can then justify the stream parameter
    used by the editable HIP launcher.  Unknown constructs intentionally fail
    closed; Event timing is preferable to a fabricated graph speedup.
    """

    if not source_paths:
        return False, "hip_source_launch_stream_unverified"

    sources: list[str] = []
    for path in source_paths:
        try:
            with open(path, encoding="utf-8") as source_file:
                raw_source = source_file.read()
        except (OSError, UnicodeError):
            return False, "hip_source_unreadable"
        if "AKA_BENCHMARK_EVENT_ONLY" in raw_source:
            return False, "hip_source_declares_event_only"
        sources.append(_strip_cpp_comments_and_literals(raw_source))

    if any(
        _CAPTURE_UNSAFE_HIP_API.search(source)
        or _CAPTURE_UNSAFE_CALLBACK_OR_ALLOCATION.search(source)
        for source in sources
    ):
        return False, "hip_source_contains_capture_unsafe_api"

    known_stream_apis = set(_EXPLICIT_LAUNCH_APIS)
    for source in sources:
        for api_match in _MEMORY_API.finditer(source):
            # Streamless memcpy/memset APIs either use the legacy default
            # stream or synchronize the host and are not safe to capture here.
            # Known *Async variants are validated below using their exact
            # stream-argument positions; all other memory APIs fail closed.
            if api_match.group(1) not in known_stream_apis:
                return False, "hip_source_contains_capture_unsafe_api"
        for api_match in _UNKNOWN_ASYNC_OR_LAUNCH_API.finditer(source):
            if api_match.group(1) not in known_stream_apis:
                return False, "hip_source_launch_stream_unverified"

    launch_streams: list[tuple[str | None, str, int]] = []

    for source in sources:
        # Native triple-chevron launch syntax. Three configuration arguments
        # (or fewer) imply the legacy default stream; the fourth is the stream.
        position = 0
        while True:
            opening = source.find("<<<", position)
            if opening < 0:
                break
            closing = source.find(">>>", opening + 3)
            if closing < 0:
                launch_streams.append((None, source, opening))
                break
            config = _split_top_level_cpp_args(source[opening + 3 : closing])
            launch_streams.append(
                (config[3] if len(config) >= 4 else None, source, opening)
            )
            position = closing + 3

        # Runtime launch and asynchronous memory APIs.
        for api_name, stream_index in _EXPLICIT_LAUNCH_APIS.items():
            pattern = re.compile(rf"\b{re.escape(api_name)}\s*\(")
            for match in pattern.finditer(source):
                arguments = _balanced_call_arguments(source, match.end() - 1)
                if arguments is None:
                    launch_streams.append((None, source, match.start()))
                    continue
                parts = _split_top_level_cpp_args(arguments)
                launch_streams.append((
                    parts[stream_index] if len(parts) > stream_index else None,
                    source,
                    match.start(),
                ))

        # Launch-config structures and multi-device launch arrays do not expose
        # a stream expression in a form this task-local helper can prove.
        if re.search(
            r"\b(?:hip|cuda)(?:LaunchKernelEx|LaunchCooperativeKernelMultiDevice)\s*\(",
            source,
        ):
            return False, "hip_source_launch_stream_unverified"

    if not launch_streams:
        return False, "hip_source_launch_stream_unverified"
    if any(stream is None for stream, _source, _position in launch_streams):
        return False, "hip_source_uses_legacy_default_stream"
    if any(
        _stream_expression_is_legacy_default(stream)
        for stream, _source, _position in launch_streams
        if stream is not None
    ):
        return False, "hip_source_uses_legacy_default_stream"
    if any(
        not _stream_expression_is_current(
            stream,
            source,
            position,
            sources,
        )
        for stream, source, position in launch_streams
        if stream is not None
    ):
        return False, "hip_source_launch_stream_unverified"
    return True, None


def _positive_int(value: int, minimum: int = 1) -> int:
    return max(minimum, int(value))


def _require_gpu_timing() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU benchmark unavailable: torch.cuda.is_available() is false; "
            "CPU wall-clock timing is not a valid kernel score"
        )


def _event_elapsed_ms(start_event: Any, end_event: Any) -> float:
    value = float(start_event.elapsed_time(end_event))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"invalid GPU event elapsed time: {value!r}")
    return value


def _wait_for_event(end_event: Any) -> None:
    # Event-local synchronization avoids stalling unrelated streams where the
    # backend supports it.  Older/mocked PyTorch implementations may only
    # expose device-wide synchronization.
    synchronize = getattr(end_event, "synchronize", None)
    if synchronize is not None:
        synchronize()
    else:
        torch.cuda.synchronize()


def benchmark_cuda_event_samples(
    fn: Callable[[], Any],
    repetition: int = 100,
    prepare_fn: Callable[[], Any] | None = None,
) -> list[float]:
    """Return eager per-call GPU-event samples in milliseconds.

    This is the explicit fallback path for callables that cannot be captured in
    a CUDA/HIP Graph.  ``prepare_fn``, when provided, is enqueued before the
    start event for each sample.  It never falls back further to a CPU timer.
    """

    _require_gpu_timing()
    repetition = _positive_int(repetition)
    samples: list[float] = []
    for _ in range(repetition):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if prepare_fn is not None:
            prepare_fn()
        start_event.record()
        fn()
        end_event.record()
        _wait_for_event(end_event)
        samples.append(_event_elapsed_ms(start_event, end_event))
    return samples


def _capture_graph(
    fn: Callable[[], Any],
    repeats: int,
    stream: Any,
    prepare_fn: Callable[[], Any] | None = None,
    output_holder: list[Any] | None = None,
) -> Any:
    graph = torch.cuda.CUDAGraph()
    if prepare_fn is not None:
        with torch.cuda.stream(stream):
            prepare_fn()
        # Preparation is intentionally outside capture.  Completing it here
        # gives the captured stateful operation a stable initial input.
        synchronize = getattr(stream, "synchronize", None)
        if synchronize is not None:
            synchronize()
        else:
            torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    captured_outputs = fn()
                    if output_holder is not None:
                        output_holder[:] = [captured_outputs]
    torch.cuda.synchronize()
    if any(
        "graph is empty" in str(item.message).lower()
        for item in captured_warnings
    ):
        raise _EmptyGraphCapture("PyTorch reported an empty CUDA/HIP Graph")
    return graph


def _graph_replay_samples(
    graph: Any,
    stream: Any,
    samples: int,
    calls_per_replay: int,
    prepare_fn: Callable[[], Any] | None = None,
) -> list[float]:
    values: list[float] = []
    for _ in range(samples):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # CUDAGraph.replay() launches on the current stream.  Keep the replay
        # and both timing events under the captured side-stream context so the
        # events cannot accidentally bracket an idle stream.
        with torch.cuda.stream(stream):
            if prepare_fn is not None:
                # Enqueue preparation before the start event. Stream ordering
                # makes replay consume the fresh state without charging the
                # preparation work to the kernel sample.
                prepare_fn()
            start_event.record(stream)
            graph.replay()
            end_event.record(stream)
        _wait_for_event(end_event)
        values.append(
            _event_elapsed_ms(start_event, end_event) / float(calls_per_replay)
        )
    return values


def _fallback_metadata(
    metadata: dict[str, Any],
    repetition: int,
    reason: str,
) -> dict[str, Any]:
    del repetition
    metadata.update(
        {
            "benchmark_method": "cuda_event_fallback",
            "benchmark_effective_repeats": 1,
            "benchmark_fallback_reason": reason,
        }
    )
    return metadata


def _event_fallback(
    fn: Callable[[], Any],
    repetition: int,
    metadata: dict[str, Any],
    reason: str,
    prepare_fn: Callable[[], Any] | None = None,
    timed_run: Any | None = None,
) -> tuple[list[float], dict[str, Any]]:
    if timed_run is not None:
        raise RuntimeError(
            f"{reason}; timed_run requires an observable CUDA-graph replay "
            "and cannot validate a separate post-timing invocation"
        )
    values = benchmark_cuda_event_samples(fn, repetition, prepare_fn=prepare_fn)
    return values, _fallback_metadata(metadata, repetition, reason)


def benchmark_cuda_graph_or_events_samples(
    fn: Callable[[], Any],
    warmup: int = 10,
    repetition: int = 100,
    target_ms: float = 1.0,
    n_retries: int = 5,
    estimate_reps: int = 5,
    max_graph_repeats: int = 1000,
    use_cuda_graph: bool = True,
    fallback_reason: str | None = None,
    prepare_fn: Callable[[], Any] | None = None,
    timed_run: Any | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Benchmark ``fn`` and return per-call millisecond samples plus metadata.

    The callable is warmed up before timing.  When enabled, a small graph is
    captured first to estimate per-call device time, then a second graph batches
    enough calls to target approximately ``target_ms`` per replay.  Capture
    failure, invalid replay timing, and an effectively empty capture explicitly
    fall back to eager GPU-event timing.

    Stateful in-place kernels may pass ``prepare_fn`` to restore their input
    before every warmup and sample. Preparation is stream-ordered but excluded
    from timing; the captured graph then contains one logical ``fn`` invocation
    per replay so no invocation consumes already-mutated state.

    ``n_retries`` remains accepted for compatibility with older task runners;
    ``repetition`` is the authoritative number of reported samples.
    """

    del n_retries
    if timed_run is not None and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; timed_run requires an observable CUDA-graph "
            "replay and cannot validate a separate post-timing invocation"
        )
    _require_gpu_timing()

    if os.environ.get(_FORCE_EVENT_ENV) == "1":
        use_cuda_graph = False
        fallback_reason = "forced_event_baseline"

    warmup = max(0, int(warmup))
    repetition = _positive_int(repetition)
    estimate_reps = _positive_int(estimate_reps)
    max_graph_repeats = _positive_int(max_graph_repeats)
    target_ms = float(target_ms)
    if not math.isfinite(target_ms) or target_ms <= 0.0:
        raise ValueError(f"target_ms must be finite and positive, got {target_ms!r}")

    for _ in range(warmup):
        if prepare_fn is not None:
            prepare_fn()
        fn()
    torch.cuda.synchronize()

    metadata: dict[str, Any] = {
        "benchmark_target_ms": target_ms,
        "benchmark_samples": repetition,
        "benchmark_max_repeats": max_graph_repeats,
        "benchmark_warmup": warmup,
    }

    if not use_cuda_graph:
        if timed_run is not None:
            raise RuntimeError(
                "CUDA-graph timing is disabled; timed_run requires an observable "
                "CUDA-graph replay and cannot validate a separate post-timing "
                "invocation"
            )
        return _event_fallback(
            fn,
            repetition,
            metadata,
            fallback_reason or "cuda_graph_disabled",
            prepare_fn=prepare_fn,
            timed_run=timed_run,
        )

    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        # A stateful workload prepared between samples can only capture one
        # logical invocation per replay; batching multiple invocations would
        # consume already-mutated state inside the graph.
        capture_estimate_reps = 1 if prepare_fn is not None else estimate_reps
        estimate_graph = _capture_graph(
            fn,
            capture_estimate_reps,
            stream,
            prepare_fn=prepare_fn,
        )
        _graph_replay_samples(
            estimate_graph,
            stream,
            samples=1,
            calls_per_replay=capture_estimate_reps,
            prepare_fn=prepare_fn,
        )
        estimate_values = _graph_replay_samples(
            estimate_graph,
            stream,
            samples=1,
            calls_per_replay=capture_estimate_reps,
            prepare_fn=prepare_fn,
        )
        estimate_ms = estimate_values[0]
        if estimate_ms < _EMPTY_GRAPH_FLOOR_MS:
            return _event_fallback(
                fn,
                repetition,
                metadata,
                fallback_reason or "empty_cuda_graph_capture",
                prepare_fn=prepare_fn,
                timed_run=timed_run,
            )

        graph_repeats = (
            1
            if prepare_fn is not None
            else min(
                max_graph_repeats,
                max(1, int(target_ms / estimate_ms)),
            )
        )
        captured_outputs: list[Any] | None = [] if timed_run is not None else None
        capture_kwargs: dict[str, Any] = {"prepare_fn": prepare_fn}
        if captured_outputs is not None:
            capture_kwargs["output_holder"] = captured_outputs
        graph = _capture_graph(fn, graph_repeats, stream, **capture_kwargs)
        # Prime the final graph executable outside the reported sample set.
        # Some backends perform lazy graph-exec setup on the first replay; if
        # the start event has already reached the head of the stream, that host
        # setup delay would otherwise contaminate the first GPU sample.
        _graph_replay_samples(
            graph,
            stream,
            samples=1,
            calls_per_replay=graph_repeats,
            prepare_fn=prepare_fn,
        )
        values = _graph_replay_samples(
            graph,
            stream,
            samples=repetition,
            calls_per_replay=graph_repeats,
            prepare_fn=prepare_fn,
        )

        if not values or any(
            not math.isfinite(value) or value < _EMPTY_GRAPH_FLOOR_MS
            for value in values
        ):
            return _event_fallback(
                fn,
                repetition,
                metadata,
                fallback_reason or "empty_or_invalid_cuda_graph_replay",
                prepare_fn=prepare_fn,
                timed_run=timed_run,
            )

        metadata.update(
            {
                "benchmark_method": "cuda_graph",
                "benchmark_effective_repeats": graph_repeats,
                "benchmark_estimate_ms": estimate_ms,
            }
        )
        if timed_run is not None:
            captured_output = captured_outputs[0] if captured_outputs else None

            def _replay_once() -> Any:
                # Callers may perturb inputs or poison outputs on the current
                # stream before requesting validation. Order the capture stream
                # after that work, then replay the exact graph that was timed.
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    if prepare_fn is not None:
                        prepare_fn()
                    graph.replay()
                torch.cuda.synchronize()
                return captured_output

            timed_run._bind(_replay_once, captured_output)
        return values, metadata
    except _EmptyGraphCapture:
        return _event_fallback(
            fn,
            repetition,
            metadata,
            fallback_reason or "empty_cuda_graph_capture",
            prepare_fn=prepare_fn,
            timed_run=timed_run,
        )
    except Exception as exc:
        # A failed capture can leave queued work behind.  Best-effort isolation
        # keeps it out of the eager fallback samples; if the backend is no longer
        # usable, the event fallback itself raises instead of fabricating a CPU
        # measurement.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        if timed_run is not None:
            raise RuntimeError(
                "CUDA-graph capture failed; timed_run cannot validate the "
                "separate CUDA-event fallback invocation"
            ) from exc
        for _ in range(min(3, max(1, warmup))):
            if prepare_fn is not None:
                prepare_fn()
            fn()
        torch.cuda.synchronize()
        detail = str(exc).replace("\n", " ")[:160]
        return _event_fallback(
            fn,
            repetition,
            metadata,
            f"cuda_graph_failed: {type(exc).__name__}: {detail}",
            prepare_fn=prepare_fn,
        )


def benchmark_cuda_graph_or_events(
    fn: Callable[[], Any],
    warmup: int = 10,
    repetition: int = 100,
    target_ms: float = 1.0,
    n_retries: int = 5,
    estimate_reps: int = 5,
    max_graph_repeats: int = 1000,
    use_cuda_graph: bool = True,
    fallback_reason: str | None = None,
    prepare_fn: Callable[[], Any] | None = None,
    timed_run: Any | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return mean device milliseconds per ``fn`` invocation plus metadata."""

    samples, metadata = benchmark_cuda_graph_or_events_samples(
        fn,
        warmup=warmup,
        repetition=repetition,
        target_ms=target_ms,
        n_retries=n_retries,
        estimate_reps=estimate_reps,
        max_graph_repeats=max_graph_repeats,
        use_cuda_graph=use_cuda_graph,
        fallback_reason=fallback_reason,
        prepare_fn=prepare_fn,
        timed_run=timed_run,
    )
    return sum(samples) / len(samples), metadata


__all__ = [
    "benchmark_cuda_event_samples",
    "benchmark_cuda_graph_or_events",
    "benchmark_cuda_graph_or_events_samples",
    "hip_source_graph_capture_policy",
]
