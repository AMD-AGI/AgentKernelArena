# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Codegen injection for held-out shape replacement.

Handles three patterns:
1. triton2triton/vllm: replace TEST_SHAPES = [...] block in task_runner.py
2. hip2hip/torch2hip gpumode: replace def get_inputs(): ... function in
   both pytorch_code_module/*.py and pytorch_code_functional/*_func.py
3. raw_replace: exact string find-and-replace, used for triton2triton/rocmbench
   @pytest.mark.parametrize decorators and any other ad-hoc patterns
"""
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _find_bracket_end(text: str, start: int) -> int:
    """Find the index of the closing bracket matching the opening bracket at `start`."""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def replace_test_shapes(source: str, replacement_code: str) -> str:
    """
    Replace a TEST_SHAPES = [...] block in Python source code.

    Finds the assignment via regex, then uses bracket balancing to locate
    the full extent of the list literal (handles multiline lists).

    Args:
        source: Original file contents
        replacement_code: Full replacement including 'TEST_SHAPES = [...]'

    Returns:
        Modified source with TEST_SHAPES block replaced

    Raises:
        ValueError: If TEST_SHAPES assignment not found or brackets unbalanced
    """
    pattern = re.compile(r'^([ \t]*)TEST_SHAPES\s*=\s*\[', re.MULTILINE)
    match = pattern.search(source)
    if not match:
        raise ValueError("Could not find 'TEST_SHAPES = [' in source")

    bracket_start = source.index('[', match.start())
    bracket_end = _find_bracket_end(source, bracket_start)
    if bracket_end == -1:
        raise ValueError("Unbalanced brackets in TEST_SHAPES definition")

    # Consume trailing newline if present
    end = bracket_end + 1
    if end < len(source) and source[end] == '\n':
        end += 1

    indent = match.group(1)
    indented_replacement = '\n'.join(
        (indent + line) if line.strip() else line
        for line in replacement_code.strip().splitlines()
    ) + '\n'

    return source[:match.start()] + indented_replacement + source[end:]


def _find_function_end(source: str, func_start: int) -> int:
    """
    Find the end of a top-level function definition starting at `func_start`.

    Uses indentation: the function body consists of all lines after the def
    line that are either blank or indented deeper than the def line.
    """
    lines = source[func_start:].split('\n')
    if not lines:
        return func_start

    def_line = lines[0]
    base_indent = len(def_line) - len(def_line.lstrip())

    consumed = len(lines[0]) + 1  # +1 for the newline
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == '' or stripped.startswith('#'):
            consumed += len(line) + 1
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= base_indent:
            break
        consumed += len(line) + 1

    return func_start + consumed


def replace_get_inputs(source: str, replacement_code: str) -> str:
    """
    Replace a def get_inputs(): ... function in Python source code.

    Detects the function via regex, finds its end via indentation analysis,
    and replaces the entire function body.

    Args:
        source: Original file contents
        replacement_code: Full replacement including 'def get_inputs():'

    Returns:
        Modified source with get_inputs replaced

    Raises:
        ValueError: If get_inputs function not found
    """
    pattern = re.compile(r'^([ \t]*)def get_inputs\s*\(', re.MULTILINE)
    match = pattern.search(source)
    if not match:
        raise ValueError("Could not find 'def get_inputs(' in source")

    func_end = _find_function_end(source, match.start())
    indent = match.group(1)

    indented_replacement = '\n'.join(
        (indent + line) if line.strip() else line
        for line in replacement_code.strip().splitlines()
    ) + '\n'

    return source[:match.start()] + indented_replacement + source[func_end:]


def apply_injection(
    workspace: Path,
    injection_spec: dict,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Apply a single injection (one file replacement) from a held_out_shapes.yaml entry.

    Args:
        workspace: Task workspace directory (the copied run workspace)
        injection_spec: Dict with keys 'file', 'find_marker', 'replacement_code'
        logger: Optional logger

    Returns:
        True if injection succeeded, False otherwise
    """
    log = logger or logging.getLogger(__name__)

    target_file = workspace / injection_spec['file']
    find_marker = injection_spec['find_marker']
    replacement_code = injection_spec['replacement_code']

    if not target_file.exists():
        log.error(f"Injection target file not found: {target_file}")
        return False

    source = target_file.read_text()

    try:
        if find_marker == "TEST_SHAPES":
            modified = replace_test_shapes(source, replacement_code)
        elif find_marker.startswith("def get_inputs"):
            modified = replace_get_inputs(source, replacement_code)
        elif find_marker == "raw_replace":
            old_code = injection_spec.get('old_code')
            if not old_code:
                log.error(f"raw_replace injection missing 'old_code' in spec for {target_file}")
                return False
            if old_code not in source:
                log.error(
                    f"raw_replace: old_code not found in {target_file}. "
                    f"First 80 chars of old_code: {old_code[:80]!r}"
                )
                return False
            modified = source.replace(old_code, replacement_code, 1)
        else:
            log.error(f"Unknown find_marker type: {find_marker}")
            return False
    except ValueError as e:
        log.error(f"Injection failed for {target_file}: {e}")
        return False

    if modified == source:
        log.warning(f"Injection produced no change in {target_file}")
        return False

    target_file.write_text(modified)
    log.info(f"Injected held-out shapes into {target_file}")
    return True


def apply_all_injections(
    workspace: Path,
    heldout_config: dict,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Apply all injections from a held_out_shapes.yaml config to a workspace.

    For hip2hip/torch2hip tasks this applies the same replacement to both
    the modular and functional Python files.

    Args:
        workspace: Task workspace directory
        heldout_config: Parsed held_out_shapes.yaml dict
        logger: Optional logger

    Returns:
        True if all injections succeeded, False if any failed
    """
    log = logger or logging.getLogger(__name__)
    injections = heldout_config.get('injections', [])

    if not injections:
        log.error("No injections defined in held-out config")
        return False

    all_ok = True
    for spec in injections:
        if not apply_injection(workspace, spec, log):
            all_ok = False

    return all_ok
