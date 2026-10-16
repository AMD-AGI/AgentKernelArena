# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Protected dense-input/full-output checks, including exact graph replay."""
import json
import re
import subprocess

CONTROL_SHAPES = [(16, 32, 48), (32, 64, 16)]
CONTROL_IDS = ['control_dense_rectangular_16_32_48', 'control_dense_rectangular_32_64_16']


def self_test(harness):
    if not harness.TEST_SHAPES or any(any(type(x) is not int or x <= 0 for x in shape)
                                      for shape in harness.TEST_SHAPES):
        raise ValueError("Native host workload must contain positive integer dimensions")


def parse_validation(output, *, mode):
    def unique(prefix):
        lines = [line[len(prefix):] for line in output.splitlines() if line.startswith(prefix)]
        if len(lines) != 1:
            raise ValueError("Missing or duplicate native matrix input/validation record")
        return json.loads(lines[0])
    inputs = unique('AKA_MATRIX_INPUTS ')
    validation = unique('AKA_MATRIX_VALIDATION ')
    expected_keys = {'schema_version', 'generator', 'fingerprint_kind', 'initial_seed', 'replay_seed',
                     'initial_A', 'initial_B', 'replay_A', 'replay_B'}
    if (set(inputs) != expected_keys or inputs['schema_version'] != 1
            or inputs['generator'] != 'splitmix64_dyadic_dense_v1'
            or inputs['fingerprint_kind'] != 'fnv1a64_le_fp32_noncryptographic'):
        raise ValueError("Unknown native matrix input distribution")
    for key in ('initial_seed', 'replay_seed'):
        text = inputs[key]
        if not isinstance(text, str) or not re.fullmatch(r'0|[1-9][0-9]{0,19}', text) or int(text) >= 2**64:
            raise ValueError("Invalid reproducibility seed")
    for key in ('initial_A', 'initial_B', 'replay_A', 'replay_B'):
        if not isinstance(inputs[key], str) or not re.fullmatch(r'[0-9a-f]{16}', inputs[key]):
            raise ValueError("Invalid noncryptographic input fingerprint")
    if (inputs['initial_seed'] == inputs['replay_seed'] or inputs['initial_A'] == inputs['replay_A']
            or inputs['initial_B'] == inputs['replay_B']):
        raise ValueError("Independent replay did not change both inputs")
    if (validation.get('inputs') != inputs or validation.get('mode') != mode
            or validation.get('initial_output_correctness') != 'PASS'
            or validation.get('changed_input_replay_correctness') != 'PASS'
            or validation.get('readonly_initial') != 'PASS'
            or validation.get('readonly_replay') != 'PASS'
            or validation.get('restored_initial_inputs') != 'PASS'
            or validation.get('output_coverage') != 'all_output_elements'
            or validation.get('replay_path') not in ('captured_graph', 'eager_callable')):
        raise ValueError("Incomplete actual matrix input/output/replay validation")
    return validation


def check_additional_paths(harness):
    evidence = []
    shapes = [*harness.TEST_SHAPES, *CONTROL_SHAPES]
    ids = [*(f'shape_{index}' for index in range(len(harness.TEST_SHAPES))), *CONTROL_IDS]
    for identity, (rows, inner, cols) in zip(ids, shapes):
        command = [harness.BENCH_BINARY, "--A_rows", str(rows), "--A_cols", str(inner),
                   "--B_cols", str(cols), "--check-only", "1"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode or "Full reference validation passed" not in completed.stdout:
            raise ValueError(f"Full independent dense product check failed for {(rows, inner, cols)}: "
                             f"{completed.stdout}\n{completed.stderr}")
        evidence.append({'test_case_id': identity, 'validation': parse_validation(completed.stdout, mode='check_only')})
    return evidence
