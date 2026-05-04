# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Test case handling for evaluator: data structures, parsing, matching, and speedup calculation.
"""
import json
import re
import logging
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class TestCaseResult:
    """Represents a single test case performance result."""
    test_case_id: str  # Unique identifier (e.g., "shape_0", "test_1")
    shape: Optional[List[Any]] = None  # Shape/size parameters (e.g., [256, 256, 256])
    execution_time_ms: float = 0.0
    metadata: Optional[Dict[str, Any]] = None  # Additional info (dtype, etc.)


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert input to float; return None for invalid values."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_time_from_dict(
    data: Dict[str, Any],
    is_baseline: bool = False,
    task_type: Optional[str] = None
) -> Tuple[float, Optional[str]]:
    """
    Extract execution time from a dictionary, handling various formats.
    
    Returns:
        Tuple of (time_ms, matched_key) or (0.0, None) if not found
    """
    # Special handling for torch2hip tasks
    if task_type == 'torch2hip':
        if is_baseline and 'ori_time' in data:
            time_val = _safe_float(data.get('ori_time'))
            if time_val is not None:
                return time_val, 'ori_time'  # Task runners already write milliseconds
        elif not is_baseline and 'opt_time' in data:
            time_val = _safe_float(data.get('opt_time'))
            if time_val is not None:
                return time_val, 'opt_time'  # Task runners already write milliseconds
    
    # Standard time keys (in order of preference)
    time_keys = ['execution_time_ms', 'execution_time', 'time_ms', 'time']
    for key in time_keys:
        if key in data:
            time_val = _safe_float(data.get(key))
            if time_val is None:
                continue
            if key.endswith('_ms') or key == 'time_ms':
                return time_val, key
            elif time_val < 1000.0:  # Likely already in ms
                return time_val, key
            else:  # Likely in seconds, convert to ms
                return time_val * 1000.0, key
    
    # Pytest benchmark format: nested timing_ms structure
    if 'timing_ms' in data:
        timing = data['timing_ms']
        if isinstance(timing, dict):
            # Prefer mean, fallback to median, then min
            if 'mean' in timing:
                time_val = _safe_float(timing.get('mean'))
                if time_val is not None:
                    return time_val, 'timing_ms.mean'
            elif 'median' in timing:
                time_val = _safe_float(timing.get('median'))
                if time_val is not None:
                    return time_val, 'timing_ms.median'
            elif 'min' in timing:
                time_val = _safe_float(timing.get('min'))
                if time_val is not None:
                    return time_val, 'timing_ms.min'

    # Throughput metrics: GB/s — higher is better.
    # calculate_average_speedup uses baseline_time / optimized_time with "lower is better".
    # Store cost = 1/throughput so speedup = (1/bps_base)/(1/bps_opt) = bps_opt/bps_base.
    if 'bytes_per_second_gs' in data:
        bps = _safe_float(data.get('bytes_per_second_gs'))
        if bps is not None and bps > 0:
            return 1.0 / bps, 'bytes_per_second_gs (inverse for speedup)'

    return 0.0, None


def _build_metadata_from_case(
    case: Dict[str, Any],
    exclude_keys: List[str]
) -> Dict[str, Any]:
    """Build metadata dict excluding specified keys."""
    metadata = {k: v for k, v in case.items() if k not in exclude_keys}
    
    # Always include params if present
    if 'params' in case:
        metadata['params'] = case['params']
    
    return metadata


def _parse_single_case_from_dict(
    case: Dict[str, Any],
    default_test_id: str,
    is_baseline: bool = False,
    task_type: Optional[str] = None
) -> Optional[TestCaseResult]:
    """Parse a single test case from a dictionary."""
    test_id = case.get('test_case_id', default_test_id)
    shape = case.get('shape') or case.get('shapes')
    
    time_ms, matched_key = _extract_time_from_dict(case, is_baseline, task_type)
    
    # Allow negative values (e.g., -1.0) as valid error indicators from task runners
    # Reject only if time is missing (0.0) and no key was matched
    if time_ms == 0.0 and matched_key is None:
        return None
    
    # Build metadata
    exclude_keys = ['test_case_id', 'shape', 'shapes', 'execution_time_ms',
                   'execution_time', 'time_ms', 'time', 'timing_ms', 'params',
                   'ori_time', 'opt_time', 'bytes_per_second_gs']
    metadata = _build_metadata_from_case(case, exclude_keys)
    
    # For torch2hip, include both ori_time and opt_time in metadata for reference
    if task_type == 'torch2hip':
        if 'ori_time' in case:
            metadata['ori_time'] = case['ori_time']
        if 'opt_time' in case:
            metadata['opt_time'] = case['opt_time']
        if 'speedup' in case:
            metadata['speedup'] = case['speedup']
    if 'bytes_per_second_gs' in case:
        metadata['bytes_per_second_gs'] = case['bytes_per_second_gs']

    return TestCaseResult(
        test_case_id=test_id,
        shape=shape,
        execution_time_ms=time_ms,
        metadata=metadata
    )


def parse_test_cases_from_json(
    report_file: Path,
    logger: Optional[logging.Logger] = None,
    is_baseline: bool = False,
    task_type: Optional[str] = None
) -> List[TestCaseResult]:
    """
    Parse multiple test case results from JSON report file.
    
    Handles:
    - Array of test cases (hip2hip, pytest benchmark format)
    - Single object with standard keys (triton2triton/vllm)
    - Single object with torch2hip keys (ori_time/opt_time)
    - Single object with custom _ms keys
    
    Args:
        report_file: Path to JSON report file
        logger: Optional logger
        is_baseline: If True, use ori_time for torch2hip; if False, use opt_time
        task_type: Task type (e.g., 'torch2hip', 'hip2hip', 'triton2triton')
        
    Returns:
        List of TestCaseResult objects
    """
    log = logger or logging.getLogger(__name__)
    test_cases = []
    
    try:
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        # Format 1: Array of test cases (hip2hip, pytest benchmark)
        if isinstance(report, list):
            for idx, case in enumerate(report):
                try:
                    if not isinstance(case, dict):
                        log.warning(f"Skipping non-dict test case at index {idx} in {report_file}")
                        continue
                    test_case = _parse_single_case_from_dict(
                        case, f"test_case_{idx}", is_baseline, task_type
                    )
                    if test_case:
                        test_cases.append(test_case)
                except Exception as e:
                    log.warning(f"Skipping invalid test case at index {idx} in {report_file}: {e}")
        
        # Format 2: Object with 'test_cases' key
        elif 'test_cases' in report:
            for idx, case in enumerate(report['test_cases']):
                try:
                    if not isinstance(case, dict):
                        log.warning(f"Skipping non-dict test case at index {idx} in {report_file}")
                        continue
                    test_case = _parse_single_case_from_dict(
                        case, f"test_case_{idx}", is_baseline, task_type
                    )
                    if test_case:
                        test_cases.append(test_case)
                except Exception as e:
                    log.warning(f"Skipping invalid test case at index {idx} in {report_file}: {e}")
        
        # Format 3: Single object
        else:
            # Try standard parsing first
            test_case = _parse_single_case_from_dict(
                report, "test_case_0", is_baseline, task_type
            )
            
            if test_case:
                test_cases.append(test_case)
            else:
                # Fallback: Look for any keys ending in '_ms' (custom format)
                ms_keys = [k for k in report.keys() if k.endswith('_ms')]
                if ms_keys:
                    for idx, ms_key in enumerate(sorted(ms_keys)):
                        time_val = _safe_float(report.get(ms_key))
                        if time_val is None:
                            log.warning(f"Skipping invalid timing value {ms_key}={report.get(ms_key)!r} in {report_file}")
                            continue
                        # Build metadata excluding timing
                        exclude_keys = ['shape', 'shapes'] + [k for k in report.keys() if k.endswith('_ms')]
                        metadata = _build_metadata_from_case(report, exclude_keys)
                        
                        # Include other _ms keys in metadata for reference
                        other_timings = {k: report[k] for k in ms_keys if k != ms_key}
                        if other_timings:
                            metadata['other_timings'] = other_timings
                        
                        # Create descriptive test_case_id
                        key_name = ms_key.replace('_ms', '')
                        test_case_id = f"{key_name}_{idx}" if len(ms_keys) > 1 else key_name
                        
                        test_cases.append(TestCaseResult(
                            test_case_id=test_case_id,
                            shape=report.get('shape') or report.get('shapes'),
                            execution_time_ms=time_val,
                            metadata=metadata
                        ))
        
        log.info(f"Parsed {len(test_cases)} test case(s) from {report_file}")
        
    except Exception as e:
        log.warning(f"Failed to parse test cases from {report_file}: {e}")
    
    return test_cases


def parse_test_cases_from_stdout(
    output: str,
    logger: Optional[logging.Logger] = None
) -> List[TestCaseResult]:
    """
    Parse multiple test case results from stdout.
    
    Looks for patterns like:
    - "Test case 0: 123.45 ms"
    - "Shape [256, 256, 256]: 123.45 ms"
    - Multiple "Performance: X ms" lines
    
    Args:
        output: Command output text
        logger: Optional logger
        
    Returns:
        List of TestCaseResult objects
    """
    log = logger or logging.getLogger(__name__)
    test_cases = []
    
    # Pattern 1: "Test case N: X ms" or "TestCase N: X ms"
    pattern1 = r'(?:Test\s+case|TestCase)\s+(\d+)[:\s]+([0-9.]+)\s*ms'
    matches1 = re.findall(pattern1, output, re.IGNORECASE)
    for match in matches1:
        test_id, time_str = match
        test_cases.append(TestCaseResult(
            test_case_id=f"test_case_{test_id}",
            execution_time_ms=float(time_str)
        ))
    
    # Pattern 2: "Shape [X, Y, Z]: X ms" or "shape: [X, Y, Z], time: X ms"
    pattern2 = r'(?:Shape|shape)[:\s]+\[([0-9,\s]+)\][:\s]+([0-9.]+)\s*ms'
    matches2 = re.findall(pattern2, output, re.IGNORECASE)
    for idx, match in enumerate(matches2):
        shape_str, time_str = match
        shape = [int(x.strip()) for x in shape_str.split(',')]
        test_cases.append(TestCaseResult(
            test_case_id=f"shape_{idx}",
            shape=shape,
            execution_time_ms=float(time_str)
        ))
    
    # Pattern 3: Multiple "Performance: X ms" lines (if no other pattern matched)
    if not test_cases:
        pattern3 = r'Performance:\s*([0-9.]+)\s*ms'
        matches3 = re.findall(pattern3, output, re.IGNORECASE)
        for idx, time_str in enumerate(matches3):
            test_cases.append(TestCaseResult(
                test_case_id=f"perf_{idx}",
                execution_time_ms=float(time_str)
            ))
    
    log.info(f"Parsed {len(test_cases)} test case(s) from stdout")
    return test_cases


def match_test_cases(
    baseline_cases: List[TestCaseResult],
    optimized_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None
) -> List[Tuple[TestCaseResult, TestCaseResult]]:
    """
    Match test cases between baseline and optimized results.
    
    Matching strategy:
    1. Match by test_case_id (exact match)
    2. Match by params (excluding distinguishing conditions)
    3. Match by shape (if available)
    4. Match by index (if same number of cases)
    
    Args:
        baseline_cases: Baseline test case results
        optimized_cases: Optimized test case results
        logger: Optional logger
        
    Returns:
        List of (baseline_case, optimized_case) tuples
    """
    log = logger or logging.getLogger(__name__)
    matched = []
    used_optimized = set()
    
    # Strategy 1: Match by test_case_id
    baseline_by_id = {case.test_case_id: case for case in baseline_cases}
    for opt_case in optimized_cases:
        if opt_case.test_case_id in baseline_by_id:
            matched.append((baseline_by_id[opt_case.test_case_id], opt_case))
            used_optimized.add(id(opt_case))
    
    # Strategy 2: Match by params (excluding distinguishing conditions)
    remaining_baseline = [b for b in baseline_cases if b not in [m[0] for m in matched]]
    remaining_optimized = [o for o in optimized_cases if id(o) not in used_optimized]
    
    # Distinguishing param keys that should be ignored for matching
    distinguishing_keys = {'mode', 'dtype', 'query_type', 'input_type', 'layout', 
                          'output_mode', 'box_type', 'pool_mode'}
    
    def get_matching_params(params_dict):
        """Extract params for matching (excluding distinguishing conditions)."""
        if not params_dict or not isinstance(params_dict, dict):
            return None
        matching = {k: v for k, v in params_dict.items() if k not in distinguishing_keys}
        return tuple(sorted(matching.items())) if matching else None
    
    for base_case in remaining_baseline:
        base_params = get_matching_params(base_case.metadata.get('params') if base_case.metadata else None)
        if base_params:
            for opt_case in remaining_optimized:
                opt_params = get_matching_params(opt_case.metadata.get('params') if opt_case.metadata else None)
                if opt_params and base_params == opt_params:
                    matched.append((base_case, opt_case))
                    used_optimized.add(id(opt_case))
                    break
    
    # Strategy 3: Fallback to shape matching if params not available
    remaining_baseline = [b for b in baseline_cases if b not in [m[0] for m in matched]]
    remaining_optimized = [o for o in optimized_cases if id(o) not in used_optimized]
    
    for base_case in remaining_baseline:
        if base_case.shape:
            for opt_case in remaining_optimized:
                if opt_case.shape and base_case.shape == opt_case.shape:
                    matched.append((base_case, opt_case))
                    used_optimized.add(id(opt_case))
                    break
    
    # Strategy 4: Match by index (for remaining cases)
    remaining_baseline = [b for b in baseline_cases if b not in [m[0] for m in matched]]
    remaining_optimized = [o for o in optimized_cases if id(o) not in used_optimized]
    
    min_len = min(len(remaining_baseline), len(remaining_optimized))
    for i in range(min_len):
        matched.append((remaining_baseline[i], remaining_optimized[i]))
    
    log.info(f"Matched {len(matched)} test case(s) between baseline and optimized")
    return matched


def calculate_average_speedup(
    baseline_cases: List[TestCaseResult],
    optimized_cases: List[TestCaseResult],
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Calculate average speedup across all matched test cases.
    
    Args:
        baseline_cases: Baseline test case results
        optimized_cases: Optimized test case results
        logger: Optional logger
        
    Returns:
        Average speedup ratio (baseline_time / optimized_time), or 0.0 if no valid matches
    """
    log = logger or logging.getLogger(__name__)
    
    matched = match_test_cases(baseline_cases, optimized_cases, logger)
    
    if not matched:
        log.warning("No test cases matched, cannot calculate speedup")
        return 0.0
    
    speedups = []
    for base_case, opt_case in matched:
        if base_case.execution_time_ms > 0 and opt_case.execution_time_ms > 0:
            speedup = base_case.execution_time_ms / opt_case.execution_time_ms
            speedups.append(speedup)
            log.debug(f"Test case {base_case.test_case_id}: {base_case.execution_time_ms:.4f} ms -> {opt_case.execution_time_ms:.4f} ms (speedup: {speedup:.2f}x)")
        else:
            log.warning(f"Invalid execution times for test case {base_case.test_case_id}: baseline={base_case.execution_time_ms}, optimized={opt_case.execution_time_ms}")
    
    if not speedups:
        log.warning("No valid speedups calculated")
        return 0.0
    
    avg_speedup = sum(speedups) / len(speedups)
    log.info(f"Average speedup across {len(speedups)} test case(s): {avg_speedup:.2f}x")
    return avg_speedup


def save_performance_results(
    test_cases: List[TestCaseResult],
    workspace: Path,
    filename: str,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Save test case results to YAML file.
    
    Only saves essential fields: test_case_id, execution_time_ms, shape (if available), and params (if available).
    All other metadata is excluded to keep the file clean.
    
    Args:
        test_cases: List of test case results
        workspace: Workspace directory
        filename: Filename (e.g., 'baseline_perf.yaml')
        logger: Optional logger
    """
    log = logger or logging.getLogger(__name__)
    
    results = {
        'test_cases': []
    }
    
    for case in test_cases:
        case_dict = {
            'test_case_id': case.test_case_id,
            'execution_time_ms': case.execution_time_ms
        }
        if case.shape:
            case_dict['shape'] = case.shape
        # Only include params from metadata, exclude everything else
        if case.metadata and 'params' in case.metadata:
            case_dict['params'] = case.metadata['params']
        results['test_cases'].append(case_dict)
    
    output_file = workspace / filename
    with open(output_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    log.info(f"Saved {len(test_cases)} test case(s) to {output_file}")


def load_performance_results(
    workspace: Path,
    filename: str,
    logger: Optional[logging.Logger] = None
) -> List[TestCaseResult]:
    """
    Load test case results from YAML file.
    
    Args:
        workspace: Workspace directory
        filename: Filename (e.g., 'baseline_perf.yaml')
        logger: Optional logger
        
    Returns:
        List of TestCaseResult objects
    """
    log = logger or logging.getLogger(__name__)
    input_file = workspace / filename
    
    if not input_file.exists():
        log.warning(f"Performance results file not found: {input_file}")
        return []
    
    try:
        with open(input_file, 'r') as f:
            data = yaml.safe_load(f)
        
        test_cases = []
        for case_dict in data.get('test_cases', []):
            test_cases.append(TestCaseResult(
                test_case_id=case_dict.get('test_case_id', 'unknown'),
                shape=case_dict.get('shape'),
                execution_time_ms=float(case_dict.get('execution_time_ms', 0.0)),
                metadata={k: v for k, v in case_dict.items() if k not in ['test_case_id', 'shape', 'execution_time_ms']}
            ))
        
        log.info(f"Loaded {len(test_cases)} test case(s) from {input_file}")
        return test_cases
        
    except Exception as e:
        log.error(f"Failed to load performance results from {input_file}: {e}")
        return []
