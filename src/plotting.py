# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Plotting utilities for visualizing baseline vs optimized performance.
"""
import logging
from pathlib import Path
from typing import Optional, List, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from .testcases import TestCaseResult, load_performance_results, match_test_cases


def format_value(value: Any) -> str:
    """
    Format a value for display (exact values, only shorten float types).
    
    Args:
        value: Value to format
        
    Returns:
        Formatted string (exact value, only fp32/fp16 shortening)
    """
    if isinstance(value, str):
        # Only shorten float types: float16 -> fp16, float32 -> fp32
        if value.startswith('float'):
            return value.replace('float', 'fp')
        return value
    else:
        # Return exact value for numbers
        return str(value)


def create_x_axis_label(case: TestCaseResult) -> str:
    """
    Create a simple x-axis label showing only values (no parameter names).
    
    Args:
        case: TestCaseResult object
        
    Returns:
        Simple label with just values (e.g., "98.4K, 1024, f16" or "Case 0")
    """
    values = []
    
    # Extract parameter values (general - works with any parameters)
    if case.metadata and 'params' in case.metadata:
        params = case.metadata['params']
        if isinstance(params, dict):
            # Sort keys for consistent ordering, but show all values
            for key in sorted(params.keys()):
                value = params[key]
                values.append(format_value(value))
    
    # Fallback to shape if no params (for backward compatibility)
    if not values and case.shape:
        for dim in case.shape:
            values.append(format_value(dim))
    
    # Fallback to test case number
    if not values:
        if case.test_case_id.startswith('test_case_'):
            num = case.test_case_id.replace('test_case_', '')
            return f"Case {num}"
        else:
            return case.test_case_id
    
    # Join values with comma
    return ", ".join(values)


def create_test_case_details(case: TestCaseResult) -> str:
    """
    Create a string with parameter names only (p1: name1, p2: name2, ...).
    
    Args:
        case: TestCaseResult object
        
    Returns:
        String with parameter names like "p1: BLOCK_SIZE_RUNTIME, p2: SIZE, p3: dtype_str"
    """
    param_names = []
    
    # Extract parameter names only
    if case.metadata and 'params' in case.metadata:
        params = case.metadata['params']
        if isinstance(params, dict):
            # Sort keys for consistent ordering
            sorted_keys = sorted(params.keys())
            for idx, key in enumerate(sorted_keys, start=1):
                param_names.append(f"p{idx}: {key}")
    
    if param_names:
        return ", ".join(param_names)
    else:
        return "No parameters"


def plot_performance_comparison(
    workspace: Path,
    kernel_name: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    save_format: str = 'png'
) -> Optional[Path]:
    """
    Create two separate plots comparing baseline and optimized performance.
    
    Creates:
    1. Execution time comparison plot (performance_execution_time.png)
    2. Speedup comparison plot (performance_speedup.png)
    
    Args:
        workspace: Workspace directory containing baseline_perf.yaml and optimized_perf.yaml
        kernel_name: Optional kernel/task name to include in title
        logger: Optional logger
        save_format: Image format ('png', 'pdf', 'svg')
        
    Returns:
        Path to execution time plot file, or None if plotting failed
    """
    log = logger or logging.getLogger(__name__)
    
    # Load performance results
    baseline_cases = load_performance_results(workspace, "baseline_perf.yaml", logger)
    optimized_cases = load_performance_results(workspace, "optimized_perf.yaml", logger)
    
    if not baseline_cases:
        log.warning("No baseline performance data found, skipping plotting")
        return None
    
    if not optimized_cases:
        log.warning("No optimized performance data found, skipping plotting")
        return None
    
    # Match test cases
    matched = match_test_cases(baseline_cases, optimized_cases, logger)
    
    if not matched:
        log.warning("No matched test cases found, skipping plotting")
        return None
    
    log.info(f"Creating performance comparison plots for {len(matched)} test case(s)")
    
    # Extract data for plotting
    labels = []
    baseline_times = []
    optimized_times = []
    speedups = []
    
    for base_case, opt_case in matched:
        labels.append(create_x_axis_label(base_case))
        baseline_times.append(base_case.execution_time_ms)
        optimized_times.append(opt_case.execution_time_ms)
        speedups.append(base_case.execution_time_ms / opt_case.execution_time_ms 
                       if opt_case.execution_time_ms > 0 else 0.0)
    
    # Get parameter names (same for all test cases)
    test_case_details = [create_test_case_details(matched[0][0])] if matched else []
    
    # Determine layout based on number of test cases
    num_cases = len(matched)
    if num_cases > 10:
        # Many test cases: wider figure, rotated labels, smaller font
        fig_width = max(16, num_cases * 0.8)
        label_rotation = 45
        label_ha = 'right'
        label_fontsize = 8
        bottom_margin = 0.12
    elif num_cases > 5:
        # Moderate number: slightly wider, rotated labels
        fig_width = max(14, num_cases * 0.7)
        label_rotation = 30
        label_ha = 'right'
        label_fontsize = 9
        bottom_margin = 0.08
    else:
        # Few test cases: standard layout
        fig_width = 12
        label_rotation = 0
        label_ha = 'center'
        label_fontsize = 9
        bottom_margin = 0.02
    
    # Create title prefix with kernel name if provided
    title_prefix = ''
    if kernel_name:
        title_prefix = f'{kernel_name} - '
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot 1: Execution time comparison
    fig1 = plt.figure(figsize=(fig_width, 7))
    gs1 = fig1.add_gridspec(2, 1, height_ratios=[1, 0.15], hspace=0.3)
    
    ax1 = fig1.add_subplot(gs1[0, 0])
    bars1 = ax1.bar(x - width/2, baseline_times, width, label='Baseline', color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, optimized_times, width, label='Optimized', color='#A23B72', alpha=0.85, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Test Case', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Execution Time (ms)', fontweight='bold', fontsize=11)
    ax1.set_title(f'{title_prefix}Execution Time Comparison', fontweight='bold', fontsize=13, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=label_rotation, ha=label_ha, fontsize=label_fontsize)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add parameter names below plot
    ax1_param = fig1.add_subplot(gs1[1, 0])
    ax1_param.axis('off')
    if test_case_details:
        param_names_str = test_case_details[0]
        ax1_param.text(0.5, 0.5, f'Parameters: {param_names_str}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, bottom_margin, 1, 0.98])
    plot_file1 = workspace / f'performance_execution_time.{save_format}'
    plt.savefig(plot_file1, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved execution time plot to {plot_file1}")
    
    # Plot 2: Speedup chart
    fig2 = plt.figure(figsize=(fig_width, 7))
    gs2 = fig2.add_gridspec(2, 1, height_ratios=[1, 0.15], hspace=0.3)
    
    ax2 = fig2.add_subplot(gs2[0, 0])
    colors = ['#06A77D' if s >= 1.0 else '#F18F01' for s in speedups]
    bars3 = ax2.bar(x, speedups, width=0.6, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No speedup (1.0x)', zorder=0)
    
    ax2.set_xlabel('Test Case', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Speedup Ratio', fontweight='bold', fontsize=11)
    ax2.set_title(f'{title_prefix}Speedup per Test Case (Baseline / Optimized)', fontweight='bold', fontsize=13, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=label_rotation, ha=label_ha, fontsize=label_fontsize)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add parameter names below plot
    ax2_param = fig2.add_subplot(gs2[1, 0])
    ax2_param.axis('off')
    if test_case_details:
        param_names_str = test_case_details[0]
        ax2_param.text(0.5, 0.5, f'Parameters: {param_names_str}', 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3, edgecolor='gray'))
    
    plt.tight_layout(rect=[0, bottom_margin, 1, 0.98])
    plot_file2 = workspace / f'performance_speedup.{save_format}'
    plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved speedup plot to {plot_file2}")
    
    return plot_file1

