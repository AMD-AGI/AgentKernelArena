# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Standalone script to compare two AgentKernelArena runs.

Usage:
    python compare_runs.py run1_path run2_path
    python compare_runs.py workspace_MI300_cursor/run_20250115_143022 workspace_MI300_cursor/run_20250115_160530
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def load_run_data(run_path: Path) -> Dict[str, Any]:
    """
    Load task_type_breakdown.json from a run directory.
    
    Args:
        run_path: Path to run directory (e.g., workspace_MI300_cursor/run_20250115_143022)
    
    Returns:
        Dictionary containing run data from JSON file
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON file is invalid
    """
    json_path = run_path / "reports" / "task_type_breakdown.json"
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"task_type_breakdown.json not found in {run_path}/reports/\n"
            f"Make sure the run directory contains a reports/ subdirectory with task_type_breakdown.json"
        )
    
    with open(json_path, 'r') as f:
        return json.load(f)


def format_difference(value1: float, value2: float, is_percentage: bool = False) -> str:
    """
    Format the difference between two values.
    
    Args:
        value1: First value (baseline)
        value2: Second value (comparison)
        is_percentage: If True, format as percentage change
    
    Returns:
        Formatted string showing difference
    """
    diff = value2 - value1
    if is_percentage:
        if value1 == 0:
            return f"{diff:+.1f}pp" if diff != 0 else "0.0pp"
        pct_change = (diff / value1 * 100) if value1 != 0 else 0
        return f"{diff:+.1f}pp ({pct_change:+.1f}%)"
    else:
        pct_change = (diff / value1 * 100) if value1 != 0 else 0
        return f"{diff:+.3f} ({pct_change:+.1f}%)"


def compare_overall(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> list:
    """
    Compare overall statistics between two runs.
    
    Returns:
        List of formatted comparison lines
    """
    overall1 = run1_data.get('overall', {})
    overall2 = run2_data.get('overall', {})
    
    lines = [
        "=" * 80,
        "OVERALL STATISTICS COMPARISON",
        "=" * 80,
        f"Run 1: {run1_data.get('run_timestamp', 'unknown')} ({run1_data.get('agent', 'unknown')})",
        f"Run 2: {run2_data.get('run_timestamp', 'unknown')} ({run2_data.get('agent', 'unknown')})",
        "=" * 80,
        "",
        f"{'Metric':<40} {'Run 1':<15} {'Run 2':<15} {'Difference':<20}",
        "-" * 80,
    ]
    
    metrics = [
        ('Total Tasks', 'total_tasks', False),
        ('Total Score', 'total_score', False),
        ('Average Score', 'average_score', False),
        ('Compilation Pass Rate', 'compilation_pass_rate', True),
        ('Correctness Pass Rate', 'correctness_pass_rate', True),
        ('Speedup > 1.0 Rate', 'speedup_gt_1_rate', True),
        ('Average Speedup', 'average_speedup', False),
    ]
    
    for label, key, is_percentage in metrics:
        val1 = overall1.get(key, 0.0)
        val2 = overall2.get(key, 0.0)
        
        if is_percentage:
            fmt1 = f"{val1:.1f}%"
            fmt2 = f"{val2:.1f}%"
        elif key == 'total_tasks':
            fmt1 = f"{int(val1)}"
            fmt2 = f"{int(val2)}"
        elif key == 'total_score':
            fmt1 = f"{val1:.2f}"
            fmt2 = f"{val2:.2f}"
        else:
            fmt1 = f"{val1:.3f}"
            fmt2 = f"{val2:.3f}"
        
        diff_str = format_difference(val1, val2, is_percentage)
        
        # Determine if improvement (green) or regression (red) - for display purposes
        if key in ['average_score', 'compilation_pass_rate', 'correctness_pass_rate', 
                   'speedup_gt_1_rate', 'average_speedup']:
            if val2 > val1:
                indicator = "↑"
            elif val2 < val1:
                indicator = "↓"
            else:
                indicator = "="
        else:
            indicator = ""
        
        lines.append(f"{label:<40} {fmt1:<15} {fmt2:<15} {diff_str:<20} {indicator}")
    
    lines.append("")
    return lines


def compare_task_types(run1_data: Dict[str, Any], run2_data: Dict[str, Any]) -> list:
    """
    Compare task type breakdowns between two runs.
    
    Returns:
        List of formatted comparison lines
    """
    types1 = run1_data.get('task_types', {})
    types2 = run2_data.get('task_types', {})
    
    # Get all unique task types from both runs
    all_types = set(types1.keys()) | set(types2.keys())
    
    if not all_types:
        return ["No task type data available for comparison."]
    
    lines = [
        "=" * 80,
        "TASK TYPE BREAKDOWN COMPARISON",
        "=" * 80,
        "",
    ]
    
    for task_type in sorted(all_types):
        stats1 = types1.get(task_type, {})
        stats2 = types2.get(task_type, {})
        
        count1 = stats1.get('count', 0)
        count2 = stats2.get('count', 0)
        
        lines.append(f"{task_type.upper()} ({count1} tasks → {count2} tasks):")
        lines.append("-" * 80)
        
        if count1 == 0 and count2 == 0:
            lines.append("  No tasks in either run")
            lines.append("")
            continue
        
        # Compare key metrics
        metrics = [
            ('Average Speedup', 'average_speedup', False),
            ('Compilation Pass Rate', 'compilation_pass_rate', True),
            ('Correctness Pass Rate', 'correctness_pass_rate', True),
            ('Speedup > 1.0 Rate', 'speedup_gt_1_rate', True),
            ('Average Score', 'average_score', False),
        ]
        
        for label, key, is_percentage in metrics:
            val1 = stats1.get(key, 0.0)
            val2 = stats2.get(key, 0.0)
            
            if count1 == 0:
                fmt1 = "N/A"
                diff_str = "N/A (new)"
            elif count2 == 0:
                fmt2 = "N/A"
                diff_str = "N/A (removed)"
            else:
                if is_percentage:
                    fmt1 = f"{val1:.1f}%"
                    fmt2 = f"{val2:.1f}%"
                elif key == 'average_score':
                    fmt1 = f"{val1:.2f}"
                    fmt2 = f"{val2:.2f}"
                else:
                    fmt1 = f"{val1:.3f}"
                    fmt2 = f"{val2:.3f}"
                
                diff_str = format_difference(val1, val2, is_percentage)
            
            if count1 > 0 and count2 > 0:
                if val2 > val1:
                    indicator = "↑ (improved)"
                elif val2 < val1:
                    indicator = "↓ (regressed)"
                else:
                    indicator = "= (same)"
            else:
                indicator = ""
            
            if count1 == 0:
                lines.append(f"  {label:<35} {'N/A':<15} {fmt2:<15} {diff_str:<20} {indicator}")
            elif count2 == 0:
                lines.append(f"  {label:<35} {fmt1:<15} {'N/A':<15} {diff_str:<20} {indicator}")
            else:
                lines.append(f"  {label:<35} {fmt1:<15} {fmt2:<15} {diff_str:<20} {indicator}")
        
        lines.append("")
    
    return lines


def generate_comparison_report(run1_path: Path, run2_path: Path, output_path: Optional[Path] = None) -> str:
    """
    Generate a comparison report between two runs.
    
    Args:
        run1_path: Path to first run directory
        run2_path: Path to second run directory
        output_path: Optional path to save report (if None, auto-generates in comparisons/ directory)
    
    Returns:
        Comparison report as string
    """
    # Load data from both runs
    try:
        run1_data = load_run_data(run1_path)
    except Exception as e:
        return f"Error loading Run 1 data from {run1_path}:\n{str(e)}\n"
    
    try:
        run2_data = load_run_data(run2_path)
    except Exception as e:
        return f"Error loading Run 2 data from {run2_path}:\n{str(e)}\n"
    
    # Generate comparison report
    lines = [
        "=" * 80,
        "AgentKernelArena Run Comparison Report",
        "=" * 80,
        "",
    ]
    
    lines.extend(compare_overall(run1_data, run2_data))
    lines.extend(compare_task_types(run1_data, run2_data))
    
    lines.extend([
        "=" * 80,
        "Legend:",
        "  ↑ = Improvement (higher is better)",
        "  ↓ = Regression (lower is worse)",
        "  = = No change",
        "  pp = percentage points",
        "=" * 80,
    ])
    
    report = "\n".join(lines)
    
    # Determine output path
    if output_path is None:
        # Auto-generate path in comparisons/ directory at project root
        # Extract run directory names (e.g., "run_20250115_143022" from full path)
        run1_name = run1_path.name
        run2_name = run2_path.name
        
        # Project root is the directory where compare_runs.py is located
        project_root = Path(__file__).resolve().parent
        
        comparisons_dir = project_root / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename: comparison_report_{run1}_{run2}.txt
        filename = f"comparison_report_{run1_name}_{run2_name}.txt"
        output_path = comparisons_dir / filename
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Comparison report written to: {output_path}")
    
    return report


def main():
    """Main entry point for comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare two AgentKernelArena runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two runs
  python compare_runs.py workspace_MI300_cursor/run_20250115_143022 workspace_MI300_cursor/run_20250115_160530
  
  # Compare and save to file
  python compare_runs.py run1 run2 --output comparison_report.txt
        """
    )
    
    parser.add_argument(
        'run1',
        type=str,
        help='Path to first run directory (e.g., workspace_MI300_cursor/run_20250115_143022)'
    )
    
    parser.add_argument(
        'run2',
        type=str,
        help='Path to second run directory (e.g., workspace_MI300_cursor/run_20250115_160530)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Optional output file path for comparison report (if not specified, auto-generates in comparisons/ directory)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    run1_path = Path(args.run1).resolve()
    run2_path = Path(args.run2).resolve()
    
    # Validate paths exist
    if not run1_path.exists():
        print(f"Error: Run 1 directory does not exist: {run1_path}", file=sys.stderr)
        sys.exit(1)
    
    if not run2_path.exists():
        print(f"Error: Run 2 directory does not exist: {run2_path}", file=sys.stderr)
        sys.exit(1)
    
    # Generate and print comparison report
    output_path = Path(args.output).resolve() if args.output else None
    report = generate_comparison_report(run1_path, run2_path, output_path)
    
    # Print to stdout
    print(report)


if __name__ == "__main__":
    main()

