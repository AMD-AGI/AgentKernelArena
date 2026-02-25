# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
OpenEvolve agent launcher for AgentKernelArena
"""
import asyncio
import importlib
import logging
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Add openevolve to path
_openevolve_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geak-openevolve')
sys.path.insert(0, _openevolve_dir)

from agents import register_agent

logger = logging.getLogger(__name__)


def ensure_openevolve_setup() -> None:
    """Run OpenEvolve setup script when local GEAK sources are missing."""
    agent_dir = Path(__file__).parent
    local_repo_dir = agent_dir / "geak-openevolve"
    local_pkg_dir = local_repo_dir / "openevolve"
    geak_eval_dir = local_repo_dir / "GEAK-eval-OE"

    # Ensure local repo path is first so imports resolve to this checkout.
    local_repo_str = str(local_repo_dir)
    if local_repo_str not in sys.path:
        sys.path.insert(0, local_repo_str)

    # Source of truth is local package directory + eval repo.
    if local_pkg_dir.exists() and geak_eval_dir.exists():
        return

    setup_script = agent_dir / "agent_setup.sh"
    if not setup_script.exists():
        raise FileNotFoundError(f"Missing setup script: {setup_script}")

    logger.info("Running OpenEvolve setup via agent_setup.sh")
    proc = subprocess.run(
        ["bash", str(setup_script)],
        cwd=str(Path(__file__).parent),
        env={**os.environ, "PYTHON_BIN": sys.executable},
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenEvolve setup failed.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    if not local_pkg_dir.exists():
        raise RuntimeError(
            f"OpenEvolve package directory not found after setup: {local_pkg_dir}"
        )


def create_evaluator(workspace: str, task_config: Dict) -> str:
    """Create evaluator that runs AgentKernelArena commands"""
    compile_cmds = task_config.get('compile_command', ['make'])
    correctness_cmds = task_config.get('correctness_command', ['./app'])
    performance_cmds = task_config.get('performance_command', [])
    source_files = task_config.get('source_file_path', [])
    
    # For instruction2triton tasks, extract filename from compile command
    target_file = None
    if not source_files or source_files[0] is None:
        # Try to extract .py filename from compile command
        for cmd in compile_cmds:
            if '.py' in cmd:
                import re
                match = re.search(r'(\w+\.py)', cmd)
                if match:
                    target_file = match.group(1)
                    break
    else:
        target_file = source_files[0]
    
    evaluator_code = f'''
import json
import os
import re
import subprocess

BASELINE_TIME = None

def parse_time(output):
    patterns = [r'([0-9.]+)\\s*ms', r'([0-9.]+)\\s*s']
    for p in patterns:
        m = re.search(p, output, re.I)
        if m:
            value = float(m.group(1))
            if p.endswith(r'\\s*s'):
                return value * 1000.0
            return value
    return None

def parse_perf_json(perf_dir):
    if not os.path.isdir(perf_dir):
        return None
    ms_values = []
    for root, _, files in os.walk(perf_dir):
        for name in files:
            if not name.endswith(".json"):
                continue
            file_path = os.path.join(root, name)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            ms_value = item.get("ms")
                            if isinstance(ms_value, (int, float)) and ms_value > 0:
                                ms_values.append(float(ms_value))
            except Exception:
                continue
    if not ms_values:
        return None
    return min(ms_values)

def evaluate(test_suite_path, program_text, ref_wrapper_path=None, wrapper_fn_name=None,
             unit_tests_path=None, n_warmup=5, n_iters=10, atol=1e-3, rtol=1e-3, 
             verbose=False, gpu_id=0, timeout=300):
    global BASELINE_TIME
    # Determine workspace directory
    program_dir = os.path.dirname(os.path.abspath(program_text))
    
    # Check if we're in an eval temp directory (evals/tmpXXX)
    if '/evals/tmp' in program_dir:
        # Go up two levels: evals/tmpXXX -> evals -> workspace
        workspace = os.path.dirname(os.path.dirname(program_dir))
    else:
        # We're directly in the workspace
        workspace = program_dir
    
    os.chdir(workspace)
    run_env = dict(os.environ)
    existing_pythonpath = run_env.get("PYTHONPATH")
    pythonpath_entries = [workspace, program_dir]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    run_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    
    if verbose:
        print(f"[EVAL] Workspace: {{workspace}}")
        print(f"[EVAL] Program dir: {{program_dir}}")
        print(f"[EVAL] Working dir: {{os.getcwd()}}")
    
    # For ROCm/Triton tasks, program_text contains the evolved kernel
    # Get relative path from workspace to the test file
    test_file_path = os.path.relpath(os.path.abspath(program_text), workspace)
    
    if verbose:
        print(f"[EVAL] Test file (relative): {{test_file_path}}")
    
    result = {{
        'success': 0.0,
        'correctness_score': 0.0,
        'combined_score': 0.0,
        'base_execution_time': 0.0,
        'best_optimized_execution_time': 0.0,
        'speedup': 0.0
    }}
    
    try:
        # Compile - replace target filename with actual test file path
        compile_timeout = min(60, timeout // 2)  # Use half of total timeout or 60s
        for cmd_template in {repr(compile_cmds)}:
            # Replace the hardcoded filename with the actual evolved test file  
            cmd = cmd_template.replace({repr(target_file if target_file else 'test.py')}, test_file_path)
            if verbose:
                print(f"\\n[COMPILE] Running: {{cmd}}")
                print(f"[COMPILE] Timeout: {{compile_timeout}} seconds")
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=compile_timeout,
                env=run_env
            )
            if proc.returncode != 0:
                error_msg = f"Compile failed (exit {{proc.returncode}})\\nSTDOUT:\\n{{proc.stdout}}\\nSTDERR:\\n{{proc.stderr}}"
                if verbose:
                    print(f"[COMPILE ERROR] {{error_msg}}")
                result['error'] = error_msg
                return result
            if verbose and proc.stdout:
                print(f"[COMPILE STDOUT] {{proc.stdout}}")
        
        result['success'] = 0.2
        result['combined_score'] = 0.2
        
        # Correctness - replace target filename with actual test file path
        for cmd_template in {repr(correctness_cmds)}:
            # Replace the hardcoded filename with the actual evolved test file
            cmd = cmd_template.replace({repr(target_file if target_file else 'test.py')}, test_file_path)
            if verbose:
                print(f"\\n[CORRECTNESS] Running: {{cmd}}")
                print(f"[CORRECTNESS] Timeout: {{timeout}} seconds")
                print(f"[CORRECTNESS] Real-time output:")
                # Run without capture to see real-time output
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    timeout=timeout,
                    env=run_env
                )
            else:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=run_env
                )
            if proc.returncode != 0:
                error_msg = f"Correctness failed (exit {{proc.returncode}})\\nSTDOUT:\\n{{proc.stdout}}\\nSTDERR:\\n{{proc.stderr}}"
                if verbose:
                    print(f"[CORRECTNESS ERROR] {{error_msg}}")
                result['error'] = error_msg
                return result
            if verbose and proc.stdout:
                print(f"[CORRECTNESS STDOUT] {{proc.stdout}}")
        
        result['correctness_score'] = 1.0
        result['success'] = 1.0
        result['combined_score'] = 1.2

        # Performance - replace target filename with actual evolved test file
        performance_outputs = []
        for cmd_template in {repr(performance_cmds)}:
            cmd = cmd_template.replace({repr(target_file if target_file else 'test.py')}, test_file_path)
            if verbose:
                print(f"\\n[PERFORMANCE] Running: {{cmd}}")
                print(f"[PERFORMANCE] Timeout: {{timeout}} seconds")
            proc = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env
            )
            performance_outputs.append((proc.stdout or "") + "\\n" + (proc.stderr or ""))
            if proc.returncode != 0:
                # Keep compile/correctness pass if performance command fails.
                perf_error = f"Performance failed (exit {{proc.returncode}})\\nSTDOUT:\\n{{proc.stdout}}\\nSTDERR:\\n{{proc.stderr}}"
                if verbose:
                    print(f"[PERFORMANCE ERROR] {{perf_error}}")
                result['performance_error'] = perf_error
                return result

        perf_time_ms = parse_perf_json(os.path.join(program_dir, "perf"))
        if perf_time_ms is None:
            perf_time_ms = parse_perf_json(os.path.join(workspace, "perf"))
        if perf_time_ms is None:
            for output in performance_outputs:
                parsed = parse_time(output)
                if parsed is not None:
                    if perf_time_ms is None or parsed < perf_time_ms:
                        perf_time_ms = parsed

        if perf_time_ms is not None and perf_time_ms > 0:
            if BASELINE_TIME is None:
                BASELINE_TIME = perf_time_ms
            result['base_execution_time'] = float(BASELINE_TIME)
            result['best_optimized_execution_time'] = float(perf_time_ms)
            result['speedup'] = float(BASELINE_TIME / perf_time_ms)
            result['combined_score'] = 1.2 + result['speedup']
        
    except Exception as e:
        result['error'] = str(e)
    
    return result
'''
    
    evaluator_path = os.path.abspath(os.path.join(workspace, 'aig_evaluator.py'))
    with open(evaluator_path, 'w') as f:
        f.write(evaluator_code)
    return evaluator_path


def create_kernel_with_separator(kernel_path: str) -> str:
    """Add OpenEvolve separator to kernel"""
    # Ensure absolute path
    kernel_path = os.path.abspath(kernel_path)
    
    with open(kernel_path, 'r') as f:
        kernel_code = f.read()
    
    separator = "#" * 146
    test_code = '''
# Test section (not used)
def test():
    pass
'''
    
    full_program = f"{kernel_code}\n\n{separator}\n\n{test_code}"
    formatted_path = kernel_path.replace('.hip', '_oe.hip')
    with open(formatted_path, 'w') as f:
        f.write(full_program)
    
    return os.path.abspath(formatted_path)


@register_agent("openevolve")
def launch_agent(eval_config: Dict, task_config_dir: str, workspace: str) -> str:
    """Launch OpenEvolve agent"""
    ensure_openevolve_setup()

    OpenEvolve = importlib.import_module("openevolve").OpenEvolve
    Config = importlib.import_module("openevolve.config").Config
    
    logger.info("=" * 80)
    logger.info("Starting OpenEvolve Agent")
    logger.info("=" * 80)
    
    # Load configs
    with open(task_config_dir, 'r') as f:
        task_config = yaml.safe_load(f)
    
    agent_config_path = Path(__file__).parent / 'agent_config_amd_claude.yaml'
    with open(agent_config_path, 'r') as f:
        agent_config = yaml.safe_load(f)
    
    # Get source file - instruction2triton tasks have null source files
    source_files = task_config.get('source_file_path', [])
    
    # For instruction2triton tasks, source_file_path is [null]
    # BUT: OpenEvolve is triton2triton, so we need an existing kernel to start with
    if not source_files or source_files[0] is None:
        logger.info("instruction2triton task detected")
        
        # OpenEvolve needs an existing kernel to optimize (triton2triton)
        # Look for the kernel file in the workspace (AgentKernelArena copies task files)
        kernel_files = list(Path(workspace).glob("*.py"))
        
        # Filter out config.yaml and find the main kernel file
        kernel_files = [f for f in kernel_files if f.name not in ['config.yaml', 'aig_evaluator.py']]
        
        if kernel_files:
            # Use the first .py file as the initial kernel
            source_file = os.path.abspath(kernel_files[0])
            logger.info(f"Using existing kernel as initial program: {source_file}")
        else:
            # Fallback: Extract filename from compile command
            compile_cmds = task_config.get('compile_command', [])
            kernel_filename = None
            for cmd in compile_cmds:
                if '.py' in cmd:
                    import re
                    match = re.search(r'(\w+\.py)', cmd)
                    if match:
                        kernel_filename = match.group(1)
                        break
            
            if not kernel_filename:
                # Last resort: use target kernel name
                target_kernels = task_config.get('target_kernel_functions', [])
                kernel_filename = f"{target_kernels[0]}.py" if target_kernels else "kernel.py"
            
            source_file = os.path.abspath(os.path.join(workspace, kernel_filename))
            
            if not os.path.exists(source_file):
                raise FileNotFoundError(
                    f"No existing kernel found for instruction2triton task. "
                    f"OpenEvolve is triton2triton and needs an initial kernel to optimize. "
                    f"Expected kernel at: {source_file}"
                )
            
            logger.info(f"Using kernel from workspace: {source_file}")
        
        formatted_kernel = source_file
    else:
        source_file = os.path.abspath(os.path.join(workspace, source_files[0]))
        # Create formatted kernel
        formatted_kernel = create_kernel_with_separator(source_file)
    
    # Create evaluator
    evaluator_path = create_evaluator(workspace, task_config)
    
    # Create OpenEvolve config
    oe_config = Config()
    oe_config.max_iterations = agent_config.get('max_iterations', 10)
    oe_config.checkpoint_interval = agent_config.get('checkpoint_interval', 5)
    oe_config.max_code_length = agent_config.get('max_code_length', oe_config.max_code_length)
    
    # Fix sampling parameter (OpenEvolve requires it)
    oe_config.llm.sampling = {'fn': 'random'}
    
    # ============================================================================
    # API Configuration (Key + Base URL)
    # ============================================================================
    # Priority order for api_base:
    # 1. Per-model config in agent_config.yaml (highest priority)
    # 2. Shared llm.api_base in agent_config.yaml
    # 3. OPENAI_API_BASE environment variable
    # 4. Default to standard OpenAI API (lowest priority)
    
    # Get API credentials from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # Default to standard OpenAI API, can be overridden by env var or config
    DEFAULT_API_BASE = 'https://api.openai.com/v1'
    api_base = os.environ.get('OPENAI_API_BASE', DEFAULT_API_BASE)
    
    # Set API key if provided
    if api_key:
        oe_config.llm.api_key = api_key
        oe_config.llm.update_model_params({'api_key': api_key}, overwrite=True)
        for model in oe_config.llm.models:
            model.api_key = api_key
    
    # Apply LLM configuration from agent_config.yaml
    if 'llm' in agent_config:
        # Shared api_base (overrides environment variable if provided)
        if 'api_base' in agent_config['llm']:
            api_base = agent_config['llm']['api_base']
            logger.info(f"Using api_base from config: {api_base}")
        elif api_base != DEFAULT_API_BASE:
            logger.info(f"Using api_base from environment: {api_base}")
        else:
            logger.info(f"Using default OpenAI API: {api_base}")
        
        # Set shared api_base for all models
        oe_config.llm.api_base = api_base
        oe_config.llm.update_model_params({'api_base': api_base}, overwrite=True)
        
        # Model-specific configuration
        if 'models' in agent_config['llm'] and len(agent_config['llm']['models']) > 0:
            model_config = agent_config['llm']['models'][0]
            oe_config.llm.models[0].name = model_config.get('name', 'gpt-4o-mini')
            
            # Per-model api_base (highest priority, overrides shared api_base)
            if 'api_base' in model_config:
                oe_config.llm.models[0].api_base = model_config['api_base']
                logger.info(f"Model '{oe_config.llm.models[0].name}' using per-model api_base: {model_config['api_base']}")
            else:
                oe_config.llm.models[0].api_base = api_base
        
        oe_config.llm.temperature = agent_config['llm'].get('temperature', 0.7)
    else:
        # No LLM config provided, use defaults with environment variables
        oe_config.llm.api_base = api_base
        oe_config.llm.update_model_params({'api_base': api_base}, overwrite=True)
        logger.info(f"No LLM config provided, using api_base: {api_base}")
    
    if 'database' in agent_config:
        oe_config.database.population_size = agent_config['database'].get('population_size', 50)
        oe_config.database.num_islands = agent_config['database'].get('num_islands', 2)
    
    # Set db_path to avoid None error
    oe_config.database.db_path = "database"
    
    if 'evaluator' in agent_config:
        oe_config.evaluator.timeout = agent_config['evaluator'].get('timeout', 120)
        oe_config.evaluator.verbose = agent_config['evaluator'].get('verbose', False)
        oe_config.evaluator.parallel_evaluations = agent_config['evaluator'].get('parallel_evaluations', oe_config.evaluator.parallel_evaluations)
        oe_config.evaluator.cascade_evaluation = agent_config['evaluator'].get('cascade_evaluation', oe_config.evaluator.cascade_evaluation)
        if oe_config.evaluator.verbose:
            logger.info("Verbose evaluation mode enabled - will print detailed stdout/stderr")
    
    # Use absolute paths to avoid issues when OpenEvolve changes working directory
    workspace_abs = os.path.abspath(workspace)
    
    output_dir = os.path.join(workspace_abs, 'openevolve_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create eval_dir for temporary evaluation files (must be absolute path)
    eval_dir = os.path.join(workspace_abs, 'evals')
    os.makedirs(eval_dir, exist_ok=True)
    oe_config.evaluator.eval_dir = eval_dir
    
    logger.info(f"Using absolute eval_dir: {eval_dir}")
    
    try:
        # Run OpenEvolve
        openevolve = OpenEvolve(
            initial_program_path=formatted_kernel,
            evaluation_file=evaluator_path,
            config=oe_config,
            output_dir=output_dir
        )
        
        logger.info(f"Running evolution for {agent_config.get('max_iterations', 10)} iterations...")
        best_program = asyncio.run(openevolve.run())
        
        if best_program is None:
            raise RuntimeError("No valid program produced")
        
        logger.info(f"Best program metrics: {best_program.metrics}")
        
        # Extract kernel code
        #separator = "#" * 146
        #best_kernel_code = best_program.code.split(separator)[0].strip() if separator in best_program.code else best_program.code
        best_kernel_code = best_program.code
        
        # Write back to source
        with open(source_file, 'w') as f:
            f.write(best_kernel_code)
        
        best_metrics = best_program.metrics or {}
        pass_compilation = best_metrics.get('success', 0) >= 0.2
        pass_correctness = best_metrics.get('correctness_score', 0) >= 1.0
        base_execution_time = float(best_metrics.get('base_execution_time', 0.0) or 0.0)
        best_optimized_execution_time = float(best_metrics.get('best_optimized_execution_time', 0.0) or 0.0)
        if base_execution_time > 0 and best_optimized_execution_time > 0:
            speedup_ratio = base_execution_time / best_optimized_execution_time
        else:
            speedup_ratio = float(best_metrics.get('speedup', 0.0) or 0.0)

        metrics_error = best_metrics.get('error')

        # Write task_result.yaml with standard framework fields.
        task_result = {
            'task_name': Path(workspace).name,
            'best_optimized_source_file_path': [os.path.basename(source_file)],
            'best_optimized_kernel_functions': task_config.get('target_kernel_functions', []),
            'pass_compilation': pass_compilation,
            'compilation_error_message': None if pass_compilation else metrics_error,
            'pass_correctness': pass_correctness,
            'correctness_error_message': None if pass_correctness else metrics_error,
            'base_execution_time': base_execution_time,
            'best_optimized_execution_time': best_optimized_execution_time,
            'speedup_ratio': speedup_ratio,
            'optimization_summary': 'OpenEvolve optimization complete'
        }
        
        with open(os.path.join(workspace, 'task_result.yaml'), 'w') as f:
            yaml.dump(task_result, f)
        
        return best_kernel_code
        
    except Exception as e:
        logger.error(f"OpenEvolve failed: {e}", exc_info=True)
        
        # Write failed result
        task_result = {
            'task_name': Path(workspace).name,
            'best_optimized_source_file_path': [os.path.basename(source_file)] if 'source_file' in locals() else [],
            'best_optimized_kernel_functions': task_config.get('target_kernel_functions', []) if 'task_config' in locals() else [],
            'pass_compilation': False,
            'compilation_error_message': str(e),
            'pass_correctness': False,
            'correctness_error_message': str(e),
            'base_execution_time': 0.0,
            'best_optimized_execution_time': 0.0,
            'speedup_ratio': 0.0,
            'optimization_summary': f'Failed: {str(e)}'
        }
        with open(os.path.join(workspace, 'task_result.yaml'), 'w') as f:
            yaml.dump(task_result, f)
        
        raise
