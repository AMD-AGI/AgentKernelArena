# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from src.tasks import get_task_config
from src.preprocessing import setup_workspace, setup_rocm_env, is_task_complete
from src.module_registration import AgentType, load_agent_launcher, load_post_processing_handler
from src.evaluator import measure_baseline, evaluate_kernel, write_task_result


parser = argparse.ArgumentParser(description="arguments for AgentKernelArena")
parser.add_argument("--config_name", type=str, default="config.yaml",help="the config of AgentKernelArena, default set to config. \
                    You can set different tasks in different config yaml file in order to run multi evaluation task in one folder.")
parser.add_argument("--resume-run", type=str, default=None,
                    help="Resume an existing run by specifying the run directory name (e.g., run_20250115_143022)")
parser.add_argument("--resume-latest", action="store_true",
                    help="Resume the most recent run in the workspace")

def main() -> None:
    """Main entry point for AgentKernelArena framework."""
    args = parser.parse_args()

    # Load config.yaml
    with open(args.config_name, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration
    tasks = config['tasks']  # Now directly a list
    agent_string = config['agent']['template']
    target_gpu_model = config['target_gpu_model']

    log_directory = config['log_directory']
    workspace_directory_prefix = config['workspace_directory_prefix']

    # Convert agent string to AgentType enum
    try:
        agent = AgentType.from_string(agent_string)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Build workspace directory name
    workspace_directory_name = f"{workspace_directory_prefix}_{target_gpu_model}_{agent.value}"    
    project_root = Path(__file__).resolve().parent
    workspace_directory = (project_root / workspace_directory_name).resolve()

    # Handle resume functionality
    resume_mode = False
    if args.resume_run:
        # Resume specific run
        run_directory_name = args.resume_run
        run_directory = workspace_directory / run_directory_name
        if not run_directory.exists():
            print(f"Error: Run directory does not exist: {run_directory}")
            return
        resume_mode = True
        # Extract timestamp from run directory name: run_20250115_143022 -> 20250115_143022
        if run_directory_name.startswith("run_"):
            timestamp = run_directory_name[4:]  # Remove "run_" prefix
        else:
            print(f"Error: Invalid run directory name format: {run_directory_name}. Expected format: run_YYYYMMDD_HHMMSS")
            return
    elif args.resume_latest:
        # Resume latest run
        # Find all run directories and get the most recent one
        run_dirs = sorted([d for d in workspace_directory.iterdir() 
                          if d.is_dir() and d.name.startswith("run_")], 
                         key=lambda x: x.name, reverse=True)
        if not run_dirs:
            print(f"Error: No run directories found in {workspace_directory}")
            return
        run_directory = run_dirs[0]
        run_directory_name = run_directory.name
        resume_mode = True
        # Extract timestamp from run directory name
        if run_directory_name.startswith("run_"):
            timestamp = run_directory_name[4:]
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        # Create new run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_directory_name = f"run_{timestamp}"
        run_directory = workspace_directory / run_directory_name
        run_directory.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{target_gpu_model}_{agent.value}_{timestamp}.log"
    log_path = log_dir / log_filename

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("AgentKernelArena Framework Started")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Agent: {agent.value}")
    logger.info(f"Target Architecture: {target_gpu_model}")
    logger.info(f"Workspace Directory: {workspace_directory}")
    logger.info(f"Run Directory: {run_directory}")
    if resume_mode:
        logger.info(f"RESUME MODE: Resuming existing run {run_directory_name}")
    else:
        logger.info(f"NEW RUN: Creating new run {run_directory_name}")

    # Set PYTORCH_ROCM_ARCH based on target_gpu_model before any task runs
    setup_rocm_env(target_gpu_model, logger)

    # Load agent launcher
    try:
        agent_launcher = load_agent_launcher(agent, logger)
    except Exception as e:
        logger.error(f"Failed to load agent launcher: {e}")
        return


    # Get task config
    if 'all' in tasks:
        task_config_dict = get_task_config()
    else:
        task_config_dict = {}
        for category in tasks:
            task_config_dict.update(get_task_config(category=category))

    # Filter out completed tasks if resuming
    if resume_mode:
        original_task_count = len(task_config_dict)
        tasks_to_run = {}
        skipped_tasks = []
        
        for task_name, task_config_dir in task_config_dict.items():
            if is_task_complete(run_directory, task_name, timestamp):
                skipped_tasks.append(task_name)
                logger.info(f"Skipping completed task: {task_name}")
            else:
                tasks_to_run[task_name] = task_config_dir
        
        task_config_dict = tasks_to_run
        
        logger.info(f"Resume mode: {len(skipped_tasks)} tasks already completed, {len(task_config_dict)} tasks remaining")
        if skipped_tasks:
            logger.info(f"Skipped tasks: {skipped_tasks}")
        if len(task_config_dict) == 0:
            logger.info("All tasks are already completed. Nothing to run.")
            return

    logger.info(f"Found {len(task_config_dict)} tasks to execute")
    logger.info(f"Tasks: {list(task_config_dict.keys())}")

    # Collect workspace paths for post-processing
    workspace_paths = []

    # Run tasks
    for idx, (task_name, task_config_dir) in enumerate(task_config_dict.items(), 1):
        logger.info("=" * 80)
        logger.info(f"Task {idx}/{len(task_config_dict)}: {task_name}")
        logger.info("=" * 80)
        
        try:
            # Setup workspace
            workspace_path = setup_workspace(task_config_dir, run_directory, timestamp, logger, task_name=task_name)
            
            # Load task config for evaluation
            with open(task_config_dir, 'r') as f:
                task_config = yaml.safe_load(f)
            
            task_type = task_config.get('task_type', '')
            if task_type == 'torch2hip':
                logger.info("torch2hip task: skipping baseline compilation, measuring PyTorch baseline directly...")
                baseline_cases = measure_baseline(workspace_path, task_config, logger)
            else:
                from src.evaluator import evaluate_compilation
                logger.info("Compiling original kernel for baseline measurement...")
                pass_compilation, comp_error = evaluate_compilation(workspace_path, task_config, logger)
                if not pass_compilation:
                    logger.warning(f"Baseline compilation failed: {comp_error}")
                    logger.warning("Baseline measurement will be skipped")
                    baseline_cases = []
                else:
                    logger.info("Measuring baseline performance...")
                    baseline_cases = measure_baseline(workspace_path, task_config, logger)
            
            # Launch agent (agent should only generate optimized kernel)
            logger.info(f"Launching agent: {agent.value}")

            # For agentic approaches (cursor, claude_code, etc.)
            result = agent_launcher(
                eval_config=config,
                task_config_dir=task_config_dir,
                workspace=str(workspace_path)
            )

            logger.info(f"Agent execution completed")
            
            # Centralized evaluation of optimized kernel
            logger.info("Running centralized evaluation...")
            evaluation_results = evaluate_kernel(
                workspace_path,
                task_config,
                baseline_cases,
                logger
            )
            
            # Write standardized task_result.yaml
            write_task_result(
                workspace_path,
                evaluation_results,
                baseline_cases,
                task_name,
                agent.value,
                logger
            )
            
            logger.info(f"Task {task_name} completed successfully")

            # Add workspace path to list for post-processing
            workspace_paths.append(str(workspace_path))

        except Exception as e:
            logger.error(f"Task {task_name} failed with error: {e}", exc_info=True)
            # Still add workspace path even if task failed (for post-processing to record failure)
            if 'workspace_path' in locals():
                workspace_paths.append(str(workspace_path))
            continue

    # Run post-processing to generate report
    logger.info("=" * 80)
    logger.info("Running Post-Processing")
    logger.info("=" * 80)

    try:
        post_processing_handler = load_post_processing_handler(agent, logger)
        post_processing_handler(workspace_paths, logger)
    except NotImplementedError as e:
        logger.warning(f"Post-processing skipped: {e}")
    except Exception as e:
        logger.error(f"Post-processing failed: {e}", exc_info=True)

    logger.info("=" * 80)
    logger.info("AgentKernelArena Framework Completed")
    logger.info("=" * 80)



if __name__ == "__main__":
    main()
