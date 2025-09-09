#!/usr/bin/env python3
"""
General experiment launcher script that reads YAML configuration files.

This script allows you to define experiments with:
- Script selection
- Fixed arguments that apply to all runs
- Variable arguments that create multiple experiment runs
- Automatic experiment naming and logging

Example usage:
    python launch_experiments.py config/rephrase_benchmark.yaml
    python launch_experiments.py config/rephrase_benchmark.yaml --dry-run
    python launch_experiments.py config/rephrase_benchmark.yaml --run-names "run1,run3"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ExperimentLauncher:
    """Launches experiments based on YAML configuration."""
    
    def __init__(self, config_path: str, dry_run: bool = False, 
                 selected_runs: Optional[List[str]] = None):
        """
        Initialize the experiment launcher.
        
        Args:
            config_path: Path to YAML configuration file
            dry_run: If True, print commands without executing
            selected_runs: Optional list of specific run names to execute
        """
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.selected_runs = selected_runs or []
        
        # Load and validate configuration
        self.config = self._load_config()
        self._validate_config()
        
        # Setup experiment metadata
        self.experiment_name = self.config.get('experiment_name', 'experiment')
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')
        
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if not config:
            raise ValueError("Empty configuration file")
            
        return config
    
    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        required_keys = ['script', 'runs']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required key in config: {key}")
        
        # Validate runs structure
        if not isinstance(self.config['runs'], list):
            raise ValueError("'runs' must be a list of dictionaries")
            
        if len(self.config['runs']) == 0:
            raise ValueError("'runs' list cannot be empty")
            
        for i, run in enumerate(self.config['runs']):
            if not isinstance(run, dict):
                raise ValueError(f"Run {i} must be a dictionary")
            if 'name' not in run:
                raise ValueError(f"Run {i} missing required 'name' field")
            # Validate that no run has 'name' in its args (since we auto-generate it)
            if 'name' in run.get('args', {}):
                raise ValueError(
                    f"Run '{run['name']}' should not have 'name' in its args. "
                    f"The run name is automatically passed as --name to the script."
                )
        
        # Validate that all runs have the same varied arguments (for fair comparison)
        self._validate_consistent_varied_args()
    
    def _validate_consistent_varied_args(self) -> None:
        """Validate that all runs vary the same arguments for fair comparison."""
        if len(self.config['runs']) < 2:
            return  # No need to validate if there's only one run
        
        # Get the set of varied arguments from the first run
        first_run_args = set(self.config['runs'][0].get('args', {}).keys())
        
        # Check that all runs have the same varied arguments
        for i, run in enumerate(self.config['runs'][1:], 1):
            run_args = set(run.get('args', {}).keys())
            if run_args != first_run_args:
                missing_in_run = first_run_args - run_args
                extra_in_run = run_args - first_run_args
                error_parts = []
                if missing_in_run:
                    error_parts.append(f"missing: {', '.join(missing_in_run)}")
                if extra_in_run:
                    error_parts.append(f"extra: {', '.join(extra_in_run)}")
                raise ValueError(
                    f"Run '{run['name']}' has inconsistent varied arguments compared to '{self.config['runs'][0]['name']}'. "
                    f"{'; '.join(error_parts)}. All runs must vary the same set of arguments for fair comparison."
                )
    
    def _build_command(self, run_config: Dict[str, Any]) -> List[str]:
        """
        Build command line from configuration.
        
        Args:
            run_config: Configuration for a specific run
            
        Returns:
            Command as list of strings
        """
        # Build base command - execute script directly
        script = self.config['script']
        if script.endswith('.py'):
            cmd = ['python', script]
        else:
            # Execute script directly (e.g., "rephrase" -> "rephrase")
            cmd = [script]
        
        # Add the run name as --name argument first
        cmd.extend(['--name', run_config['name']])
        
        # Add fixed arguments (apply to all runs)
        fixed_args = self.config.get('fixed_args', {})
        for key, value in fixed_args.items():
            self._add_argument_to_command(cmd, key, value)
        
        # Add variable arguments (specific to this run)
        var_args = run_config.get('args', {})
        for key, value in var_args.items():
            self._add_argument_to_command(cmd, key, value)
        
        return cmd
    
    def _add_argument_to_command(self, cmd: List[str], key: str, value: Any) -> None:
        """Add an argument to the command list based on its type."""
        if isinstance(value, bool):
            if value:  # Only add flag if True
                cmd.append(f"--{key}")
        elif isinstance(value, list):
            # Handle multiple values (e.g., --data_paths path1 path2)
            cmd.append(f"--{key}")
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([f"--{key}", str(value)])
    
    def _should_run(self, run_name: str) -> bool:
        """Check if a specific run should be executed."""
        if not self.selected_runs:
            return True
        return run_name in self.selected_runs
    
    def _execute_command(self, cmd: List[str], run_name: str) -> int:
        """Execute a command (Slurm job submission) and return the exit code."""
        print(f"\n{'='*60}")
        print(f"Submitting Slurm job for run: {run_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        if self.dry_run:
            print("[DRY RUN] Slurm job would be submitted")
            return 0
        
        try:
            # Execute the command (which will submit a Slurm job)
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            
            # Print stdout/stderr for job submission feedback
            if result.stdout:
                print(f"Job submission output: {result.stdout.strip()}")
            if result.stderr:
                print(f"Job submission errors: {result.stderr.strip()}")
                
            if result.returncode == 0:
                print(f"✅ Slurm job for {run_name} submitted successfully")
            else:
                print(f"❌ Failed to submit Slurm job for {run_name}")
                
            return result.returncode
        except KeyboardInterrupt:
            print(f"\nInterrupted during job submission for run: {run_name}")
            raise
        except Exception as e:
            print(f"Error submitting job for run {run_name}: {e}")
            return 1
    
    def launch(self) -> Dict[str, int]:
        """
        Launch all configured experiments as Slurm jobs.
        
        Returns:
            Dictionary mapping run names to job submission exit codes
        """
        print(f"Launching experiment: {self.experiment_name}")
        print(f"Configuration: {self.config_path}")
        print(f"Timestamp: {self.timestamp}")
        
        if self.dry_run:
            print("\n*** DRY RUN MODE - No Slurm jobs will be submitted ***")
        else:
            print("\n*** SUBMITTING SLURM JOBS ***")
        
        if self.selected_runs:
            print(f"Selected runs: {', '.join(self.selected_runs)}")
        
        results = {}
        
        for run_config in self.config['runs']:
            run_name = run_config['name']
            
            if not self._should_run(run_name):
                print(f"Skipping run: {run_name}")
                continue
            
            # Build and submit Slurm job
            cmd = self._build_command(run_config)
            exit_code = self._execute_command(cmd, run_name)
            results[run_name] = exit_code
            
            if exit_code != 0:
                print(f"❌ Slurm job submission failed for {run_name} (exit code {exit_code})")
                
                # Check if we should continue on failure
                if not self.config.get('continue_on_failure', True):
                    print("Stopping job submissions due to failure")
                    break
            elif not self.dry_run:
                print(f"✅ Slurm job submitted successfully for {run_name}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("JOB SUBMISSION SUMMARY")
        print(f"{'='*60}")
        
        for run_name, exit_code in results.items():
            if self.dry_run:
                status = "✅ WOULD SUBMIT" if exit_code == 0 else f"❌ WOULD FAIL ({exit_code})"
            else:
                status = "✅ SUBMITTED" if exit_code == 0 else f"❌ FAILED ({exit_code})"
            print(f"{run_name:<30} {status}")
        
        if results:
            successful_submissions = sum(1 for code in results.values() if code == 0)
            if self.dry_run:
                print(f"\n{successful_submissions}/{len(results)} jobs would be submitted successfully")
            else:
                print(f"\n{successful_submissions}/{len(results)} jobs submitted successfully")
                if successful_submissions > 0:
                    print("Use 'squeue -u $USER' to monitor job status")
                    print("Use 'scancel <job_id>' to cancel jobs if needed")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Submit Slurm experiments from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Submit multiple Slurm jobs based on YAML configuration. Each run submits a separate 
Slurm job with the specified parameters. The scripts handle their own Slurm submission.

Example YAML configuration:

experiment_name: "rephrasing_benchmark"
script: "rephrase_dataset"  # Script name or script.py
continue_on_failure: true

fixed_args:
  data_paths: "/path/to/data"
  limit: 1000
  n_tasks: 4
  time: "20:00:00"
  partition: "hopper-cpu"
  gpus: 1

runs:
  - name: "qwen_0.6b_100M_tokens"  # Used as --name argument
    args:
      model_name_or_path: "Qwen/Qwen3-0.6B-FP8"
      
  - name: "llama_1b_100M_tokens"
    args:
      model_name_or_path: "hynky/Llama-3.2-1B-no-bos"
        """
    )
    
    parser.add_argument(
        "config",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without submitting Slurm jobs"
    )
    
    parser.add_argument(
        "--run-names",
        type=str,
        help="Comma-separated list of specific run names to submit"
    )
    
    args = parser.parse_args()
    
    # Parse selected runs
    selected_runs = None
    if args.run_names:
        selected_runs = [name.strip() for name in args.run_names.split(',')]
    
    try:
        launcher = ExperimentLauncher(
            config_path=args.config,
            dry_run=args.dry_run,
            selected_runs=selected_runs
        )
        
        results = launcher.launch()
        
        # Exit with non-zero code if any job submissions failed
        failed_submissions = [name for name, code in results.items() if code != 0]
        if failed_submissions:
            if args.dry_run:
                print(f"\nDry run completed with {len(failed_submissions)} jobs that would fail")
            else:
                print(f"\nExperiment completed with {len(failed_submissions)} failed job submissions")
            sys.exit(1)
        else:
            if args.dry_run:
                print(f"\nDry run completed - all {len(results)} jobs would submit successfully")
            else:
                print(f"\nAll {len(results)} Slurm jobs submitted successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
