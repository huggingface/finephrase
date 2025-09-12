#!/usr/bin/env python3
"""
SLURM job submission script for VLLM benchmarks.
Submits jobs with different model, tensor parallelism, and length configurations.
"""

import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv

load_dotenv() # Load the HF_TOKEN for gated models

from utils import LOG_BASE_PATH

# Configuration constants
DEFAULT_MODELS = [
    #"Qwen/Qwen3-0.6B",
    #"Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B-Base",
    #"Qwen/Qwen3-8B",
    #"Qwen/Qwen3-14B",
    #"Qwen/Qwen3-32B",
    #"Qwen/Qwen3-30B-A3B",
    #"google/gemma-3-270m-it",
    #"google/gemma-3-1b-it",
    #"google/gemma-3-4b-it",
    #"google/gemma-3-12b-it",
    #"google/gemma-3-27b-it",

    #"microsoft/Phi-4-mini-instruct",
    #"microsoft/phi-4",
    #"baidu/ERNIE-4.5-0.3B-PT",
    #"baidu/ERNIE-4.5-21B-A3B-PT",
]

DEFAULT_TP_VALUES = [1, 2, 4]

# Length configurations: (INPUT_LEN, OUTPUT_LEN, MAX_MODEL_LEN)
DEFAULT_LENGTH_CONFIGS = [
    (2048, 1024 + 512, 4096),     # Short context
    (4096, 2048 + 1024, 8192),    # Medium context
    (8192, 4096 + 2048, 16384),   # Long context
]

MAX_NUM_SEQS_LIST = "256"  # Throughput is not very sensitive to this parameter
MAX_NUM_BATCHED_TOKENS_LIST = "512 1024 2048 4096" # Cannot be larger than max model length


class BenchmarkJobSubmitter:
    def __init__(self, base_dir: str = "/fsx/joel_niklaus/projects/finephrase/vllm_benchmark", 
                 experiments: List[str] = None):
        self.base_dir = Path(base_dir)
        
        # Log base path from utils
        self.log_base_path = Path(LOG_BASE_PATH)
        
        # Configuration sets - either from specific experiments or defaults
        if experiments:
            self.experiment_configs = self.parse_experiments(experiments)
        else:
            # Use default sweep configuration
            self.experiment_configs = self.generate_default_experiments()
    
    def parse_experiments(self, experiments: List[str]) -> List[Dict]:
        """
        Parse experiment strings into configuration dictionaries.
        Format: model/tp/length_config
        Example: google_gemma_3_27b_it/tp1/8192_6144_16384
        """
        configs = []
        for exp in experiments:
            try:
                parts = exp.strip().split('/')
                if len(parts) != 3:
                    print(f"Warning: Invalid experiment format '{exp}', expected 'model/tp/length_config'")
                    continue
                
                model_safe, tp_str, length_config_str = parts
                
                # Convert model name back from safe format
                model = model_safe.replace('_', '/', 1).replace('_', '-')
                
                # Extract TP value
                if not tp_str.startswith('tp'):
                    print(f"Warning: Invalid TP format '{tp_str}' in experiment '{exp}'")
                    continue
                tp = int(tp_str[2:])
                
                # Parse length config
                length_parts = length_config_str.split('_')
                if len(length_parts) != 3:
                    print(f"Warning: Invalid length config format '{length_config_str}' in experiment '{exp}'")
                    continue
                
                input_len = int(length_parts[0])
                output_len = int(length_parts[1])
                max_model_len = int(length_parts[2])
                length_config = (input_len, output_len, max_model_len)
                
                configs.append({
                    'model': model,
                    'tp': tp,
                    'length_config': length_config
                })
                
            except Exception as e:
                print(f"Error parsing experiment '{exp}': {e}")
                continue
        
        return configs
    
    def generate_default_experiments(self) -> List[Dict]:
        """Generate default experiment configurations from constants."""
        configs = []
        for model in DEFAULT_MODELS:
            for tp in DEFAULT_TP_VALUES:
                for length_config in DEFAULT_LENGTH_CONFIGS:
                    configs.append({
                        'model': model,
                        'tp': tp,
                        'length_config': length_config
                    })
        return configs

    def get_model_name_safe(self, model: str) -> str:
        """Convert model name to filesystem-safe string."""
        return model.replace("/", "_").replace("-", "_")

    def get_log_dir_path(self, model: str, tp: int, length_config: Tuple[int, int, int]) -> Path:
        """Generate log directory path for a specific configuration."""
        input_len, output_len, max_model_len = length_config
        model_safe = self.get_model_name_safe(model)
        
        # Create directory structure: LOG_BASE_PATH/benchmarking/{model}/tp{n}/{length_config}
        length_config_str = f"{input_len}_{output_len}_{max_model_len}"
        log_dir = self.log_base_path / "benchmarking" / model_safe / f"tp{tp}" / length_config_str
        
        # Create directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create subdirectories for SLURM output files and results
        slurm_logs_dir = log_dir / "slurm_logs"
        slurm_logs_dir.mkdir(exist_ok=True)
        
        results_dir = log_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        return log_dir

    def get_slurm_script_content(self, model: str, tp: int, length_config: Tuple[int, int, int]) -> str:
        """Generate SLURM script content for a specific configuration."""
        input_len, output_len, max_model_len = length_config
        model_safe = self.get_model_name_safe(model)
        log_dir = self.get_log_dir_path(model, tp, length_config)
        
        script_content = f"""#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --job-name=vllm_benchmark_{model_safe}_tp{tp}
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={10 * tp}
#SBATCH --mem-per-cpu=20G
#SBATCH --qos=normal
#SBATCH --gres=gpu:{tp}
#SBATCH --output={log_dir}/slurm_logs/%j.out

# Environment setup
export BASE="{self.base_dir}"
export MODEL="{model}"
export SYSTEM="GPU"
export TP={tp}
export DOWNLOAD_DIR=""
export INPUT_LEN={input_len}
export OUTPUT_LEN={output_len}
export MAX_MODEL_LEN={max_model_len}
export MIN_CACHE_HIT_PCT=5
export MAX_LATENCY_ALLOWED_MS=100000000000
export NUM_SEQS_LIST="{MAX_NUM_SEQS_LIST}"
export NUM_BATCHED_TOKENS_LIST="{MAX_NUM_BATCHED_TOKENS_LIST}"
export NUM_PROMPTS_MAIN=200
export NUM_PROMPTS_SUB=200
export VLLM_LOGGING_LEVEL="DEBUG"
export LOG_FOLDER="{log_dir}/results"

# Job info
echo "Starting benchmark job:"
echo "Model: $MODEL"
echo "Tensor Parallelism: $TP"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN" 
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "================================"

# Change to benchmark directory
cd "{self.base_dir}"

# Run the benchmark
bash autotune.sh

echo "================================"
echo "Job completed at: $(date)"
"""
        return script_content

    def create_job_script(self, model: str, tp: int, length_config: Tuple[int, int, int]) -> Path:
        """Create a SLURM job script file for the given configuration."""
        log_dir = self.get_log_dir_path(model, tp, length_config)
        script_path = log_dir / "launch_script.slurm"
        
        script_content = self.get_slurm_script_content(model, tp, length_config)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        return script_path

    def submit_job(self, script_path: Path) -> str:
        """Submit a job script to SLURM and return the job ID."""
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            # Extract job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job {script_path}: {e}")
            print(f"Error output: {e.stderr}")
            return None

    def submit_all_jobs(self, dry_run: bool = False) -> List[Dict]:
        """Submit all benchmark jobs with different configurations."""
        submitted_jobs = []
        
        print(f"Preparing to submit {len(self.experiment_configs)} jobs...")
        print(f"Experiments: {len(self.experiment_configs)}")
        print(f"Log base path: {self.log_base_path}/benchmarking")
        print("="*60)
        
        for config in self.experiment_configs:
            model = config['model']
            tp = config['tp']
            length_config = config['length_config']
            input_len, output_len, max_model_len = length_config
            
            # Skip if total length exceeds max model length
            if input_len + output_len > max_model_len:
                print(f"Skipping {model:<30} | TP={tp} | ({input_len}+{output_len}>{max_model_len})")
                continue
            
            # Create job script
            script_path = self.create_job_script(model, tp, length_config)
            
            job_info = {
                "model": model,
                "tp": tp,
                "length_config": length_config,
                "script_path": str(script_path),
                "job_id": None
            }
            
            if dry_run:
                print(f"DRY RUN: Would submit {model:<30} | TP={tp} | {input_len}/{output_len}/{max_model_len} → {script_path}")
                job_info["job_id"] = "DRY_RUN"
            else:
                # Submit job
                job_id = self.submit_job(script_path)
                if job_id:
                    job_info["job_id"] = job_id
                    print(f"Submitted job {job_id}: {model:<30} | TP={tp} | {input_len}/{output_len}/{max_model_len}")
                else:
                    print(f"Failed to submit: {model:<30} | TP={tp} | {input_len}/{output_len}/{max_model_len}")
            
            submitted_jobs.append(job_info)
        
        return submitted_jobs




def main():
    parser = argparse.ArgumentParser(description="Submit VLLM benchmark jobs to SLURM")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Create job scripts but don't submit to SLURM")
    parser.add_argument("--base-dir", default="/fsx/joel_niklaus/projects/finephrase/vllm_benchmark",
                       help="Base directory for benchmark scripts")
    parser.add_argument("--experiments", type=str,
                       help="Comma-separated list of experiments to run (format: model/tp/length_config)")
    
    args = parser.parse_args()
    
    # Parse experiments list if provided
    experiments = None
    if args.experiments:
        experiments = [exp.strip() for exp in args.experiments.split(',')]
        print(f"Running specific experiments: {experiments}")
    
    submitter = BenchmarkJobSubmitter(base_dir=args.base_dir, experiments=experiments)
    
    print("VLLM Benchmark Job Submitter")
    print(f"Base directory: {submitter.base_dir}")
    print(f"Log directory: {submitter.log_base_path}/benchmarking")
    
    if experiments:
        print(f"Running {len(experiments)} specific experiments")
    else:
        print("Running default sweep configuration")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No jobs will be submitted ***\n")
    
    submitted_jobs = submitter.submit_all_jobs(dry_run=args.dry_run)
    
    print(f"\nCompleted! {len(submitted_jobs)} jobs processed.")


if __name__ == "__main__":
    main()
