#!/usr/bin/env python3
"""
SLURM job submission script for VLLM benchmarks.
Submits jobs with different model, tensor parallelism, and length configurations.
"""

import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict


class BenchmarkJobSubmitter:
    def __init__(self, base_dir: str = "/fsx/joel_niklaus/projects/finephrase/vllm_benchmark"):
        self.base_dir = Path(base_dir)
        self.job_dir = self.base_dir / "slurm_jobs"
        self.job_dir.mkdir(exist_ok=True)
        
        # Configuration sets
        self.models = [
            "Qwen/Qwen3-8B-FP8",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-32B",
            "google/gemma-3-270m-it",
            "google/gemma-3-1b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "google/gemma-3-27b-it",
            "microsoft/Phi-4-mini-instruct",
            "microsoft/phi-4",
            "baidu/ERNIE-4.5-0.3B-PT",
            "baidu/ERNIE-4.5-21B-A3B-PT",
        ]
        
        self.tp_values = [1, 2, 4]
        
        # Length configurations: (INPUT_LEN, OUTPUT_LEN, MAX_MODEL_LEN)
        self.length_configs = [
            (2048, 1024 + 512, 4096),     # Short context
            (4096, 2048 + 1024, 8192),    # Medium context
            (8192, 4096 + 2048, 16384),   # Long context  
        ]

    def get_model_name_safe(self, model: str) -> str:
        """Convert model name to filesystem-safe string."""
        return model.replace("/", "_").replace("-", "_")

    def get_slurm_script_content(self, model: str, tp: int, length_config: Tuple[int, int, int]) -> str:
        """Generate SLURM script content for a specific configuration."""
        input_len, output_len, max_model_len = length_config
        model_safe = self.get_model_name_safe(model)
        
        script_content = f"""#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --job-name=vllm_benchmark_{model_safe}_tp{tp}
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={10 * tp}
#SBATCH --mem-per-cpu=20G
#SBATCH --qos=normal
#SBATCH --gres=gpu:{tp}
#SBATCH --output={self.job_dir}/benchmark_%j_{model_safe}_tp{tp}_{input_len}_{output_len}.out
#SBATCH --error={self.job_dir}/benchmark_%j_{model_safe}_tp{tp}_{input_len}_{output_len}.err

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
export NUM_SEQS_LIST="128 256"
export NUM_BATCHED_TOKENS_LIST="512 1024 2048 4096"
export NUM_PROMPTS_MAIN=200
export NUM_PROMPTS_SUB=50
export VLLM_LOGGING_LEVEL="DEBUG"

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
        input_len, output_len, max_model_len = length_config
        model_safe = self.get_model_name_safe(model)
        
        script_name = f"benchmark_{model_safe}_tp{tp}_{input_len}_{output_len}.sh"
        script_path = self.job_dir / script_name
        
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
        
        print(f"Preparing to submit {len(self.models) * len(self.tp_values) * len(self.length_configs)} jobs...")
        print(f"Models: {len(self.models)}")
        print(f"TP values: {self.tp_values}")
        print(f"Length configs: {len(self.length_configs)}")
        print(f"Job directory: {self.job_dir}")
        print("="*60)
        
        for model in self.models:
            for tp in self.tp_values:
                for length_config in self.length_configs:
                    input_len, output_len, max_model_len = length_config
                    
                    # Skip if total length exceeds max model length
                    if input_len + output_len > max_model_len:
                        print(f"Skipping {model} TP={tp} ({input_len}+{output_len}>{max_model_len})")
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
                        print(f"DRY RUN: Would submit {script_path.name}")
                        job_info["job_id"] = "DRY_RUN"
                    else:
                        # Submit job
                        job_id = self.submit_job(script_path)
                        if job_id:
                            job_info["job_id"] = job_id
                            print(f"Submitted job {job_id}: {script_path.name}")
                        else:
                            print(f"Failed to submit: {script_path.name}")
                    
                    submitted_jobs.append(job_info)
        
        return submitted_jobs

    def save_job_summary(self, submitted_jobs: List[Dict]) -> None:
        """Save a summary of submitted jobs to a file."""
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        summary_file = self.job_dir / f"job_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"VLLM Benchmark Jobs Submitted at {timestamp}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total jobs: {len(submitted_jobs)}\n")
            f.write(f"Successful submissions: {len([j for j in submitted_jobs if j['job_id'] and j['job_id'] != 'DRY_RUN'])}\n\n")
            
            f.write("Job Details:\n")
            f.write("-" * 60 + "\n")
            
            for job in submitted_jobs:
                input_len, output_len, max_model_len = job['length_config']
                f.write(f"Job ID: {job['job_id']}\n")
                f.write(f"  Model: {job['model']}\n")
                f.write(f"  TP: {job['tp']}\n")
                f.write(f"  Lengths: {input_len}/{output_len}/{max_model_len}\n")
                f.write(f"  Script: {job['script_path']}\n")
                f.write("\n")
        
        print(f"Job summary saved to: {summary_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Submit VLLM benchmark jobs to SLURM")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Create job scripts but don't submit to SLURM")
    parser.add_argument("--base-dir", default="/fsx/joel_niklaus/projects/finephrase/vllm_benchmark",
                       help="Base directory for benchmark scripts")
    
    args = parser.parse_args()
    
    submitter = BenchmarkJobSubmitter(base_dir=args.base_dir)
    
    print("VLLM Benchmark Job Submitter")
    print(f"Base directory: {submitter.base_dir}")
    print(f"Job scripts directory: {submitter.job_dir}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No jobs will be submitted ***\n")
    
    # Submit all jobs
    submitted_jobs = submitter.submit_all_jobs(dry_run=args.dry_run)
    
    # Save summary
    submitter.save_job_summary(submitted_jobs)
    
    print("\nJob submission completed!")
    
    if not args.dry_run:
        print(f"\nTo monitor jobs, use:")
        print(f"  squeue -u $USER")
        print(f"  ls {submitter.job_dir}/*.out")
        print(f"  ls {submitter.job_dir}/*.err")


if __name__ == "__main__":
    main()
