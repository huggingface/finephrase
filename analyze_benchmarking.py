#!/usr/bin/env python3
"""
Analyze benchmarking results and generate summary tables.

This script scans the benchmarking directory, extracts metrics from successful experiments,
computes per-TP throughput metrics, and saves results as CSV.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import LOG_BASE_PATH


def parse_results_file(results_path: Path) -> Tuple[Optional[Dict], bool]:
    """
    Parse results.txt file to extract best configuration.
    
    Returns:
        Tuple of (best_config_dict, is_failed)
        best_config_dict is None if parsing fails
        is_failed is True if best_throughput is 0
    """
    try:
        with open(results_path, 'r') as f:
            lines = f.readlines()
        
        # Find the best configuration line
        best_line = None
        for line in lines:
            if line.startswith('best_'):
                best_line = line.strip()
                break
        
        if not best_line:
            return None, True
        
        # Parse best configuration
        # Format: best_max_num_seqs: 256, best_num_batched_tokens: 4096, best_throughput: 8.04, profile saved in: ...
        pattern = r'best_max_num_seqs: (\d+), best_num_batched_tokens: (\d+), best_throughput: ([\d.]+)'
        match = re.search(pattern, best_line)
        
        if not match:
            return None, True
        
        max_num_seqs = int(match.group(1))
        max_num_batched_tokens = int(match.group(2))
        best_throughput = float(match.group(3))
        
        # Check if experiment failed
        if best_throughput == 0:
            return {
                'max_num_seqs': max_num_seqs,
                'max_num_batched_tokens': max_num_batched_tokens,
                'best_throughput': best_throughput
            }, True
        
        return {
            'max_num_seqs': max_num_seqs,
            'max_num_batched_tokens': max_num_batched_tokens,
            'best_throughput': best_throughput
        }, False
        
    except Exception as e:
        print(f"Error parsing {results_path}: {e}")
        return None, True


def parse_benchmark_log(log_path: Path) -> Optional[Dict]:
    """
    Parse benchmark log file to extract metrics from "Serving Benchmark Result" table.
    """
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # Find the Serving Benchmark Result section
        start_idx = None
        end_idx = None
        
        for i, line in enumerate(lines):
            if "============ Serving Benchmark Result ============" in line:
                start_idx = i
            elif start_idx is not None and "==================================================" in line:
                end_idx = i
                break
        
        if start_idx is None or end_idx is None:
            return None
        
        # Extract metrics from the table
        metrics = {}
        for i in range(start_idx + 1, end_idx):
            line = lines[i].strip()
            if ':' in line:
                # Split on the last colon to handle cases like "Time per Output Token (excl. 1st token)"
                parts = line.rsplit(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    
                    # Try to convert to float
                    try:
                        value = float(value_str)
                        metrics[key] = value
                    except ValueError:
                        # Keep as string if not a number
                        metrics[key] = value_str
        
        return metrics
        
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None


def find_best_log_file(results_dir: Path, best_config: Dict) -> Optional[Path]:
    """
    Find the log file corresponding to the best configuration.
    """
    max_num_seqs = best_config['max_num_seqs']
    max_num_batched_tokens = best_config['max_num_batched_tokens']
    
    # First try the BEST_PROFILE file
    best_profile_path = results_dir / "bm_log_BEST_PROFILE.txt"
    if best_profile_path.exists():
        return best_profile_path
    
    # Otherwise, look for the specific config file
    pattern = f"bm_log_{max_num_seqs}_{max_num_batched_tokens}_requestrate_inf.txt"
    log_path = results_dir / pattern
    
    if log_path.exists():
        return log_path
    
    # If exact match not found, list available files for debugging
    log_files = list(results_dir.glob("bm_log_*.txt"))
    print(f"Warning: Could not find log file for config {max_num_seqs}_{max_num_batched_tokens}")
    print(f"Available log files: {[f.name for f in log_files]}")
    
    return None


def extract_experiment_info(experiment_path: Path) -> Tuple[str, int, str]:
    """
    Extract model name, TP, and length configuration from experiment path.
    
    Path format: .../benchmarking/{model}/{tp}/{length_config}/results/
    """
    parts = experiment_path.parts
    
    # Find the benchmarking directory and extract info
    benchmarking_idx = None
    for i, part in enumerate(parts):
        if part == "benchmarking":
            benchmarking_idx = i
            break
    
    if benchmarking_idx is None or len(parts) < benchmarking_idx + 4:
        raise ValueError(f"Invalid experiment path: {experiment_path}")
    
    model = parts[benchmarking_idx + 1]
    tp_str = parts[benchmarking_idx + 2]
    length_config = parts[benchmarking_idx + 3]
    
    # Extract TP number
    tp = int(tp_str.replace('tp', ''))
    
    return model, tp, length_config


def analyze_benchmarking_results(base_path: Path) -> Tuple[List[Dict], List[str]]:
    """
    Analyze all benchmarking results.
    
    Returns:
        Tuple of (successful_experiments, failed_experiments)
    """
    successful_experiments = []
    failed_experiments = []
    
    benchmarking_path = base_path / "benchmarking"
    
    if not benchmarking_path.exists():
        print(f"Benchmarking directory not found: {benchmarking_path}")
        return successful_experiments, failed_experiments
    
    # Find all results.txt files
    results_files = list(benchmarking_path.glob("*/*/*/results/result.txt"))
    
    print(f"Found {len(results_files)} experiment result files")
    
    for results_file in results_files:
        try:
            # Extract experiment information
            model, tp, length_config = extract_experiment_info(results_file.parent)
            
            # Parse results file
            best_config, is_failed = parse_results_file(results_file)
            
            if best_config is None:
                failed_experiments.append(f"{model}/tp{tp}/{length_config} - Failed to parse results")
                continue
            
            if is_failed:
                failed_experiments.append(f"{model}/tp{tp}/{length_config} - best_throughput: {best_config['best_throughput']}")
                continue
            
            # Find and parse the best log file
            log_file = find_best_log_file(results_file.parent, best_config)
            if log_file is None:
                failed_experiments.append(f"{model}/tp{tp}/{length_config} - Log file not found")
                continue
            
            metrics = parse_benchmark_log(log_file)
            if metrics is None:
                failed_experiments.append(f"{model}/tp{tp}/{length_config} - Failed to parse log file")
                continue
            
            # Compute per-TP metrics
            output_token_throughput = metrics.get("Output token throughput (tok/s)", 0)
            total_token_throughput = metrics.get("Total Token throughput (tok/s)", 0)
            
            output_token_throughput_per_tp = output_token_throughput / tp if tp > 0 else 0
            total_token_throughput_per_tp = total_token_throughput / tp if tp > 0 else 0
            
            # Create experiment record
            experiment = {
                # Model information
                'model': model,
                'tp': tp,
                'length_config': length_config,
                'max_num_batched_tokens': best_config['max_num_batched_tokens'],
                
                # Per-TP throughput metrics
                'output_token_throughput_per_tp': output_token_throughput_per_tp,
                'total_token_throughput_per_tp': total_token_throughput_per_tp,
                
                # Absolute throughput metrics
                'output_token_throughput': output_token_throughput,
                'total_token_throughput': total_token_throughput,
                
                # All other metrics from benchmark result (excluding Request goodput)
                'successful_requests': metrics.get("Successful requests", 0),
                'benchmark_duration': metrics.get("Benchmark duration (s)", 0),
                'total_input_tokens': metrics.get("Total input tokens", 0),
                'total_generated_tokens': metrics.get("Total generated tokens", 0),
                'request_throughput': metrics.get("Request throughput (req/s)", 0),
                'mean_ttft': metrics.get("Mean TTFT (ms)", 0),
                'median_ttft': metrics.get("Median TTFT (ms)", 0),
                'p99_ttft': metrics.get("P99 TTFT (ms)", 0),
                'mean_tpot': metrics.get("Mean TPOT (ms)", 0),
                'median_tpot': metrics.get("Median TPOT (ms)", 0),
                'p99_tpot': metrics.get("P99 TPOT (ms)", 0),
                'mean_itl': metrics.get("Mean ITL (ms)", 0),
                'median_itl': metrics.get("Median ITL (ms)", 0),
                'p99_itl': metrics.get("P99 ITL (ms)", 0),
                'mean_e2el': metrics.get("Mean E2EL (ms)", 0),
                'median_e2el': metrics.get("Median E2EL (ms)", 0),
                'p99_e2el': metrics.get("P99 E2EL (ms)", 0),
            }
            
            successful_experiments.append(experiment)
            print(f"✓ {model:<30}/tp{tp}/{length_config} - throughput: {best_config['best_throughput']:.2f}")
            
        except Exception as e:
            failed_experiments.append(f"{results_file} - Error: {e}")
            print(f"✗ Error processing {results_file}: {e}")
    
    return successful_experiments, failed_experiments


def save_results_to_csv(experiments: List[Dict], output_path: Path):
    """
    Save experiment results to CSV file.
    """
    if not experiments:
        print("No successful experiments to save")
        return
    
    # Define column order as specified
    columns = [
        # Model information
        'model', 'tp', 'length_config', 'max_num_batched_tokens',
        
        # Per-TP throughput metrics
        'output_token_throughput_per_tp', 'total_token_throughput_per_tp',
        
        # Absolute throughput metrics
        'output_token_throughput', 'total_token_throughput',
        
        # All other metrics from benchmark result
        'successful_requests', 'benchmark_duration', 'total_input_tokens', 
        'total_generated_tokens', 'request_throughput',
        'mean_ttft', 'median_ttft', 'p99_ttft',
        'mean_tpot', 'median_tpot', 'p99_tpot',
        'mean_itl', 'median_itl', 'p99_itl',
        'mean_e2el', 'median_e2el', 'p99_e2el',
    ]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(experiments)


def save_failed_experiments(failed_experiments: List[str], output_path: Path):
    """
    Save failed experiments list to a text file for easy rerunning.
    """
    if not failed_experiments:
        print("No failed experiments to save")
        return
    
    with open(output_path, 'w') as f:
        f.write("# Failed Benchmarking Experiments\n")
        f.write("# Format: model/tp/length_config - reason\n")
        f.write("# Use this list to identify experiments to rerun\n\n")
        
        for failed in failed_experiments:
            f.write(f"{failed}\n")


def generate_rerun_command(failed_experiments: List[str]) -> str:
    """
    Generate a benchmark-vllm command to rerun failed experiments.
    """
    if not failed_experiments:
        return ""
    
    # Extract experiment configs from failed experiment strings
    experiment_configs = []
    for failed in failed_experiments:
        # Split on ' - ' to separate the config from the reason
        parts = failed.split(' - ', 1)
        if len(parts) >= 1:
            config = parts[0].strip()
            # Only include valid experiment configs (model/tp/length_config format)
            # Should have exactly 2 slashes
            if config.count('/') == 2 and not config.startswith('/'):
                experiment_configs.append(config)
    
    if not experiment_configs:
        return ""
    
    # Join configs with commas
    experiments_str = ','.join(experiment_configs)
    
    return f"benchmark-vllm --experiments {experiments_str}"


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmarking results")
    parser.add_argument(
        "--base-path", 
        type=Path,
        default=Path(LOG_BASE_PATH),
        help=f"Base path for logs (default: {LOG_BASE_PATH})"
    )
    parser.add_argument(
        "--output", 
        type=Path,
        default="benchmarking_results.csv",
        help="Output CSV file path (default: benchmarking_results.csv)"
    )
    parser.add_argument(
        "--failed-output", 
        type=Path,
        default="failed_experiments.txt",
        help="Output file for failed experiments list (default: failed_experiments.txt)"
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing benchmarking results in: {args.base_path}")
    
    # Analyze results
    successful_experiments, failed_experiments = analyze_benchmarking_results(args.base_path)
    
    if failed_experiments:
        print(f"\n=== FAILED EXPERIMENTS ===")
        for failed in failed_experiments:
            print(f"  {failed}")
    
    # Save results
    if successful_experiments:
        save_results_to_csv(successful_experiments, args.output)
        print(f"\n✓ Analysis complete! Results saved to {args.output}")
    else:
        print(f"\n✗ No successful experiments found")
    
    # Save failed experiments list
    if failed_experiments:
        save_failed_experiments(failed_experiments, args.failed_output)
        print(f"✓ Failed experiments list saved to {args.failed_output}")
        
        # Generate rerun command
        rerun_command = generate_rerun_command(failed_experiments)
        if rerun_command:
            print(f"\n=== RERUN COMMAND ===")
            print(f"To rerun all failed experiments, use:")
            print(f"{rerun_command}")

    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Successful experiments: {len(successful_experiments)}")
    print(f"Failed experiments: {len(failed_experiments)}")


if __name__ == "__main__":
    main()
