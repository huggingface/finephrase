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
def parse_gpu_memory_utilization(log_path: Path) -> Optional[float]:
    """
    Parse vLLM server log to extract gpu_memory_utilization value.
    Looks for patterns like 'gpu_memory_utilization=0.98' or 'gpu_memory_utilization 0.98'.
    """
    try:
        text = log_path.read_text()
    except Exception:
        return None

    # Unified detection: gpu_memory_utilization followed by '=' or space
    import re
    m = re.search(r"gpu_memory_utilization(?:\s*=\s*|\s+)([0-9]*\.?[0-9]+)", text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    return None



def find_best_log_file(results_dir: Path, best_config: Dict) -> Optional[Path]:
    """
    Find the log file corresponding to the best configuration.
    """
    max_num_seqs = best_config['max_num_seqs']
    max_num_batched_tokens = best_config['max_num_batched_tokens']
    
    # Look for the specific config file
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

            # Parse gpu_memory_utilization from vllm server log for this best config
            vllm_log_file = results_file.parent / f"vllm_log_{best_config['max_num_seqs']}_{best_config['max_num_batched_tokens']}.txt"
            gpu_memory_utilization = parse_gpu_memory_utilization(vllm_log_file) if vllm_log_file.exists() else None
            
            # Compute per-TP metrics
            output_token_throughput = metrics.get("Output token throughput (tok/s)", 0)
            total_token_throughput = metrics.get("Total Token throughput (tok/s)", 0)
            
            output_token_throughput_per_tp = output_token_throughput / tp if tp > 0 else 0
            total_token_throughput_per_tp = total_token_throughput / tp if tp > 0 else 0
            
            # Parse length_config into components
            try:
                input_len_str, output_len_str, max_model_len_str = length_config.split('_')
                input_len = int(input_len_str)
                output_len = int(output_len_str)
                max_model_len = int(max_model_len_str)
            except Exception:
                input_len = 0
                output_len = 0
                max_model_len = 0

            # Create experiment record
            experiment = {
                # Model information
                'model': model,
                'tp': tp,
                'length_config': length_config,
                'input_len': input_len,
                'output_len': output_len,
                'max_model_len': max_model_len,
                'max_num_batched_tokens': best_config['max_num_batched_tokens'],
                'gpu_memory_utilization': gpu_memory_utilization if gpu_memory_utilization is not None else 0,
                
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
            
            # Keep best throughput on record for uniform summary printing later
            experiment['best_throughput'] = best_config['best_throughput']
            successful_experiments.append(experiment)
            
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
        'model', 'tp', 'input_len', 'output_len', 'max_model_len', 'max_num_batched_tokens', 'gpu_memory_utilization',
        
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
        writer = csv.DictWriter(csvfile, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(experiments)


def format_summary_line(symbol: str, model: str, tp: int, length_config: str, right_text: str) -> str:
    """Return a uniformly formatted summary line for success/failure lists."""
    return f"{symbol} {model:<30} /tp{tp}/{length_config:<20} - {right_text}"


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
    # Note: we no longer save failed experiments to a file.
    
    args = parser.parse_args()
    
    print(f"Analyzing benchmarking results in: {args.base_path}")
    
    # Analyze results
    successful_experiments, failed_experiments = analyze_benchmarking_results(args.base_path)
    
    # Print successful experiments section (uniform formatting)
    if successful_experiments:
        print(f"\n=== SUCCESSFUL EXPERIMENTS ===")
        # Sort for stable output
        successful_experiments.sort(key=lambda x: (x['model'], x['tp'], x['length_config']))
        for exp in successful_experiments:
            line = format_summary_line(
                symbol="✓",
                model=exp['model'],
                tp=exp['tp'],
                length_config=exp['length_config'],
                right_text=f"throughput: {exp['best_throughput']:>6.2f}",
            )
            print(line)

    # Print failed experiments section (uniform formatting)
    if failed_experiments:
        print(f"\n=== FAILED EXPERIMENTS ===")
        for failed in failed_experiments:
            # Split experiment path from reason for better alignment
            if ' - ' in failed:
                exp_path, reason = failed.split(' - ', 1)
                parts = exp_path.split('/')
                if len(parts) >= 3:
                    model = parts[0]
                    tp_str = parts[1]
                    length_config = parts[2]
                    try:
                        tp = int(tp_str.replace('tp', ''))
                    except Exception:
                        # Fallback if parsing fails
                        print(f"✗ {failed}")
                        continue
                    line = format_summary_line("✗", model, tp, length_config, reason)
                    print(line)
                else:
                    print(f"✗ {failed}")
            else:
                print(f"✗ {failed}")
    
    # Save results
    if successful_experiments:
        # Sort experiments by model, tp, and length_config
        successful_experiments.sort(key=lambda x: (x['model'], x['tp'], x['length_config']))
        save_results_to_csv(successful_experiments, args.output)
        print(f"\n✓ Analysis complete! Results saved to {args.output}")
    else:
        print(f"\n✗ No successful experiments found")
    
    # Generate rerun command for failures
    if failed_experiments:
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
