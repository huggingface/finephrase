#!/usr/bin/env python3
import os
import argparse
import random
import subprocess
import yaml
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from utils import BASE_PATH, FAULTY_NODES, LOG_BASE_PATH, PROJECT_NAME, PROJECT_PATH, S3_BASE_PATH, LOCAL_TMP_PATH_ON_NODE

EVAL_LOGS_PATH = f"{LOG_BASE_PATH}/evals"
TRAINING_RUNS_PATH = f"{LOG_BASE_PATH}/training"

NANOTRON_PATH = f"{PROJECT_PATH}/nanotron"
S5CMD_PATH = f"{PROJECT_PATH}/.venv/bin/s5cmd"

S3_CHECKPOINTS_PREFIX = f"{S3_BASE_PATH}/checkpoints"
EVALS_OUTPUT_PATH = f"{S3_BASE_PATH}/evals-test"

TASKS_PATH = f"{PROJECT_PATH}/tasks.txt"
TASK_LIST_PATH = f"{PROJECT_PATH}/task_list.py"

NUM_GPUS = 8
NUM_CPUS_IN_NODE = 88

GLOBAL_BATCH_SIZE = 512
MICRO_BATCH_SIZE = 2 # We cannot fit more with the current setup

SEQUENCE_LENGTH = 4096

DEFAULT_SEED_VALUE = 6

def launch_slurm_job(launch_file_contents, job_id, nodes, background, name, timestamp, *args):
    """
    Small helper function to save a sbatch script and call it.
    Args:
        launch_file_contents: Contents of the sbatch script
        *args: any other arguments to pass to the sbatch command

    Returns: the id of the launched slurm job
    """
    run_dir = f"{TRAINING_RUNS_PATH}/{name}"
    slurm_logs_dir = f"{run_dir}/slurm_logs"
    with open(f"{run_dir}/launch_script.slurm", "w") as f:
        f.write(launch_file_contents)
        f.flush()
      
    # Make the script executable
    os.chmod(f.name, 0o755)
    if job_id:
      srun_args = ["srun", "--jobid", job_id, "--ntasks-per-node", "1", "--nodes", str(nodes)]
      if background:
        srun_args += ["--output", f"{slurm_logs_dir}/train-{timestamp}.out", "--error", f"{slurm_logs_dir}/train-{timestamp}.err"]
        # Run the job in background - DOES NOT WAIT
        subprocess.Popen(srun_args + list(args) + [f.name])
        subprocess
        print(f"Running in background. Logs: {slurm_logs_dir}/train-{timestamp}.out {slurm_logs_dir}/train-{timestamp}.err")
      else:
        # Run in foreground till job is done - WAITS
        subprocess.check_call(srun_args + list(args) + [f.name])
    else:
      return subprocess.run(["sbatch", *args, f.name], capture_output=True, text=True).stdout.split()[-1]


parser = argparse.ArgumentParser(description="Launch training job with updated configuration")
parser.add_argument("--data", help="Dataset folder paths (can be S3 path)", type=str, required=True)
parser.add_argument("--name", help="Run name", type=str, required=True)
parser.add_argument("--tokenizer", help="Tokenizer name or path", type=str, default="hynky/Llama-3.2-1B-no-bos")
parser.add_argument("--seed", help="Seed", type=int, default=DEFAULT_SEED_VALUE)
parser.add_argument("--data-seed", help="Data seed", type=int, default=DEFAULT_SEED_VALUE)
parser.add_argument("--train-steps", help="Training steps", type=int, default=10_000)
parser.add_argument("--qos", help="QoS to use", type=str, default="normal")
parser.add_argument("--nodes", help="Number of nodes", type=int, default=8)
parser.add_argument("--debug", help="Enable debug mode", action="store_true")
parser.add_argument("--job-id", help="Job ID", type=str, default=None)
parser.add_argument("--lr", help="Learning rate", type=float, default=5e-4)
parser.add_argument("--lr-schedule", help="Learning rate schedule", type=str, default="wsd")
parser.add_argument("--background", help="Run in background", action="store_true")
parser.add_argument("--reservation", help="SLURM reservation name", type=str, default=None)
parser.add_argument("--time", help="SLURM time", type=str, default="1-00:00:00") # It should finish within 3 hours already with 8 nodes
parser.add_argument("--resume-checkpoint-path", help="Path to the checkpoint to resume from", type=str, default=None)
parser.add_argument("--decay-exp", help="Run a decay experiment", action="store_true")
parser.add_argument("--dep-job-id", help="Dependency job", type=str, default=None)


def main():
    args = parser.parse_args()

    # Debug mode settings
    if args.debug:
        args.nodes = 1 # Only use one node for debugging

    # For decay experiments, we need to set up two data stages
    if args.decay_exp:
      # Hard code to the lq checkpoint because we see larger differences there
      args.resume_checkpoint_path = "s3://finephrase/experiments/checkpoints/fw_edu_lq/9000/"
      # Extract step number from checkpoint path (e.g., "path/9000" -> 9000)
      checkpoint_step = int(args.resume_checkpoint_path.rstrip('/').split('/')[-1])
      assert args.lr_schedule == "wsd", "LR schedule must be wsd for decay experiments"

    
    # batch size == batch_accumulation_per_replica * micro_batch_size * dp: 4 * 2 * 64 = 512
    batch_accumulation_per_replica = GLOBAL_BATCH_SIZE // (MICRO_BATCH_SIZE * NUM_GPUS * args.nodes)
    print(f"Training on {args.nodes} nodes with {NUM_GPUS} GPUs per node and {batch_accumulation_per_replica} batch accumulation per replica")
    batch_size = batch_accumulation_per_replica * MICRO_BATCH_SIZE * NUM_GPUS * args.nodes
    assert batch_size == GLOBAL_BATCH_SIZE, f"Batch size {batch_size} is not equal to global batch size {GLOBAL_BATCH_SIZE}"
    tokens_per_step = SEQUENCE_LENGTH * batch_size # sequence length * batch size: 4096 * 512 = 2097152
    print(f"Batch size: {batch_size} examples / {tokens_per_step} tokens")
    total_tokens_consumed = round(tokens_per_step * args.train_steps / 1e9) # in billions
    print(f"Total tokens consumed: {total_tokens_consumed}B") # 8 GPUs, 8 nodes, 10K steps: 20.97152BT
    
    # Naming convention: {stage1_data}-decay-{stage2_data}-seed-{data_seed*100+seed}
    name = args.name.replace(" ", "_") 
    if args.data_seed != DEFAULT_SEED_VALUE and args.seed != DEFAULT_SEED_VALUE: # Only add them if they are not the default values
      name += f"-seed-{(args.data_seed * 100) + args.seed}"
    
    # Parse multiple comma-separated data paths
    data_paths = [path.strip() for path in args.data.split(",")]
    
    # Calculate local dataset paths if using S3
    local_dataset_paths = []
    dataset_folders = []
    for data_path in data_paths:
        if data_path.startswith("s3://"):
            # Extract the last item from the path (e.g., s3://.../my-dataset/ -> my-dataset)
            dataset_name = data_path.rstrip('/').split('/')[-1]
            local_path = f"{LOCAL_TMP_PATH_ON_NODE}/dataset/{dataset_name}/"
            local_dataset_paths.append((data_path, local_path))
            dataset_folders.append(local_path)
        else:
            dataset_folders.append(data_path)

    warmup_steps = round(args.train_steps * 0.01) # Warmup for 1% of training steps
    min_decay_lr = f"{args.lr / 10:.8f}"

    if args.lr_schedule == "wsd":
      lr_decay_style = "linear"
      decay_start_step = round(args.train_steps * 0.9) # Decaying for 10% of training steps
    elif args.lr_schedule == "cosine":
      lr_decay_style = "cosine"
      decay_start_step = warmup_steps
    else:
      raise ValueError(f"Unsupported learning rate schedule: {args.lr_schedule}")

    # Build data_stages configuration
    # Helper function to create a stage config
    def make_stage(name, start_step, folders):
        num_folders = len(folders)
        weights = [1.0 / num_folders] * num_folders
        folders_str = ", ".join(folders)
        weights_str = ", ".join([str(w) for w in weights])
        return f"""- data:
    dataset:
      dataset_folder: [{folders_str}]
      dataset_weights: [{weights_str}]
      pad_samples_to_global_batch_size: false
      return_positions: true
      token_size_in_bytes: 4
      use_old_brrr_dataloader: false
      tokenizer_name: {args.tokenizer}
      vocab_size: 128256
    num_loading_workers: 0
    seed: {args.data_seed}
  name: {name}
  start_training_step: {start_step}"""
    
    # For decay experiments, we need TWO stages:
    # 1. First stage at step 1 (satisfies nanotron's requirement, won't be used since we resume from step 9000)
    # 2. Second stage at checkpoint_step + 1 (the new dataset for decay phase)
    # Note: Both stages point to the same dataset since the first stage is never actually used
    if args.decay_exp:
        dummy_stage = make_stage("warmup_stable_phase", 1, dataset_folders)
        decay_stage = make_stage("decay_phase", checkpoint_step + 1, dataset_folders)
        data_stages_config = f"{dummy_stage}\n{decay_stage}"
    else:
        data_stages_config = make_stage("stable", 1, dataset_folders)

    MODEL_CONFIG = f"""
checkpoints:
  checkpoint_interval: 500
  checkpoints_path: {LOCAL_TMP_PATH_ON_NODE}/checkpoints/{name}
  checkpoints_path_is_shared_file_system: false
  load_lr_scheduler: true
  load_optimizer: true
  resume_checkpoint_path: {args.resume_checkpoint_path if args.resume_checkpoint_path else "null"}
  save_final_state: true
  save_initial_state: false
data_stages:
{data_stages_config}
general:
  benchmark_csv_path: null
  consumed_train_samples: null
  ignore_sanity_checks: true
  project: {PROJECT_NAME}
  run: {name}
  seed: {args.seed}
  step: null
logging:
  iteration_step_info_interval: 5
  log_level: info
  log_level_replica: info
model:
  ddp_bucket_cap_mb: 50
  dtype: bfloat16
  init_method:
    std: 0.02
  make_vocab_size_divisible_by: 1
  model_config:
    _attn_implementation: flash_attention_2
    _fused_rms_norm: true
    _fused_rotary_emb: true
    _use_doc_masking: true
    _use_qkv_packed: true
    attention_bias: false
    bos_token_id: 128000
    eos_token_id: 128001
    flex_attention_mask: null
    hidden_act: silu
    hidden_size: 2048
    initializer_range: 0.02
    intermediate_size: 6144
    is_qwen2_config: true
    max_position_embeddings: {SEQUENCE_LENGTH}
    moe_config: null
    num_attention_heads: 16
    num_hidden_layers: 28
    num_key_value_heads: 8
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-06
    rope_interleaved: false
    rope_scaling: null
    rope_theta: 10000
    sliding_window_size: null
    tie_word_embeddings: true
    use_cache: true
    vocab_size: 128256
    z_loss_coefficient: 1.0e-05
    z_loss_enabled: false
    no_rope_layer: null
optimizer:
  accumulate_grad_in_fp32: true
  clip_grad: 1.0
  learning_rate_scheduler:
    learning_rate: {args.lr}
    lr_decay_starting_step: {decay_start_step}
    lr_decay_steps: {args.train_steps - decay_start_step}
    lr_decay_style: {lr_decay_style}
    lr_warmup_steps: {warmup_steps}
    lr_warmup_style: linear
    min_decay_lr: {min_decay_lr}
  optimizer_factory:
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW
    torch_adam_is_fused: true
  weight_decay: 0.1
  weight_decay_exclude_named_params:
  - .*token_embedding.*
  zero_stage: 0
parallelism:
  context_parallel_size: 1
  dp: {args.nodes * NUM_GPUS}
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  recompute_layer: false
  tp: 1
  tp_linear_async_communication: true
  tp_mode: REDUCE_SCATTER
  tp_recompute_allgather: true
profiler: null
s3_upload:
  remove_after_upload: true
  s5cmd_concurrency: 10
  s5cmd_numworkers: 32
  s5cmd_path: {S5CMD_PATH}
  upload_s3_path: {S3_CHECKPOINTS_PREFIX}/{name}
tokenizer:
  tokenizer_max_length: {SEQUENCE_LENGTH}
  tokenizer_name_or_path: {args.tokenizer}
  tokenizer_revision: null
metrics_logging:
  log_level: 1
  log_detail_interval: 200
tokens:
  batch_accumulation_per_replica: {batch_accumulation_per_replica}
  limit_test_batches: 0
  limit_val_batches: 0
  micro_batch_size: {MICRO_BATCH_SIZE}
  sequence_length: {SEQUENCE_LENGTH}
  train_steps: {args.train_steps}
  val_check_interval: 0
lighteval:
  output_dir: {EVALS_OUTPUT_PATH}
  logs_path: {EVAL_LOGS_PATH}
  local_checkpoint_dir: {LOCAL_TMP_PATH_ON_NODE}/evals-ckpt
  upload_to_wandb: false
  eval_interval: 500
  eval_interval_file: null
  nanotron_path: {NANOTRON_PATH}
  batch_size: 8
  slurm:
    gpus_per_node: {NUM_GPUS}
    hf_cache: "{BASE_PATH}/.cache/huggingface"
    partition: "hopper-prod"
    cpus_per_task: {11*NUM_GPUS}
    qos: "normal"
    time: "1:00:00"
  tasks:
    tasks: {TASKS_PATH}
    custom_tasks: {TASK_LIST_PATH}
    max_samples: 1000
"""

    # Load the config
    config = yaml.safe_load(MODEL_CONFIG)

    
    # Save the updated config
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"{TRAINING_RUNS_PATH}/{name}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/slurm_logs", exist_ok=True)
    config_path_yaml = f"{run_dir}/config.yaml"
    
    with open(config_path_yaml, "w") as f:
        yaml.dump(config, f)
    
    # Build dataset download command if needed
    dataset_download_cmd = ""
    run_cmd = "" if args.job_id else "srun"
    if local_dataset_paths:
        dataset_download_cmd = f"{run_cmd} rm -rf {LOCAL_TMP_PATH_ON_NODE}/dataset\n"
        for s3_path, local_path in local_dataset_paths:
            dataset_download_cmd += f"{run_cmd} {S5CMD_PATH} cp '{s3_path.removesuffix('/')}/*' {local_path}\n"
        dataset_download_cmd += "# "
    
    # Build SLURM job script
    job_name = name
    
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node=1  # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task={NUM_CPUS_IN_NODE}
#SBATCH --gres=gpu:{NUM_GPUS}
#SBATCH --partition=hopper-prod
#SBATCH --output={run_dir}/slurm_logs/train-{timestamp}-%x-%j
#SBATCH --qos={args.qos}
#SBATCH --begin=now+0minutes
#SBATCH --time={args.time}
#SBATCH --exclusive
#SBATCH --exclude={FAULTY_NODES}
{"#SBATCH --dependency=afterok:" + args.dep_job_id if args.dep_job_id else ""}
{"#SBATCH --reservation=" + args.reservation if args.reservation else ""}

set -x -e

echo "START TIME: $(date)"
secs_to_human(){{
    echo "$(( ${{1}} / 3600 )):$(( (${{1}} / 60) % 60 )):$(( ${{1}} % 60 ))"
}}
start=$(date +%s)
echo "$(date -d @${{start}} "+%Y-%m-%d %H:%M:%S"): ${{SLURM_JOB_NAME}} start id=${{SLURM_JOB_ID}}"

{dataset_download_cmd}

# SLURM setup
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT={1024 + random.randint(0, 64511)}
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export TMPDIR={LOCAL_TMP_PATH_ON_NODE}
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module load cuda/12.4

hf auth login --token {os.getenv('HF_TOKEN')}
hf auth whoami

echo go $COUNT_NODE
echo $HOSTNAMES

CMD=" \
    {NANOTRON_PATH}/run_train.py \
    --config-file {config_path_yaml}
    "
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node {NUM_GPUS} \
    --nnodes $COUNT_NODE \
    --rdzv-backend c10d \
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv-id $SLURM_JOB_ID \
    --node_rank $SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --max_restarts 0 \
    --tee 3 \
    "

# Add small random delay to avoid concurrent hub requests
random_milliseconds=$(( RANDOM % 1001 ))
sleep_time=$(bc <<< "scale=3; $random_milliseconds / 1000")
echo "Sleeping for $sleep_time seconds..."
sleep $sleep_time

{run_cmd} bash -c "$LAUNCHER $CMD"
echo "END TIME: $(date)"

# Clean up dataset if downloaded from S3
{
    "" if not local_dataset_paths else f"{run_cmd} rm -rf {LOCAL_TMP_PATH_ON_NODE}/dataset"
}
"""
    
    # Launch the job
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    job_id = launch_slurm_job(sbatch_script, args.job_id, args.nodes, args.background, name, timestamp)
    slurm_log_path = f"{run_dir}/slurm_logs/train-{timestamp}-{job_name}-{job_id}.out"
    
    print(f"Launched with Slurm job id = {job_id}")
    print(f"To view the logs, use: tail -f {slurm_log_path}")

if __name__ == "__main__":
    main()