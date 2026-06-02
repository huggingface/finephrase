#!/usr/bin/env python3
import os
import argparse
import random
import subprocess
import yaml
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from finephrase.utils import BASE_PATH, FAULTY_NODES, LOG_BASE_PATH, PROJECT_NAME, PROJECT_PATH, S3_BASE_PATH, LOCAL_TMP_PATH_ON_NODE

EVAL_LOGS_PATH = f"{LOG_BASE_PATH}/evals"
TRAINING_RUNS_PATH = f"{LOG_BASE_PATH}/training"

NANOTRON_PATH = f"{PROJECT_PATH}/nanotron"
S5CMD_PATH = f"{PROJECT_PATH}/.venv/bin/s5cmd"

S3_CHECKPOINTS_PREFIX = f"{S3_BASE_PATH}/checkpoints"
EVALS_OUTPUT_PATH = f"{S3_BASE_PATH}/evals-test"

TASKS_PATH = f"{PROJECT_PATH}/finephrase/tasks.txt"
TASK_LIST_PATH = f"{PROJECT_PATH}/finephrase/task_list.py"

NUM_GPUS = 8
NUM_CPUS_IN_NODE = 88

GLOBAL_BATCH_SIZE = 512
VOCAB_SIZE = 128256
NUM_HIDDEN_LAYERS = 28
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 8

QWEN_SIZE_PRESETS = {
    "0.5b": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "tp": 1,
        "recompute_layer": False,
        "micro_batch_size": 4,
        "eval_batch_size": 32,
    },
    "1.7b": {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "tp": 1,
        "recompute_layer": False,
        "micro_batch_size": 2,
        "eval_batch_size": 16,
    },
    "2.9b": {
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "tp": 1,
        "recompute_layer": True,
        "micro_batch_size": 1,
        "eval_batch_size": 8,
    },
    "6.2b": {
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "tp": 2,
        "recompute_layer": True,
        "micro_batch_size": 1,
        "eval_batch_size": 4,
    },
    # Stop scaling because due to flash attention 2, I could not go beyond hidden size 4096 with 16 attention heads
}

DEFAULT_MODEL_SIZE = "1.7b"

SEQUENCE_LENGTH = 4096

DEFAULT_SEED_VALUE = 6


def calculate_qwen_parameter_count(
    model_preset: dict[str, int | bool],
    vocab_size: int,
    tie_word_embeddings: bool = True,
) -> int:
    """Return total trainable parameter count for the configured Qwen architecture."""
    hidden_size = model_preset["hidden_size"]
    intermediate_size = model_preset["intermediate_size"]
    num_hidden_layers = NUM_HIDDEN_LAYERS
    num_attention_heads = NUM_ATTENTION_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
        )

    head_dim = hidden_size // num_attention_heads

    # Per-layer parameters for Qwen2-style blocks (no linear biases):
    # attention: q_proj + k_proj + v_proj + o_proj
    attention_params = (
        (hidden_size * hidden_size)
        + (hidden_size * num_key_value_heads * head_dim)
        + (hidden_size * num_key_value_heads * head_dim)
        + (hidden_size * hidden_size)
    )
    # MLP: gate_proj + up_proj + down_proj
    mlp_params = (hidden_size * intermediate_size) + (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
    # Norms: input + post-attention RMSNorm
    layer_norm_params = hidden_size + hidden_size
    params_per_layer = attention_params + mlp_params + layer_norm_params

    embedding_params = vocab_size * hidden_size
    final_norm_params = hidden_size
    lm_head_params = 0 if tie_word_embeddings else (vocab_size * hidden_size)

    return embedding_params + (num_hidden_layers * params_per_layer) + final_norm_params + lm_head_params

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
        print(f"Running in background. Logs: {slurm_logs_dir}/train-{timestamp}.out {slurm_logs_dir}/train-{timestamp}.err")
      else:
        # Run in foreground till job is done - WAITS
        subprocess.check_call(srun_args + list(args) + [f.name])
    else:
      return subprocess.run(["sbatch", *args, f.name], capture_output=True, text=True).stdout.split()[-1]


parser = argparse.ArgumentParser(description="Launch training job with updated configuration")
parser.add_argument("--data", help="Dataset folder paths (can be S3 path)", type=str, required=True)
parser.add_argument(
    "--data-weights",
    help=(
        "Comma-separated blending weights, one per --data path. "
        "Defaults to uniform (1/N per dataset). Weight 0 disables a dataset."
    ),
    type=str,
    default=None,
)
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
parser.add_argument("--decay-exp", help="Run a decay experiment", action="store_true")
parser.add_argument("--dep-job-id", help="Dependency job", type=str, default=None)
parser.add_argument(
    "--model-size",
    help="Qwen model size preset",
    type=str,
    default=DEFAULT_MODEL_SIZE,
    choices=tuple(QWEN_SIZE_PRESETS.keys()),
)


def main():
    args = parser.parse_args()
    model_preset = QWEN_SIZE_PRESETS[args.model_size]
    micro_batch_size = model_preset["micro_batch_size"]
    eval_batch_size = int(model_preset["eval_batch_size"])
    tp_size = int(model_preset["tp"])
    recompute_layer = bool(model_preset["recompute_layer"])
    model_param_count = calculate_qwen_parameter_count(
        model_preset=model_preset,
        vocab_size=VOCAB_SIZE,
        tie_word_embeddings=True,
    )
    print(f"Model parameters: {model_param_count:,} ({model_param_count / 1e9:.3f}B)")

    # Debug mode settings
    if args.debug:
        args.nodes = 1 # Only use one node for debugging

    world_size = args.nodes * NUM_GPUS
    if world_size % tp_size != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by tensor parallel size ({tp_size})"
        )
    dp_size = world_size // tp_size

    # For decay experiments, resume from a fixed checkpoint and set up two data stages
    resume_checkpoint_path: str | None = None
    if args.decay_exp:
      resume_checkpoint_path = "s3://finephrase/experiments/checkpoints/fw_edu_lq/9000/"
      checkpoint_step = int(resume_checkpoint_path.rstrip('/').split('/')[-1])
      assert args.lr_schedule == "wsd", "LR schedule must be wsd for decay experiments"

    # Parse multiple comma-separated data paths and their blending weights.
    # Defaults to uniform (1/N) when --data-weights is not given. Weight 0 disables
    # a dataset, which is useful for endpoints of a synthetic-fraction sweep.
    data_paths = [path.strip() for path in args.data.split(",")]
    if args.data_weights is not None:
        dataset_weights = [float(w.strip()) for w in args.data_weights.split(",")]
        if len(dataset_weights) != len(data_paths):
            raise ValueError(
                f"--data-weights has {len(dataset_weights)} values but --data has "
                f"{len(data_paths)} paths; they must match."
            )
        if sum(dataset_weights) <= 0:
            raise ValueError("--data-weights must sum to a positive value.")
    else:
        dataset_weights = [1.0 / len(data_paths)] * len(data_paths)

    # batch size == batch_accumulation_per_replica * micro_batch_size * dp
    per_accum_batch = micro_batch_size * dp_size
    batch_accumulation_per_replica, remainder = divmod(GLOBAL_BATCH_SIZE, per_accum_batch)
    if remainder:
        raise ValueError(
            f"GLOBAL_BATCH_SIZE ({GLOBAL_BATCH_SIZE}) must be divisible by micro_batch_size * dp "
            f"({micro_batch_size} * {dp_size} = {per_accum_batch})"
        )
    print(
        f"Training Qwen-{args.model_size} on {args.nodes} nodes with {NUM_GPUS} GPUs per node, "
        f"tp={tp_size}, dp={dp_size}, micro batch size {micro_batch_size}, "
        f"and {batch_accumulation_per_replica} batch accumulation per replica"
    )
    print(f"Data sources ({len(data_paths)}):")
    for path, weight in zip(data_paths, dataset_weights):
        print(f"  weight={weight:.4f}  {path}")
    batch_size = batch_accumulation_per_replica * per_accum_batch
    assert batch_size == GLOBAL_BATCH_SIZE, f"Batch size {batch_size} is not equal to global batch size {GLOBAL_BATCH_SIZE}"
    tokens_per_step = SEQUENCE_LENGTH * batch_size # sequence length * batch size: 4096 * 512 = 2097152
    print(f"Batch size: {batch_size} examples / {tokens_per_step} tokens")
    total_tokens_consumed = round(tokens_per_step * args.train_steps / 1e9) # in billions
    print(f"Total tokens consumed: {total_tokens_consumed}B") # 8 GPUs, 8 nodes, 10K steps: 20.97152BT
    
    # Naming convention: {stage1_data}-decay-{stage2_data}-seed-{data_seed*100+seed}
    name = args.name.replace(" ", "_") 
    if args.data_seed != DEFAULT_SEED_VALUE and args.seed != DEFAULT_SEED_VALUE: # Only add them if they are not the default values
      name += f"-seed-{(args.data_seed * 100) + args.seed}"

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
    def make_stage(name, start_step, folders, weights):
        if len(folders) != len(weights):
            raise ValueError(
                f"folders ({len(folders)}) and weights ({len(weights)}) must have the same length"
            )
        folders_str = ", ".join(folders)
        weights_str = ", ".join(str(w) for w in weights)
        return f"""- data:
    dataset:
      dataset_folder: [{folders_str}]
      dataset_weights: [{weights_str}]
      pad_samples_to_global_batch_size: false
      return_positions: true
      token_size_in_bytes: 4
      use_old_brrr_dataloader: false
      tokenizer_name: {args.tokenizer}
      vocab_size: {VOCAB_SIZE}
    num_loading_workers: 0
    seed: {args.data_seed}
  name: {name}
  start_training_step: {start_step}"""

    # For decay experiments, we need TWO stages:
    # 1. First stage at step 1 (satisfies nanotron's requirement, won't be used since we resume from step 9000)
    # 2. Second stage at checkpoint_step + 1 (the new dataset for decay phase)
    # Note: Both stages point to the same dataset since the first stage is never actually used
    if args.decay_exp:
        dummy_stage = make_stage("warmup_stable_phase", 1, dataset_folders, dataset_weights)
        decay_stage = make_stage("decay_phase", checkpoint_step + 1, dataset_folders, dataset_weights)
        data_stages_config = f"{dummy_stage}\n{decay_stage}"
    else:
        data_stages_config = make_stage("stable", 1, dataset_folders, dataset_weights)

    MODEL_CONFIG = f"""
checkpoints:
  checkpoint_interval: 500
  checkpoints_path: {LOCAL_TMP_PATH_ON_NODE}/checkpoints/{name}
  checkpoints_path_is_shared_file_system: false
  load_lr_scheduler: true
  load_optimizer: true
  resume_checkpoint_path: {resume_checkpoint_path if resume_checkpoint_path else "null"}
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
    hidden_size: {model_preset["hidden_size"]}
    initializer_range: 0.02
    intermediate_size: {model_preset["intermediate_size"]}
    is_qwen2_config: true
    max_position_embeddings: {SEQUENCE_LENGTH}
    moe_config: null
    num_attention_heads: {NUM_ATTENTION_HEADS}
    num_hidden_layers: {NUM_HIDDEN_LAYERS}
    num_key_value_heads: {NUM_KEY_VALUE_HEADS}
    pad_token_id: null
    pretraining_tp: 1
    rms_norm_eps: 1.0e-06
    rope_interleaved: false
    rope_scaling: null
    rope_theta: 10000
    sliding_window_size: null
    tie_word_embeddings: true
    use_cache: true
    vocab_size: {VOCAB_SIZE}
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
  dp: {dp_size}
  expert_parallel_size: 1
  pp: 1
  pp_engine: 1f1b
  recompute_layer: {str(recompute_layer).lower()}
  tp: {tp_size}
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
  micro_batch_size: {micro_batch_size}
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
  batch_size: {eval_batch_size}
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
#SBATCH --requeue
#SBATCH --exclude={FAULTY_NODES}
{"#SBATCH --dependency=afterok:" + args.dep_job_id if args.dep_job_id else ""}
{"#SBATCH --reservation=" + args.reservation if args.reservation else ""}

set -x -e

# Handle preemption: SIGTERM is sent ~30s before SIGKILL on preemption.
# Exit cleanly so SLURM requeues the job; nanotron's periodic checkpointing
# ensures progress is saved.
trap 'echo "SIGTERM received (preemption), exiting for requeue..."; exit 0' SIGTERM

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

# Auto-detect latest S3 checkpoint for resume (--requeue). Skip training if already done
# (latest checkpoint step >= train_steps) so requeued jobs exit cleanly without restarting.
{"" if args.decay_exp else f'''set +e
python3 -c "
import subprocess
import sys
import yaml

train_steps = {args.train_steps}
config_path = '{config_path_yaml}'
s5cmd = '{S5CMD_PATH}'
prefix = '{S3_CHECKPOINTS_PREFIX}/{name}/'

with open(config_path) as f:
    cfg = yaml.safe_load(f)

result = subprocess.run([s5cmd, 'ls', prefix], capture_output=True, text=True)
steps = sorted(
    int(p.split()[-1].strip('/'))
    for p in result.stdout.strip().splitlines()
    if 'DIR' in p and p.split()[-1].strip('/').isdigit()
)

if steps and steps[-1] >= train_steps:
    print(
        f'Latest checkpoint step {{steps[-1]}} >= train_steps {{train_steps}}; '
        'run already finished, not resuming'
    )
    cfg['checkpoints']['resume_checkpoint_path'] = None
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
    sys.exit(2)

if steps:
    resume = prefix + str(steps[-1]) + '/'
    print(f'Auto-resuming from checkpoint: {{resume}}')
    cfg['checkpoints']['resume_checkpoint_path'] = resume
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
else:
    print('No existing checkpoints found, starting from scratch')
sys.exit(0)
"
_ckpt_rc=$?
set -e
if [ "$_ckpt_rc" -eq 2 ]; then
  echo "Skipping training: checkpoint already reached train_steps."
  exit 0
fi
if [ "$_ckpt_rc" -ne 0 ]; then
  echo "WARNING: Auto-checkpoint detection failed (exit $_ckpt_rc), continuing with existing config"
fi'''}

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