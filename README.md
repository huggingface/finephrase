# FinePhrase
Synthetic pretraining data by rephrasing the web

[Blog post](https://huggingface.co/spaces/HuggingFaceFW/finephrase)

![Training Progression](assets/training_progression.png)

We ran 90 experiments, generated over 1 trillion tokens, and spent 12.7 GPU years to find the best recipe for synthetic pretraining data. The result is FinePhrase, a 486B token dataset that clearly outperforms all existing synthetic data baselines. It's [available on the Hub](https://huggingface.co/datasets/HuggingFaceFW/finephrase), and this post walks you through everything we learned along the way.

## Setup

### Install uv and setup a venv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
```

### Clone repos, apply patches, and install

FinePhrase depends on custom branches of nanotron, lighteval, and datatrove. Each needs a small patch on top:

| Patch                                              | Why                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------- |
| `datatrove-folder-dataset-repeat.patch`            | Wraps sample indices with modulo so nanotron blending doesn't `IndexError`        |
| `nanotron-recursive-dataloader.patch`              | Sets `recursive=True` so `build_dataset` discovers `.ds` files in subdirectories |
| `lighteval-trust-remote-code.patch`                | Removes `trust_remote_code` arg dropped in newer HF libraries                    |

From the FinePhrase repo root, on a GPU node (`srun --gpus=1 --qos=high --time="01:59:00" --pty bash && module load cuda/12.4`):

```bash
git clone -b nanotron-working-branch git@github.com:huggingface/nanotron.git
git clone -b lighteval-experiment-setup git@github.com:huggingface/lighteval.git
git clone -b fix-nanotron git@github.com:joelniklaus/datatrove.git

patch -p1 -d datatrove  < patches/datatrove-folder-dataset-repeat.patch
patch -p1 -d nanotron   < patches/nanotron-recursive-dataloader.patch
patch -p1 -d lighteval  < patches/lighteval-trust-remote-code.patch

uv pip install setuptools
uv pip install --find-links https://download.pytorch.org/whl/cu124/torch/ "torch==2.6.0+cu124"
uv pip install --find-links https://download.pytorch.org/whl/cu124/torchvision/ "torchvision==0.21.0+cu124"
uv pip install --no-build-isolation "flash_attn==2.7.4.post1"
uv pip install -e "nanotron" && uv pip install -e "lighteval[math,multilingual]" && uv pip install -e "datatrove[s3,io,processing]"
uv pip install -e .
```

### Test the installation
```bash
python -c "import nanotron"
```

## Available Commands

After installation, you can use these console commands for different aspects of the data pipeline:

- `report-tokens`         - Report tokens (report_tokens.py)
- `filter`                - Filter datasets with selectable filters (filter_dataset.py)
- `rephrase`              - Rephrase datasets (rephrase_dataset.py)
- `tokenize`              - Tokenize datasets (tokenize_dataset.py)
- `train`                 - Train models (train_model.py)
- `evaluate`              - Evaluate checkpoints (evaluate_checkpoints.py)
- `launch-experiments`    - Launch multiple Slurm experiments from YAML configs (launch_experiments.py)
- `benchmark-vllm`        - Submit vLLM serving benchmarks to Slurm (benchmark_vllm.py)
- `analyze-benchmarking` - Analyze benchmark results and export CSV (analyze_benchmarking.py)

All commands support `--help` to see available options.

## Data Processing

### Token Statistics
Get comprehensive token statistics for any dataset:

```bash
report-tokens --data s3://path/to/dataset1,hf://datasets/owner/dataset2
```

### Filtering Data
Use `filter` to filter datasets using predefined filter functions. Supported filters include:
- `fineweb_edu_hq` (FineWeb-Edu HQ: rounded int_score 4,5)
- `fineweb_edu_lq` (FineWeb-Edu LQ: rounded int_score 0,1)
- `noop` (no filtering, copy all data)

**HQ data**:
```bash
filter \
  --data hf://datasets/HuggingFaceFW/fineweb-edu/data \
  --filter fineweb_edu_hq \
  --name fineweb-edu-hq-20BT \
  --subset-tokens 21.5e9 \
  --total-tokens 217715141792
```

**LQ data**:
```bash
filter \
  --data s3://fineweb-data-processing-us-east-1/edu_annotated/score1_2 \
  --filter fineweb_edu_lq \
  --name fineweb-edu-lq-20BT \
  --subset-tokens 21.5e9 \
  --total-tokens 1644271223950
```

**Noop filter (copy all data)**:
```bash
filter \
  --data hf://datasets/mlfoundations/dclm-baseline-1.0-parquet/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_01_of_10/local-shard_0_of_10 \
  --filter noop \
  --name dclm-37BT

filter \
  --data hf://datasets/HuggingFaceTB/cosmopedia/data \
  --filter noop \
  --name cosmopedia-25BT
```

**Data sources:**
- Hub (fineweb-edu): all dumps with score >=3
- Hub (fineweb-edu-score-2): all dumps with score >= 2
- S3 bucket: new dumps, data for all scores (score1_2 for <3, score3 for >=3)

Note: Token counting is handled separately via `report-tokens`.

### Tokenizing Datasets
Prepare datasets for training by tokenizing them:

```bash
tokenize --data s3://finephrase/experiments/filtered/fineweb-edu-hq-20BT --name fw_edu_hq
tokenize --data s3://finephrase/experiments/filtered/fineweb-edu-lq-20BT --name fw_edu_lq
```

## Model Training & Evaluation

### Training Models
Train Qwen-size models (`0.5b`, `1.7b`, `2.9b`, `6.2b`) on your tokenized datasets.
The launcher prints the exact parameter count for the selected preset before submitting training.
Tensor parallelism and recomputation are configured per model preset in `train_model.py`.
Lighteval batch size in the generated Nanotron config scales with model size (`eval_batch_size` in `QWEN_SIZE_PRESETS`: e.g. `1.7b` → 8, `6.2b` → 2) to reduce eval OOM on larger models. Run names use a trailing `-0.5b` / `-1.7b` / `-2.9b` / `-6.2b` suffix; if missing, the `1.7b` preset is used.
All presets keep depth/head topology fixed and only scale `hidden_size` + `intermediate_size`.

Slurm jobs use `--qos low` by default, `--requeue`, and a script that picks the **latest** checkpoint under `s3://finephrase/experiments/checkpoints/<run>/` before each run. If that step is already `>= train_steps`, training is **skipped** (clean exit) so finished runs do not reload step-`train_steps` checkpoints and crash.

**Repeating / blended data:** Nanotron’s blend builds sample indices up to `train_steps × global_batch_size` per dataset stream. With the [datatrove patch](#patch-datatrove-folder-dataset-index-wrap), `DatatroveFolderDataset` maps any global index with `index % len(dataset)` before resolving the `.ds` file, so a smaller corpus is cycled instead of raising `IndexError` (same idea as `OldTokenizedBytesFolderDataset` in nanotron). Empty folders still fail with a clear `IndexError`.

```bash
train --data s3://finephrase/experiments/tokenized/fw_edu_hq --name fw_edu_hq
train --data s3://finephrase/experiments/tokenized/fw_edu_lq --name fw_edu_lq
train --data s3://finephrase/experiments/tokenized/fw_edu_hq --name fw_edu_hq_2.9b --model-size 2.9b
```

### Evaluating Checkpoints
Run evaluations manually if automatic ones fail during training:

```bash
evaluate --name fw_edu_hq,fw_edu_lq
```

`evaluate_checkpoints.py` infers the same preset from the run folder name (suffix like `-6.2b`). Override the batch size if needed: `evaluate --batch-size 1`.

Run all missing evaluations:
```bash
evaluate --all
```

## Prompt Optimization

Optimize prompts for text generation tasks using DSPy GEPA:

```bash
optimize-prompt --budget 10
```

Iterate on prompts manually:
```bash
iterate-prompt --prompt format/faq.md --model-size 1b --data-path hf://datasets/HuggingFaceFW/fineweb-edu
```

## Data Generation

### Rephrasing Datasets
Generate synthetic training data by rephrasing existing content:

```bash
rephrase --data s3://finephrase/experiments/filtered/fineweb-edu-lq-20BT --prompt dspy/rephrase/gemma-3-1b-it/budget-10.md --name dspy-rephrase-budget-10 --debug
```

## Data Inspection

Quickly inspect data

```bash
inspect-data --data s3://finephrase/experiments/rephrased/dspy/rephrase/gemma-3-27b-it/ --limit 5
```

**Available prompts for data quality improvement:**

*For LQ data:*
- `prompts/rewire/guided_rewrite_corrected.md` - Guided rewriting with expert reasoning
- `prompts/nemotron/wikipedia_style_rephrasing.md` - Wikipedia-style paraphrasing
- `prompts/dspy/5-max_full_evals.md` - DSPy GEPA optimized prompt

*For HQ data:*
- `prompts/nemotron/distill.md` - Text condensation and paraphrasing
- `prompts/nemotron/extract_knowledge.md` - Knowledge extraction and rewriting
- `prompts/nemotron/diverse_qa_pairs.md` - Question-answer pair generation
- `prompts/nemotron/knowledge_list.md` - Factual information extraction
- `prompts/nemotron/wikipedia_style_rephrasing.md` - Wikipedia-style paraphrasing

## Experiment Management

### Experiment Launcher
Submit multiple Slurm experiments with different configurations using YAML files:

```bash
# Submit all Slurm jobs in a configuration
launch-experiments configs/rephrase_benchmark.yaml

# Test configuration without submitting jobs (dry run)
launch-experiments configs/rephrase_benchmark.yaml --dry-run

# Submit only specific experiments
launch-experiments configs/rephrase_benchmark.yaml --run-names "qwen_0.6b_thinking,qwen_1.7b_thinking"
```

## Citation

```bibtex
@misc{niklaus2026_the_synthetic_data_playbook_generating_trillions_of_the_finest_tokens,
  title={The Synthetic Data Playbook: Generating Trillions of the Finest Tokens},
  author={Joel Niklaus and Guilherme Penedo and Hynek Kydlicek and Elie Bakouch and Lewis Tunstall and Ed Beeching and Thibaud Frere and Colin Raffel and Leandro von Werra and Thomas Wolf},
  year={2026},
}
```

