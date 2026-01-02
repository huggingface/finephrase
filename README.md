# FinePhrase
Synthetic pretraining data by rephrasing the web

## Setup

### Install uv and setup a venv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
```

### Clone the core repos from correct branch
```bash
git clone -b nanotron-working-branch git@github.com:huggingface/nanotron.git
git clone -b lighteval-experiment-setup  git@github.com:huggingface/lighteval.git
git clone -b fix-nanotron git@github.com:joelniklaus/datatrove.git
```

### Enable training with recursive dataloaders
In `nanotron/src/nanotron/data/tokenized_bytes.py`, update lines 414 and 430 to set `recursive=True`.

### Enter a GPU node for installation
```bash
srun --gpus=1 --qos=high --time="01:59:00" --pty bash
module load cuda/12.4
```

### Install dependencies (order is important)
```bash
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
Train 1B parameter models on your tokenized datasets:

```bash
train --data s3://finephrase/experiments/tokenized/fw_edu_hq --name fw_edu_hq
train --data s3://finephrase/experiments/tokenized/fw_edu_lq --name fw_edu_lq
```

### Evaluating Checkpoints
Run evaluations manually if automatic ones fail during training:

```bash
evaluate --name fw_edu_hq,fw_edu_lq
```

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

## Analysis & Visualization

### Visualizing Rephrasing Statistics
Generate bar charts comparing educational score improvements across rephrasing experiments:

```bash
python /fsx/joel_niklaus/projects/finephrase/plot_rephrasing_stats.py
```

This writes a high-DPI PNG to `plots/rephrasing_edu_score_difference_means.png`.


