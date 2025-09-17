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
srun --gpus=1 --qos=high --time="04:00:00" --pty bash
module load cuda/12.4
```

### Install dependencies (order is important)
```bash
uv pip install setuptools
uv pip install --find-links https://download.pytorch.org/whl/cu124/torch/ "torch==2.6.0+cu124"
uv pip install --find-links https://download.pytorch.org/whl/cu124/torchvision/ "torchvision==0.21.0+cu124"
uv pip install --no-build-isolation  "flash_attn==2.7.4.post1"
uv pip install -e "nanotron" && uv pip install -e "lighteval[math,multilingual]" && uv pip install -e "datatrove[s3,io,processing]"
uv pip install -e .
```

### Test the installation
```bash
python -c "import nanotron"
```

## Available Commands

After installation, you can use these console commands for different aspects of the data pipeline:

- `report-tokens`       - Report tokens (report_tokens.py)
- `filter-fineweb-edu`  - Filter fineweb edu data (filter_fineweb_edu.py)
- `rephrase`            - Rephrase datasets (rephrase_dataset.py)
- `tokenize`            - Tokenize datasets (tokenize_dataset.py)
- `train`               - Train models (train_model.py)
- `evaluate`            - Evaluate checkpoints (evaluate_checkpoints.py)
- `launch-experiments`  - Launch multiple Slurm experiments from YAML configs (launch_experiments.py)

All commands support `--help` to see available options.

## Data Processing

### Token Statistics
Get comprehensive token statistics for any dataset:

```bash
report-tokens --data-paths s3://path/to/dataset1,hf://datasets/owner/dataset2
```

### Filtering Educational Data
Use `filter-fineweb-edu` to create datasets with specific educational quality scores.

**HQ data** (rounded int_score 4,5 or score > 3.5):
```bash
filter-fineweb-edu \
  --data-paths hf://datasets/HuggingFaceFW/fineweb-edu/data \
  --quality hq \
  --name fineweb-edu-hq-20BT \
  --subset-tokens 21.5e9 \
  --total-tokens 217715141792
```

**LQ data** (rounded int_score 0,1 or score < 1.5):
```bash
filter-fineweb-edu \
  --data-paths s3://fineweb-data-processing-us-east-1/edu_annotated/score1_2 \
  --quality lq \
  --name fineweb-edu-lq-20BT \
  --subset-tokens 21.5e9 \
  --total-tokens 1644271223950
```

**Data sources:**
- Hub (fineweb-edu): all dumps with score >=3
- Hub (fineweb-edu-score-2): all dumps with score >= 2
- S3 bucket: new dumps, data for all scores (score1_2 for <3, score3 for >=3)

Note: `TokensCounter` runs before writing, adding `token_count` to metadata.

### Tokenizing Datasets
Prepare datasets for training by tokenizing them:

```bash
tokenize --data-paths s3://finephrase/experiments/filtered/fineweb-edu-hq-20BT --name fineweb-edu-hq-20BT --sample 1
tokenize --data-paths s3://finephrase/experiments/filtered/fineweb-edu-lq-20BT --name fineweb-edu-lq-20BT --sample 1
```

## Model Training & Evaluation

### Training Models
Train 1B parameter models on your tokenized datasets:

```bash
train s3://finephrase/experiments/tokenized/fineweb-edu-hq-20BT fineweb-edu-hq-20BT
train s3://finephrase/experiments/tokenized/fineweb-edu-lq-20BT fineweb-edu-lq-20BT
```

### Evaluating Checkpoints
Run evaluations manually if automatic ones fail during training:

```bash
evaluate fineweb-edu-hq-20BT-21B-seed-606,fineweb-edu-lq-20BT-21B-seed-606
```

## Prompt Optimization

Optimize prompts for text generation tasks using DSPy GEPA:

```bash
optimize-prompt --budget 10
```

## Data Generation

### Rephrasing Datasets
Generate synthetic training data by rephrasing existing content:

```bash
rephrase --data-paths s3://finephrase/experiments/filtered/fineweb-edu-lq-20BT --prompt dspy/rephrase-budget-20.md --name dspy-rephrase-budget-20 --limit 100
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


