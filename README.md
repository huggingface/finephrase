# finephrase
Synthetic pretraining data by rephrasing the web

## Setup

### Install uv and setup a venv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
```

### Clone the core repos from correct branch
```
git clone -b nanotron-working-branch git@github.com:huggingface/nanotron.git
git clone -b lighteval-experiment-setup  git@github.com:huggingface/lighteval.git
git clone -b fix-nanotron git@github.com:huggingface/datatrove.git
```

### Add latest datatrove changes so we can use the InferenceRunner for rephrasing
```
(cd datatrove && git rebase origin/main)
```

### Enter a GPU node for installation
```
srun --gpus=1 --qos=high --time="01:59:00"  --pty bash
module load cuda/12.4
```

### Install dependencies (order is important)
```
uv pip install --find-links https://download.pytorch.org/whl/cu124/torch/ "torch==2.6.0+cu124"
uv pip install --find-links https://download.pytorch.org/whl/cu124/torchvision/ "torchvision==0.21.0+cu124"
uv pip install --no-build-isolation  "flash_attn==2.7.4.post1"
uv pip install -e "nanotron" && uv pip install -e "lighteval[math,multilingual]" && uv pip install -e "datatrove[s3,io,processing]"
uv pip install -e .
```

### Test the installation
```
python -c "import nanotron"
```

### Available console commands
After installation, you can use these short commands:

- `filter-fineweb-edu`  - Filter fineweb edu data (filter_fineweb_edu.py)
- `rephrase`            - Rephrase datasets (rephrase_dataset.py)
- `tokenize`            - Tokenize datasets (tokenize_dataset.py)
- `train`               - Train models (train_model.py)
- `evaluate`            - Evaluate checkpoints (evaluate_checkpoints.py) 

All commands support `--help` to see available options.

### Before running ablations
1. Create a bucket on s3 for your project
2. Modify the `training_script.py` constants
3. Set the default output path for `tokenize_dataset.py` script.

### Count the total tokens

HQ data (rounded int_score 4,5 or score > 3.5):
```
filter-fineweb-edu \
  --data_paths hf://datasets/HuggingFaceFW/fineweb-edu/data \
  --quality hq \
  --name fineweb-edu-hq \
  --total_tokens 217715141792
```

LQ data (rounded int_score 0,1 or score < 1.5):
```
filter-fineweb-edu \
  --data_paths s3://fineweb-data-processing-us-east-1/edu_annotated/score1_2 \
  --quality lq \
  --name fineweb-edu-lq \
  --total_tokens 1644271223950
```

Hub (fineweb-edu): all dumps with score >=3
Hub (fineweb-edu-score-2): all dumps with score >= 2
My S3 bucket: new dumps, data for all scores (score1_2 for <3, score3 for >=3)

Note: `TokensCounter` runs before writing, adding `token_count` to metadata.

### Running ablations
1. Tokenize your datasets with tokenize_dataset.py
```
tokenize --data_paths s3://finephrase/experiments/filtered/fineweb-edu-hq --name fineweb-edu-hq
tokenize --data_paths s3://finephrase/experiments/filtered/fineweb-edu-lq --name fineweb-edu-lq
```

2. Train small model for 36B tokens
```
train s3://finephrase/experiments/tokenized/fineweb-edu-hq {ablation_name}
```

3. Run evaluations manually (if automatic ones fail during training)
```
evaluate fineweb-edu-hq-36B-seed-606,fineweb-edu-lq-36B-seed-606
```

4. Inference with different rephrasing prompts
```
rephrase \
  --data_paths s3://finephrase/experiments/filtered/fineweb-edu-lq \
  --name rewire \
  --prompt_template rewire/guided_rewrite_improved.md \
  --limit 5 --debug --disable_checkpoints
```

Use different prompts for data quality improvement:

**For LQ data:**
- `prompts/rewire/guided_rewrite_corrected.md` - Guided rewriting with expert reasoning
- `prompts/nemotron/wikipedia_style_rephrasing.md` - Wikipedia-style paraphrasing

**For HQ data:**
- Any of the Nemotron prompts in `prompts/nemotron/`:
  - `distill.md` - Text condensation and paraphrasing
  - `extract_knowledge.md` - Knowledge extraction and rewriting
  - `diverse_qa_pairs.md` - Question-answer pair generation
  - `knowledge_list.md` - Factual information extraction
  - `wikipedia_style_rephrasing.md` - Wikipedia-style paraphrasing`

