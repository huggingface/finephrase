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

### Enter a GPU node for installation
```
srun --gpus=1 --qos=high --time="01:59:00"  --pty bash
```

### Install deps and correct cuda
```
module load cuda/12.4
uv pip install --find-links https://download.pytorch.org/whl/cu124/torch/ "torch==2.6.0+cu124"
uv pip install setuptools s5cmd && uv pip install --no-build-isolation  flash_attn=="2.7.4.post1"
uv pip install --find-links https://download.pytorch.org/whl/cu124/torchvision/ "torchvision==0.21.0+cu124"
uv pip install -e nanotron && uv pip install -e "lighteval[math,multilingual]" && uv pip install -e "datatrove[s3,io,processing]"
uv pip install hf-transfer datasets==3.5.1
```

### This is because nanotron env checks (for uv)
```
uv pip install pip pybind11 pydantic "huggingface_hub[hf_xet]"
```

### Test the installation
```
python -c "import nanotron"
```

### Before running ablations
1. Create a bucket on s3 for your project
2. Modify the `training_script.py` constants
3. Set the default output path for `tokenize_dataset.py` script.

### Count the total tokens

HQ data (rounded int_score 4,5 or score > 3.5):
```
python filter_fineweb_edu.py \
  --data_paths hf://datasets/HuggingFaceFW/fineweb-edu/data \
  --quality hq \
  --name fineweb-edu-hq
```

LQ data (rounded int_score 0,1 or score < 1.5):
```
python filter_fineweb_edu.py \
  --data_paths s3://fineweb-data-processing-us-east-1/edu_annotated/score1_2 \
  --quality lq \
  --name fineweb-edu-lq
```

Hub (fineweb-edu): all dumps with score >=3
Hub (fineweb-edu-score-2): all dumps with score >= 2
My S3 bucket: new dumps, data for all scores (score1_2 for <3, score3 for >=3)

Note: `TokensCounter` runs before writing, adding `token_count` to metadata.

### Running ablations
1. Tokenize your datasets with tokenize_dataset.py
```
python tokenize_dataset.py --data_paths s3://finephrase/experiments/filtered/fineweb-edu-hq --name fineweb-edu-hq
python tokenize_dataset.py --data_paths s3://finephrase/experiments/filtered/fineweb-edu-lq --name fineweb-edu-lq
```

2. Train small model for 36B tokens
```
python training_script.py s3://finephrase/experiments/tokenized/fineweb-edu-hq {ablation_name}
```
