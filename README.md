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


### Running ablations
1. Tokenize your dataset with tokenize_dataset.py
```
python tokenize_dataset.py --data_paths s3://cosmopedia-data/fineweb_edu_samples/100BT/ --name fineweb-edu
```

2. Train small model for 33B tokens
```
python training_script.py s3://finephrase/experiments/tokenized/fineweb-edu {ablation_name}
```
