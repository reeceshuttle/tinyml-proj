# TinyML Final Project

### By Reece Shuttleworth, Simon Opsahl, Ziyad Hassan, Nicky Medearis, and Abdul-Kareem Aliu

## Setup:

1. create and activate environment:

venv:

```
python3.10 -m venv tinyml-env
source tinyml-env/bin/activate
```

conda:

```
conda create --name tinyml python=3.10
conda activate tinyml
```

2. install dependencies:

```
pip install -r requirements.txt
```

## Usage:

Mixed Precision:

```
python main.py --method=mixed --model=meta-llama/Meta-Llama-3-8B --a_bits=4 --w_bits=4 --q_group_size=128 --salient_weight_p=1
```

AWQ:

```
python main.py --method=awq --model=meta-llama/Meta-Llama-3-8B --a_bits=4 --w_bits=4 --q_group_size=128
```
