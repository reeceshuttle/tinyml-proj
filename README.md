# TinyML Final Project

### By Reece Shuttleworth, Simon Opsahl, Ziyad Hassan, Nicky Medearis, and Abdul-Kareem Aliu

## Setup:

1. create and install environment:

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

mixed precision:

```
python main.py --method=mixed --model=meta-llama/Llama-3.2-1B --a_bits=8 --w_bits=4 --salient_weight_p=2
```
