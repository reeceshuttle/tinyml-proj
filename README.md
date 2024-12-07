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

```
python main.py --model_name=meta-llama/Llama-3.2-1B \
               --method=awq
```
