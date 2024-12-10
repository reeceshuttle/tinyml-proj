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

Entry is via `main.py`. Pass in any transformers model via the `--model` command line flag. Specify the method (`awq` or `mixed`) via the `--method` flag.

All flags:

`--model`: transformers model name.

`--method`: quantization method. Either `awq` or `mixed`.

`--a_bits`: The number of bits to use to represent the activations.

`--w_bits`: The number of bits to use to represent the weights.

`--q_group_size`: The quantizaton group size to use.

`--salient_weight_p`: For Mixed Precision. The number of salient channels to preserve in full precision.

Examples:

Mixed Precision:

```
python main.py --method=mixed --model=meta-llama/Meta-Llama-3-8B --a_bits=4 --w_bits=4 --q_group_size=128 --salient_weight_p=1
```

AWQ:

```
python main.py --method=awq --model=meta-llama/Meta-Llama-3-8B --a_bits=4 --w_bits=4 --q_group_size=128
```
