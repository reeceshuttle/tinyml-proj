import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import evaluate

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--method", type=str, required=True, help="\'mixed\',\'awq\',\'naive_zeropoint\'")
    argparser.add_argument("--model_name", type=str, required=True)
    argparser.add_argument("--a_bits", type=int, required=False, default=4, help="number of bits to use when quantizing activations.")
    argparser.add_argument("--w_bits", type=int, required=False, default=4, help="number of bits to use when quantizing weights.")
    argparser.add_argument("--salient_weight_p", type=int, required=False, default=1, help="percentage of salient weights/activations to protect in mixed precision.")
    argparser.add_argument("--q_group_size", type=int, required=False, default=128, help="group size when doing quantization.")
    args = argparser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # TODO: log into hf here so other can use the model?

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16)

    print('before quantization (full precision):')
    evaluate(model)

    if args.method == 'mixed':
        raise NotImplementedError('Implement me!')
    
    elif args.method == 'awq':
        raise NotImplementedError('Implement me!')
    
    elif args.method == 'naive_zeropoint':
        raise NotImplementedError('Implement me!')
    
    else:
        raise ValueError("Not a supported method, must be \'awq\', \'mixed\', or \'naive_zeropoint\'")
    
    print(f'after quantization (a{args.a_bits}w{args.w_bits}):')
    evaluate(model)
    # dont just do ppl, also to task performance with lm_eval_harness?


    

    
    