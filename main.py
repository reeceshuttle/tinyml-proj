import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import *

from methods import *

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--method", type=str, required=True, help="\'mixed\',\'awq\',\'naive_zeropoint\'")
    argparser.add_argument("--model_name", type=str, required=True)
    argparser.add_argument("--a_bits", type=float, required=False, default=4, help="number of bits to use when quantizing activations.")
    argparser.add_argument("--w_bits", type=float, required=False, default=4, help="number of bits to use when quantizing weights.")
    argparser.add_argument("--salient_weight_p", type=int, required=False, default=1, help="percentage of salient weights/activations to protect in mixed precision.")
    argparser.add_argument("--q_group_size", type=int, required=False, default=128, help="group size when doing quantization.")
    args = argparser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    # TODO: log into hf here so other can use the model?

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(args.device)

    print(f'full precision ppl: {evaluate(model, tokenizer)}')

    input_feat = get_calib_feat(model, tokenizer)
    if args.method == 'mixed':
        pseudo_quantize_mixed_precision(model, w_bit=args.w_bits, a_bit=args.a_bits, q_group_size=args.q_group_size, input_feat=input_feat, salient_weight_p=args.salient_weight_p)
    
    elif args.method == 'awq':
        raise NotImplementedError('Implement me!')
    
    elif args.method == 'naive_zeropoint': # for debugging
        # (this is only weights, not activations)
        args.a_bits = None
        pseudo_quantize_model_weights(model, w_bit=args.w_bits, q_group_size=256)
    
    else:
        raise ValueError("Not a supported method, must be \'awq\', \'mixed\', or \'naive_zeropoint\'")
    
    print(f'quantized (a{args.a_bits}w{args.w_bits}) ppl: {evaluate(model, tokenizer)}')
    # dont just do ppl, also to task performance with lm_eval_harness?


    

    
    