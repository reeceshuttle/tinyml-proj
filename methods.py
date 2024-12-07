import torch
import torch.nn as nn

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    """From TinyML Pset 4."""
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w

@torch.no_grad()
def pseudo_quantize_model_weights(
    model, w_bit, q_group_size,
):
    """From TinyML Pset 4."""
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)


@torch.no_grad()
def pseudo_quantize_mixed_precision(
model, w_bit, a_bit, q_group_size, input_feat, salient_weight_p
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()
            outlier_indices = torch.topk(importance, int(len(importance)*0.01*salient_weight_p)).indices
            assert outlier_indices.dim() == 1
            m.outlier_indices = outlier_indices.clone()
            outlier = m.weight.data[:, outlier_indices].clone()
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)
            m.weight.data[:, outlier_indices] = outlier

            def quantize_activations(outlier_indices, input):
                if len(input.shape) == 3:
                    outlier = input[:, :, outlier_indices].clone()
                elif len(input.shape) == 2:
                    outlier = input[:, outlier_indices].clone()
                input = pseudo_quantize_tensor(input, n_bit=a_bit, q_group_size=q_group_size)
                if len(input.shape) == 3:
                    input[:, :, outlier_indices] = outlier
                elif len(input.shape) == 2:
                    input[:, outlier_indices] = outlier
                
                return input

            def forward_hook(module, input):
                return quantize_activations(module.outlier_indices, input[0])
            
            m.register_forward_pre_hook(forward_hook)

@torch.no_grad()
def pseudo_quantize_awq_naive(
    model, w_bit, a_bit, q_group_size, input_feat, salient_weight_p, scaling_factor
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()
            outlier_indices = torch.topk(importance, int(len(importance)*0.01*salient_weight_p)).indices
            assert outlier_indices.dim() == 1
            m.outlier_indices = outlier_indices.clone()
            m.scaling_factor = scaling_factor

            # scale weights up
            m.weight.data[:, outlier_indices] *= scaling_factor
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            def quantize_activations(outlier_indices, input, scaling_factor):
                # scale activations down
                if len(input.shape) == 3:
                    input[:, :, outlier_indices] /= scaling_factor
                elif len(input.shape) == 2:
                    input[:, outlier_indices] /= scaling_factor
                input = pseudo_quantize_tensor(input, n_bit=a_bit, q_group_size=q_group_size)
                return input

            def forward_hook(module, input):
                return quantize_activations(module.outlier_indices, input[0], module.scaling_factor)
            
            m.register_forward_pre_hook(forward_hook)
    