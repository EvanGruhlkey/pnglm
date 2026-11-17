import torch
import torch.nn.functional as F
from .attention import attention
from .mlp import mlp

def layer_norm(x, layer, eps):
    return F.layer_norm(x, (x.shape[-1],), layer.weight, layer.bias, eps)

def transformer_block(x, block, config):
    # normalize
    norm = layer_norm(x, block.ln_1, config.layer_norm_epsilon)

    # att + residual
    x = x + attention(norm, block, config.n_head)
    
    # do same for mlp
    norm = layer_norm(x, block.ln_2, config.layer_norm_epsilon)
    x = x + mlp(norm, block)
    
    return x

# inference
def forward(input_ids, weights):
    cfg = weights.config
    transformer = weights.transformer
    
    batch, seq = input_ids.shape
    if seq > cfg.n_positions:
        raise ValueError("Prompt is too long")
    
    positions = torch.arange(seq, device=input_ids.device)
    
    # tokens into vectors
    x = transformer.wte.weight[input_ids]
    
    # add positional info
    x = x + transformer.wpe.weight[positions][None, :, :]
    
    # pass through all blocks
    for block in transformer.h:
        x = transformer_block(x, block, cfg)
    
    # normalize
    x = layer_norm(x, transformer.ln_f, cfg.layer_norm_epsilon)
    
    # vectors ro vocab scores
    logits = x @ weights.lm_head.weight.T
    return logits