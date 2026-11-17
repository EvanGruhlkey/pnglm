"""Processes the resulting features of each token independently"""

import torch
import torch.nn.functional as F

def mlp(x, block):
    hidden = x @ block.mlp.c_fc.weight
    hidden = hidden + block.mlp.c_fc.bias
    
    # activation func
    hidden = F.gelu(hidden, approximate="tanh")
    
    output = hidden @ block.mlp.c_proj.weight
    output = output + block.mlp.c_proj.bias
    return output