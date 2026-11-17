"""Combines information from preceding tokens"""

import math
import torch
import torch.nn.functional as F

def attention(x, block, num_heads):
    # input tensor
    batch, seq, channels = x.shape
    
    head_dim = channels // num_heads
    
    # create queries, keys, and vals (xW)
    qkv = x @ block.attn.c_attn.weight
    
    # + bias
    qkv = qkv + block.attn.c_attn.bias
    
    q,k,v = qkv.split(channels, dim=-1)
    
    # group by heads
    def make_heads(t):
        t = t.reshape(batch, seq, num_heads, head_dim)
        return t.transpose(1,2)
    
    q = make_heads(q)
    k = make_heads(k)
    v = make_heads(v)
    
    # att between tokens
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)
    
    # don't look at future tokens
    causal = torch.ones(seq, seq, dtype=torch.bool, device=x.device).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    
    # att probs
    probs = F.softmax(scores, dim=-1)

    # combine val vectors
    context = probs @ v
    
    # combine att heads
    context = context.transpose(1,2)
    context = context.contiguous().reshape(batch,seq,channels)
    
    return (context @ block.attn.c_proj.weight + block.attn.c_proj.bias)