import torch
from .model import forward

def generate(prompt, model, tokenizer, max_tokens=30):
    
    # text to tokens
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    # one token at a time
    with torch.inference_mode():
        for _ in range(max_tokens):
            
            # transformer
            logits = forward(input_ids, model)
            
            # select the most likely next token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # append token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # break cases
            if next_token.item() == tokenizer.eos_token_id:
                break
            if input_ids.shape[1] >= model.config.n_positions:
                break
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)