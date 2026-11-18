"""Rn we predict the next token"""

import torch

from engine.loader import load_weights
from engine.tokenizer import load_tokenizer
from engine.model import forward

model = load_weights()
tokenizer = load_tokenizer()

prompt = input("Enter prompt: ")

# text to tokens
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

# inference
with torch.no_grad():
    logits = forward(input_ids, model)
   
# next token prediction 
next_token = logits[0, -1].argmax().item()

print("Prediction:", tokenizer.decode([next_token]))
