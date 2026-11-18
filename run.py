"""Rn we predict the next token"""

import torch

from engine.loader import load_weights
from engine.tokenizer import load_tokenizer
from engine.generate import generate

model = load_weights()
tokenizer = load_tokenizer()

prompt = input("Enter prompt: ")

print("Generated:", generate(prompt, model, tokenizer))
