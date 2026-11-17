import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilbert/distilgpt2"

# load model weights and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)