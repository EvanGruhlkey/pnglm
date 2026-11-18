from transformers import AutoModelForCausalLM
from .tokenizer import MODEL_NAME

def load_weights():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    model.eval()
    return model