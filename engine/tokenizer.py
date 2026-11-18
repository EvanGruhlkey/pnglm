from transformers import AutoTokenizer

MODEL_NAME = "distilbert/distilgpt2"

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME, clean_up_tokenization_spaces=False)