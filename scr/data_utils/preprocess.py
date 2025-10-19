# data_utils/preprocess.py

from transformers import AutoTokenizer

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer, max_length=512):
    """
    Dùng cho Dataset.map() — token hóa văn bản đầu vào.
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
