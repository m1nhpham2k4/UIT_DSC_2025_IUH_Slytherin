import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

import warnings
import datasets
from typing import Tuple, Any, Dict, List, Union
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import PeftModel

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.prompt.base_prompt import PROMPT



NUM_CLASSES = 3
max_seq_length = 2700
dtype = None
load_in_4bit = False

model_name = 'unsloth/Qwen3-4B-Base'
save_dir = "./result/checkpoint-438"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=load_in_4bit,
    max_seq_length=max_seq_length,
    dtype=dtype,
    # device_map = "balanced",
)

number_token_ids = []
for i in range(0, NUM_CLASSES+1):
    number_token_ids.append(tokenizer.encode(str(i), add_special_tokens=False)[0])

par = torch.nn.Parameter(model.lm_head.weight[number_token_ids, :])

old_shape = model.lm_head.weight.shape
old_size = old_shape[0]
print(par.shape)
print(old_shape)

model.lm_head.weight = par

reverse_map = {value: idx for idx, value in enumerate(number_token_ids)}
print(reverse_map)

model = PeftModel.from_pretrained(model, save_dir)

trimmed_lm_head = model.lm_head.weight.data.clone()
trimmed_lm_head_bias = model.lm_head.bias.data.clone() if hasattr(model.lm_head, "bias") and model.lm_head.bias is not None else torch.zeros(len(number_token_ids), device=trimmed_lm_head.device)

hidden_dim = trimmed_lm_head.shape[1]
new_lm_head = torch.full((old_size, hidden_dim), 0, dtype=trimmed_lm_head.dtype, device=trimmed_lm_head.device)
new_lm_head_bias = torch.full((old_size,), -1000.0, dtype=trimmed_lm_head_bias.dtype, device=trimmed_lm_head_bias.device)


for new_idx, orig_token_id in enumerate(number_token_ids):
    new_lm_head[orig_token_id] = trimmed_lm_head[new_idx]
    new_lm_head_bias[orig_token_id] = trimmed_lm_head_bias[new_idx]


with torch.no_grad():
    new_lm_head_module = torch.nn.Linear(hidden_dim, old_size, bias=True, device=model.device)
    new_lm_head_module.weight.data.copy_(new_lm_head)
    new_lm_head_module.bias.data.copy_(new_lm_head_bias)
    model.lm_head.modules_to_save["default"] = new_lm_head_module

print(f"Remade lm_head: shape = {model.lm_head.weight.shape}. Allowed tokens: {number_token_ids}")

test_df = pd.read_csv("./data/test/vihallu-private-test.csv")
prompt_ = PROMPT
inference_prompt_template = prompt_.split("class {}")[0] + "class "
train_df = pd.read_csv("./data/train/vihallu-train.csv")


_text2id = {"no": 1, "intrinsic": 2, "extrinsic": 3}
def to_123(x):
    s = str(x).strip().lower()
    if s.isdigit():           
        return int(s)
    return _text2id.get(s, 1)

def collect_slice3_logits(model, tokenizer, texts, number_token_ids, max_len=3072, bs=8, device=None):
    device = device or (model.device if hasattr(model, "device") else "cuda")
    outs = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            batch_txt = texts[i:i+bs]
            inputs = tokenizer(
                batch_txt, return_tensors="pt", padding=True, truncation=True, max_length=max_len
            ).to(device)

            logits = model(**inputs).logits
            last_idx = inputs.attention_mask.sum(1).to(torch.long) - 1
            last_logits = logits[torch.arange(len(batch_txt), device=device), last_idx, :]

            slice3 = last_logits[:, number_token_ids[1:]].to(torch.float32)
            outs.append(slice3.cpu())

    if not outs:
        return np.empty((0, 3), dtype=np.float32)
    return torch.cat(outs, dim=0).cpu().numpy()

def find_best_bias(logits3, y_true_123, lo=-1.0, hi=1.0, step=0.1):
    y_true = np.array([y-1 for y in y_true_123])
    grid = np.arange(lo, hi+1e-9, step)
    best, best_f1 = (0.0, 0.0, 0.0), -1.0
    for b0 in grid:
        for b1 in grid:
            for b2 in grid:
                pred = (logits3 + np.array([b0,b1,b2])[None,:]).argmax(-1)
                f1 = f1_score(y_true, pred, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1, best = f1, (float(b0), float(b1), float(b2))
    return best, best_f1


calib_df = train_df.sample(frac=0.10, random_state=3407).reset_index(drop=True)

calib_texts = []
for i in range(len(calib_df)):
    calib_texts.append(inference_prompt_template.format(
        calib_df["context"].iloc[i],
        calib_df["prompt"].iloc[i],
        calib_df["response"].iloc[i],
    ))

calib_logits3 = collect_slice3_logits(model, tokenizer, calib_texts, number_token_ids,
                                      max_len=max_seq_length, bs=8, device=model.device)
calib_y = calib_df["label"].apply(to_123).tolist()

best_bias_3, f1c = find_best_bias(calib_logits3, calib_y, lo=-1.0, hi=1.0, step=0.1)
print("[CALIB] bias =", best_bias_3, "| calib F1 =", f1c)

bias4 = torch.tensor([0.0, best_bias_3[0], best_bias_3[1], best_bias_3[2]],
                     device=model.device, dtype=torch.float32)


def formatting_prompts_func(dataset_):
    texts = []
    for i in range(len(dataset_)):
        context_ = dataset_['context'].iloc[i]
        prompt_ = dataset_['prompt'].iloc[i]
        response_ = dataset_['response'].iloc[i]

        text = inference_prompt_template.format(context_, prompt_, response_)
        texts.append(text)
    return texts

test_df['text'] = formatting_prompts_func(test_df)
test_df['token_length'] = test_df['text'].apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
test_df_sorted = test_df.sort_values(by='token_length').reset_index(drop=True)
display = 10
batch_size = 2
device = model.device
correct = 0
results = []

id2label = {
    1: "no",
    2: "intrinsic",
    3: "extrinsic"
}

with torch.inference_mode():
    for i in tqdm(range(0, len(test_df_sorted), batch_size), desc="Inference"):
        batch = test_df_sorted.iloc[i:i+batch_size]
        prompts = [text for text in batch['text']]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_seq_length
        ).to(device)

        logits = model(**inputs).logits
        last_idxs = inputs.attention_mask.sum(1) - 1
        last_logits = logits[torch.arange(len(batch)), last_idxs, :]

        logits_slice = last_logits[:, number_token_ids]
        logits_slice = logits_slice + bias4


        preds = torch.argmax(logits_slice, dim=-1).cpu().numpy()

        for j in range(len(batch)):
            results.append({
                "id": batch['id'].iloc[j],
                "predict_label": id2label.get(int(preds[j]), "no")
            })

df = pd.DataFrame(results)