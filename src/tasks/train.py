# train.py

import os
import torch
from transformers import Trainer, TrainingArguments
from unsloth import FastLanguageModel
from peft import prepare_model_for_kbit_training

# Import các module nội bộ
from data_utils.dataset_builder import load_dataset
from data_utils.preprocess import get_tokenizer, tokenize_function
from data_utils.collator import DataCollatorForLastTokenLM
from models.lm_head_utils import build_number_token_mapping, trim_lm_head_to_numbers, restore_full_lm_head
from models.qwen3_instruct import load_model   # file chứa hàm load_model bạn viết


class Args:
    model_name = "unsloth/Qwen3-4B-Base"
    train_path = "./data/vihallu-train.csv"
    test_path = "./data/vihallu-private-test.csv"
    output_dir = "./checkpoints/lora-qwen3-hallucination"

    # LoRA
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0

    num_train_epochs = 1
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 8
    learning_rate = 1e-4
    max_seq_len = 3072
    seed = 3407

    load_in_8bit = False

def main():
    args = Args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("🔹 Loading model & tokenizer...")
    model, tokenizer = load_model(args)


    print("🔹 Loading datasets...")
    datasets = load_dataset(args.train_path, args.test_path)

    tokenized_datasets = {}
    tokenized_datasets["train"] = datasets["train"].map(
        lambda e: tokenize_function(e, tokenizer, max_length=args.max_seq_len),
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    tokenized_datasets["test"] = datasets["test"].map(
        lambda e: tokenize_function(e, tokenizer, max_length=args.max_seq_len),
        batched=True,
        remove_columns=datasets["test"].column_names,
    )

    print("🔹 Trimming lm_head...")
    number_token_ids = build_number_token_mapping(tokenizer, num_classes=3)
    lm_meta = trim_lm_head_to_numbers(model, number_token_ids)
    reverse_map = lm_meta["reverse_map"]

    data_collator = DataCollatorForLastTokenLM(
        tokenizer=tokenizer,
        reverse_map=reverse_map,
        mlm=False,
    )

    print("🔹 Preparing Trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        save_steps = 500,
        bf16=True,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()

    print("💾 Saving model...")
    restore_full_lm_head(model, lm_meta["old_size"], lm_meta["number_token_ids"])
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Training completed and model saved!")


if __name__ == "__main__":
    main()
