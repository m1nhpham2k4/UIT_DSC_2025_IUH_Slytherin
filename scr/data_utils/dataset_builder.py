# data_utils/dataset_builder.py

import pandas as pd
from datasets import Dataset
from prompt.format_prompt import build_training_prompts, build_inference_prompts

def load_dataset(train_path: str, test_path: str):
    """
    Load dữ liệu train và test riêng biệt (không chia val).
    CSV phải có cột: context, prompt, response, label (train).
    
    Args:
        train_path (str): Đường dẫn file CSV tập huấn luyện.
        test_path (str): Đường dẫn file CSV tập kiểm thử.
    
    Returns:
        dict: {"train": Dataset, "test": Dataset}
    """
    # Đọc dữ liệu
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Gắn text prompt cho từng tập
    train_df["text"] = build_training_prompts(train_df)
    test_df["text"] = build_inference_prompts(test_df)
    
    # Chuyển sang HuggingFace Dataset
    datasets = {
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    }
    
    return datasets
