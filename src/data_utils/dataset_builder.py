# data_utils/dataset_builder.py

import pandas as pd
from datasets import Dataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from prompt.format_prompt import build_training_prompts, build_inference_prompts

# Ánh xạ nhãn text sang số
LABEL_MAP = {
    "no": 1,
    "intrinsic": 2,
    "extrinsic": 3
}

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
    
    # Chuẩn hóa nhãn về số
    if "label" in train_df.columns:
        train_df["label"] = train_df["label"].map(LABEL_MAP)
        if train_df["label"].isnull().any():
            raise ValueError("⚠️ Có nhãn không hợp lệ trong train.csv. Chỉ chấp nhận: 'no', 'intrinsic', 'extrinsic'.")
    
    if "label" in test_df.columns:
        test_df["label"] = test_df["label"].map(LABEL_MAP)
        if test_df["label"].isnull().any():
            raise ValueError("⚠️ Có nhãn không hợp lệ trong test_df.csv. Chỉ chấp nhận: 'no', 'intrinsic', 'extrinsic'.")

    # Tạo text prompt
    train_df["text"] = build_training_prompts(train_df)
    test_df["text"] = build_inference_prompts(test_df)

    # Tạo HuggingFace Dataset
    datasets = {
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    }
    
    return datasets
