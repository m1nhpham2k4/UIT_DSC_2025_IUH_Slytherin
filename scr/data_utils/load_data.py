# import unsloth
import datasets
import pandas as pd
import numpy as np

class TrainDatasetBuilder:
    """
    Load, map and build Dataset from CSV for training for SFTTrainer 
    """
    def __init__(self, tokenizer, mock=False):
        self.tokenizer = tokenizer
        self.mock = mock

    def _validate_columns(self, df):
        required = ["id", "context", "prompt", "response"]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            
    