from typing import List
import pandas as pd
from .base_prompt import PROMPT

def build_training_prompts(df: pd.DataFrame) -> List[str]:
    texts = []
    for i in range(len(df)):
        context_ = df['context'].iloc[i]
        prompt_ = df['prompt'].iloc[i]
        response_ = df['response'].iloc[i]
        label_ = df['label'].iloc[i]
        texts.append(PROMPT.format(context_, prompt_, response_, label_))

    return texts

def build_inference_prompts(df: pd.DataFrame) -> List[str]:
    inference_template = PROMPT.split("class {}")[0] + "class "
    texts = []
    for i in range(len(df)):
        context_ = df['context'].iloc[i]
        prompt_ = df['prompt'].iloc[i]
        response_ = df['response'].iloc[i]
        texts.append(inference_template.format(context_, prompt_, response_))
    return texts