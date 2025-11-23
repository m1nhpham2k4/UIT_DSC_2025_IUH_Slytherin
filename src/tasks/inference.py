import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.prompt.base_prompt import PROMPT

test_df = pd.read_csv("./data/test/vihallu-private-test.csv")
prompt_ = PROMPT
print(prompt_)