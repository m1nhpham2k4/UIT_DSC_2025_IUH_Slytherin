# 🐍 UIT_DSC_2025_IUH_Slytherin

---

## ⚠️ Note
Due to the use of **Large Language Models (LLMs)**, the output might slightly vary across runs (some predictions may differ by a few samples).

---

## 🧠 Overview

This project is a **Vietnamese Hallucination Detection System**, designed to classify whether a model-generated response is consistent, distorted, or hallucinated relative to its context.

The system is built upon the **Qwen3-4B-Base** model and fine-tuned using **LoRA (Low-Rank Adaptation)** to optimize memory usage and training speed.

A custom **`lm_head` modification strategy** is applied — only retaining weight parameters corresponding to label tokens (`1`, `2`, `3`) during training.  
This helps the model focus on **classification** rather than **text generation**.  
After fine-tuning, the full `lm_head` is restored for precise **inference**.

During inference, **dynamic bias calibration** is used to balance prediction probabilities among three label classes:

- **`class 1 – no`**: Response is fully consistent with the context.  
- **`class 2 – intrinsic`**: Response contradicts or distorts the context.  
- **`class 3 – extrinsic`**: Response adds external information not present in the context.

---

## Project Structure
```
UIT_DSC_2025_IUH_SLYTHERIN/
│
├── data/                         # 📂 Original datasets for training and evaluation
│   ├── test/                     # 📁 Test data
│   │   ├── vihallu-private-test.csv
│   │   └── vihallu-public-test.csv
│   └── train/                    # 📁 Training data
│       └── vihallu-train.csv
│
├── src/                          # 📂 Main source code
│   ├── data_utils/               # 🧩 Data processing utilities
│   │   ├── __init__.py
│   │   ├── collator.py           # Custom collate_fn for DataLoader
│   │   ├── dataset_builder.py    # Dataset builder for CSV files
│   │   └── preprocess.py         # Preprocessing (cleaning, normalization, tokenization,…)
│   │
│   ├── models/                   # 🧠 Model-related components
│   │   ├── __init__.py
│   │   ├── lm_head_utils.py      # Utilities for language model linear head manipulation
│   │   └── qwen3_instruct.py     # Qwen3-Instruct model configuration and setup
│   │
│   ├── prompt/                   # 💬 Prompt design and formatting
│   │   ├── __init__.py
│   │   ├── base_prompt.py        # Base prompt structure
│   │   └── format_prompt.py      # Prompt formatting for training/inference tasks
│   │
│   └── tasks/                    # 🚀 Core tasks (training & inference)
│       ├── __init__.py
│       ├── inference.py          # Run inference/prediction
│       └── train.py              # Model fine-tuning script
│
└── LICENSE
└── README.md
```

## ⚙️ Installation

### 1. System Requirements

- **Python** ≥ 3.10  
- **CUDA-compatible GPU** (≥ 32GB VRAM recommended)  
- Example hardware: NVIDIA RTX 5090 (32GB VRAM), 32GB RAM, 512GB SSD

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

```csv
id,context,prompt,response,label
1,"<context text>","<question>","<response>","no"
2,"<context text>","<question>","<response>","intrinsic"
2,"<context text>","<question>","<response>","extrinsic"
```
## 📊 Data Format

### Training Data (CSV)

**Required Columns:**
- `id`: Unique sample ID
- `context`: Background or reference text
- `prompt`: Instruction or question
- `response`: Response from csv
- `label`: Classification label(`no`, `intrinsic`, `extrinsic`)

## 🎛️ Configuration Parameters

### Training Parameters
| Parameter | Default | Description |
|-----------|----------|-------------|
| `--model_name` | `unsloth/Qwen3-4B-Base` | Base model |
| `--max_seq_len` | 3072 | Maximum sequence length |
| `--epochs` | 1 | Number of training epochs |
| `--per_device_train_batch_size` | 4 | Batch size per GPU |
| `--gradient_accumulation_steps` | 8 | Gradient accumulation steps |
| `--lr` | 1e-4 | Learning rate |
| `--save_steps` | 500 | Number of steps between checkpoint saves |

### LoRA Parameters

| Parameter | Default | Description |
|-----------|----------|-------------|
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha scaling factor |
| `--lora_dropout` | 0 | Dropout rate |
