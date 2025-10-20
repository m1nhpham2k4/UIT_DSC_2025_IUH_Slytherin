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

## Kiến trúc thư mục
```
UIT_DSC_2025_IUH_SLYTHERIN/
│
├── data/                         # 📂 Dữ liệu gốc dùng cho huấn luyện và kiểm thử
│   ├── test/                     # 📁 Tập dữ liệu kiểm thử
│   │   ├── vihallu-private-test.csv
│   │   └── vihallu-public-test.csv
│   └── train/                    # 📁 Tập dữ liệu huấn luyện
│       └── vihallu-train.csv
│
├── src/                          # 📂 Mã nguồn chính của dự án
│   ├── data_utils/               # 🧩 Tiện ích xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── collator.py           # Định nghĩa hàm collate_fn cho DataLoader
│   │   ├── dataset_builder.py    # Xây dựng Dataset từ file CSV
│   │   └── preprocess.py         # Hàm tiền xử lý dữ liệu (chuẩn hóa, làm sạch, tokenize,…)
│   │
│   ├── models/                   # 🧠 Các mô hình hoặc module liên quan đến LLM
│   │   ├── __init__.py
│   │   ├── lm_head_utils.py      # Tiện ích cho phần Linear head của mô hình ngôn ngữ
│   │   └── qwen3_instruct.py     # Cấu hình hoặc triển khai mô hình Qwen3-Instruct
│   │
│   ├── prompt/                   # 💬 Xử lý prompt cho huấn luyện/inference
│   │   ├── __init__.py
│   │   ├── base_prompt.py        # Cấu trúc prompt cơ bản
│   │   └── format_prompt.py      # Định dạng và xây dựng prompt cho từng nhiệm vụ
│   │
│   └── tasks/                    # 🚀 Các tác vụ chính của project
│       ├── __init__.py
│       ├── inference.py          # Chạy suy luận (inference)
│       └── train.py              # Script huấn luyện mô hình
│
└── LICENSE
└── README.mb
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
