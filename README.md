# 🚀 DPO Fine-Tuning on Qwen2.5-7B-Instruct
### Fine-tune the powerful Qwen2.5-7B-Instruct model using Direct Preference Optimization (DPO) on the Intel/orca_dpo_pairs dataset — with support for LoRA, 4-bit quantization, and W&B logging.

### 🧠 Overview
### This project modularizes the full DPO training pipeline using:

### 🤗 Hugging Face Transformers, PEFT, and TRL

### 📦 4-bit quantization using bitsandbytes

### 🔧 LoRA for efficient parameter tuning

### 📊 W&B for experiment tracking

### 🧩 Environment-variable-based configuration

### ✅ Designed to run on any machine (Kaggle-independent)

```bash
dpo_training_project/
├── config/                # Auth and model configuration
├── data/                  # Dataset loading and preprocessing
├── training/              # Trainer setup
├── utils/                 # Formatting logic (ChatML)
├── main.py                # Main training & merge script
├── .env                   # Your API keys (not committed)
├── requirements.txt       # Dependencies
└── README.md              # You're here

```
