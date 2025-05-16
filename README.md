# 🚀 DPO Fine-Tuning on Qwen2.5-7B-Instruct
### Fine-tune the powerful Qwen2.5-7B-Instruct model using Direct Preference Optimization (DPO) on the Intel/orca_dpo_pairs dataset — with support for LoRA, 4-bit quantization, and W&B logging.

### 🧠 Features Overview
| Feature                   | Description                                                |
| ------------------------- | ---------------------------------------------------------- |
| 🤗 Hugging Face Ecosystem | Uses Transformers, TRL, and PEFT for modern LLM finetuning |
| 📦 Quantization           | Optimized with 4-bit `bitsandbytes` inference and training |
| 🔧 Efficient Training     | Lightweight LoRA adaptation for cost-effective finetuning  |
| 📊 Experiment Tracking    | Integrated W\&B logging for robust model analysis          |
| 🔐 Secure Configuration   | `.env` based secret management for API tokens              |
| 🧩 Modular Codebase       | Clean, scalable, and reusable module structure             |
| ✅ Fully Offline Friendly  | Runs on any local machine — **no Kaggle dependency**       |

### Project Structure
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
### Clone this repository
```bash
git clone https://github.com/vebg25/DPO_finetuning_project.git
```
### Install the requirements.txt
```bash
pip install -r requirements.txt
```

### Put your credentials
```ini
WANDB_API_KEY=your_wandb_api_key
HF_TOKEN=your_huggingface_token
```

### Run the main app
```python
pyhton main.py
```
