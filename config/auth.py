import os
import wandb
from dotenv import load_dotenv

def authenticate():
    load_dotenv()

    wandb_api_key = os.getenv("WANDB_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    if not wandb_api_key or not hf_token:
        raise ValueError("WANDB_API_KEY or HF_TOKEN not set in .env")

    wandb.login(key=wandb_api_key)
    return hf_token
