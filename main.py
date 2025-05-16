import gc
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoConfig
)
from peft import PeftModel

from config.auth import authenticate
from config.model_config import get_bnb_config, get_lora_config
from data.dataset import load_and_prepare_dataset
from training.trainer import get_training_args, initialize_trainer

def main():
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    output_model_name = "DPO-Qwen2.5-7B-v1"

    # Authenticate and get Hugging Face token
    hf_token = authenticate()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load and format dataset
    train_ds, eval_ds = load_and_prepare_dataset(tokenizer)
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    # Load quantized base model
    config = AutoConfig.from_pretrained(base_model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )
    model.config.use_cache = False

    # Training setup
    args = get_training_args(output_model_name)
    peft_config = get_lora_config()

    trainer = initialize_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        args=args,
        peft_config=peft_config
    )

    # Train
    trainer.train()

    # Save adapter weights
    trainer.model.save_pretrained("final_checkpoint")
    tokenizer.save_pretrained("final_checkpoint")

    # Free memory
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Merge LoRA with base
    print("🔁 Reloading base model in FP16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        token=hf_token
    )
    print("🔗 Merging base model with LoRA weights...")
    model = PeftModel.from_pretrained(base_model, "final_checkpoint")
    model = model.merge_and_unload()

    print("💾 Saving merged full model...")
    model.save_pretrained(output_model_name)
    tokenizer.save_pretrained(output_model_name)

    # (Optional) Push to HF Hub
    print("🚀 Uploading to Hugging Face Hub...")
    model.push_to_hub(output_model_name, use_temp_dir=False, token=hf_token)
    tokenizer.push_to_hub(output_model_name, use_temp_dir=False, token=hf_token)

if __name__ == "__main__":
    main()
