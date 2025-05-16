from trl import DPOTrainer, DPOConfig

def get_training_args(output_dir):
    return DPOConfig(                      
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        max_steps=200,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        logging_steps=50,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        warmup_steps=50,
        bf16=False,
        fp16=True,
        eval_strategy="steps",
        eval_steps=50,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_rewards/chosen",
        greater_is_better=False,
        model_init_kwargs=None,
        max_prompt_length=768,
        max_length=1536,
        beta=0.2,
    )

def initialize_trainer(model, tokenizer, train_ds, eval_ds, args, peft_config):
    return DPOTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
