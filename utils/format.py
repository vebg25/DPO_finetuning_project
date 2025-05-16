def chatml_format(example, tokenizer):
    system = ""
    if len(example['system']) > 0:
        system_msg = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([system_msg], tokenize=False)

    user_msg = {"role": "user", "content": example['question']}
    prompt = tokenizer.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)

    return {
        "prompt": system + prompt,
        "chosen": example['chosen'] + "<|im_end|>\n",
        "rejected": example['rejected'] + "<|im_end|>\n",
    }
