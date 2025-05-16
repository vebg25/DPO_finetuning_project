from datasets import load_dataset

def load_and_prepare_dataset(tokenizer):
    from utils.format import chatml_format

    dataset = load_dataset("Intel/orca_dpo_pairs")['train']
    formatted = dataset.map(lambda x: chatml_format(x, tokenizer), remove_columns=dataset.column_names)
    split = formatted.train_test_split(test_size=0.1, seed=42)
    return split['train'], split['test']
