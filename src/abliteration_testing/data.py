

from datasets import load_dataset

def reformat_texts(texts):
    return [[{"role": "user", "content": text}] for text in texts]

# Get harmful and harmless datasets
def get_harmful_instructions():
    dataset = load_dataset('mlabonne/harmful_behaviors')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

def get_harmless_instructions():
    dataset = load_dataset('mlabonne/harmless_alpaca')
    return reformat_texts(dataset['train']['text']), reformat_texts(dataset['test']['text'])

harmful_inst_train, harmful_inst_test = get_harmful_instructions()
harmless_inst_train, harmless_inst_test = get_harmless_instructions()


def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids

