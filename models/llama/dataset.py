import json
from sklearn.model_selection import train_test_split
from transformers import LlamaTokenizer, AutoTokenizer

# Load the prompt-completion data from JSONL
def load_prompt_completion_data(jsonl_path):
    prompts = []
    completions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['prompt'])
            completions.append(data['completion'])
    return prompts, completions

# Prepare datasets for training and validation
def prepare_datasets(prompts, completions):
    train_prompts, val_prompts, train_completions, val_completions = train_test_split(
        prompts, completions, test_size=0.2, random_state=42)
    return train_prompts, val_prompts, train_completions, val_completions

# Tokenize prompt-completion pairs using LLaMA tokenizer
def tokenize_data(train_prompts, train_completions, val_prompts, val_completions, max_length=64):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", token="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")
 
    tokenizer.pad_token = tokenizer.eos_token
 
    # Tokenize prompts (inputs) only
    train_encodings = tokenizer(
        train_prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
 
    val_encodings = tokenizer(
        val_prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return train_encodings, val_encodings

