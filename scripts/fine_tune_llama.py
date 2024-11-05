from models.llama.dataset import load_prompt_completion_data, prepare_datasets, tokenize_data
from models.llama.train import train_model

if __name__ == "__main__":
    print("Starting fine-tuning LLaMA model..")
    jsonl_path = "dataset/HDFSv1/preprocessed/prompt_completion_data.jsonl"
    prompts, completions = load_prompt_completion_data(jsonl_path)
    print("Loaded prompt-completion data..")
    train_prompts, val_prompts, train_completions, val_completions = prepare_datasets(prompts, completions)
    print("Prepared datasets..")
    train_encodings, val_encodings = tokenize_data(train_prompts, train_completions, val_prompts, val_completions)
    print("Tokenized data..")
    train_model(train_prompts, train_completions, val_prompts, val_completions, train_encodings, val_encodings)


