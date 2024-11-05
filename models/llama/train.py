import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from models.llama.model import get_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class PromptCompletionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # Labels are used as target responses for each prompt
        input_length = self.encodings['input_ids'][idx].size(0)
        label = torch.tensor([self.labels[idx]] * input_length, dtype=torch.long)

        item['labels'] = label
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def prepare_labels(completions):
    labels = [1 if completion == "Label: Anomaly" else 0 for completion in completions]
    return labels

def train_model(train_prompts, train_completions, val_prompts, val_completions, train_encodings, val_encodings):
    # Convert the completion labels into binary labels
    train_labels = [1 if completion == "Label: Anomaly" else 0 for completion in train_completions]
    val_labels = [1 if completion == "Label: Anomaly" else 0 for completion in val_completions]

    # Prepare the dataset objects for training and validation
    train_dataset = PromptCompletionDataset(train_encodings, train_labels)
    val_dataset = PromptCompletionDataset(val_encodings, val_labels)
    
    model = get_model()
    print("Loaded LLaMA..")

    # Check CUDA version
    print(torch.version.cuda)
    print(torch.cuda.is_available)
    # Check if CUDA is available and move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    training_args = TrainingArguments(
        output_dir='./results_llama',
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir='./logs_llama',
        logging_steps=10,
        fp16=True if torch.cuda.is_available() else False,
        max_grad_norm=1.0,  
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("Cleaning cache..")
    torch.cuda.empty_cache()
    print("Starting training LLaMA..")
    trainer.train()

    print("Fine Tuning Finished")
    save_dir = "/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/result/llama_sft"
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

