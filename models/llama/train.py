from transformers import Trainer, TrainingArguments
from models.llama.callbacks import MetricsCallback
import torch

def create_trainer(model, tokenizer, train_dataset, val_dataset, output_dir):
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        warmup_steps=50,
        learning_rate=2e-5,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        dataloader_num_workers=6,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    return trainer
