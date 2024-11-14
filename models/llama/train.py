from transformers import Trainer, TrainingArguments
from models.llama.callbacks import MetricsCallback
import torch

def create_trainer(model, tokenizer, train_dataset, val_dataset, output_dir):
    
    
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8, # modified
        per_device_eval_batch_size=8, #modified from 4 to 8
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        warmup_steps=100,
        learning_rate=2e-5,
        
        # Modified precision handling
        bf16=True,  # Use bfloat16 instead of fp16
        bf16_full_eval=True,
        logging_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,  # Add this to prevent column removal
        # Add gradient clipping
        max_grad_norm=1.0,
        # Add optimizations for A100
        dataloader_num_workers=4,
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        optim="adamw_torch"
    )
    



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
  #return trainer, metrics_callback
    return trainer
