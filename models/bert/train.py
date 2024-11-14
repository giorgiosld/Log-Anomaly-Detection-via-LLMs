import torch
import numpy as np
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from models.bert.model import get_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class HDFSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

class MetricsCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.01):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )
        # Initialize with steps and epochs tracking
        self.training_metrics = {
            'steps': [], 'epochs': [], 'loss': [],
            'accuracy': [], 'f1': [], 'precision': [], 'recall': []
        }
        self.evaluation_metrics = {
            'steps': [], 'epochs': [], 'loss': [],
            'accuracy': [], 'f1': [], 'precision': [], 'recall': []
        }
        self.current_train_metrics = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics:
            step = state.global_step
            epoch = state.epoch
            
            self.evaluation_metrics['steps'].append(step)
            self.evaluation_metrics['epochs'].append(epoch)
            self.evaluation_metrics['loss'].append(metrics.get('eval_loss', 0))
            self.evaluation_metrics['accuracy'].append(metrics.get('eval_accuracy', 0))
            self.evaluation_metrics['f1'].append(metrics.get('eval_f1', 0))
            self.evaluation_metrics['precision'].append(metrics.get('eval_precision', 0))
            self.evaluation_metrics['recall'].append(metrics.get('eval_recall', 0))

    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step"""
        if self.current_train_metrics and state.global_step % args.logging_steps == 0:
            self.training_metrics['steps'].append(state.global_step)
            self.training_metrics['epochs'].append(state.epoch)
            self.training_metrics['loss'].append(self.current_train_metrics.get('loss', 0))
            self.training_metrics['accuracy'].append(self.current_train_metrics.get('accuracy', 0))
            self.training_metrics['f1'].append(self.current_train_metrics.get('f1', 0))
            self.training_metrics['precision'].append(self.current_train_metrics.get('precision', 0))
            self.training_metrics['recall'].append(self.current_train_metrics.get('recall', 0))

    def on_prediction_step(self, args, state, control, outputs=None, **kwargs):
        """Called after each prediction step during training"""
        if outputs and hasattr(outputs, 'metrics'):
            self.current_train_metrics = outputs.metrics

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Existing log handling"""
        if logs:
            # Update early stopping as before
            metric_value = logs.get("eval_f1")
            if metric_value:
                self.check_metric_value(args, state, control, metric_value)

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Get class weights tensor
        weight_tensor = torch.tensor(
            [self.class_weights[0], self.class_weights[1]],
            device=model.device,
            dtype=torch.float32
        )

        # Apply class weights to loss calculation
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def train_model(train_encodings, train_labels, val_encodings, val_labels):
    train_dataset = HDFSDataset(train_encodings, train_labels)
    val_dataset = HDFSDataset(val_encodings, val_labels)
    
    model = get_model()
    print("Loaded BERT..")

    # Check if CUDA is available and move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Calculate steps
    num_epochs = 5
    batch_size = 32
    num_training_steps = (len(train_dataset) // batch_size) * num_epochs
    num_warmup_steps = num_training_steps // 10  

    # Compute class weights to handle unbalanced dataset
    total_samples = len(train_labels)
    num_class_0 = sum(1 for label in train_labels if label == 0)
    num_class_1 = sum(1 for label in train_labels if label == 1)
    
    weight_class_0 = 1.0
    weight_class_1 = min(num_class_0 / num_class_1, 10.0)

    class_weights = {0: weight_class_0, 1: weight_class_1}
    print(f"\nClass weights: {class_weights}")

    training_args = TrainingArguments(
        output_dir='./results_bert',
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir='./logs_bert',
        logging_steps=10,
        fp16=True if torch.cuda.is_available() else False,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        gradient_checkpointing=True,  
        gradient_accumulation_steps=2,
        label_smoothing_factor=0.05,
        save_total_limit=3,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        seed=42,                 
        dataloader_drop_last=False,
        dataloader_num_workers=4,
    )
    
    # Create callbacks
    metrics_callback = MetricsCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.01
    )

    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback] 
    )

    print("\nTraining configuration:")
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(val_dataset)}")
    print(f"Number of training steps: {num_training_steps}")
    print(f"Number of warmup steps: {num_warmup_steps}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}\n")

    print("Starting training BERT..")
    trainer.train()

    # Print final evaluation
    final_metrics = trainer.evaluate()
    print("\nFinal Evaluation Metrics:")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")

    print("Fine Tuning finished")
    save_dir = "/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/result/bert_sft"
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    return trainer, metrics_callback
