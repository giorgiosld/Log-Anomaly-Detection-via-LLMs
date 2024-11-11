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

'''
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    
    # Calculate metrics with focus on minority class
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    # Calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None
    )
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(labels, preds)
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "normal_f1": f1_per_class[0],
        "anomaly_f1": f1_per_class[1],
        "normal_precision": precision_per_class[0],
        "anomaly_precision": precision_per_class[1],
        "normal_recall": recall_per_class[0],
        "anomaly_recall": recall_per_class[1]
    }
'''

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

class F1EarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.01):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.metric_name = "eval_f1"

    def check_metric_value(self, args, state, control, metric_value):
        # Only stop if we've trained for at least 1 epoch
        if state.epoch < 1.0:
            return False
            
        if not self.best_metric:
            self.best_metric = metric_value
            return False
            
        if metric_value < self.best_metric + self.early_stopping_threshold:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.best_metric = metric_value
            
        if self.no_improvement_count >= self.early_stopping_patience:
            control.should_training_stop = True
        return False



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
    
    #weight_class_0 = total_samples / (2 * num_class_0)
    #weight_class_1 = total_samples / (2 * num_class_1)
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
        #warmup_steps=num_warmup_steps,
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
    '''training_args = TrainingArguments(
        output_dir='./results_bert',
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir='./logs_bert',
        logging_steps=10,
        fp16=True if torch.cuda.is_available() else False,
        max_grad_norm=1.0,
        load_best_model_at_end=True
    )'''
    
    class MetricsCallback(EarlyStoppingCallback):
        def __init__(self, early_stopping_patience=5, early_stopping_threshold=0.01):
            super().__init__(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
            self.training_metrics = {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
            self.evaluation_metrics = {'loss': [], 'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                step_metrics = {}
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        step_metrics[k] = v

                if 'loss' in logs:
                    if 'eval' in logs.get('step', ''):
                        for metric, value in step_metrics.items():
                            if metric.startswith('eval_') and metric[5:] in self.evaluation_metrics:
                                self.evaluation_metrics[metric[5:]].append(value)
                    else:
                        for metric, value in step_metrics.items():
                            if metric in self.training_metrics:
                                self.training_metrics[metric].append(value)

    # Create callbacks
    metrics_callback = MetricsCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.01
    )

    early_stopping = F1EarlyStoppingCallback(
        early_stopping_patience=5,       # Increased patience
        early_stopping_threshold=0.01    # Need 1% improvement in F1
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
    
    '''trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        callbacks=[metrics_callback]
    )'''

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
