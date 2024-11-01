import torch
from transformers import Trainer, TrainingArguments

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

def train_model(train_encodings, train_labels, val_encodings, val_labels):
    train_dataset = HDFSDataset(train_encodings, train_labels)
    val_dataset = HDFSDataset(val_encodings, val_labels)
    
    model = get_model()

    training_args = TrainingArguments(
        output_dir='./results_bert',
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs_bert',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

