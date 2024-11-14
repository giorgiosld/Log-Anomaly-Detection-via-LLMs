import json
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class LogTraceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        completion = item['completion']

        # Combine prompt and completion for training
        full_text = f"{prompt}\n{completion}"
        
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

def load_and_split_data(data_path, test_size=0.1, val_size=0.1):
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # First split: separate test set
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Second split: separate validation set from training set
    val_ratio = val_size / (1 - test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=val_ratio, random_state=42)
    
    return train_data, val_data, test_data
