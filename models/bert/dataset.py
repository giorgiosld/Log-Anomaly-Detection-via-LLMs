import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    x_data, y_data = data['x_data'], data['y_data']
    x_data_str = [' '.join(events) for events in x_data]
    return x_data_str, y_data

def prepare_datasets(x_data_str, y_data):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        x_data_str, y_data, test_size=0.2, random_state=42)
    return train_texts, val_texts, train_labels, val_labels

def tokenize_data(train_texts, val_texts, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    return train_encodings, val_encodings

