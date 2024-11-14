import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    x_data, y_data = data['x_data'], data['y_data']
    x_data_str = [' '.join(events) for events in x_data]
    
    # Print initial class distribution
    unique, counts = np.unique(y_data, return_counts=True)
    total = len(y_data)
    print("\nFull Dataset Class Distribution:")
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples ({count/total*100:.2f}%)")
    
    return x_data_str, y_data

def prepare_datasets(x_data_str, y_data):
    """
    Perform a three-way split: train, validation, and test.
    First splits out test set, then splits remaining data into train/val.
    """
    # First split: separate out test set (20% of total data)
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        x_data_str, y_data,
        test_size=0.2,
        random_state=42,
        stratify=y_data
    )
    
    # Second split: split remaining data into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_val_labels
    )
    
    # Print distribution for each split
    def print_distribution(labels, name):
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        print(f"\n{name} Set Distribution:")
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/total*100:.2f}%)")
    
    print_distribution(train_labels, "Training")
    print_distribution(val_labels, "Validation")
    print_distribution(test_labels, "Test")
    
    print("\nSplit Sizes:")
    print(f"Training set: {len(train_texts)} samples ({len(train_texts)/len(x_data_str)*100:.1f}%)")
    print(f"Validation set: {len(val_texts)} samples ({len(val_texts)/len(x_data_str)*100:.1f}%)")
    print(f"Test set: {len(test_texts)} samples ({len(test_texts)/len(x_data_str)*100:.1f}%)")
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def tokenize_data(train_texts, val_texts, test_texts, max_length=128):
    """
    Tokenize all three datasets with proper error handling and batch processing
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def safe_tokenize(texts):
        """Helper function to safely tokenize a batch of texts"""
        try:
            return tokenizer(
                list(texts),  # Ensure texts is a list
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors=None  # Return dictionary of lists
            )
        except Exception as e:
            print(f"Error during tokenization: {str(e)}")
            print(f"Sample text: {texts[0][:100]}...")  # Print sample for debugging
            raise
    
    print("Tokenizing training set...")
    train_encodings = safe_tokenize(train_texts)
    
    print("Tokenizing validation set...")
    val_encodings = safe_tokenize(val_texts)
    
    print("Tokenizing test set...")
    test_encodings = safe_tokenize(test_texts)
    
    return train_encodings, val_encodings, test_encodings
