from models.bert.dataset import load_data, prepare_datasets, tokenize_data
from models.bert.train import HDFSDataset, train_model
from models.bert.evaluation import evaluate_model

if __name__ == "__main__":
    print("Starting fine tuning BERT model..")
    npz_path = "dataset/HDFSv1/preprocessed/HDFS.npz"
    x_data_str, y_data = load_data(npz_path)
    print("Loaded data..")

    train_texts, val_texts, train_labels, val_labels = prepare_datasets(x_data_str, y_data)
    print("Prepared datasets..")
    print("\nSplit sizes:")
    print(f"Training set: {len(train_texts)} samples")
    print(f"Validation set: {len(val_texts)} samples")

    train_encodings, val_encodings = tokenize_data(train_texts, val_texts)
    print("Tokenized data..")
    
    trainer, metrics_callback = train_model(train_encodings, train_labels, val_encodings, val_labels)
    
    val_dataset = HDFSDataset(val_encodings, val_labels)
    evaluate_model(trainer, metrics_callback, val_dataset)
