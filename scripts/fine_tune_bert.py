from models.bert.dataset import load_data, prepare_datasets, tokenize_data
from models.bert.train import HDFSDataset, train_model
from models.bert.evaluation import evaluate_model

if __name__ == "__main__":
    print("Starting fine tuning BERT model..")
    npz_path = "dataset/HDFSv1/preprocessed/HDFS.npz"
    x_data_str, y_data = load_data(npz_path)
    print("Loaded data..")

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = prepare_datasets(x_data_str, y_data)

    train_encodings, val_encodings, test_encodings = tokenize_data(train_texts, val_texts, test_texts)
    print("Tokenized data..")
    
    trainer, metrics_callback = train_model(train_encodings, train_labels, val_encodings, val_labels)
    
    # Create datasets for evaluation
    val_dataset = HDFSDataset(val_encodings, val_labels)
    test_dataset = HDFSDataset(test_encodings, test_labels)

    # Evaluate on validation set
    print("\nValidation Set Evaluation:")
    evaluate_model(trainer, metrics_callback, val_dataset, save_dir='./results_bert/validation')

    # Final evaluation on test set
    print("\nTest Set Evaluation (Final Results):")
    evaluate_model(trainer, metrics_callback, test_dataset, save_dir='./results_bert/test')
