from models.bert.dataset import load_data, prepare_datasets, tokenize_data
from models.bert.train import train_model

if __name__ == "__main__":
    npz_path = "dataset/HDFSv1/HDFS.npz"
    x_data_str, y_data = load_data(npz_path)
    train_texts, val_texts, train_labels, val_labels = prepare_datasets(x_data_str, y_data)
    train_encodings, val_encodings = tokenize_data(train_texts, val_texts)
    train_model(train_encodings, train_labels, val_encodings, val_labels)

