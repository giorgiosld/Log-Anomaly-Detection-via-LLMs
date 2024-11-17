import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.llama.dataset import load_and_split_data, LogTraceDataset
from models.llama.model import load_model_and_tokenizer
from models.llama.train import create_trainer
from models.llama.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve

def main():
    # Configuration
    data_path = "dataset/HDFSv1/preprocessed/prompt_completion_data.jsonl" 
    output_dir = "results_llama_2"
    model_name = "meta-llama/Llama-3.2-1B"
    hf_token = "your_huggingface_token"
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split data
    train_data, val_data, test_data = load_and_split_data(data_path)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, hf_token=hf_token)
    
    # Create datasets
    train_dataset = LogTraceDataset(train_data, tokenizer)
    val_dataset = LogTraceDataset(val_data, tokenizer)
    test_dataset = LogTraceDataset(test_data, tokenizer)
    
    # Create and start trainer
    trainer = create_trainer(model, tokenizer, train_dataset, val_dataset, output_dir)
    trainer.train()

    # Evaluate model
    predictions, true_labels = evaluate_model(model, tokenizer, test_dataset)
    
    # Generate evaluation plots
    plot_confusion_matrix(true_labels, predictions, f"{output_dir}/confusion_matrix.png")
    plot_roc_curve(true_labels, predictions, f"{output_dir}/roc_curve.png")

if __name__ == "__main__":
    main()
