import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

def plot_training_metrics(metrics_callback, save_dir):
    """Plot training and evaluation metrics over time."""
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Evaluation Metrics')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        if metric in metrics_callback.training_metrics:
            ax.plot(metrics_callback.training_metrics[metric], label='Training')
        if metric in metrics_callback.evaluation_metrics:
            ax.plot(metrics_callback.evaluation_metrics[metric], label='Evaluation')
        ax.set_title(f'{metric.capitalize()} over Time')
        ax.set_xlabel('Evaluation Step')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics.png')
    plt.close()

def plot_confusion_matrix(trainer, eval_dataset, save_dir):
    """Plot confusion matrix from model predictions."""
    # Get predictions
    predictions = trainer.predict(eval_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids

    # Create confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()

def plot_roc_curve(trainer, eval_dataset, save_dir):
    """Plot ROC curve and calculate AUC."""
    predictions = trainer.predict(eval_dataset)
    probs = predictions.predictions[:, 1]  # Probability of positive class
    labels = predictions.label_ids

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/roc_curve.png')
    plt.close()

def save_metrics_to_file(metrics_callback, trainer, save_dir):
    """Save all metrics to JSON file."""
    final_eval = trainer.evaluate()
    
    all_metrics = {
        'training_history': metrics_callback.training_metrics,
        'evaluation_history': metrics_callback.evaluation_metrics,
        'final_evaluation': final_eval
    }
    
    with open(f'{save_dir}/all_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)

def evaluate_model(trainer, metrics_callback, eval_dataset, save_dir='./results_bert'):
    """Complete evaluation pipeline."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating evaluation metrics and plots...")
    
    # Generate all plots
    plot_training_metrics(metrics_callback, save_dir)
    plot_confusion_matrix(trainer, eval_dataset, save_dir)
    plot_roc_curve(trainer, eval_dataset, save_dir)
    
    # Save metrics
    save_metrics_to_file(metrics_callback, trainer, save_dir)
    
    # Print final evaluation metrics
    final_metrics = trainer.evaluate()
    print("\nFinal Evaluation Metrics:")
    print(json.dumps(final_metrics, indent=2))
