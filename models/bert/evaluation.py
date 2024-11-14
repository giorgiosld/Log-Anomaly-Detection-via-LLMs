import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
import pandas as pd

def plot_training_metrics(metrics_callback, save_dir):
    """Plot training and evaluation metrics over time."""
    metrics = ['loss', 'accuracy', 'f1', 'precision', 'recall']
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Training and Evaluation Metrics', fontsize=16)
    
    for idx, metric in enumerate(metrics):
        if idx < 5:  # We have 5 plots (3x2 grid, last spot empty)
            ax = axes[idx // 2, idx % 2]
            
            if len(metrics_callback.evaluation_metrics['epochs']) > 0:
                if metric in metrics_callback.training_metrics and len(metrics_callback.training_metrics[metric]) > 0:
                    ax.plot(metrics_callback.training_metrics['epochs'], 
                           metrics_callback.training_metrics[metric], 
                           'o-', label=f'Training', color='blue', markersize=2)
                ax.plot(metrics_callback.evaluation_metrics['epochs'], 
                       metrics_callback.evaluation_metrics[metric], 
                       'o-', label=f'Validation', color='orange', markersize=2)
                
            ax.set_title(f'{metric.capitalize()} over Time', fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Format y-axis to 4 decimal places
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    axes[2, 1].remove()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(trainer, dataset, save_dir, dataset_name="validation"):
    """Plot confusion matrix with improved visualization."""
    predictions = trainer.predict(dataset)
    preds = predictions.predictions.argmax(-1)
    labels = dataset.labels
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    
    # Calculate metrics for title
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Plot with improved styling
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    
    plt.title(f'Confusion Matrix ({dataset_name} set)\n' +
             f'Accuracy: {accuracy:.4f} | F1: {f1:.4f}\n' +
             f'Precision: {precision:.4f} | Recall: {recall:.4f}',
             fontsize=12, pad=20)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm, accuracy, precision, recall, f1

def plot_roc_curve(trainer, dataset, save_dir, dataset_name="validation"):
    """Plot ROC curve with improved visualization."""
    predictions = trainer.predict(dataset)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    labels = dataset.labels
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random classifier')
    
    # Plot optimal threshold point
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ko',
             label=f'Optimal threshold ({optimal_threshold:.4f})')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve ({dataset_name} set)\nAUC = {roc_auc:.4f}',
             fontsize=14, pad=20)
    
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def evaluate_model(trainer, metrics_callback, dataset, save_dir='./results_bert', dataset_name="validation"):
    """Complete evaluation pipeline with improved metrics saving."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating evaluation metrics and plots for {dataset_name} set...")
    
    # Generate plots and collect metrics
    plot_training_metrics(metrics_callback, save_dir)
    cm, accuracy, precision, recall, f1 = plot_confusion_matrix(
        trainer, dataset, save_dir, dataset_name)
    roc_auc = plot_roc_curve(trainer, dataset, save_dir, dataset_name)
    
    # Collect detailed metrics
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "dataset": dataset_name,
        "accuracy": float(f"{accuracy:.4f}"),
        "precision": float(f"{precision:.4f}"),
        "recall": float(f"{recall:.4f}"),
        "f1": float(f"{f1:.4f}"),
        "auc": float(f"{roc_auc:.4f}"),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "class_metrics": {
            "normal": {
                "precision": float(f"{tn/(tn+fn):.4f}") if (tn+fn) > 0 else 0,
                "recall": float(f"{tn/(tn+fp):.4f}") if (tn+fp) > 0 else 0
            },
            "anomaly": {
                "precision": float(f"{tp/(tp+fp):.4f}") if (tp+fp) > 0 else 0,
                "recall": float(f"{tp/(tp+fn):.4f}") if (tp+fn) > 0 else 0
            }
        }
    }
    
    # Save metrics to file
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n{dataset_name} Set Metrics:")
    print(json.dumps(metrics, indent=2))
    
    return metrics
