import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, tokenizer, test_dataset):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for item in test_dataset:
            inputs = {
                'input_ids': item['input_ids'].unsqueeze(0).cuda(),
                'attention_mask': item['attention_mask'].unsqueeze(0).cuda()
            }
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,  
                temperature=1.0,
            )
            
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            true_text = tokenizer.decode(item['labels'], skip_special_tokens=True)
            
            # Extract labels
            pred_label = 1 if "Anomaly" in pred_text else 0
            true_label = 1 if "Anomaly" in true_text else 0
            
            predictions.append(pred_label)
            true_labels.append(true_label)
    
    return np.array(predictions), np.array(true_labels)

def plot_confusion_matrix(true_labels, predictions, output_path):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(true_labels, predictions, output_path):
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
