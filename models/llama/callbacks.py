from transformers import TrainerCallback
import json

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.training_metrics = {
            "loss": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": []
        }
        self.evaluation_metrics = {
            "loss": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": []
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Training metrics
            if "loss" in logs:
                self.training_metrics["loss"].append(round(logs["loss"], 4))
            
            # Evaluation metrics
            if "eval_loss" in logs:
                self.evaluation_metrics["loss"].append(round(logs["eval_loss"], 4))
                if "eval_accuracy" in logs:
                    self.evaluation_metrics["accuracy"].append(round(logs["eval_accuracy"], 4))
                if "eval_f1" in logs:
                    self.evaluation_metrics["f1"].append(round(logs["eval_f1"], 4))
                if "eval_precision" in logs:
                    self.evaluation_metrics["precision"].append(round(logs["eval_precision"], 4))
                if "eval_recall" in logs:
                    self.evaluation_metrics["recall"].append(round(logs["eval_recall"], 4))

def save_metrics_to_file(metrics_callback, trainer, save_dir):
    """Save all metrics to JSON file."""
    import os
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get final evaluation
    final_eval = trainer.evaluate()
    
    # Round all final evaluation metrics
    final_eval = {k: round(v, 4) if isinstance(v, float) else v 
                 for k, v in final_eval.items()}
    
    all_metrics = {
        'training_history': metrics_callback.training_metrics,
        'evaluation_history': metrics_callback.evaluation_metrics,
        'final_evaluation': final_eval
    }
    
    with open(f'{save_dir}/all_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
