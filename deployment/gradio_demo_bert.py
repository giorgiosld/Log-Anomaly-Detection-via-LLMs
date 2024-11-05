import gradio as gr
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the model and tokenizer from the saved directory
model_directory = "/hpchome/giorgio24/Log-Anomaly-Detection-via-LLMs/result/bert_sft"
model = BertForSequenceClassification.from_pretrained(model_directory)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a function that will take an input log and predict whether it's normal or anomalous
def predict_log_classification(log_text):
    # Tokenize the input text
    inputs = tokenizer(log_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Make a prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Map prediction to human-readable label
    label = "Anomaly" if predicted_class == 1 else "Normal"
    return label

# Create a Gradio interface
interface = gr.Interface(
    fn=predict_log_classification,
    inputs="text",
    outputs="text",
    title="Log Anomaly Detection",
    description="Enter a log sequence to determine if it is anomalous or normal.",
    examples=[
        ["E5 E22 E11 E11 E9 E26 -- Example labeled as normal"],  
        ["E5 E5 E22 E7          -- Example labeled as anomaly"]
    ]
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch(share=True)

